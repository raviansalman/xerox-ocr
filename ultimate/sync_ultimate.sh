#!/usr/bin/env bash
# Rsync ultimate/ → remote host(s), restart API container(s), curl /health.
#
# Usage:
#   ./sync_ultimate.sh                    # both servers, full tree
#   ./sync_ultimate.sh search             # search only
#   ./sync_ultimate.sh processing         # processing only
#   ./sync_ultimate.sh both               # explicit both
#   ./sync_ultimate.sh search ultimate/src/foo.py   # one or more files → that role only
#
# Role can also be set with ULTIMATE_SYNC_ROLE=search|processing|both (overridden by first arg).
#
# Environment (staging defaults; set ULTIMATE_* for production):
#   ULTIMATE_DEPLOY_PROFILE       staging | production
#   ULTIMATE_REMOTE_DIR           path to ultimate/ on remote (shared default)
#   ULTIMATE_SEARCH_SSH / ULTIMATE_PROCESSING_SSH
#   ULTIMATE_SEARCH_CONTAINER / ULTIMATE_PROCESSING_CONTAINER
#   ULTIMATE_HEALTH_URL (search) / ULTIMATE_PROCESSING_HEALTH_URL (processing)
#   ULTIMATE_IDENTITY_FILE        SSH key
#   ULTIMATE_DOCKER_CMD           optional global override; else per-role staging defaults
#   ULTIMATE_RSYNC_OVERWRITE_ENV  1 = include docker.staging.env / docker.production.env (docker.env is never rsynced)
#   ULTIMATE_HEALTH_SSH_FALLBACK_URL  if public :8000 is blocked, health is checked via SSH to this URL (default http://127.0.0.1:8000/health)
#
# Legacy: STAGING_SEARCH_SSH, STAGING_PROCESSING_SSH, STAGING_REMOTE_ULTIMATE, etc.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULTIMATE_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "$ULTIMATE_DIR/.." && pwd)"

ULTIMATE_DEPLOY_PROFILE="${ULTIMATE_DEPLOY_PROFILE:-staging}"
ULTIMATE_IDENTITY_FILE="${ULTIMATE_IDENTITY_FILE:-${STAGING_IDENTITY_FILE:-${HOME}/.ssh/id_rsa_invozone}}"
ULTIMATE_REMOTE_DIR="${ULTIMATE_REMOTE_DIR:-${STAGING_REMOTE_ULTIMATE:-}}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; exit 1; }

# --- parse role + file args -------------------------------------------------
ROLE="${ULTIMATE_SYNC_ROLE:-}"
FILE_ARGS=()
if [[ $# -gt 0 && "$1" =~ ^(search|processing|both)$ ]]; then
  ROLE="$1"
  shift
fi
ROLE="${ROLE:-both}"
FILE_ARGS=("$@")

case "$ROLE" in
  search|processing|both) ;;
  *) die "Invalid role '$ROLE' (use search, processing, both, or ULTIMATE_SYNC_ROLE)" ;;
esac

SSH_OPTS=(-o "StrictHostKeyChecking=accept-new" -o "ConnectTimeout=15" -i "$ULTIMATE_IDENTITY_FILE")
RSYNC_SSH="ssh"
for opt in "${SSH_OPTS[@]}"; do
  RSYNC_SSH+=" $(printf '%q' "$opt")"
done

EXCL=(
  --exclude=.git
  --exclude=__pycache__
  --exclude='*.pyc'
  --exclude=temp_uploads
  --exclude=logs
  --exclude=data
  --exclude=milvus
  --exclude=local_documents
  --exclude=node_modules
  --exclude=test_files
  --exclude='.env.*'
  --exclude='*.log'
)
# Never rsync legacy docker.env (canonical: docker.staging.env / docker.production.env).
EXCL+=(
  --exclude=docker.env
  --exclude=docker_byoc.env
  --exclude=docker_byoc_staging.env
)
if [[ "${ULTIMATE_RSYNC_OVERWRITE_ENV:-${STAGING_RSYNC_OVERWRITE_DOCKER_ENV:-0}}" != "1" ]]; then
  EXCL+=(
    --exclude=docker.staging.env
    --exclude=docker.production.env
  )
fi

ssh_orch() { ssh "${SSH_OPTS[@]}" "$@"; }

# Sets: _SSH _CONTAINER _HEALTH _DOCKER (per target)
resolve_one_target() {
  local r="$1"
  if [[ "$ULTIMATE_DEPLOY_PROFILE" == "production" ]]; then
    : "${ULTIMATE_REMOTE_DIR:?Set ULTIMATE_REMOTE_DIR for production}"
  else
    : "${ULTIMATE_REMOTE_DIR:=/home/ubuntu/ultimate-staging-deploy/ultimate}"
  fi

  case "$r" in
    search)
      _SSH="${ULTIMATE_SEARCH_SSH:-${STAGING_SEARCH_SSH:-}}"
      _CONTAINER="${ULTIMATE_SEARCH_CONTAINER:-${STAGING_SEARCH_CONTAINER:-}}"
      _HEALTH="${ULTIMATE_HEALTH_URL:-${STAGING_HEALTH_URL:-}}"
      if [[ "$ULTIMATE_DEPLOY_PROFILE" == "production" ]]; then
        : "${_SSH:?Set ULTIMATE_SEARCH_SSH for production}"
        : "${_CONTAINER:?Set ULTIMATE_SEARCH_CONTAINER for production}"
        : "${_HEALTH:?Set ULTIMATE_HEALTH_URL for production}"
        _DOCKER="${ULTIMATE_DOCKER_CMD:-sudo docker}"
      else
        : "${_SSH:?Set ULTIMATE_*_SSH for this role}"
        # After compose up with container_name: ultimate-search-api (see docker-compose.search-only.yml).
        : "${_CONTAINER:=ultimate-search-api}"
        : "${_HEALTH:?Set the health URL for this role}"
        _DOCKER="${ULTIMATE_DOCKER_CMD:-docker}"
      fi
      ;;
    processing)
      _SSH="${ULTIMATE_PROCESSING_SSH:-${STAGING_PROCESSING_SSH:-}}"
      _CONTAINER="${ULTIMATE_PROCESSING_CONTAINER:-${STAGING_PROCESSING_CONTAINER:-}}"
      _HEALTH="${ULTIMATE_PROCESSING_HEALTH_URL:-${STAGING_PROCESSING_HEALTH_URL:-}}"
      if [[ "$ULTIMATE_DEPLOY_PROFILE" == "production" ]]; then
        : "${_SSH:?Set ULTIMATE_PROCESSING_SSH for production}"
        : "${_CONTAINER:?Set ULTIMATE_PROCESSING_CONTAINER for production}"
        : "${_HEALTH:?Set ULTIMATE_PROCESSING_HEALTH_URL for production}"
        _DOCKER="${ULTIMATE_DOCKER_CMD:-sudo docker}"
      else
        : "${_SSH:?Set ULTIMATE_*_SSH for this role}"
        : "${_CONTAINER:=ultimate-processing-api}"
        : "${_HEALTH:?Set the health URL for this role}"
        _DOCKER="${ULTIMATE_DOCKER_CMD:-sudo docker}"
      fi
      ;;
    *) die "internal: bad target $r" ;;
  esac
}

do_rsync_and_restart() {
  local r="$1"
  resolve_one_target "$r"

  if [[ ${#FILE_ARGS[@]} -eq 0 ]]; then
    if [[ "${ULTIMATE_RSYNC_OVERWRITE_ENV:-${STAGING_RSYNC_OVERWRITE_DOCKER_ENV:-0}}" != "1" ]]; then
      log "[$r] rsync: excluding env templates (ULTIMATE_RSYNC_OVERWRITE_ENV=1 to overwrite)"
    else
      log "[$r] WARN: ULTIMATE_RSYNC_OVERWRITE_ENV=1 — env files will be overwritten on server"
    fi
    log "[$r] rsync ultimate/ → ${_SSH}:${ULTIMATE_REMOTE_DIR}/ (profile=$ULTIMATE_DEPLOY_PROFILE)"
    ssh_orch "${_SSH}" "mkdir -p $(printf '%q' "$ULTIMATE_REMOTE_DIR")"
    rsync -az -e "$RSYNC_SSH" "${EXCL[@]}" "$ULTIMATE_DIR/" "${_SSH}:${ULTIMATE_REMOTE_DIR}/"
  else
    for f in "${FILE_ARGS[@]}"; do
      abs="$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
      [[ -f "$abs" ]] || die "Not a file: $f"
      bn="$(basename "$abs")"
      case "$bn" in
        docker.env|docker_byoc.env|docker_byoc_staging.env)
          die "Refusing to rsync ${bn} (use docker.staging.env / docker.production.env)"
          ;;
      esac
      if [[ "${ULTIMATE_RSYNC_OVERWRITE_ENV:-${STAGING_RSYNC_OVERWRITE_DOCKER_ENV:-0}}" != "1" ]]; then
        case "$bn" in
        docker.staging.env|docker.production.env)
          die "Refusing to rsync ${bn} without ULTIMATE_RSYNC_OVERWRITE_ENV=1"
          ;;
      esac
      fi
      rel="${abs#$REPO_ROOT/}"
      case "$rel" in
        ultimate/*) ;;
        *) die "File must live under repo ultimate/: $rel" ;;
      esac
      remote_path="${ULTIMATE_REMOTE_DIR}/${rel#ultimate/}"
      log "[$r] rsync → ${_SSH}:$remote_path"
      ssh_orch "${_SSH}" "mkdir -p $(printf '%q' "$(dirname "$remote_path")")"
      rsync -az -e "$RSYNC_SSH" "$abs" "${_SSH}:$remote_path"
    done
  fi

  log "[$r] ${_DOCKER} restart ${_CONTAINER}"
  local restart_ok=0
  if ssh_orch "${_SSH}" "${_DOCKER} restart $(printf '%q' "$_CONTAINER")" 2>/dev/null; then
    restart_ok=1
  else
    local alts=()
    case "$r" in
      search)
        alts=(ultimate-staging-search_ultimate-search_1 ultimate-search_ultimate-search_1)
        ;;
      processing)
        alts=(ultimate-processing_ultimate-search_1)
        ;;
    esac
    for alt in "${alts[@]}"; do
      [[ "$alt" == "$_CONTAINER" ]] && continue
      log "[$r] retry: ${alt}"
      if ssh_orch "${_SSH}" "${_DOCKER} restart $(printf '%q' "$alt")" 2>/dev/null; then
        _CONTAINER="$alt"
        restart_ok=1
        break
      fi
    done
  fi
  if [[ "$restart_ok" -ne 1 ]]; then
    die "[$r] docker restart failed (set ULTIMATE_SEARCH_CONTAINER or ULTIMATE_PROCESSING_CONTAINER if needed)"
  fi

  log "[$r] waiting for health: ${_HEALTH}"
  local ok=0
  # Public URL often blocked by security groups; fall back to curl on the host via SSH.
  local local_url="${ULTIMATE_HEALTH_SSH_FALLBACK_URL:-http://127.0.0.1:8000/health}"
  for _ in $(seq 1 30); do
    code="$(curl -sS -m 5 -o /dev/null -w '%{http_code}' "$_HEALTH" 2>/dev/null || true)"
    if [[ "$code" != "200" ]]; then
      code="$(ssh_orch "${_SSH}" "curl -sS -m 5 -o /dev/null -w '%{http_code}' $(printf '%q' "$local_url")" 2>/dev/null || true)"
    fi
    if [[ "$code" == "200" ]]; then
      log "[$r] OK HTTP $code"
      if out="$(curl -fsS -m 5 "$_HEALTH" 2>/dev/null)"; then
        echo "$out" | head -c 240
      else
        ssh_orch "${_SSH}" "curl -fsS -m 5 $(printf '%q' "$local_url")" 2>/dev/null | head -c 240
      fi
      echo ""
      ok=1
      break
    fi
    sleep 1
  done
  if [[ "$ok" -ne 1 ]]; then
    die "[$r] health check failed after 30s ($_HEALTH and SSH $local_url)"
  fi
}

if [[ "$ROLE" == "both" ]]; then
  do_rsync_and_restart processing
  do_rsync_and_restart search
else
  do_rsync_and_restart "$ROLE"
fi
