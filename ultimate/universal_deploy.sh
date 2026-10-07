#!/usr/bin/env bash
# =============================================================================
# universal_deploy.sh — processing + search (staging or production)
# =============================================================================
#
# ARCHITECTURE
#   Processing : Milvus, Redis, Celery, API (ultimate_ui), embedder.  Project: ultimate-processing
#   Search     : search_api + embedder-search → remote Milvus/Redis.   Project: ultimate-search
#
# USAGE (on host or laptop)
#   ULTIMATE_DEPLOY_PROFILE=staging|production  STAGING_ROLE=processing ./universal_deploy.sh
#   ULTIMATE_DEPLOY_PROFILE=staging             STAGING_ROLE=search     ./universal_deploy.sh
#   STAGING_ROLE=all  (orchestrator from laptop; set SSH vars below)
#
# ─── PROFILE & CONTAINER ENV FILE ───────────────────────────────────────────
#   ULTIMATE_DEPLOY_PROFILE   staging (default) | production
#   ULTIMATE_ENV_FILE         docker.staging.env | docker.production.env (defaults from profile)
#   Compose env_file:         ${ULTIMATE_ENV_FILE:-docker.staging.env} in docker-compose.*.yml
#   Do NOT put COMPOSE_PROJECT_NAME in those env files (host CLI only).
#
# ─── REQUIRED ───────────────────────────────────────────────────────────────
#   STAGING_ROLE              processing | search | all
#
# Orchestrator (STAGING_ROLE=all) — prefer ULTIMATE_* (STAGING_* still accepted):
#   ULTIMATE_PROCESSING_SSH   user@processing-host
#   ULTIMATE_SEARCH_SSH       user@search-host
#   ULTIMATE_REMOTE_DIR       same absolute path on BOTH hosts (e.g. /home/ubuntu/ultimate-deploy/ultimate)
#   ULTIMATE_BACKEND_HOST     processing IP/DNS as seen FROM search (Milvus :19530, Redis :6379)
#
# Single-host search:
#   MILVUS_HOST               processing host reachable from this machine
#   REDIS_URL                 redis://<host>:6379/0
#
# Optional (all roles):
#   STAGING_SCALE_PROFILE     fast | production | ultimate
#   ULTIMATE_SCALE_PROFILE    alias for STAGING_SCALE_PROFILE
#   STAGING_AUTO_PROFILE      1 (default): auto production when RAM≥28GB & cores≥8
#   STAGING_FORCE_PROFILE     XL | L | M | S
#
# Optional (orchestrator):
#   STAGING_SYNC              1 = rsync repo to both servers before deploy
#   STAGING_LOCAL_REPO_ROOT   repo root (parent of ultimate/) when SYNC=1
#   STAGING_IDENTITY_FILE     SSH key for ssh/rsync
#   STAGING_REDIS_PORT / STAGING_REDIS_DB / STAGING_SEARCH_REDIS_URL / STAGING_STRICT_HOST_KEY
#   ULTIMATE_RSYNC_OVERWRITE_ENV  1 = rsync docker.staging.env / docker.production.env (default: exclude; docker.env never rsynced)
#   STAGING_RSYNC_OVERWRITE_DOCKER_ENV  alias for ULTIMATE_RSYNC_OVERWRITE_ENV
#
# Quick code sync (one script: search, processing, or both):
#   ./sync_ultimate.sh
#   ./sync_ultimate.sh search | processing | both
#   (ULTIMATE_DEPLOY_PROFILE=production + set ULTIMATE_* SSH/containers/health URLs)
#
# Compose project names (explicit -p on every compose invocation):
#   ultimate-processing | ultimate-search
#
# Metadata cache (in docker.staging.env / docker.production.env):
#   METADATA_CACHE_DIR=/app/data/metadata_cache  (volume ./data:/app/data)
#
# Optional (deploy behavior):
#   STAGING_SKIP_BUILD        1 = skip --build (use existing images)
#   STAGING_NO_DOWN           1 = skip compose down before up (hot restart)
#   STAGING_PRUNE_IMAGES      0 = skip Docker cleanup
#                             1 = default: buildx/builder cache prune -af + dangling images (image prune -f)
#                             2 = also remove all unused images (image prune -af) — run after compose down
#   STAGING_PRUNE_AUTO        0 = do not bump level when disk is low (default: 1 = auto)
#   STAGING_REQUIRE_SMOKE     1 = fail if HTTP smoke test cannot reach /health
#
# ─── NETWORK REQUIREMENTS ────────────────────────────────────────────────────
#
# Search server → Processing server:
#   TCP 19530 (Milvus gRPC)
#   TCP 6379  (Redis)
#   Ensure VPC / security groups allow these paths.
#
# ─── .dockerignore ───────────────────────────────────────────────────────────
#
# The repo-root .dockerignore (parent of ultimate/) MUST exist on deploy hosts.
# Without it, Docker build context tars Milvus data / temp_uploads → disk full.
# STAGING_SYNC=1 copies it automatically; otherwise place it manually.
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Deploy profile & env file loaded into containers (compose env_file) ────
ULTIMATE_DEPLOY_PROFILE="${ULTIMATE_DEPLOY_PROFILE:-staging}"
export ULTIMATE_DEPLOY_PROFILE
case "$ULTIMATE_DEPLOY_PROFILE" in
  production)
    export ULTIMATE_ENV_FILE="${ULTIMATE_ENV_FILE:-docker.production.env}"
    ;;
  staging|*)
    export ULTIMATE_ENV_FILE="${ULTIMATE_ENV_FILE:-docker.staging.env}"
    ;;
esac
export ULTIMATE_ENV_FILE

GEN_FILE="docker-compose.deploy.generated.yml"

# ---------------------------------------------------------------------------
# Utility: log with timestamp
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }
warn() { echo "[$(date '+%H:%M:%S')] WARN: $*" >&2; }
die() { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Docker command: use sudo if the current user cannot talk to the docker socket.
# Called only for processing/search roles (not 'all' orchestrator).
# ---------------------------------------------------------------------------
DOCKER_CMD="docker"
detect_docker_cmd() {
  if ! docker info >/dev/null 2>&1; then
    if sudo docker info >/dev/null 2>&1; then
      DOCKER_CMD="sudo docker"
      log "Using 'sudo docker' (current user not in docker group)"
    else
      die "Cannot run docker (not in docker group and sudo docker also fails)."
    fi
  fi
}

# ---------------------------------------------------------------------------
# Compose command detection (--compatibility for deploy.replicas on v1)
# Must use sudo if DOCKER_CMD uses sudo, because compose talks to the same socket.
# ---------------------------------------------------------------------------
detect_compose() {
  if [[ "$DOCKER_CMD" == "sudo docker" ]]; then
    # sudo is required — try sudo variants first
    if sudo docker compose version >/dev/null 2>&1; then
      COMPOSE=(sudo docker compose --compatibility)
    elif sudo docker-compose --version >/dev/null 2>&1; then
      COMPOSE=(sudo docker-compose --compatibility)
    else
      die "Neither 'sudo docker compose' nor 'sudo docker-compose' found."
    fi
  else
    # Non-sudo docker works
    if docker compose version >/dev/null 2>&1; then
      COMPOSE=(docker compose --compatibility)
    elif docker-compose --version >/dev/null 2>&1; then
      COMPOSE=(docker-compose --compatibility)
    else
      die "Neither 'docker compose' nor 'docker-compose' found."
    fi
  fi
  log "Compose command: ${COMPOSE[*]}"
}

# ---------------------------------------------------------------------------
# ORCHESTRATOR: deploy processing → wait → deploy search (STAGING_ROLE=all)
# ---------------------------------------------------------------------------
orchestrate_both_servers() {
  ULTIMATE_PROCESSING_SSH="${ULTIMATE_PROCESSING_SSH:-${STAGING_PROCESSING_SSH:-}}"
  ULTIMATE_SEARCH_SSH="${ULTIMATE_SEARCH_SSH:-${STAGING_SEARCH_SSH:-}}"
  ULTIMATE_REMOTE_DIR="${ULTIMATE_REMOTE_DIR:-${STAGING_REMOTE_ULTIMATE:-}}"
  ULTIMATE_BACKEND_HOST="${ULTIMATE_BACKEND_HOST:-${STAGING_BACKEND_HOST:-}}"
  : "${ULTIMATE_PROCESSING_SSH:?Set ULTIMATE_PROCESSING_SSH (or STAGING_PROCESSING_SSH)=user@processing-host}"
  : "${ULTIMATE_SEARCH_SSH:?Set ULTIMATE_SEARCH_SSH (or STAGING_SEARCH_SSH)=user@search-host}"
  : "${ULTIMATE_REMOTE_DIR:?Set ULTIMATE_REMOTE_DIR (or STAGING_REMOTE_ULTIMATE)=/path/to/ultimate on BOTH servers}"
  : "${ULTIMATE_BACKEND_HOST:?Set ULTIMATE_BACKEND_HOST (or STAGING_BACKEND_HOST)=processing IP/DNS FROM search}"

  local REDIS_PORT="${STAGING_REDIS_PORT:-6379}"
  local REDIS_DB="${STAGING_REDIS_DB:-0}"
  local RURL="${STAGING_SEARCH_REDIS_URL:-redis://${ULTIMATE_BACKEND_HOST}:${REDIS_PORT}/${REDIS_DB}}"

  # Build SSH options array — identity file used for BOTH ssh and rsync
  local SSH_OPTS=(-o "StrictHostKeyChecking=${STAGING_STRICT_HOST_KEY:-accept-new}" -o "ConnectTimeout=15")
  [[ -n "${STAGING_IDENTITY_FILE:-}" ]] && SSH_OPTS+=(-i "$STAGING_IDENTITY_FILE")

  _orch_ssh() { ssh "${SSH_OPTS[@]}" "$@"; }

  # rsync -e must pass the SAME key/options (plain rsync defaults to ssh without -i)
  local RSYNC_SSH="ssh"
  for opt in "${SSH_OPTS[@]}"; do
    RSYNC_SSH+=" $(printf '%q' "$opt")"
  done

  # ── SYNC ──────────────────────────────────────────────────────────────────
  if [[ "${STAGING_SYNC:-0}" == "1" ]]; then
    : "${STAGING_LOCAL_REPO_ROOT:?STAGING_SYNC=1 requires STAGING_LOCAL_REPO_ROOT (directory with ultimate/ and .dockerignore)}"
    local RROOT="${STAGING_LOCAL_REPO_ROOT%/}"
    local REMOTE_PARENT
    REMOTE_PARENT=$(dirname "$ULTIMATE_REMOTE_DIR")

    [[ -f "$RROOT/.dockerignore" ]] || warn "$RROOT/.dockerignore missing — Docker builds on remote may tar huge data dirs"
    [[ -d "$RROOT/ultimate" ]]      || die "$RROOT/ultimate directory not found — check STAGING_LOCAL_REPO_ROOT"

    # Exclude runtime data (redis/milvus files often root-owned → rsync permission errors)
    local EXCL=(
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
    # Never rsync legacy docker.env (use docker.staging.env / docker.production.env in repo only).
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
      log "rsync: excluding docker.staging.env / docker.production.env (ULTIMATE_RSYNC_OVERWRITE_ENV=1 to push them)"
    else
      warn "ULTIMATE_RSYNC_OVERWRITE_ENV=1 — syncing docker.staging.env / docker.production.env; server values may be overwritten."
    fi

    log "Syncing .dockerignore + ultimate/ → processing host..."
    _orch_ssh "${ULTIMATE_PROCESSING_SSH}" "mkdir -p $(printf '%q' "$ULTIMATE_REMOTE_DIR")"
    rsync -az --delete -e "$RSYNC_SSH" "${EXCL[@]}" "$RROOT/.dockerignore" "${ULTIMATE_PROCESSING_SSH}:$REMOTE_PARENT/"
    rsync -az --delete -e "$RSYNC_SSH" "${EXCL[@]}" "$RROOT/ultimate/" "${ULTIMATE_PROCESSING_SSH}:${ULTIMATE_REMOTE_DIR}/"

    log "Syncing .dockerignore + ultimate/ → search host..."
    _orch_ssh "${ULTIMATE_SEARCH_SSH}" "mkdir -p $(printf '%q' "$ULTIMATE_REMOTE_DIR")"
    rsync -az --delete -e "$RSYNC_SSH" "${EXCL[@]}" "$RROOT/.dockerignore" "${ULTIMATE_SEARCH_SSH}:$REMOTE_PARENT/"
    rsync -az --delete -e "$RSYNC_SSH" "${EXCL[@]}" "$RROOT/ultimate/" "${ULTIMATE_SEARCH_SSH}:${ULTIMATE_REMOTE_DIR}/"

    log "Sync complete."
  fi

  # ── Build remote env string ─────────────────────────────────────────────
  local REMOTE_EXTRA=""
  [[ -n "${STAGING_SCALE_PROFILE:-}" ]]   && REMOTE_EXTRA+="STAGING_SCALE_PROFILE=$(printf '%q' "$STAGING_SCALE_PROFILE") "
  [[ -n "${ULTIMATE_SCALE_PROFILE:-}" ]]  && REMOTE_EXTRA+="ULTIMATE_SCALE_PROFILE=$(printf '%q' "$ULTIMATE_SCALE_PROFILE") "
  [[ "${STAGING_SKIP_BUILD:-0}" == "1" ]] && REMOTE_EXTRA+="STAGING_SKIP_BUILD=1 "
  if [[ "${STAGING_PRUNE_IMAGES+x}" = x ]]; then
    REMOTE_EXTRA+="STAGING_PRUNE_IMAGES=$(printf '%q' "$STAGING_PRUNE_IMAGES") "
  fi
  if [[ "${STAGING_PRUNE_AUTO+x}" = x ]]; then
    REMOTE_EXTRA+="STAGING_PRUNE_AUTO=$(printf '%q' "$STAGING_PRUNE_AUTO") "
  fi
  [[ "${STAGING_NO_DOWN:-0}" == "1" ]]    && REMOTE_EXTRA+="STAGING_NO_DOWN=1 "
  [[ -n "${STAGING_FORCE_PROFILE:-}" ]]   && REMOTE_EXTRA+="STAGING_FORCE_PROFILE=$(printf '%q' "$STAGING_FORCE_PROFILE") "
  [[ "${STAGING_AUTO_PROFILE:-1}" != "1" ]] && REMOTE_EXTRA+="STAGING_AUTO_PROFILE=$(printf '%q' "$STAGING_AUTO_PROFILE") "
  [[ "${STAGING_REQUIRE_SMOKE:-0}" == "1" ]] && REMOTE_EXTRA+="STAGING_REQUIRE_SMOKE=1 "
  REMOTE_EXTRA+="ULTIMATE_DEPLOY_PROFILE=$(printf '%q' "$ULTIMATE_DEPLOY_PROFILE") "
  REMOTE_EXTRA+="ULTIMATE_ENV_FILE=$(printf '%q' "$ULTIMATE_ENV_FILE") "

  # ── STEP 1: Deploy processing ───────────────────────────────────────────
  echo ""
  log "========== 1/2: PROCESSING @ ${ULTIMATE_PROCESSING_SSH} (profile=${ULTIMATE_DEPLOY_PROFILE}) =========="
  _orch_ssh "${ULTIMATE_PROCESSING_SSH}" \
    "cd $(printf '%q' "$ULTIMATE_REMOTE_DIR") && chmod +x universal_deploy.sh 2>/dev/null; \
       ${REMOTE_EXTRA} STAGING_ROLE=processing bash ./universal_deploy.sh"

  # ── STEP 2: Wait for Milvus+Redis reachable from search ─────────────────
  echo ""
  log "========== Waiting: ${ULTIMATE_BACKEND_HOST}:19530 + :${REDIS_PORT} (from search host) =========="
  _orch_ssh "${ULTIMATE_SEARCH_SSH}" bash -s -- "$ULTIMATE_BACKEND_HOST" "$REDIS_PORT" <<'EOCHECK'
set -e
H="$1"; R="$2"
MAX_WAIT=72   # 72 × 5s = 6 minutes max
for i in $(seq 1 $MAX_WAIT); do
  if (echo >/dev/tcp/$H/19530) 2>/dev/null && (echo >/dev/tcp/$H/$R) 2>/dev/null; then
    echo "OK: backend ports reachable from search host (Milvus:19530 + Redis:$R)."
    exit 0
  fi
  echo "Waiting for $H:19530 and :$R ($i/$MAX_WAIT)..."
  sleep 5
done
echo "ERROR: search host CANNOT reach processing Milvus/Redis."
echo "  Check: ULTIMATE_BACKEND_HOST=$H, security groups, VPC peering."
exit 1
EOCHECK

  # ── STEP 3: Deploy search ───────────────────────────────────────────────
  echo ""
  log "========== 2/2: SEARCH @ ${ULTIMATE_SEARCH_SSH} (profile=${ULTIMATE_DEPLOY_PROFILE}) =========="
  _orch_ssh "${ULTIMATE_SEARCH_SSH}" \
    "cd $(printf '%q' "$ULTIMATE_REMOTE_DIR") && chmod +x universal_deploy.sh 2>/dev/null; \
       ${REMOTE_EXTRA} MILVUS_HOST=$(printf '%q' "$ULTIMATE_BACKEND_HOST") REDIS_URL=$(printf '%q' "$RURL") \
       STAGING_ROLE=search bash ./universal_deploy.sh"

  echo ""
  log "========== ORCHESTRATOR: DONE (processing → search) profile=${ULTIMATE_DEPLOY_PROFILE} =========="
}

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

STAGING_ROLE="${STAGING_ROLE:-}"

if [[ "$STAGING_ROLE" == "all" ]]; then
  orchestrate_both_servers
  exit 0
fi

if [[ "$STAGING_ROLE" != "processing" && "$STAGING_ROLE" != "search" ]]; then
  die "Set STAGING_ROLE=processing, search, or all.  Example: STAGING_ROLE=processing ./universal_deploy.sh"
fi

# ── Pre-flight checks ─────────────────────────────────────────────────────
[[ -f "$ULTIMATE_ENV_FILE" ]] || die "Missing $ULTIMATE_ENV_FILE in $(pwd) — copy from repo or set ULTIMATE_ENV_FILE."

PARENT_DIR="$(dirname "$SCRIPT_DIR")"
if [[ ! -f "$PARENT_DIR/.dockerignore" ]]; then
  warn "Missing $PARENT_DIR/.dockerignore — Docker build context will include data dirs and may fill disk."
  warn "Copy repo-root .dockerignore next to the 'ultimate' folder before building."
fi

# Source optional local overrides (e.g. docker.env.search.local for custom ports)
if [[ -f docker.env.search.local ]]; then
  log "Sourcing docker.env.search.local"
  set -a; source docker.env.search.local; set +a
fi

detect_docker_cmd
detect_compose

# Prefer BuildKit to avoid legacy builder deprecation warnings and speed up builds.
# Can be disabled explicitly via STAGING_USE_BUILDKIT=0.
# Guard: only enable when docker buildx is actually available on host.
if [[ "${STAGING_USE_BUILDKIT:-1}" == "1" ]]; then
  if eval "$DOCKER_CMD buildx version >/dev/null 2>&1"; then
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    # Plain progress keeps CI/SSH logs readable.
    export BUILDKIT_PROGRESS="${BUILDKIT_PROGRESS:-plain}"
    log "BuildKit enabled (DOCKER_BUILDKIT=1, COMPOSE_DOCKER_CLI_BUILD=1)"
  else
    warn "BuildKit requested but buildx is unavailable; using legacy builder on this host."
  fi
fi

# ── Hardware detection (same pattern as universal_deploy.sh) ──────────────
if [[ -f /proc/meminfo ]]; then
  RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
  RAM_GB=$((RAM_KB / 1024 / 1024))
  CORES=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || echo 4)
  DISK_MB=$(df -m . 2>/dev/null | awk 'NR==2 {print $4}' | awk '{print $1}')
  DISK_GB=$((DISK_MB / 1024))
elif [[ "$(uname)" == "Darwin" ]]; then
  RAM_GB=$(sysctl hw.memsize | awk '{print int($2/1073741824)}')
  CORES=$(sysctl -n hw.ncpu)
  DISK_MB=$(df -m . 2>/dev/null | awk 'NR==2 {print $4}' | awk '{print $1}')
  DISK_GB=$((DISK_MB / 1024))
else
  RAM_GB=16; CORES=8; DISK_GB=20
fi

# Profile alias: ULTIMATE_SCALE_PROFILE → STAGING_SCALE_PROFILE
if [[ -z "${STAGING_SCALE_PROFILE:-}" && -n "${ULTIMATE_SCALE_PROFILE:-}" ]]; then
  STAGING_SCALE_PROFILE="$ULTIMATE_SCALE_PROFILE"
fi

# Auto-detect: production for beefy boxes
STAGING_AUTO_PROFILE="${STAGING_AUTO_PROFILE:-1}"
if [[ -z "${STAGING_SCALE_PROFILE:-}" && "$STAGING_AUTO_PROFILE" == "1" && "$RAM_GB" -ge 28 && "$CORES" -ge 8 ]]; then
  STAGING_SCALE_PROFILE="production"
  log "Auto-selected STAGING_SCALE_PROFILE=production (${RAM_GB}GB RAM, ${CORES} cores)"
fi

echo ""
log "Hardware : ${RAM_GB}GB RAM | ${CORES} cores | ${DISK_GB:-?}GB free disk"
log "Deploy   : STAGING_ROLE=$STAGING_ROLE | profile=$ULTIMATE_DEPLOY_PROFILE | env_file=$ULTIMATE_ENV_FILE | STAGING_SCALE_PROFILE=${STAGING_SCALE_PROFILE:-auto-RAM}"

# Docker cleanup: default on so search/small disks do not fail mid-build (see staging_docker_reclaim_space).
STAGING_PRUNE_IMAGES="${STAGING_PRUNE_IMAGES:-1}"
STAGING_PRUNE_AUTO="${STAGING_PRUNE_AUTO:-1}"

if [[ "${DISK_GB:-0}" -lt 5 ]]; then
  warn "<5GB free disk — build may fail; STAGING_PRUNE_IMAGES=${STAGING_PRUNE_IMAGES} runs after compose down"
fi
if [[ "${STAGING_PRUNE_AUTO}" == "1" && "${STAGING_PRUNE_IMAGES}" != "0" && "${DISK_GB:-99}" -lt 12 ]]; then
  if [[ "${STAGING_PRUNE_IMAGES}" == "1" && "${STAGING_SKIP_BUILD:-0}" != "1" ]]; then
    STAGING_PRUNE_IMAGES=2
    warn "Low disk (${DISK_GB}GB free) — STAGING_PRUNE_IMAGES bumped to 2 (prune unused images + build cache)"
  fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# SCALE PROFILES — Processing Server
# ═══════════════════════════════════════════════════════════════════════════
#
# Tuning guide (compose-only, NO code changes):
#   • Milvus memory (MM): higher = more vectors cached in RAM → faster search.
#     XL=10G is safe for 500K+ chunks; S=4G for dev/small corpora.
#   • Embedder replicas (ER) × workers (EW): controls embedding throughput.
#     Each worker loads ~800MB model; 2 replicas × 4 workers = 8 concurrent.
#   • API workers (APIW): uvicorn workers for the processing API.
#     1 is fine unless you need concurrent /process-file uploads.
#   • Celery replicas: each replica = 1 concurrent task of that type.
#     PDF+OCR are the heaviest; scale those first.
#
# Memory budget (approximate, XL profile on 32GB host):
#   Milvus 10G + Redis 0.5G + API 2G + 2×Embedder 6G + 26 Celery ~20G = ~38G
#   → XL needs ≥32GB. L needs ≥18GB. M needs ≥12GB. S fits in 8GB.
# ═══════════════════════════════════════════════════════════════════════════

apply_ram_tier_processing() {
  local PF="${STAGING_FORCE_PROFILE:-}"
  if [[ -n "$PF" ]]; then
    PROFILE="$PF"
  elif [[ "$RAM_GB" -ge 28 ]]; then PROFILE="XL"
  elif [[ "$RAM_GB" -ge 18 ]]; then PROFILE="L"
  elif [[ "$RAM_GB" -ge 12 ]]; then PROFILE="M"
  else                               PROFILE="S"
  fi
  log "Processing RAM tier: $PROFILE"

  case "$PROFILE" in
    XL) # ≥28GB host — high throughput
      MM=10G; MC=4.0; REDIS_MEM=512M
      # EW=2: model ~1.5GB/worker; spike during large-batch inference can hit ~2.5GB/worker.
      # 5G limit gives safe headroom (2×1.5GB=3G steady, 2×2.5GB=5G peak).
      ER=2; EW=2; EM_MEM=5G; EM_CPU=3.0; EM_RES=2G
      APIW=4; API_MEM=2G; API_CPU=2.0; PROC_SRCH_MEM=2G
      R_PDF=4; R_OCR=4; R_SHEET=3; R_WORD=2; R_PPT=2; R_IMG=2
      M_PDF=1400M; C_PDF=1.2; M_OCR=1800M; C_OCR=1.1
      M_SHEET=2000M; C_SHEET=1.2; M_WORD=1200M; C_WORD=1.0
      M_PPT=1200M; C_PPT=1.0; M_IMG=1800M; C_IMG=1.0
      ;;
    L)  # ≥18GB host — balanced
      MM=8G; MC=4.0; REDIS_MEM=400M
      # EW=2: model ~1.5GB/worker; 4 workers OOM on 3G limit. 2×1.5G=3G fits.
      ER=1; EW=2; EM_MEM=3G; EM_CPU=3.0; EM_RES=2G
      APIW=3; API_MEM=2G; API_CPU=2.0; PROC_SRCH_MEM=1500M
      R_PDF=3; R_OCR=3; R_SHEET=2; R_WORD=2; R_PPT=2; R_IMG=2
      M_PDF=1300M; C_PDF=1.1; M_OCR=1700M; C_OCR=1.0
      M_SHEET=1800M; C_SHEET=1.1; M_WORD=1100M; C_WORD=0.9
      M_PPT=1100M; C_PPT=0.9; M_IMG=1600M; C_IMG=1.0
      ;;
    M)  # ≥12GB host — conservative
      MM=6G; MC=3.0; REDIS_MEM=300M
      # EW=1: on a 12GB host, even 2 workers (2×1.5G=3G) vs 2G limit OOMs.
      ER=1; EW=1; EM_MEM=2G; EM_CPU=2.0; EM_RES=1500M
      APIW=1; API_MEM=1500M; API_CPU=1.0; PROC_SRCH_MEM=1G
      R_PDF=2; R_OCR=2; R_SHEET=1; R_WORD=1; R_PPT=1; R_IMG=1
      M_PDF=1200M; C_PDF=1.0; M_OCR=1500M; C_OCR=1.0
      M_SHEET=1500M; C_SHEET=1.0; M_WORD=1000M; C_WORD=0.7
      M_PPT=1000M; C_PPT=0.7; M_IMG=1500M; C_IMG=1.0
      ;;
    S|*) # <12GB host — minimal
      MM=4G; MC=2.0; REDIS_MEM=256M
      # EW=1: on small hosts, model alone ~1.5G fills the 1.5G limit. 1 worker.
      ER=1; EW=1; EM_MEM=1500M; EM_CPU=1.5; EM_RES=1G
      APIW=1; API_MEM=1200M; API_CPU=0.8; PROC_SRCH_MEM=800M
      R_PDF=1; R_OCR=1; R_SHEET=1; R_WORD=1; R_PPT=1; R_IMG=1
      M_PDF=1000M; C_PDF=0.8; M_OCR=1200M; C_OCR=0.8
      M_SHEET=1200M; C_SHEET=0.8; M_WORD=800M; C_WORD=0.5
      M_PPT=800M; C_PPT=0.5; M_IMG=1200M; C_IMG=0.8
      ;;
  esac
}

write_processing_generated() {
  local SP="${STAGING_SCALE_PROFILE:-}"

  if [[ -n "$SP" ]]; then
    log "Processing STAGING_SCALE_PROFILE=$SP"
    case "$SP" in
      fast)
        # Max embedder concurrency → fast queue drain. Needs ≥24GB.
        # APIW=2: each worker loads ~500MB CrossEncoder; 4 workers OOMs on 2G.
        MM=10G; MC=4.0; REDIS_MEM=512M
        # EW=2 (not 4): sentence-transformers model = ~1.5GB per uvicorn worker.
        # 4 workers × 1.5GB = 6GB > 4GB limit → OOM kill loop. 2 workers = 3GB,
        # safely under the 4GB limit with 1GB headroom for batch spikes.
        # EM_RES=2G: pins model in RAM so kernel won't swap it under pressure.
        ER=2; EW=2; EM_MEM=6G; EM_CPU=4.0; EM_RES=2G
        # PROC_SRCH_MEM=2G: search on processing node is metadata-publish only,
        # not a real search worker — cap at 2G to leave room for embedders.
        APIW=2; API_MEM=4G; API_CPU=2.0; PROC_SRCH_MEM=2G
        R_PDF=5; R_OCR=5; R_SHEET=4; R_WORD=3; R_PPT=3; R_IMG=3
        M_PDF=1400M; C_PDF=1.2; M_OCR=1800M; C_OCR=1.1
        M_SHEET=2000M; C_SHEET=1.2; M_WORD=1200M; C_WORD=1.0
        M_PPT=1200M; C_PPT=1.0; M_IMG=1800M; C_IMG=1.0
        ;;
      production)
        # Balanced for 30-32GB class hosts — high throughput without OOM risk.
        # ER=2 (not 4): on a 30GB host, 4×4G embedders + Milvus 12G + workers
        # exhausts RAM and triggers OOM kills. 2 embedders handle steady ingestion
        # without blowing out the host.
        # EW=2 (not 4): sentence-transformers model = ~1.5GB per uvicorn worker.
        # 4 workers × 1.5GB = 6GB > 4GB limit → OOM kill loop. 2 workers = 3GB.
        # EM_RES=2G: pins model in RAM so kernel won't swap it under pressure.
        MM=12G; MC=4.0; REDIS_MEM=512M
        # EM_MEM=6G: 2 workers × ~1.5GB model = 3GB steady-state; during large-batch
        # inference one worker can spike to ~2.5GB, making combined usage ~4GB.
        # 4G limit triggers OOM kill → worker respawn death loop. 6G gives safe headroom.
        ER=2; EW=2; EM_MEM=6G; EM_CPU=4.0; EM_RES=2G
        # PROC_SRCH_MEM=2G: search on processing node is metadata-publish only,
        # not a real search worker — cap at 2G to leave room for embedders.
        APIW=2; API_MEM=4G; API_CPU=2.0; PROC_SRCH_MEM=2G
        R_PDF=6; R_OCR=6; R_SHEET=4; R_WORD=2; R_PPT=2; R_IMG=2
        M_PDF=1400M; C_PDF=1.2; M_OCR=1800M; C_OCR=1.1
        M_SHEET=2000M; C_SHEET=1.2; M_WORD=1200M; C_WORD=1.0
        M_PPT=1200M; C_PPT=1.0; M_IMG=1800M; C_IMG=1.0
        ;;
      ultimate)
        # Maximum replicas — needs ≥48GB host or will OOM.
        # On 30GB hosts use ER=3, EM_MEM=4G to avoid kernel OOM kills.
        # EW=2 (not 4): sentence-transformers model = ~1.5GB per uvicorn worker.
        # 4 workers × 1.5GB = 6GB > 4GB limit → OOM kill loop. 2 workers = 3GB.
        # EM_RES=2G: pins model in RAM so kernel won't swap it under pressure.
        MM=14G; MC=4.0; REDIS_MEM=512M
        ER=3; EW=2; EM_MEM=6G; EM_CPU=4.0; EM_RES=2G
        # PROC_SRCH_MEM=2G: search on processing node is metadata-publish only,
        # not a real search worker — cap at 2G to leave room for embedders.
        APIW=2; API_MEM=4G; API_CPU=2.0; PROC_SRCH_MEM=2G
        R_PDF=8; R_OCR=8; R_SHEET=5; R_WORD=3; R_PPT=3; R_IMG=3
        M_PDF=1400M; C_PDF=1.2; M_OCR=1800M; C_OCR=1.1
        M_SHEET=2000M; C_SHEET=1.2; M_WORD=1200M; C_WORD=1.0
        M_PPT=1200M; C_PPT=1.0; M_IMG=1800M; C_IMG=1.0
        ;;
      *)
        warn "Unknown STAGING_SCALE_PROFILE=$SP — falling back to RAM tier"
        apply_ram_tier_processing
        ;;
    esac
  else
    apply_ram_tier_processing
  fi

  # Expected container count: milvus + redis + API + (ER × embedder) + all celery replicas
  WANT_MIN_UP=$((3 + ER + R_PDF + R_OCR + R_SHEET + R_WORD + R_PPT + R_IMG))

  cat >"$GEN_FILE" <<EOF
# AUTO-GENERATED by universal_deploy.sh — do not commit
# Profile: ${SP:-ram-tier-$PROFILE} | Milvus ${MM} | Embedder ${ER}×${EW}w | API ${APIW}w
services:
  milvus:
    deploy:
      resources:
        limits:
          memory: ${MM}
          cpus: "${MC}"
        reservations:
          memory: 1G
  redis:
    deploy:
      resources:
        limits:
          memory: ${REDIS_MEM}
  ultimate-search:
    command:
      - uvicorn
      - ultimate_ui:app
      - --host
      - 0.0.0.0
      - --port
      - "8000"
      - --workers
      - "${APIW}"
      - --timeout-keep-alive
      - "600"
    environment:
      # Processing server: Milvus and Redis are LOCAL Docker services.
      # These lock in the correct hostnames so docker.env values cannot
      # be accidentally overridden by a stale MILVUS_HOST shell variable.
      MILVUS_HOST: "milvus"
      MILVUS_PORT: "19530"
      REDIS_HOST: "redis"
      REDIS_PORT: "6379"
      REDIS_DB: "0"
      REDIS_URL: "redis://redis:6379/0"
      METADATA_INDEX_PUBLISHER: "true"
      QUEUE_SHARD_COUNT: "0"
      EMBEDDER_URL: "http://ultimate-embedder:8080/embed"
    deploy:
      resources:
        limits:
          # PROC_SRCH_MEM (not API_MEM): the search container on the processing
          # node only publishes metadata to Redis — it does NOT serve real search
          # traffic. Capping at 2G frees memory for the embedders and prevents
          # it from competing with them under load.
          memory: ${PROC_SRCH_MEM}
          cpus: "${API_CPU}"
    healthcheck:
      interval: 15s
      timeout: 15s
      retries: 10
      start_period: 180s
  ultimate-embedder:
    command:
      - uvicorn
      - src.embedder_service:app
      - --host
      - 0.0.0.0
      - --port
      - "8080"
      - --workers
      - "${EW}"
      - --timeout-keep-alive
      - "600"
    # stop_grace_period: Give uvicorn workers time to finish current batches before
    # Docker SIGKILL's the master. Without this (default 10s), Docker kills the
    # master mid-computation, workers become orphaned host processes that consume
    # CPU/RAM outside Docker's control, causing memory pressure and slow processing.
    stop_grace_period: 120s
    deploy:
      replicas: ${ER}
      resources:
        limits:
          memory: ${EM_MEM}
          cpus: "${EM_CPU}"
        reservations:
          # EM_RES pins the embedding model in physical RAM. Without this the
          # Linux kernel will swap model weights to disk under memory pressure,
          # turning sub-second embeddings into 60-180s cold-reload stalls and
          # making files appear stuck. Set to ~half of EM_MEM so the kernel
          # knows this memory must stay resident even when other containers grow.
          memory: ${EM_RES}
    healthcheck:
      interval: 15s
      timeout: 15s
      retries: 10
      start_period: 120s
  ultimate-celery-pdf:
    deploy:
      replicas: ${R_PDF}
      resources:
        limits: { memory: ${M_PDF}, cpus: "${C_PDF}" }
  ultimate-celery-spreadsheet:
    deploy:
      replicas: ${R_SHEET}
      resources:
        limits: { memory: ${M_SHEET}, cpus: "${C_SHEET}" }
  ultimate-celery-word:
    deploy:
      replicas: ${R_WORD}
      resources:
        limits: { memory: ${M_WORD}, cpus: "${C_WORD}" }
  ultimate-celery-ppt:
    deploy:
      replicas: ${R_PPT}
      resources:
        limits: { memory: ${M_PPT}, cpus: "${C_PPT}" }
  ultimate-celery-image:
    deploy:
      replicas: ${R_IMG}
      resources:
        limits: { memory: ${M_IMG}, cpus: "${C_IMG}" }
  ultimate-celery-ocr:
    deploy:
      replicas: ${R_OCR}
      resources:
        limits: { memory: ${M_OCR}, cpus: "${C_OCR}" }
EOF
  export WANT_MIN_UP
}

# ═══════════════════════════════════════════════════════════════════════════
# SCALE PROFILES — Search Server
# ═══════════════════════════════════════════════════════════════════════════
#
# Tuning guide (compose-only, NO code changes):
#   • Search API workers (SW): each worker handles one request at a time.
#     More workers = more concurrent searches. 4–6 is a sweet spot.
#   • Search embedder workers (SEW): generates query embeddings.
#     Each worker loads ~800MB model. 4 workers on 16GB host is comfortable.
#   • API memory (SAPI_M): higher with more workers (each ~300MB baseline).
#   • Embedder memory (SEM): ~800MB per worker + safety margin.
#
# Sub-second latency tips (ops-only):
#   • Ensure low network latency to Milvus (same AZ / VPC peering).
#   • MILVUS_SEARCH_EF in docker.env controls HNSW recall vs speed.
#   • MetadataIndex disk cache (data/metadata_cache/) avoids Milvus round-trips.
# ═══════════════════════════════════════════════════════════════════════════

apply_ram_tier_search() {
  if [[ "$RAM_GB" -ge 14 ]]; then   PROFILE="XL"
  elif [[ "$RAM_GB" -ge 8 ]]; then  PROFILE="L"
  else                               PROFILE="S"
  fi
  log "Search RAM tier: $PROFILE"
  case "$PROFILE" in
    XL) SW=4; SER=2; SEW=3; SEM=7G; SEC=4.0; SAPI_M=14G; SAPI_C=6.0; STH=2 ;;
    L)  SW=3; SER=1; SEW=4; SEM=6G; SEC=4.0; SAPI_M=8G;  SAPI_C=4.0; STH=2 ;;
    S|*) SW=2; SER=1; SEW=2; SEM=3G; SEC=2.0; SAPI_M=4G;  SAPI_C=2.0; STH=1 ;;
  esac
}

write_search_generated() {
  local SP="${STAGING_SCALE_PROFILE:-}"
  if [[ -n "$SP" ]]; then
    log "Search STAGING_SCALE_PROFILE=$SP"
    case "$SP" in
      # 32GB-class search host profiles
      # production target: search=14G, embedder total=14G via 2x7G replicas
      fast)       SW=4; SER=2; SEW=3; SEM=6G; SEC=4.0; SAPI_M=10G; SAPI_C=5.0; STH=2 ;;
      production) SW=4; SER=2; SEW=3; SEM=7G; SEC=4.0; SAPI_M=14G; SAPI_C=6.0; STH=2 ;;
      ultimate)   SW=6; SER=2; SEW=4; SEM=7G; SEC=5.0; SAPI_M=14G; SAPI_C=7.0; STH=2 ;;
      *)          apply_ram_tier_search ;;
    esac
  else
    apply_ram_tier_search
  fi

  # Optional hard override for embedder replicas
  if [[ -n "${STAGING_SEARCH_EMBEDDER_REPLICAS:-}" ]]; then
    SER="${STAGING_SEARCH_EMBEDDER_REPLICAS}"
  fi

  WANT_MIN_UP=$((1 + SER))  # search API + N embedder replicas

  # Resolve the processing server IPs to bake into the generated file.
  # After deploy (restarts, docker start, etc.) the generated file is the
  # source of truth — shell env vars are gone after the initial compose up.
  #
  # Use private addresses only; Redis and Milvus must not be reachable from the internet.
  # Pass MILVUS_HOST / REDIS_URL at deploy time if the public IP changes, e.g.:
  #   MILVUS_HOST=<new-ip> REDIS_URL=redis://<new-ip>:6379/0 STAGING_ROLE=search ./universal_deploy.sh
  local GEN_MILVUS_HOST="${MILVUS_HOST:?Set MILVUS_HOST to the processing server private address}"
  local GEN_REDIS_URL="${REDIS_URL:?Set REDIS_URL (redis://:password@private-host:6379/0)}"
  # Extract REDIS_HOST from REDIS_URL (strip redis:// prefix and port/db)
  local GEN_REDIS_HOST
  GEN_REDIS_HOST=$(echo "$GEN_REDIS_URL" | sed 's|redis://||' | cut -d: -f1 | cut -d/ -f1)
  local GEN_REDIS_PORT
  GEN_REDIS_PORT=$(echo "$GEN_REDIS_URL" | sed 's|redis://||' | cut -d: -f2 | cut -d/ -f1)
  GEN_REDIS_PORT="${GEN_REDIS_PORT:-6379}"

  cat >"$GEN_FILE" <<EOF
# AUTO-GENERATED by universal_deploy.sh — do not commit
# Profile: ${SP:-ram-tier-$PROFILE} | Search API ${SW}w | Embedder ${SER}x${SEW}w
# Processing server: MILVUS=${GEN_MILVUS_HOST}:19530  REDIS=${GEN_REDIS_HOST}:${GEN_REDIS_PORT}
services:
  ultimate-search:
    environment:
      # ── Processing server connections (baked in at deploy time) ───────────
      # These survive container restarts because they live in this generated
      # file, not in shell env vars which are gone after the initial compose up.
      MILVUS_HOST: "${GEN_MILVUS_HOST}"
      MILVUS_PORT: "19530"
      REDIS_URL: "${GEN_REDIS_URL}"
      REDIS_HOST: "${GEN_REDIS_HOST}"
      REDIS_PORT: "${GEN_REDIS_PORT}"
      REDIS_DB: "0"
      # ─────────────────────────────────────────────────────────────────────
      EMBEDDER_URL: "http://ultimate-embedder-search:8080/embed"
      EMBEDDER_SEARCH_URL: "http://ultimate-embedder-search:8080/embed"
      METADATA_INDEX_PUBLISHER: "false"
      OMP_NUM_THREADS: "${STH}"
      MKL_NUM_THREADS: "${STH}"
      OPENBLAS_NUM_THREADS: "${STH}"
      VECLIB_MAXIMUM_THREADS: "${STH}"
      NUMEXPR_NUM_THREADS: "${STH}"
    command:
      - uvicorn
      - search_api:app
      - --host
      - 0.0.0.0
      - --port
      - "8000"
      - --workers
      - "4"
      - --timeout-keep-alive
      - "600"
    deploy:
      resources:
        limits:
          memory: ${SAPI_M}
          cpus: "${SAPI_C}"
    healthcheck:
      interval: 15s
      timeout: 15s
      retries: 10
      start_period: 240s
  ultimate-embedder-search:
    command:
      - uvicorn
      - src.embedder_service:app
      - --host
      - 0.0.0.0
      - --port
      - "8080"
      - --workers
      - "${SEW}"
      - --timeout-keep-alive
      - "600"
    deploy:
      replicas: ${SER}
      resources:
        limits:
          memory: ${SEM}
          cpus: "${SEC}"
    healthcheck:
      interval: 15s
      timeout: 20s
      retries: 10
      start_period: 180s
EOF
  export WANT_MIN_UP
}

# ═══════════════════════════════════════════════════════════════════════════
# SINGLE-HOST DEPLOY (processing or search)
# ═══════════════════════════════════════════════════════════════════════════

# Reclaim Docker disk after compose down (old stack images are then unused).
staging_docker_reclaim_space() {
  local level="${STAGING_PRUNE_IMAGES:-1}"
  [[ "$level" == "0" ]] && return 0
  local effective="$level"
  # image prune -af after `down` removes tags needed for `up` without --build
  if [[ "${STAGING_SKIP_BUILD:-0}" == "1" && "$effective" == "2" ]]; then
    effective=1
    log "STAGING_SKIP_BUILD=1 — using prune level 1 only (keep unused tagged images for up without rebuild)"
  fi
  log "Docker disk cleanup (STAGING_PRUNE_IMAGES=$level, effective=$effective)..."
  # Build cache (multi-GB after failed embedder builds)
  if ! $DOCKER_CMD buildx prune -af 2>/dev/null; then
    $DOCKER_CMD builder prune -af 2>/dev/null || $DOCKER_CMD builder prune -f 2>/dev/null || true
  fi
  if [[ "$effective" == "2" ]]; then
    log "Pruning all unused images (not referenced by any running container)..."
    $DOCKER_CMD image prune -af || true
  else
    $DOCKER_CMD image prune -f || true
  fi
  log "Docker disk summary:"
  $DOCKER_CMD system df 2>/dev/null | head -8 || true
}

run_single_deploy() {
  rm -f "$GEN_FILE"

  # ── Compose project name (documented in docker-compose.*-only.yml headers; always pass -p below) ──
  if [[ "$STAGING_ROLE" == "processing" ]]; then
    export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-ultimate-processing}"
  else
    export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-ultimate-search}"
  fi
  log "COMPOSE_PROJECT_NAME=$COMPOSE_PROJECT_NAME (explicit -p on all compose invocations)"

  # ── Generate compose override ─────────────────────────────────────────
  if [[ "$STAGING_ROLE" == "processing" ]]; then
    write_processing_generated
  elif [[ "$STAGING_ROLE" == "search" ]]; then
    if [[ -z "${MILVUS_HOST:-}" || -z "${REDIS_URL:-}" ]]; then
      die "STAGING_ROLE=search requires MILVUS_HOST and REDIS_URL pointing at the processing server."
    fi
    export MILVUS_HOST REDIS_URL
    log "Search remote connections: MILVUS_HOST=$MILVUS_HOST  REDIS_URL=$REDIS_URL"
    write_search_generated
  fi

  [[ -f "$GEN_FILE" ]] || die "Failed to write $GEN_FILE"

  # ── Create runtime dirs (create with correct perms; safe for idempotent reruns) ──
  for d in temp_uploads logs data data/milvus data/redis data/metadata_cache; do
    mkdir -p "$d" 2>/dev/null || sudo mkdir -p "$d"
  done
  chmod 777 temp_uploads logs data data/milvus data/redis data/metadata_cache 2>/dev/null \
    || sudo chmod -R 777 temp_uploads logs data 2>/dev/null || true

  # ── Resolve port conflicts (search host: old stacks may hold :8000) ──
  if [[ "$STAGING_ROLE" == "search" ]]; then
    local old_container
    old_container=$($DOCKER_CMD ps -q --filter "publish=8000" 2>/dev/null | head -1 || true)
    if [[ -n "${old_container:-}" ]]; then
      warn "Port 8000 already in use by container $old_container — stopping it."
      $DOCKER_CMD stop "$old_container" 2>/dev/null || true
    fi
  fi

  # ── Health target calculations ──────────────────────────────────────────
  # Containers WITH Docker healthchecks:
  #   processing: milvus + redis + API + (ER × embedder) = 3 + ER
  #   search:     search API + embedder-search = 2
  # Celery workers have no healthcheck; counted in WANT_MIN_UP (all containers) but not MIN_HEALTHY.
  if [[ "$STAGING_ROLE" == "processing" ]]; then
    MIN_HEALTHY=$((3 + ER))
    log "Generated $GEN_FILE — targets: up≥${WANT_MIN_UP}, healthy≥${MIN_HEALTHY} (embedder ${ER}×${EW}w, API ${APIW}w)"
  else
    MIN_HEALTHY=2
    log "Generated $GEN_FILE — targets: up≥${WANT_MIN_UP}, healthy≥${MIN_HEALTHY} (search ${SW}w, embedder ${SEW}w)"
  fi

  # ── Build + Up ─────────────────────────────────────────────────────────
  # Never pass an empty "${BUILD_ARGS[@]}" — docker-compose treats it as a service name and errors:
  #   No such service:
  _compose_up() {
    local base="$1"
    local proj="$2"
    if [[ "${STAGING_SKIP_BUILD:-0}" == "1" ]]; then
      "${COMPOSE[@]}" -f "$base" -f "$GEN_FILE" -p "$proj" up -d --remove-orphans
    else
      "${COMPOSE[@]}" -f "$base" -f "$GEN_FILE" -p "$proj" up -d --build --remove-orphans
    fi
  }

  if [[ "$STAGING_ROLE" == "processing" ]]; then
    PROJECT="ultimate-processing"
    BASE="docker-compose.processing-only.yml"
    log "=== Deploy PROCESSING: $BASE + $GEN_FILE ==="
    if [[ "${STAGING_NO_DOWN:-0}" != "1" ]]; then
      "${COMPOSE[@]}" -f "$BASE" -f "$GEN_FILE" -p "$PROJECT" down --remove-orphans 2>/dev/null || true
    fi
    staging_docker_reclaim_space
    _compose_up "$BASE" "$PROJECT"
  else
    PROJECT="ultimate-search"
    BASE="docker-compose.search-only.yml"
    log "=== Deploy SEARCH: $BASE + $GEN_FILE ==="
    if [[ "${STAGING_NO_DOWN:-0}" != "1" ]]; then
      MILVUS_HOST="$MILVUS_HOST" REDIS_URL="$REDIS_URL" \
        "${COMPOSE[@]}" -f "$BASE" -f "$GEN_FILE" -p "$PROJECT" down --remove-orphans 2>/dev/null || true
    fi
    staging_docker_reclaim_space
    if [[ "${STAGING_SKIP_BUILD:-0}" == "1" ]]; then
      MILVUS_HOST="$MILVUS_HOST" REDIS_URL="$REDIS_URL" \
        "${COMPOSE[@]}" -f "$BASE" -f "$GEN_FILE" -p "$PROJECT" up -d --remove-orphans
    else
      MILVUS_HOST="$MILVUS_HOST" REDIS_URL="$REDIS_URL" \
        "${COMPOSE[@]}" -f "$BASE" -f "$GEN_FILE" -p "$PROJECT" up -d --build --remove-orphans
    fi
  fi

  # ── Health check loop (≤6 minutes) ─────────────────────────────────────
  log "Verifying health..."
  ok=0
  for i in $(seq 1 36); do
    up=$($DOCKER_CMD ps --filter "name=${PROJECT}" --format '{{.Names}} {{.Status}}' 2>/dev/null | grep -c ' Up ' || true)
    healthy=$($DOCKER_CMD ps --filter "name=${PROJECT}" --filter "health=healthy" --format '{{.Names}}' 2>/dev/null | wc -l | tr -d ' ')
    echo "  [$i/36] up=${up:-0} healthy=${healthy:-0} (need up≥${WANT_MIN_UP}, healthy≥${MIN_HEALTHY})"
    if [[ "${up:-0}" -ge "$WANT_MIN_UP" && "${healthy:-0}" -ge "$MIN_HEALTHY" ]]; then
      ok=1
      break
    fi
    sleep 10
done

  echo ""
  $DOCKER_CMD ps --format "table {{.Names}}\t{{.Status}}" | grep -E "NAME|${PROJECT}" || true

  # ── Sanity check: processing workers use correct EMBEDDER_URL ──────────
  if [[ "$STAGING_ROLE" == "processing" ]]; then
    local wid
    wid=$($DOCKER_CMD ps -qf "name=${PROJECT}.*celery-pdf" | head -1 || true)
    if [[ -n "${wid:-}" ]]; then
      local emb_url
      emb_url=$($DOCKER_CMD exec "$wid" printenv EMBEDDER_URL 2>/dev/null || echo "n/a")
      log "EMBEDDER_URL (celery-pdf sample): $emb_url"
      if [[ "$emb_url" == *"embedder-search"* ]]; then
        warn "Celery workers are using search embedder hostname — should be http://ultimate-embedder:8080/embed"
      fi
    fi
  fi

  if [[ "$ok" -eq 1 ]]; then
    echo ""
    log "DEPLOYMENT COMPLETE — STAGING_ROLE=$STAGING_ROLE  project=$PROJECT"
  else
    echo ""
    warn "Health/up targets not met after 6 minutes. Debug: $DOCKER_CMD logs <container>"
    exit 1
  fi

  # ── HTTP smoke test ────────────────────────────────────────────────────
  # Prefer in-container curl to bypass host networking edge cases.
  smoke_ok=0
  local api_cid=""
  for t in $(seq 1 45); do
    # Stable container_name from docker-compose.*-only.yml (see container_name: …-api)
    if [[ "$STAGING_ROLE" == "processing" ]]; then
      api_cid=$($DOCKER_CMD ps -qf "name=ultimate-processing-api" | head -1 || true)
      [[ -z "${api_cid:-}" ]] && api_cid=$($DOCKER_CMD ps -qf "name=${PROJECT}_ultimate-search_" | head -1 || true)
    else
      api_cid=$($DOCKER_CMD ps -qf "name=ultimate-search-api" | head -1 || true)
      [[ -z "${api_cid:-}" ]] && api_cid=$($DOCKER_CMD ps -qf "name=${PROJECT}_ultimate-search_" | head -1 || true)
    fi
    [[ -z "${api_cid:-}" ]] && api_cid=$($DOCKER_CMD ps -qf "name=${PROJECT}-ultimate-search-" | head -1 || true)

    if [[ -n "${api_cid:-}" ]]; then
      if $DOCKER_CMD exec "$api_cid" curl -sfS --max-time 12 "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
        smoke_ok=1
        log "Smoke test: docker exec ${api_cid:0:12}... GET /health ✓"
        break
      fi
    fi
    # Host-level fallback
    if curl -sfS --max-time 12 "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
      smoke_ok=1
      log "Smoke test: GET http://127.0.0.1:8000/health ✓ (host)"
      break
    fi
    sleep 3
  done

  if [[ "$smoke_ok" -ne 1 ]]; then
    warn "HTTP smoke test failed after ~135s (API not ready, workers saturated, or port not bound)"
    if [[ "${STAGING_REQUIRE_SMOKE:-0}" == "1" ]]; then
      exit 1
    fi
  fi
}

run_single_deploy
