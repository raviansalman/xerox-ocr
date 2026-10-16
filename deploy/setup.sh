#!/usr/bin/env bash
# One-command deployment: installs Docker if needed, creates deploy/.env with random passwords, creates an admin
# login key, builds and starts the whole stack and waits until it is ready.
#
#   ./deploy/setup.sh                 tenant "default"
#   ./deploy/setup.sh invozone        tenant "invozone"
#
# Safe to run again: passwords and keys that are already set are kept (PostgreSQL keeps the password it was first
# started with), missing or empty ones are filled in, and the stack is rebuilt and restarted with the current code.
set -euo pipefail

cd "$(dirname "$0")/.."
TENANT="${1:-default}"
ENV_FILE=deploy/.env
KEYS_FILE=deploy/api_keys.json
KEY_NOTE=deploy/ADMIN_LOGIN_KEY.txt

if ! [[ "$TENANT" =~ ^[a-z0-9][a-z0-9_-]{0,62}$ ]]; then
  echo "Tenant name must be lowercase letters, digits, - or _ (got '$TENANT')." >&2
  exit 1
fi

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

# 1. Docker
if ! command -v docker >/dev/null 2>&1; then
  echo "Installing Docker..."
  curl -fsSL https://get.docker.com | $SUDO sh
fi
DOCKER="docker"
docker info >/dev/null 2>&1 || DOCKER="$SUDO docker"     # user not in the docker group yet
if ! $DOCKER compose version >/dev/null 2>&1; then
  echo "Docker Compose v2 is missing; install the docker-compose-plugin package and run this again." >&2
  exit 1
fi

random() { openssl rand -hex "$1" 2>/dev/null || head -c "$1" /dev/urandom | od -An -tx1 | tr -d ' \n'; }

# 2. Passwords (.env): created once; in an existing file, only missing or empty values are filled in. A value that
#    is already set is never changed (PostgreSQL keeps the password it was first started with).
if [ ! -f "$ENV_FILE" ]; then
  cp deploy/.env.example "$ENV_FILE"
  echo "Created $ENV_FILE."
fi
for name in POSTGRES_PASSWORD DOCINTEL_DB_PASSWORD DOCINTEL_REDIS_PASSWORD DOCINTEL_METRICS_TOKEN; do
  if grep -Eq "^$name=.+" "$ENV_FILE"; then
    continue
  elif grep -Eq "^$name=" "$ENV_FILE"; then
    sed -i.bak "s|^$name=.*|$name=$(random 24)|" "$ENV_FILE" && rm -f "$ENV_FILE.bak"
  else
    printf '\n%s=%s\n' "$name" "$(random 24)" >> "$ENV_FILE"
  fi
  echo "Set $name in $ENV_FILE."
done
chmod 600 "$ENV_FILE"

# 3. Admin login key: created when there is no key file, or when the existing one holds no usable key (for example
#    a copy of the example file with placeholders, which is set aside as api_keys.json.bak).
if [ -f "$KEYS_FILE" ] && ! grep -Eq '"key_sha256" *: *"[0-9a-f]{64}"' "$KEYS_FILE"; then
  mv "$KEYS_FILE" "$KEYS_FILE.bak"
  echo "$KEYS_FILE had no usable key; moved it to $KEYS_FILE.bak."
fi
if [ ! -f "$KEYS_FILE" ]; then
  KEY=$(random 32)
  HASH=$(printf '%s' "$KEY" | sha256sum | cut -d' ' -f1)
  printf '[\n  {"name": "%s-admin", "tenant": "%s", "roles": ["admin"], "key_sha256": "%s"}\n]\n' \
    "$TENANT" "$TENANT" "$HASH" > "$KEYS_FILE"
  printf 'Admin login key for tenant %s:\n%s\n' "$TENANT" "$KEY" > "$KEY_NOTE"
  chmod 600 "$KEY_NOTE"
  echo "Created an admin login key for tenant '$TENANT'."
else
  KEY=""
  echo "Keeping the existing $KEYS_FILE."
fi
chmod 644 "$KEYS_FILE"                   # hashes only; the app's non-root user must be able to read it

# 4. Build and start
echo "Building and starting (the first time takes several minutes)..."
$DOCKER compose -f deploy/docker-compose.yml --env-file "$ENV_FILE" up -d --build

PORT=$(grep -E '^DOCINTEL_PORT=' "$ENV_FILE" | cut -d= -f2)
PORT="${PORT:-8000}"
echo "Waiting for the system to be ready..."
for _ in $(seq 1 120); do
  if curl -fs "http://localhost:$PORT/health/ready" >/dev/null 2>&1; then
    IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    echo
    echo "Ready: open http://${IP:-localhost}:$PORT in a browser."   # hardcoded-ok(url): this server's own address
    if [ -n "$KEY" ]; then
      echo "Your admin login key (also saved in $KEY_NOTE):"
      echo "$KEY"
    else
      echo "Log in with the key in $KEY_NOTE (or one you created)."
    fi
    exit 0
  fi
  sleep 5
done
echo "Not ready after 10 minutes. Check: $DOCKER compose -f deploy/docker-compose.yml --env-file $ENV_FILE logs --tail 50" >&2
exit 1
