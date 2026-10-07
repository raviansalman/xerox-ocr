#!/bin/bash
# Copy updated code to all Ultimate Docker containers and restart.
# Run from repo root: ./ultimate/copy_code_to_containers.sh
# Or from ultimate/: ./copy_code_to_containers.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Ensure we run from ultimate/ so src/ and ultimate_ui.py paths exist
cd "$SCRIPT_DIR"

echo "🔄 Copying code to all Ultimate containers and restarting..."

# All Ultimate app containers (API + every Celery worker)
CONTAINERS=($(docker ps --filter "name=ultimate" --format "{{.Names}}" | grep -E "ultimate-search|ultimate-celery-worker" || true))

if [ ${#CONTAINERS[@]} -eq 0 ]; then
    echo "⚠️ No Ultimate containers running. Starting stack..."
    cd "$SCRIPT_DIR"
    docker compose -f docker-compose.ultimate.yml up -d --build
    echo "⏳ Waiting for containers to start..."
    sleep 45
    CONTAINERS=($(docker ps --filter "name=ultimate" --format "{{.Names}}" | grep -E "ultimate-search|ultimate-celery-worker" || true))
fi

if [ ${#CONTAINERS[@]} -eq 0 ]; then
    echo "❌ No Ultimate search or worker containers found."
    exit 1
fi

echo "📦 Containers to update (${#CONTAINERS[@]}):"
for c in "${CONTAINERS[@]}"; do
    echo "   - $c"
done

# Copy source and UI into each container
echo "📋 Copying src/ and ultimate_ui.py..."
for container in "${CONTAINERS[@]}"; do
    if docker cp src/ "$container:/app/src" 2>/dev/null; then
        echo "   ✅ $container: src/ copied"
    else
        echo "   ⚠️ $container: failed to copy src/"
    fi
    if docker cp ultimate_ui.py "$container:/app/ultimate_ui.py" 2>/dev/null; then
        echo "   ✅ $container: ultimate_ui.py copied"
    else
        echo "   ⚠️ $container: failed to copy ultimate_ui.py"
    fi
done

# Restart all so they load new code
echo "🔄 Restarting containers..."
for container in "${CONTAINERS[@]}"; do
    if docker restart "$container" 2>/dev/null; then
        echo "   ✅ $container restarted"
    else
        echo "   ⚠️ $container restart failed"
    fi
done

echo "⏳ Waiting for services to be ready..."
sleep 25

echo "✅ Code copied and all Ultimate containers restarted."
echo "   Status: docker ps --filter 'name=ultimate'"
