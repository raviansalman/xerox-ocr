#!/bin/bash
# Smart deployment script - only rebuilds when needed, uses Docker cache

set -e

cd /var/www/vector-storage-processing-service-python

echo "== Smart Deployment Script =="
echo ""

# Pull latest code
echo "=== Pulling latest code ==="
sudo git fetch origin
sudo git pull origin vector-indexing-v1

echo ""
echo "=== Latest commit ==="
sudo git log -1 --oneline

echo ""
echo "=== Checking what changed ==="
CHANGED_FILES=$(sudo git diff --name-only HEAD~1 HEAD 2>/dev/null | grep -E "ultimate/|Dockerfile|requirements" || echo "")

# Check if Python code, Dockerfile, or requirements changed
NEEDS_REBUILD=false
if echo "$CHANGED_FILES" | grep -qE "\.py$|Dockerfile|requirements"; then
    NEEDS_REBUILD=true
    echo "✅ Python code or Dockerfile changed - rebuild needed"
    echo "Changed files: $CHANGED_FILES"
else
    echo "✅ Only config files changed - restart only"
fi

cd ultimate

# Determine which docker compose command to use
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="sudo docker-compose"
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE="sudo docker compose"
else
    echo "❌ Error: docker-compose not found"
    exit 1
fi

if [ "$NEEDS_REBUILD" = true ]; then
    echo ""
    echo "=== Rebuilding with cache (faster - only rebuilds changed layers) ==="
    # Use cache for faster builds - Docker will only rebuild layers that changed
    $DOCKER_COMPOSE -f docker-compose.ultimate.yml build ultimate-search ultimate-celery-worker ultimate-celery-worker-large
    
    echo ""
    echo "=== Restarting services with new images ==="
    $DOCKER_COMPOSE -f docker-compose.ultimate.yml up -d ultimate-search ultimate-celery-worker ultimate-celery-worker-large
else
    echo ""
    echo "=== No rebuild needed, just restarting services ==="
    $DOCKER_COMPOSE -f docker-compose.ultimate.yml restart ultimate-search ultimate-celery-worker ultimate-celery-worker-large
fi

echo ""
echo "=== Waiting for services to start (15 seconds) ==="
sleep 15

echo ""
echo "=== Service status ==="
sudo docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -E "NAME|ultimate"

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Note: If you need to force a full rebuild (without cache), use:"
echo "  $DOCKER_COMPOSE -f docker-compose.ultimate.yml build --no-cache <service>"

