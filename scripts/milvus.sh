#!/usr/bin/env bash
# Milvus standalone 管理(Windows + Docker Desktop 环境专用)。
#   bash scripts/milvus.sh up|down|status|ps
# 说明:
#  - Docker Desktop 的 bin 不在 PATH 时,docker/docker compose 及其凭据助手会找不到,故此处显式加路径。
#  - 卷目录用绝对路径,避免 compose 以「文件所在目录」为基准把卷落到 docker/docker/volumes。
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"
export DOCKER_VOLUME_DIRECTORY="$REPO/docker"
COMPOSE="$REPO/docker/milvus-compose.yml"

case "${1:-status}" in
  up)
    docker compose -f "$COMPOSE" up -d
    echo "-- 等待健康检查 --"
    for i in $(seq 1 30); do
      sleep 5
      st=$(docker inspect -f '{{.State.Health.Status}}' milvus-standalone 2>/dev/null || echo "starting")
      echo "  $i: milvus-standalone=$st"
      [ "$st" = "healthy" ] && { echo "Milvus 就绪: 19530 (gRPC) / 9091 (healthz)"; exit 0; }
    done
    echo "!! 超时未 healthy,查 docker logs milvus-standalone" >&2; exit 1 ;;
  down)
    docker compose -f "$COMPOSE" down ;;
  status)
    hz=$(curl -s --max-time 3 http://127.0.0.1:9091/healthz || true)
    echo "healthz: ${hz:-<无响应>}"
    docker ps --filter name=milvus --format '{{.Names}}\t{{.Image}}\t{{.Status}}' ;;
  ps)
    docker ps -a --format '{{.Names}}\t{{.Status}}' ;;
  *)
    echo "用法: bash scripts/milvus.sh {up|down|status|ps}" >&2; exit 2 ;;
esac
