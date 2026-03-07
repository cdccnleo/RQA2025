#!/bin/bash
# 服务健康检查与自动重启脚本
SERVICE_NAME=myservice
HEALTH_URL=http://localhost:8000/health

if ! curl -sf $HEALTH_URL > /dev/null; then
  echo "[ERROR] $SERVICE_NAME health check failed, restarting..."
  systemctl restart $SERVICE_NAME
else
  echo "[OK] $SERVICE_NAME is healthy."
fi 