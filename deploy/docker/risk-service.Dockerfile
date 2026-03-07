# 风控服务Dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements-docker.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements-docker.txt

# 复制风控服务代码
COPY src/risk/ ./src/risk/
COPY src/infrastructure/ ./src/infrastructure/
COPY src/utils/ ./src/utils/

# 创建必要的目录
RUN mkdir -p /app/logs /app/cache /app/config

# 设置环境变量
ENV PYTHONPATH=/app
ENV SERVICE_NAME=risk-service
ENV SERVICE_PORT=8005
ENV DATABASE_URL=postgresql://user:password@database-service:5432/rqa2025

# 暴露端口
EXPOSE 8005

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# 启动命令
CMD ["uvicorn", "src.risk.api:app", "--host", "0.0.0.0", "--port", "8005"] 