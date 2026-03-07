# 数据服务 Dockerfile
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
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements-docker.txt ./requirements.txt

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制数据服务相关代码
COPY src/data/ ./src/data/
COPY src/infrastructure/ ./src/infrastructure/
COPY src/engine/ ./src/engine/

# 创建必要的目录
RUN mkdir -p /app/cache /app/logs /app/data /app/backups

# 设置环境变量
ENV PYTHONPATH=/app
ENV SERVICE_NAME=data-service
ENV SERVICE_PORT=8002

# 暴露端口
EXPOSE 8002

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "src.data.api:app", "--host", "0.0.0.0", "--port", "8002"] 