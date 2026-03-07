# 模型服务 Dockerfile
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
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements-docker.txt ./requirements.txt

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制模型服务相关代码
COPY src/models/ ./src/models/
COPY src/ml/ ./src/ml/
COPY src/infrastructure/ ./src/infrastructure/
COPY src/engine/ ./src/engine/

# 创建必要的目录
RUN mkdir -p /app/models /app/logs /app/cache /app/training_data

# 设置环境变量
ENV PYTHONPATH=/app
ENV SERVICE_NAME=model-service
ENV SERVICE_PORT=8003

# 暴露端口
EXPOSE 8003

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "src.models.api:app", "--host", "0.0.0.0", "--port", "8003"] 