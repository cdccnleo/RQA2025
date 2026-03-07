# 特征工程服务 Dockerfile
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
COPY requirements-docker.txt ./requirements.txt

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制特征工程相关代码
COPY src/features/ ./src/features/
COPY src/engine/ ./src/engine/
COPY src/infrastructure/ ./src/infrastructure/

# 创建必要的目录
RUN mkdir -p /app/cache /app/logs /app/data

# 设置环境变量
ENV PYTHONPATH=/app
ENV SERVICE_NAME=features-service
ENV SERVICE_PORT=8001

# 暴露端口
EXPOSE 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "src.features.api:app", "--host", "0.0.0.0", "--port", "8001"] 