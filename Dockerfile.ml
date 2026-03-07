# RQA2025 ML推理节点 Dockerfile
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY setup.py .
COPY src/ ./src/
COPY scripts/ ./scripts/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# 创建必要的目录
RUN mkdir -p /app/logs /app/data /app/models

# 设置权限
RUN chmod +x scripts/*.py

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "print('ML node is healthy')"

# 暴露端口
EXPOSE 8080

# 启动命令
CMD ["python", "scripts/run_distributed_system.py", "--node-type", "ml"]
