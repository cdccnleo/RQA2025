# RQA2025 Health Monitor Dockerfile
# 预投产环境使用

FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV ENV=preprod

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .
COPY requirements-dev.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# 安装额外的监控和健康检查依赖
RUN pip install --no-cache-dir \
    prometheus_client \
    psutil \
    aiohttp \
    fastapi \
    uvicorn[standard] \
    sqlalchemy \
    asyncpg \
    redis[hiredis] \
    influxdb-client \
    elasticsearch \
    requests \
    pyyaml

# 复制应用代码
COPY src/ ./src/
COPY tests/ ./tests/

# 创建非root用户
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# 暴露端口
EXPOSE 8000 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "src.infrastructure.health.fastapi_health_checker:app", "--host", "0.0.0.0", "--port", "8000"]

