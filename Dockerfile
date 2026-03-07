# RQA2025 量化交易系统 Dockerfile - 简化版本
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app:/app/src:/app/scripts
ENV PYTHONUNBUFFERED=1
ENV RQA_ENV=production

# 复制依赖文件并安装依赖
# 注意：使用 DOCKER_BUILDKIT=1 docker build 可以启用缓存以加速构建
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件（先复制代码，避免依赖变化时重新安装）
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY logs/ ./logs/
COPY config/ ./config/
COPY monitoring/ ./monitoring/
COPY web-static/ ./web-static/
COPY main.py ./main.py


# 创建必要的目录
RUN mkdir -p /app/data /app/logs /app/config /app/cache

# 设置时区为北京时间
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 暴露端口
EXPOSE 8000

# 健康检查 - 使用curl检查HTTP端点
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f --max-time 10 http://localhost:8000/health || exit 1

# 启动命令 - 支持多种启动方式
CMD ["python", "scripts/start_api_server.py"]