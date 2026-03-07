# 配置管理Web服务部署与架构说明

## 一、系统架构概览

- 基于 FastAPI + Uvicorn 实现后端API服务
- 前端为静态HTML+JS，支持登录、配置树展示、仪表盘等
- 支持多节点配置同步、加密、权限控制
- 支持容器化部署（Docker/Docker Compose）

## 二、主要功能

- 配置可视化管理与编辑
- 配置同步与冲突检测
- 配置加密与安全校验
- 用户登录与权限控制
- 配置历史与状态监控

## 三、部署方式

### 1. 本地开发环境

```bash
pip install -r requirements.txt
python src/infrastructure/config/web_app.py
# 访问 http://localhost:8080/static/index.html
```

### 2. Docker 部署

```bash
docker-compose build
docker-compose up -d
# 访问 http://localhost:8080/static/index.html
```

### 3. 主要端口与环境变量

- 服务端口：8080
- 环境变量：
  - PYTHONUNBUFFERED=1

## 四、目录结构

- src/infrastructure/config/web_app.py  # FastAPI主入口
- src/infrastructure/config/static/     # 前端静态资源
- src/infrastructure/config/services/   # 后端核心服务
- Dockerfile, docker-compose.yml        # 容器化部署文件

## 五、常见问题FAQ

- 登录失败：请检查用户名/密码（默认：admin/admin123）
- 端口冲突：请确保8080端口未被占用
- 配置同步异常：请检查节点注册与网络连通性

## 六、业务集成建议

- 可通过REST API与其他微服务集成，实现配置集中管理
- 支持多环境（开发/测试/生产）配置隔离
- 推荐结合CI/CD流程自动化配置下发与回滚