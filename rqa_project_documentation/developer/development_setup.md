# RQA开发环境搭建

## 系统要求
- Python 3.9+
- Node.js 16+
- Go 1.19+
- Git

## 安装步骤
1. 克隆仓库
2. 安装Python依赖
3. 安装Node.js依赖
4. 安装Go依赖
5. 配置数据库

## 运行开发环境
```bash
# 启动后端服务
python run.py

# 启动前端服务
cd frontend && npm start

# 启动数据库
docker-compose up -d postgres redis
```

---
*版本: 1.0*
