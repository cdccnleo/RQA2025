#!/bin/bash
# RQA2025 部署脚本

set -e  # 遇到错误立即退出

echo "🚀 开始部署RQA2025..."

# 创建部署目录
DEPLOY_DIR="/opt/rqa2025"
echo "📁 创建部署目录: $DEPLOY_DIR"
sudo mkdir -p $DEPLOY_DIR
sudo chown $USER:$USER $DEPLOY_DIR

# 复制应用代码
echo "📋 复制应用代码..."
cp -r . $DEPLOY_DIR/
cd $DEPLOY_DIR

# 设置Python虚拟环境
echo "🐍 设置Python虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 安装依赖
echo "📦 安装Python依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# 运行数据库迁移（如果有的话）
echo "🗄️ 执行数据库迁移..."
# python manage.py migrate  # 根据实际框架调整

# 收集静态文件（如果有的话）
echo "📄 收集静态文件..."
# python manage.py collectstatic --noinput  # 根据实际框架调整

# 设置权限
echo "🔐 设置文件权限..."
chmod +x scripts/*.sh
chmod 644 config/*.json

# 创建日志目录
echo "📝 创建日志目录..."
mkdir -p logs
chmod 755 logs

# 创建systemd服务文件（可选）
echo "⚙️ 配置systemd服务..."
sudo tee /etc/systemd/system/rqa2025.service > /dev/null <<EOF
[Unit]
Description=RQA2025 Quantitative Trading System
After=network.target

[Service]
User=$USER
Group=$USER
WorkingDirectory=$DEPLOY_DIR
Environment=PATH=$DEPLOY_DIR/venv/bin
ExecStart=$DEPLOY_DIR/venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 重新加载systemd
sudo systemctl daemon-reload

echo "🎉 RQA2025部署完成！"
echo ""
echo "启动服务命令:"
echo "  sudo systemctl start rqa2025"
echo "  sudo systemctl enable rqa2025"
echo ""
echo "查看状态命令:"
echo "  sudo systemctl status rqa2025"
echo ""
echo "查看日志命令:"
echo "  sudo journalctl -u rqa2025 -f"
