#!/usr/bin/env python3
"""
RQA2025 FastAPI 服务器启动脚本（简化版）
直接启动 uvicorn，用于诊断问题
"""

import sys
import os
from pathlib import Path

# 设置Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# 禁用输出缓冲
os.environ['PYTHONUNBUFFERED'] = '1'

if __name__ == "__main__":
    print("🚀 启动 RQA2025 FastAPI 服务器（简化版）...")
    
    try:
        # 直接导入应用
        print("📦 导入应用...")
        from src.gateway.web.app_factory import create_app
        app = create_app()
        print(f"✅ 应用创建成功，路由数: {len(app.routes)}")
        
        # 直接使用 uvicorn.run()
        print("🚀 启动 uvicorn 服务器...")
        import uvicorn
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
