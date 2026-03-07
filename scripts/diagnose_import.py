#!/usr/bin/env python3
"""
诊断脚本：逐步导入模块并记录时间，定位卡住的位置
"""

import sys
import os
import time
import traceback
from pathlib import Path

# 设置Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# 禁用输出缓冲
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

def log_step(step_name, start_time=None):
    """记录步骤和时间"""
    if start_time:
        elapsed = time.time() - start_time
        print(f"[{time.strftime('%H:%M:%S')}] [{elapsed:.2f}s] {step_name}", flush=True)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] {step_name}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    return time.time()

def diagnose_import():
    """诊断导入过程"""
    overall_start = time.time()
    log_step("=" * 70)
    log_step("开始诊断导入过程")
    log_step("=" * 70)
    
    steps = [
        ("导入基础模块", lambda: __import__('sys')),
        ("导入 pathlib", lambda: __import__('pathlib')),
        ("导入 config_manager", lambda: __import__('src.gateway.web.config_manager', fromlist=[''])),
        ("导入 websocket_manager", lambda: __import__('src.gateway.web.websocket_manager', fromlist=[''])),
        ("导入 basic_routes", lambda: __import__('src.gateway.web.basic_routes', fromlist=[''])),
        ("导入 strategy_routes", lambda: __import__('src.gateway.web.strategy_routes', fromlist=[''])),
        ("导入 data_collectors", lambda: __import__('src.gateway.web.data_collectors', fromlist=[''])),
        ("导入 api_utils", lambda: __import__('src.gateway.web.api_utils', fromlist=[''])),
        ("导入 DataManager (可能阻塞)", lambda: __import__('src.data', fromlist=[''])),
        ("导入 app_factory (关键步骤)", lambda: __import__('src.gateway.web.app_factory', fromlist=[''])),
        ("导入 api 模块 (可能阻塞)", lambda: __import__('src.gateway.web.api', fromlist=[''])),
    ]
    
    for step_name, import_func in steps:
        step_start = time.time()
        log_step(f"开始: {step_name}")
        
        try:
            result = import_func()
            elapsed = time.time() - step_start
            log_step(f"✅ 完成: {step_name} (耗时: {elapsed:.2f}s)")
            
            # 如果耗时超过5秒，发出警告
            if elapsed > 5:
                log_step(f"⚠️ 警告: {step_name} 耗时较长 ({elapsed:.2f}s)")
                
        except Exception as e:
            elapsed = time.time() - step_start
            log_step(f"❌ 失败: {step_name} (耗时: {elapsed:.2f}s)")
            log_step(f"错误: {e}")
            traceback.print_exc()
            sys.stderr.flush()
            break
    
    # 测试创建应用
    log_step("=" * 70)
    log_step("测试创建应用")
    log_step("=" * 70)
    
    try:
        step_start = time.time()
        log_step("执行: from src.gateway.web.app_factory import create_app")
        from src.gateway.web.app_factory import create_app
        elapsed = time.time() - step_start
        log_step(f"✅ app_factory 导入完成 (耗时: {elapsed:.2f}s)")
        
        step_start = time.time()
        log_step("执行: app = create_app()")
        app = create_app()
        elapsed = time.time() - step_start
        log_step(f"✅ create_app() 完成 (耗时: {elapsed:.2f}s)")
        log_step(f"应用路由数: {len(app.routes)}")
        
    except Exception as e:
        elapsed = time.time() - step_start
        log_step(f"❌ 创建应用失败 (耗时: {elapsed:.2f}s)")
        log_step(f"错误: {e}")
        traceback.print_exc()
        sys.stderr.flush()
    
    total_time = time.time() - overall_start
    log_step("=" * 70)
    log_step(f"诊断完成，总耗时: {total_time:.2f}s")
    log_step("=" * 70)

if __name__ == "__main__":
    try:
        diagnose_import()
    except KeyboardInterrupt:
        log_step("\n⚠️ 诊断被用户中断")
        sys.exit(1)
    except Exception as e:
        log_step(f"\n❌ 诊断过程发生异常: {e}")
        traceback.print_exc()
        sys.stderr.flush()
        sys.exit(1)
