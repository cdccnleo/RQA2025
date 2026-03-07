#!/usr/bin/env python3
"""
简单的后端服务，只提供数据源配置的 API 端点
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from typing import List, Dict

# 创建 FastAPI 应用
app = FastAPI(
    title="RQA2025 简单 API",
    description="只提供数据源配置的 API 端点",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 获取项目根目录的绝对路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
print(f"DEBUG: 项目根目录: {PROJECT_ROOT}")

# 环境感知的配置文件路径
def _get_config_file_path():
    """根据环境获取配置文件路径"""
    env = os.getenv("RQA_ENV", "development").lower()

    if env == "production":
        # 生产环境也使用主配置文件，确保配置一致性
        config_file = os.path.join(PROJECT_ROOT, "data", "data_sources_config.json")
    elif env == "testing":
        # 测试环境使用测试目录
        config_file = os.path.join(PROJECT_ROOT, "data", "testing", "data_sources_config.json")
    else:
        # 开发环境使用默认目录
        config_file = os.path.join(PROJECT_ROOT, "data", "data_sources_config.json")

    print(f"DEBUG: 配置文件绝对路径: {config_file}")
    print(f"DEBUG: 配置文件是否存在: {os.path.exists(config_file)}")
    return config_file

# 加载数据源配置
def load_data_sources() -> List[Dict]:
    """加载数据源配置"""
    try:
        config_file = _get_config_file_path()
        
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                raw_content = f.read()
                print(f"DEBUG: 文件原始内容长度: {len(raw_content)}")
                
                # 检查文件内容是否包含Baostock
                if 'baostock' in raw_content.lower():
                    print("DEBUG: 文件中包含Baostock数据源")
                else:
                    print("DEBUG: 文件中不包含Baostock数据源")

                config_data = json.loads(raw_content)

                # 处理不同格式的配置文件
                if isinstance(config_data, dict):
                    print(f"DEBUG: 解析后的配置键: {list(config_data.keys())}")
                    data_sources = config_data.get('data_sources', [])
                elif isinstance(config_data, list):
                    print(f"DEBUG: 配置是直接的列表格式")
                    data_sources = config_data
                else:
                    print(f"DEBUG: 未知配置格式: {type(config_data)}")
                    data_sources = []

            print(f"DEBUG: 从配置中提取的数据源数量: {len(data_sources)}")

            # 检查是否包含Baostock数据源
            baostock_count = len([s for s in data_sources if 'baostock' in s.get('id', '').lower()])
            print(f"DEBUG: 从文件加载的数据源中包含 {baostock_count} 个Baostock数据源")

            for i, source in enumerate(data_sources):
                print(f"DEBUG: 加载的数据源 {i}: id={repr(source.get('id'))}, name={repr(source.get('name'))}")

            # 检查并修复null ID - 多重保护
            print(f"DEBUG: 开始修复 {len(data_sources)} 个数据源的ID")
            for source in data_sources:
                id_value = source.get('id')
                name = source.get('name', 'unknown')
                print(f"DEBUG: 检查数据源 {name}, 当前ID: {repr(id_value)}")

                # 强制修复任何形式的null ID
                needs_fix = (
                    id_value is None or
                    str(id_value).lower() in ['null', 'none', ''] or
                    id_value == 'null' or
                    id_value == 'None'
                )

                if needs_fix:
                    if '新浪财经' in name:
                        source['id'] = 'sinafinance'
                    elif '宏观经济' in name:
                        source['id'] = 'macrodata'
                    elif '财联社' in name:
                        source['id'] = 'cls'
                    else:
                        source['id'] = name.lower().replace(' ', '_').replace('（', '_').replace('）', '_')
                    print(f"DEBUG: 修复数据源 {name} 的null ID: {repr(id_value)} -> {source['id']}")
                else:
                    print(f"DEBUG: 数据源 {name} ID正常: {repr(id_value)}")

            return data_sources
        else:
            print(f"WARNING: 配置文件不存在: {config_file}")
            return []

    except Exception as e:
        print(f"ERROR: 加载数据源配置失败: {e}")
        import traceback
        traceback.print_exc()
        return []

# 根路径
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "RQA2025 简单 API",
        "version": "1.0.0",
        "status": "running"
    }

# 健康检查
@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

# 获取数据源配置
@app.get("/api/v1/data/sources")
async def get_data_sources():
    """获取所有数据源配置"""
    try:
        sources = load_data_sources()
        active_count = len([s for s in sources if s.get("enabled", True)])
        return {
            "data": sources,
            "data_sources": sources,  # 兼容前端期望的字段名
            "total": len(sources),
            "active": active_count,
            "message": "数据源加载成功"
        }
    except Exception as e:
        print(f"ERROR: 获取数据源失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": f"获取数据源失败: {str(e)}"
        }

# 运行应用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
