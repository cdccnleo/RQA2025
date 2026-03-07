#!/usr/bin/env python3
"""
调试财联社ID保存问题的服务器
"""

import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
        allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_SOURCES_CONFIG_FILE = "data/data_sources_config.json"

def load_data_sources():
    """加载数据源配置"""
    try:
        if os.path.exists(DATA_SOURCES_CONFIG_FILE):
            with open(DATA_SOURCES_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return config_data.get('data_sources', [])
        return []
    except Exception as e:
        print(f"加载数据源配置失败: {e}")
        return []

def save_data_sources(sources):
    """保存数据源配置"""
    try:
        os.makedirs(os.path.dirname(DATA_SOURCES_CONFIG_FILE), exist_ok=True)
        with open(DATA_SOURCES_CONFIG_FILE, 'w', encoding='utf-8') as f:
            data = {"data_sources": sources}
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"数据源配置已保存 ({len(sources)} 个数据源)")
        return True
    except Exception as e:
        print(f"保存数据源配置失败: {e}")
        return False

@app.post("/api/v1/data/sources")
async def create_data_source(source_config: dict):
    """创建新的数据源配置"""
    print(f"收到新增请求: {source_config}")

    print(f"收到新增请求: {source_config}")

    # 设置默认值
    new_source = source_config.copy()
    new_source.setdefault("enabled", True)
    new_source.setdefault("rate_limit", "100次/分钟")
    new_source.setdefault("last_test", None)
    new_source.setdefault("status", "未测试")

    print(f"处理后的数据源: id={new_source.get('id')}, name={new_source.get('name')}")

    # 创建返回用的副本
    response_source = new_source.copy()

    # 直接添加到文件
    sources = load_data_sources()
    print(f"当前数据源数量: {len(sources)}")

    sources.append(new_source)
    save_data_sources(sources)

    print(f"添加成功，最终ID: {new_source.get('id')}")
    print(f"返回数据源ID: {response_source.get('id')}")

    return {
        "success": True,
        "message": f"数据源 {new_source['name']} 创建成功",
        "data_source": response_source,
    }

@app.get("/api/v1/data/sources")
async def get_data_sources():
    """获取所有数据源"""
    sources = load_data_sources()
    return {"data_sources": sources}

if __name__ == "__main__":
    import uvicorn
    print("启动调试服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
