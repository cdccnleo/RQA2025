#!/usr/bin/env python3
"""
验证财联社数据源删除的简单服务器
"""

import json
import os
from fastapi import FastAPI
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

@app.get("/api/v1/data/sources")
async def get_data_sources():
    """获取所有数据源"""
    print(f"加载配置文件: {DATA_SOURCES_CONFIG_FILE}")
    sources = load_data_sources()
    print(f"返回 {len(sources)} 个数据源")

    # 打印详细信息用于调试
    for i, source in enumerate(sources):
        if isinstance(source, dict):
            name = source.get('name', 'Unknown')
            id_val = source.get('id')
            print(f"  {i}: {name} (ID: {repr(id_val)})")

    return {"data_sources": sources}

@app.get("/api/v1/data/sources/{source_id}")
async def get_data_source(source_id: str):
    """获取指定的数据源配置"""
    sources = load_data_sources()
    for source in sources:
        if source["id"] == source_id:
            return source
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail=f"数据源 {source_id} 不存在")

@app.delete("/api/v1/data/sources/{source_id}")
async def delete_data_source(source_id: str):
    """删除数据源配置"""
    sources = load_data_sources()
    for i, source in enumerate(sources):
        if source["id"] == source_id:
            deleted_source = sources.pop(i)
            # 保存到文件
            try:
                config_data = {"data_sources": sources}
                with open(DATA_SOURCES_CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)
                return {
                    "message": f"数据源 {source_id} 删除成功",
                    "deleted_source": deleted_source,
                    "remaining_count": len(sources)
                }
            except Exception as e:
                from fastapi import HTTPException
                raise HTTPException(status_code=500, detail=f"保存配置失败: {e}")

    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail=f"数据源 {source_id} 不存在")

if __name__ == "__main__":
    import uvicorn
    print("启动验证服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
