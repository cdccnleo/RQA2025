#!/usr/bin/env python3
"""
简化的测试服务器，用于测试财联社功能
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
    """加载数据源配置并修复ID"""
    try:
        if os.path.exists(DATA_SOURCES_CONFIG_FILE):
            with open(DATA_SOURCES_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            data_sources = config_data.get('data_sources', [])

            print(f"加载数据源配置，共有 {len(data_sources)} 个数据源")

            # 修复null ID
            for source in data_sources:
                id_value = source.get('id')
                name = source.get('name', 'unknown')

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
                    print(f"🔧 修复数据源 {name} 的ID -> {source['id']}")

            return data_sources
        else:
            print("配置文件不存在")
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

@app.get("/api/v1/data/sources")
async def get_data_sources():
    """获取所有数据源"""
    print("🔍 调用 get_data_sources API")
    sources = load_data_sources()
    print(f"📋 返回 {len(sources)} 个数据源")

    # 检查财联社ID
    for source in sources:
        if source['name'] == '财联社':
            print(f"🎯 财联社ID: {repr(source['id'])}")
            break

    return {"data_sources": sources}

@app.get("/api/v1/data/sources/{source_id}")
async def get_data_source(source_id: str):
    """获取指定的数据源配置"""
    sources = load_data_sources()
    for source in sources:
        if source["id"] == source_id:
            return source
    raise HTTPException(status_code=404, detail=f"数据源 {source_id} 不存在")

@app.put("/api/v1/data/sources/{source_id}")
async def update_data_source(source_id: str, updated_source: dict):
    """更新数据源配置"""
    sources = load_data_sources()
    found = False
    for i, source in enumerate(sources):
        if source["id"] == source_id:
            sources[i] = updated_source
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail=f"数据源 {source_id} 不存在")

    save_data_sources(sources)
    return {"message": f"数据源 {source_id} 更新成功", "source": updated_source}

@app.delete("/api/v1/data/sources/{source_id}")
async def delete_data_source(source_id: str):
    """删除数据源配置"""
    sources = load_data_sources()
    found = False
    for i, source in enumerate(sources):
        if source["id"] == source_id:
            deleted_source = sources.pop(i)
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail=f"数据源 {source_id} 不存在")

    save_data_sources(sources)
    return {
        "message": f"数据源 {source_id} 删除成功",
        "deleted_source": deleted_source,
        "remaining_count": len(sources)
    }

if __name__ == "__main__":
    import uvicorn
    print("启动简化的测试服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
