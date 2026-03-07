#!/usr/bin/env python3
"""
简单的测试服务器，用于测试财联社编辑和删除功能
"""

import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
                data = json.load(f)

            data_sources = data.get('data_sources', [])

            print(f"加载数据源配置，共有 {len(data_sources)} 个数据源")

            # 修复null ID
            for source in data_sources:
                id_value = source.get('id')
                name = source.get('name', 'unknown')

                print(f"检查数据源: {name}, ID: {repr(id_value)} (type: {type(id_value).__name__})")

                # 强制修复任何形式的null ID
                needs_fix = (
                    id_value is None or
                    str(id_value).lower() in ['null', 'none', ''] or
                    id_value == 'null' or
                    id_value == 'None'
                )

                print(f"  needs_fix: {needs_fix}")

                if needs_fix:
                    print(f"  准备修复数据源: {name}")
                    if '新浪财经' in name:
                        source['id'] = 'sinafinance'
                        print("  匹配新浪财经规则")
                    elif '宏观经济' in name:
                        source['id'] = 'macrodata'
                        print("  匹配宏观经济规则")
                    elif '财联社' in name:
                        source['id'] = 'cls'
                        print("  匹配财联社规则")
                    else:
                        source['id'] = name.lower().replace(' ', '_').replace('（', '_').replace('）', '_')
                        print(f"  使用默认规则: {source['id']}")
                    print(f"🔧 修复数据源 {name} 的null ID: {repr(id_value)} -> {source['id']}")
                else:
                    print(f"✅ 数据源 {name} ID正常: {repr(id_value)}")

            return data_sources
        else:
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

    # 特别检查财联社
    cls_source = next((s for s in sources if s.get('name') == '财联社'), None)
    if cls_source:
        print(f"🎯 API返回的财联社ID: {repr(cls_source.get('id'))}")
    else:
        print("❌ API中未找到财联社数据源")

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
    print("启动简单测试服务器...")

    # 测试load_data_sources函数
    print("测试load_data_sources函数:")
    sources = load_data_sources()
    print(f"加载了 {len(sources)} 个数据源")

    uvicorn.run(app, host="0.0.0.0", port=8000)
