#!/usr/bin/env python3
"""
检查并同步数据源配置
从生产环境配置文件加载完整的数据源配置并同步到PostgreSQL
"""
import sys
import os
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_and_sync():
    """检查并同步数据源配置"""
    print("=" * 60)
    print("数据源配置检查与同步工具")
    print("=" * 60)
    
    # 读取生产环境配置文件
    prod_file = "data/production/data_sources_config.json"
    if not os.path.exists(prod_file):
        print(f"❌ 生产环境配置文件不存在: {prod_file}")
        return False
    
    print(f"\n📂 读取配置文件: {prod_file}")
    with open(prod_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理不同格式
    sources = data if isinstance(data, list) else data.get('data_sources', [])
    print(f"📊 文件总条目数: {len(sources)}")
    
    # 筛选完整的数据源配置
    complete_sources = []
    incomplete_sources = []
    
    for s in sources:
        if isinstance(s, dict):
            if 'name' in s and 'type' in s:
                complete_sources.append(s)
            else:
                incomplete_sources.append(s)
    
    print(f"✅ 完整数据源配置: {len(complete_sources)} 个")
    print(f"⚠️  不完整条目: {len(incomplete_sources)} 个")
    
    if incomplete_sources:
        print("\n不完整条目ID:")
        for s in incomplete_sources:
            print(f"  - {s.get('id', 'NO_ID')} (enabled: {s.get('enabled', 'N/A')})")
    
    print("\n完整数据源列表:")
    for i, s in enumerate(complete_sources, 1):
        print(f"  {i:2d}. {s.get('id', 'NO_ID'):30s} - {s.get('name', 'NO_NAME')}")
    
    # 同步到PostgreSQL
    print("\n" + "=" * 60)
    print("同步到PostgreSQL...")
    print("=" * 60)
    
    try:
        from src.gateway.web.data_source_config_manager import get_data_source_config_manager
        
        manager = get_data_source_config_manager()
        
        # 准备配置数据
        config_data = {
            'data_sources': complete_sources,
            'metadata': {
                'version': '1.0.0',
                'last_updated': '2026-01-12T22:45:00',
                'environment': 'production',
                'total_sources': len(complete_sources),
                'incomplete_entries': len(incomplete_sources)
            }
        }
        
        # 保存到PostgreSQL
        success = manager._save_to_postgresql(config_data)
        
        if success:
            print(f"✅ 成功同步 {len(complete_sources)} 个数据源到PostgreSQL")
            
            # 验证
            pg_config = manager._load_from_postgresql()
            if pg_config:
                pg_sources = pg_config.get('data_sources', [])
                print(f"✅ 验证：PostgreSQL中有 {len(pg_sources)} 个数据源")
                
                if len(pg_sources) == len(complete_sources):
                    print("✅ 数据源数量匹配！")
                    return True
                else:
                    print(f"⚠️  数量不匹配：期望 {len(complete_sources)} 个，实际 {len(pg_sources)} 个")
                    return False
            else:
                print("⚠️  无法从PostgreSQL验证配置")
                return False
        else:
            print("❌ 保存到PostgreSQL失败")
            return False
            
    except Exception as e:
        print(f"❌ 同步失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_and_sync()
    sys.exit(0 if success else 1)
