#!/usr/bin/env python3
"""
同步数据源配置到PostgreSQL
从文件加载配置并保存到PostgreSQL数据库
"""
import sys
import os
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gateway.web.data_source_config_manager import get_data_source_config_manager

def sync_data_sources_to_postgresql():
    """同步数据源配置到PostgreSQL"""
    print("🔄 开始同步数据源配置到PostgreSQL...")
    
    try:
        # 获取数据源配置管理器
        manager = get_data_source_config_manager()
        
        # 强制从文件重新加载配置
        print("📂 从文件加载配置...")
        manager.load_config()
        
        # 获取当前配置
        config_data = manager._get_config_from_manager()
        sources = config_data.get('data_sources', [])
        
        print(f"📊 找到 {len(sources)} 个数据源配置")
        
        # 保存到PostgreSQL
        print("💾 保存到PostgreSQL...")
        success = manager._save_to_postgresql(config_data)
        
        if success:
            print(f"✅ 成功同步 {len(sources)} 个数据源配置到PostgreSQL")
            
            # 验证保存结果
            pg_config = manager._load_from_postgresql()
            if pg_config:
                pg_sources = pg_config.get('data_sources', [])
                print(f"✅ 验证：PostgreSQL中有 {len(pg_sources)} 个数据源配置")
                if len(pg_sources) == len(sources):
                    print("✅ 数据源数量匹配！")
                else:
                    print(f"⚠️  数据源数量不匹配：文件中有 {len(sources)} 个，PostgreSQL中有 {len(pg_sources)} 个")
            else:
                print("⚠️  无法从PostgreSQL验证配置")
        else:
            print("❌ 保存到PostgreSQL失败")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ 同步失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = sync_data_sources_to_postgresql()
    sys.exit(0 if success else 1)
