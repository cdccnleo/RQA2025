#!/usr/bin/env python3
"""修改数据源配置的最后测试时间"""

import json
import sys
from datetime import datetime, timedelta

# 直接修改JSON配置文件
config_file = '/app/data/data_sources_config.json'

def update_last_test_time():
    """修改akshare_stock_a的最后测试时间为昨天"""
    try:
        # 读取配置
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 查找akshare_stock_a
        target_source = None
        for source in data:
            if source.get('id') == 'akshare_stock_a':
                target_source = source
                break
        
        if not target_source:
            print("❌ 未找到akshare_stock_a数据源配置")
            return False
        
        # 获取当前最后测试时间
        current_last_test = target_source.get('last_test')
        print(f"📊 当前akshare_stock_a最后测试时间: {current_last_test}")
        
        # 修改为昨天的时间
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        target_source['last_test'] = yesterday
        target_source['status'] = 'pending'  # 重置状态为待测试
        
        # 保存回文件
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 已修改akshare_stock_a最后测试时间为: {yesterday}")
        print(f"✅ 状态已重置为: pending")
        return True
        
    except Exception as e:
        print(f"❌ 修改失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = update_last_test_time()
    sys.exit(0 if success else 1)
