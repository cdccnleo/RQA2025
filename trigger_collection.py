#!/usr/bin/env python3
"""手工触发数据采集流程"""

import sys
import json
import time
sys.path.insert(0, '/app/src')

def trigger_data_collection():
    """触发akshare_stock_a数据源的数据采集"""
    try:
        # 1. 加载数据源配置
        config_file = '/app/data/data_sources_config.json'
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
        
        print(f"✅ 找到数据源: {target_source['id']}")
        print(f"📊 数据源名称: {target_source.get('name', 'N/A')}")
        print(f"📊 当前状态: {target_source.get('status', 'N/A')}")
        
        # 2. 获取EventBus并初始化特征工程事件监听器
        try:
            from src.core.event_bus import get_event_bus
            from src.core.event_bus.types import EventType
            
            event_bus = get_event_bus()
            if not event_bus:
                print("❌ EventBus未初始化")
                return False
            
            # 初始化特征工程事件监听器
            print("\n🔧 初始化特征工程事件监听器...")
            try:
                from src.features.core.event_listeners import initialize_event_listeners
                from src.core.orchestration.scheduler import get_unified_scheduler
                
                scheduler = get_unified_scheduler()
                listeners = initialize_event_listeners(event_bus, scheduler)
                print("✅ 特征工程事件监听器初始化成功")
                
                # 等待一下确保监听器注册完成
                time.sleep(1)
            except Exception as e:
                print(f"⚠️ 初始化特征工程事件监听器失败: {e}")
                print("   继续尝试发布事件...")
            
            # 3. 直接调用特征工程的事件处理函数
            print("\n🔧 直接调用特征工程事件处理函数...")
            try:
                from src.features.core.event_listeners import get_feature_event_listeners
                
                listeners = get_feature_event_listeners()
                if listeners and listeners.event_bus:
                    # 构建事件数据
                    event_data = {
                        "source_id": target_source['id'],
                        "source_name": target_source.get('name', ''),
                        "source_type": target_source.get('type', ''),
                        "source_config": target_source,
                        "timestamp": time.time(),
                        "triggered_by": "manual",
                        "stocks": target_source.get('config', {}).get('custom_stocks', [])
                    }
                    
                    print(f"📢 直接调用特征工程处理函数...")
                    print(f"   数据源ID: {event_data['source_id']}")
                    print(f"   股票数量: {len(event_data.get('stocks', []))}")
                    
                    # 直接调用处理函数
                    listeners._handle_data_collection_completed(event_data)
                    print(f"✅ 特征工程处理函数调用成功!")
                else:
                    print("❌ 特征工程事件监听器未正确初始化")
                    return False
                    
            except Exception as e:
                print(f"❌ 调用特征工程处理函数失败: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            # 4. 更新数据源状态
            target_source['status'] = 'collecting'
            target_source['last_test'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ 数据源状态已更新为: collecting")
            print(f"✅ 数据采集流程已触发!")
            print(f"\n⏳ 等待特征工程模块响应...")
            print(f"   预期流程: 数据采集 → 特征提取 → 特征选择")
            
            return True
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ 触发数据采集失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 手工触发数据采集流程")
    print("=" * 60)
    success = trigger_data_collection()
    print("=" * 60)
    sys.exit(0 if success else 1)
