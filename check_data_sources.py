#!/usr/bin/env python3
"""
检查数据源配置和活跃数据源采集情况
"""
import json
import os
import sys

# 添加项目路径
sys.path.insert(0, '/app')

def check_data_sources_config():
    print('=' * 70)
    print('🔍 数据源配置与活跃采集检查')
    print('=' * 70)
    
    # 1. 检查配置文件
    print('\n📁 1. 数据源配置文件检查')
    print('-' * 70)
    
    config_paths = [
        '/app/data/data_sources_config.json',
        '/app/data/production/data_sources_config.json',
        'data/data_sources_config.json',
    ]
    
    config_data = None
    for path in config_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                print(f"  ✅ 找到配置文件: {path}")
                print(f"  📊 配置数据源数量: {len(config_data)}")
                break
            except Exception as e:
                print(f"  ⚠️ 读取失败 {path}: {e}")
    
    if not config_data:
        print("  ❌ 未找到任何数据源配置文件")
        return
    
    # 2. 分析数据源状态
    print('\n📊 2. 数据源状态分析')
    print('-' * 70)
    
    enabled_sources = [s for s in config_data if s.get('enabled', False)]
    disabled_sources = [s for s in config_data if not s.get('enabled', False)]
    
    print(f"  总数据源: {len(config_data)}")
    print(f"  ✅ 已启用 (enabled=true): {len(enabled_sources)}")
    print(f"  ⏸️ 已禁用 (enabled=false): {len(disabled_sources)}")
    
    # 按状态分类
    status_count = {}
    for source in config_data:
        status = source.get('status', '未知')
        status_count[status] = status_count.get(status, 0) + 1
    
    print(f"\n  连接状态分布:")
    for status, count in status_count.items():
        icon = '✅' if '正常' in status or '200' in status else '⚠️' if '失败' in status or '超时' in status else '❓'
        print(f"    {icon} {status}: {count}个")
    
    # 3. 显示活跃数据源详情
    print('\n📡 3. 活跃数据源详情 (enabled=true)')
    print('-' * 70)
    
    if enabled_sources:
        for i, source in enumerate(enabled_sources[:10], 1):
            name = source.get('name', 'N/A')
            source_type = source.get('type', 'N/A')
            status = source.get('status', '未知')
            last_test = source.get('last_test', '从未')
            
            status_icon = '✅' if '正常' in status or '200' in status else '⚠️'
            print(f"  {i}. {name}")
            print(f"     类型: {source_type}")
            print(f"     状态: {status_icon} {status}")
            print(f"     最后测试: {last_test}")
            print()
        
        if len(enabled_sources) > 10:
            print(f"  ... 还有 {len(enabled_sources) - 10} 个活跃数据源")
    else:
        print("  ⚠️ 没有启用的数据源！")
    
    # 4. 检查系统实际加载的数据源
    print('\n🔧 4. 系统加载的数据源检查')
    print('-' * 70)
    
    try:
        from src.gateway.web.config_manager import load_data_sources
        system_sources = load_data_sources()
        
        print(f"  系统加载的数据源: {len(system_sources)}")
        
        system_enabled = [s for s in system_sources if s.get('enabled', False)]
        print(f"  系统中活跃数据源: {len(system_enabled)}")
        
        if len(system_sources) != len(config_data):
            print(f"  ⚠️ 警告: 配置文件与系统加载数量不一致!")
            print(f"     配置文件: {len(config_data)}")
            print(f"     系统加载: {len(system_sources)}")
        else:
            print(f"  ✅ 配置文件与系统加载一致")
            
    except Exception as e:
        print(f"  ❌ 检查系统数据源失败: {e}")
    
    # 5. 检查调度器中的数据源
    print('\n🎯 5. 调度器数据源状态')
    print('-' * 70)
    
    try:
        import requests
        
        # 获取调度器仪表盘
        response = requests.get('http://localhost:8000/api/v1/data/scheduler/dashboard', timeout=5)
        dashboard = response.json()
        
        scheduler_info = dashboard.get('scheduler', {})
        print(f"  调度器运行: {'✅' if scheduler_info.get('running') else '❌'}")
        print(f"  活跃数据源: {scheduler_info.get('active_sources', 0)}")
        print(f"  总数据源: {scheduler_info.get('total_sources', 0)}")
        
        # 获取健康检查
        response = requests.get('http://localhost:8000/api/v1/health', timeout=5)
        health = response.json()
        
        data_health = health.get('data', {})
        print(f"\n  健康检查:")
        print(f"    数据源总数: {data_health.get('data_sources', 0)}")
        print(f"    活跃数据源: {data_health.get('active_sources', 0)}")
        
    except Exception as e:
        print(f"  ❌ 获取调度器状态失败: {e}")
    
    # 6. 总结和建议
    print('\n' + '=' * 70)
    print('📋 检查总结与建议')
    print('=' * 70)
    
    config_enabled_count = len(enabled_sources) if config_data else 0
    
    print(f"\n  📊 数据源统计:")
    print(f"    配置文件活跃数据源: {config_enabled_count}")
    print(f"    系统加载数据源: {len(system_sources) if 'system_sources' in locals() else 'N/A'}")
    
    if config_enabled_count == 0:
        print(f"\n  ❌ 严重问题: 配置文件中没有启用的数据源!")
        print(f"     建议: 检查 data/data_sources_config.json 中的 enabled 字段")
    elif config_enabled_count > 0 and scheduler_info.get('active_sources', 0) == 0:
        print(f"\n  ⚠️ 问题: 配置文件有 {config_enabled_count} 个活跃数据源，但调度器显示 0 个!")
        print(f"     可能原因:")
        print(f"       1. 数据源未正确加载到数据库")
        print(f"       2. 调度器使用的是不同的数据源列表")
        print(f"       3. 数据源加载逻辑有bug")
        print(f"\n     建议检查:")
        print(f"       - config_manager.load_data_sources() 是否正确加载")
        print(f"       - 数据库中是否有数据源记录")
        print(f"       - 调度器 dashboard API 的数据源统计逻辑")
    else:
        print(f"\n  ✅ 数据源配置正常")
        print(f"     活跃数据源: {config_enabled_count}个")
        print(f"     调度器识别: {scheduler_info.get('active_sources', 0)}个")
    
    print('\n' + '=' * 70)

if __name__ == '__main__':
    check_data_sources_config()
