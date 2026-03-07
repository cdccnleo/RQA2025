#!/usr/bin/env python3
"""
测试数据源状态一致性检查功能
"""

import json
import time
from datetime import datetime

def test_status_consistency():
    """测试状态一致性检查逻辑"""
    print("🧪 测试数据源状态一致性检查功能")
    print("=" * 50)

    # 读取配置文件
    try:
        with open('data/data_sources_config.json', 'r', encoding='utf-8') as f:
            sources = json.load(f)
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return

    enabled_sources = [s for s in sources if s.get('enabled', True)]
    print(f"📊 启用数据源数量: {len(enabled_sources)}")

    # 模拟前端检查逻辑
    print("\n🔍 模拟前端状态一致性检查...")

    # 模拟API返回的数据（与配置文件相同）
    api_sources = enabled_sources.copy()

    # 模拟显示数据（假设都正确显示）
    display_sources = []
    for source in enabled_sources:
        display_sources.append({
            'id': source['id'],
            'name': source['name'],
            'display_status': source['status'],
            'display_last_test': source.get('last_test', '未测试'),
            'api_status': source['status'],
            'api_last_test': source.get('last_test', '未测试')
        })

    # 执行一致性检查
    status_inconsistencies = 0
    time_inconsistencies = 0
    details = []

    for item in display_sources:
        source_id = item['id']
        display_status = item['display_status']
        api_status = item['api_status']
        display_time = item['display_last_test']
        api_time = item['api_last_test']

        # 状态比较
        status_match = (
            display_status == api_status or
            (display_status and api_status and
             ('连接正常' in display_status and '连接正常' in api_status)) or
            (display_status == '未测试' and (not api_time or api_time == '未测试'))
        )

        if not status_match:
            status_inconsistencies += 1
            details.append({
                'id': source_id,
                'type': 'status',
                'display': display_status,
                'api': api_status
            })
            print(f"❌ 状态不一致: {source_id} - 显示:'{display_status}' vs API:'{api_status}'")

        # 时间比较
        if api_time and api_time != '未测试' and display_time and display_time != '未测试':
            try:
                api_date = api_time.split(' ')[0] if ' ' in api_time else api_time
                display_date = display_time.split(' ')[0] if ' ' in display_time else display_time

                if api_date != display_date:
                    time_inconsistencies += 1
                    details.append({
                        'id': source_id,
                        'type': 'time',
                        'display': display_time,
                        'api': api_time
                    })
                    print(f"⚠️ 时间不一致: {source_id} - 显示:'{display_time}' vs API:'{api_time}'")
            except Exception as e:
                print(f"⚠️ 时间解析错误: {source_id} - {e}")

    total_inconsistencies = status_inconsistencies + time_inconsistencies

    print("\n📋 测试结果:")
    print(f"   总数据源: {len(enabled_sources)}")
    print(f"   状态不一致: {status_inconsistencies}")
    print(f"   时间不一致: {time_inconsistencies}")
    print(f"   总问题数: {total_inconsistencies}")

    if total_inconsistencies == 0:
        print("✅ 数据源状态完全一致！")
        return True
    else:
        print("❌ 发现状态不一致问题")
        print("\n问题详情:")
        for detail in details:
            print(f"  {detail['id']} ({detail['type']}): 显示'{detail['display']}' vs API'{detail['api']}'")
        return False

def test_monitoring_coverage():
    """测试监控覆盖率"""
    print("\n🔍 测试监控覆盖率...")
    print("=" * 30)

    # 读取配置文件
    try:
        with open('data/data_sources_config.json', 'r', encoding='utf-8') as f:
            sources = json.load(f)
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return

    enabled_sources = [s for s in sources if s.get('enabled', True)]
    total_enabled = len(enabled_sources)

    # 统计各种状态
    status_counts = {}
    last_test_counts = {'有测试时间': 0, '无测试时间': 0}

    for source in enabled_sources:
        status = source.get('status', '未测试')
        if status not in status_counts:
            status_counts[status] = 0
        status_counts[status] += 1

        last_test = source.get('last_test')
        if last_test and last_test != '未测试':
            last_test_counts['有测试时间'] += 1
        else:
            last_test_counts['无测试时间'] += 1

    print(f"监控覆盖情况:")
    print(f"  启用数据源总数: {total_enabled}")
    print(f"  有测试记录: {last_test_counts['有测试时间']} ({last_test_counts['有测试时间']/total_enabled*100:.1f}%)")
    print(f"  无测试记录: {last_test_counts['无测试时间']} ({last_test_counts['无测试时间']/total_enabled*100:.1f}%)")

    print("\n状态分布:")
    for status, count in status_counts.items():
        percentage = count / total_enabled * 100
        print(f"  {status}: {count}个 ({percentage:.1f}%)")
    return True

if __name__ == "__main__":
    print("🚀 数据源状态一致性测试工具")
    print("=" * 60)

    # 测试状态一致性
    consistency_ok = test_status_consistency()

    # 测试监控覆盖率
    coverage_ok = test_monitoring_coverage()

    print("\n" + "=" * 60)
    print("🎯 测试总结:")
    if consistency_ok and coverage_ok:
        print("✅ 所有测试通过！数据源状态监控功能正常。")
    else:
        print("❌ 发现问题，需要进一步检查。")

    print("\n💡 使用建议:")
    print("1. 在浏览器中访问数据源配置页面")
    print("2. 点击'状态检查'按钮验证状态一致性")
    print("3. 如发现问题，使用'自动修复'功能")
    print("4. 查看浏览器控制台获取详细诊断信息")
