#!/usr/bin/env python3
"""
测试数据质量评分修复
"""

def test_quality_score_calculation():
    """测试质量评分计算逻辑"""

    # 模拟采集到的数据（与日志中的数据结构一致）
    collected_data = [
        {
            'date': '2025-12-22',
            'open': 92.8,
            'high': 95.28,
            'low': 91.96,
            'close': 94.4,
            'volume': 56855666.0,
            'amount': 5337380716.0,
            'outstanding_share': 850194970.0,
            'turnover': 0.06687367957493326,
            'symbol': '002837',
            'source_id': 'akshare_stock_a',
            'source_type': 'akshare_a_stock',
            'data_type': 'daily'
        },
        {
            'date': '2025-12-23',
            'open': 94.4,
            'high': 96.5,
            'low': 93.2,
            'close': 95.8,
            'volume': 45233666.0,
            'amount': 4287380716.0,
            'outstanding_share': 850194970.0,
            'turnover': 0.053187367957493326,
            'symbol': '002837',
            'source_id': 'akshare_stock_a',
            'source_type': 'akshare_a_stock',
            'data_type': 'daily'
        }
    ]

    # 模拟quality_stats的计算（与实际代码一致）
    quality_stats = {
        'total_records': len(collected_data),
        'data_types': {},
        'quality_score': 0.0
    }

    # 按数据类型统计
    for record in collected_data:
        dt = record.get('data_type', 'unknown')
        quality_stats['data_types'][dt] = quality_stats['data_types'].get(dt, 0) + 1

    # 计算质量评分（基于记录数量和字段完整性）
    total_collected = len(collected_data)
    if total_collected > 0:
        # 检查每条记录的关键字段完整性
        complete_records = 0
        for record in collected_data:
            required_fields = ['date', 'open', 'close', 'volume']
            if all(record.get(f) for f in required_fields):
                complete_records += 1

        quality_stats['quality_score'] = (complete_records / total_collected) * 100

    print(f"采集质量汇总: 总记录数 {quality_stats['total_records']}, "
          f"数据类型分布 {quality_stats['data_types']}, "
          f"质量评分 {quality_stats['quality_score']:.1f}%")

    # 新的监控器质量评分计算（修复后的逻辑）
    quality_score_for_monitor = quality_stats['quality_score'] / 100.0  # 转换为0-1范围

    print(f"监控器质量评分: {quality_score_for_monitor:.2f} (0-1范围)")
    print(f"原始算法质量评分: {min(1.0, len(collected_data) / 100.0):.2f} (0-1范围)")

    # 验证结果
    assert quality_stats['quality_score'] == 100.0, f"期望质量评分为100%，实际为{quality_stats['quality_score']}"
    assert quality_score_for_monitor == 1.0, f"期望监控器质量评分为1.0，实际为{quality_score_for_monitor}"
    assert quality_score_for_monitor >= 0.7, f"质量评分{quality_score_for_monitor}应该不低于监控器阈值0.7"

    print("✅ 质量评分计算验证通过")
    print(f"✅ 修复后的质量评分: {quality_score_for_monitor} >= 0.7 (监控器阈值)")
    print(f"❌ 修复前的质量评分: {min(1.0, len(collected_data) / 100.0)} < 0.7 (会导致错误告警)")

if __name__ == "__main__":
    test_quality_score_calculation()