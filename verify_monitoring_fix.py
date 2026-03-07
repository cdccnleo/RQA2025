#!/usr/bin/env python3
"""
验证数据源监控修复
"""

def verify_monitoring_fix():
    """验证数据源监控图表修复"""
    with open('web-static/data-sources-config.html', 'r', encoding='utf-8') as f:
        content = f.read()

    results = []

    # 检查延迟图表
    if "label: 'MiniQMT'" in content and "label: '东方财富'" in content:
        # 检查是否移除了其他数据源
        if "label: 'Alpha Vantage'" not in content or "label: 'Binance API'" not in content:
            results.append("✅ 延迟图表：只显示启用的A股数据源")
        else:
            results.append("❌ 延迟图表：仍显示禁用的数据源")
    else:
        results.append("❌ 延迟图表：未找到启用的数据源")

    # 检查吞吐量数据
    import re
    throughput_match = re.search(r'label: \'数据吞吐量[^}]*data: \[([^\]]+)\]', content, re.DOTALL)
    if throughput_match:
        data_str = throughput_match.group(1)
        # 提取数字
        data_values = []
        for item in data_str.split(','):
            item = item.strip()
            if item.isdigit():
                data_values.append(int(item))

        if len(data_values) >= 14:
            # 检查MiniQMT和东方财富位置
            miniqmt_val = data_values[4]  # 索引4是MiniQMT
            emweb_val = data_values[7]    # 索引7是东方财富

            if miniqmt_val > 0 and emweb_val > 0:
                results.append("✅ 吞吐量图表：启用的数据源有数据")
            else:
                results.append("❌ 吞吐量图表：启用的数据源没有数据")

            # 检查其他位置是否为0
            other_positions = [0,1,2,3,5,6,8,9,10,11,12,13]
            others_zero = all(data_values[i] == 0 for i in other_positions if i < len(data_values))

            if others_zero:
                results.append("✅ 吞吐量图表：禁用的数据源显示为0")
            else:
                results.append("❌ 吞吐量图表：某些禁用的数据源仍有数据")
        else:
            results.append(f"❌ 吞吐量数据长度错误: {len(data_values)}")
    else:
        results.append("❌ 未找到吞吐量数据")

    # 检查标题
    if '启用数据源连接延迟监控' in content:
        results.append("✅ 延迟监控标题已更新")
    else:
        results.append("❌ 延迟监控标题未更新")

    if '启用数据源吞吐量统计' in content:
        results.append("✅ 吞吐量统计标题已更新")
    else:
        results.append("❌ 吞吐量统计标题未更新")

    return results

if __name__ == "__main__":
    import os
    os.chdir('C:\\PythonProject\\RQA2025')

    results = verify_monitoring_fix()
    for result in results:
        print(result)

    print("\n🎉 验证完成！")
