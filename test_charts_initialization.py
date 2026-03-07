#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re

def test_charts_initialization():
    """测试图表初始化功能"""
    print("📈 开始图表初始化功能测试...")

    try:
        with open('web-static/data-sources-config.html', 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")
        return False

    # 1. 检查Chart.js依赖
    chartjs_dependencies = [
        ('Chart.js库加载', 'loadChartJS' in content),
        ('Chart构造函数检查', 'window.Chart' in content),
        ('Chart.js加载状态', 'chartJSLoaded' in content),
        ('备用Chart对象', 'Mock Chart constructor' in content)
    ]

    print("\n📚 Chart.js依赖检查:")
    dependency_score = 0
    for check_name, result in chartjs_dependencies:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            dependency_score += 1

    # 2. 检查图表Canvas元素
    canvas_elements = re.findall(r'<canvas[^>]*id="([^"]*)"[^>]*>', content)
    expected_charts = ['latencyChart', 'throughputChart', 'errorRateChart', 'availabilityChart', 'healthChart']

    print(f"\n🎨 Canvas元素检查:")
    canvas_score = 0
    for chart_id in expected_charts:
        found = chart_id in canvas_elements
        status = "✅" if found else "❌"
        print(f"  {status} {chart_id}")
        if found:
            canvas_score += 1

    # 3. 检查图表初始化函数
    init_functions = [
        ('initCharts主函数', 'function initCharts()' in content),
        ('updateCharts函数', 'function updateCharts()' in content),
        ('图表销毁逻辑', 'destroy()' in content),
        ('图表重置机制', 'data.datasets = []' in content)
    ]

    print(f"\n🔧 图表初始化函数检查:")
    init_score = 0
    for check_name, result in init_functions:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            init_score += 1

    # 4. 检查图表配置
    chart_configs = [
        ('延迟图表配置', 'type: \'line\'' in content and 'latencyChart'),
        ('吞吐量图表配置', 'type: \'bar\'' in content and 'throughputChart'),
        ('错误率图表配置', 'errorRateChart' in content),
        ('可用性图表配置', 'availabilityChart' in content),
        ('健康评分图表配置', 'type: \'radar\'' in content and 'healthChart')
    ]

    print(f"\n⚙️ 图表配置检查:")
    config_score = 0
    for check_name, result in chart_configs:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            config_score += 1

    # 5. 检查图表数据源
    data_sources = [
        ('API指标数据', 'getDataSourceMetricsUrl' in content),
        ('延迟数据处理', 'latency_data' in content),
        ('吞吐量数据处理', 'throughput_data' in content),
        ('错误率数据处理', 'error_rates' in content),
        ('可用性数据处理', 'availability' in content),
        ('健康评分数据处理', 'health_scores' in content)
    ]

    print(f"\n📊 图表数据源检查:")
    data_score = 0
    for check_name, result in data_sources:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            data_score += 1

    # 6. 检查响应式设计
    responsive_features = [
        ('响应式配置', 'responsive: true' in content),
        ('图表重绘', 'update()' in content),
        ('动态尺寸调整', 'resize' in content or 'Responsive' in content),
        ('图表自适应', 'maintainAspectRatio' in content or 'aspectRatio' in content)
    ]

    print(f"\n📱 响应式设计检查:")
    responsive_score = 0
    for check_name, result in responsive_features:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            responsive_score += 1

    # 7. 检查图表样式和主题
    styling_features = [
        ('颜色方案', 'rgb(' in content or 'rgba(' in content),
        ('图表标题', 'title:' in content and 'display: true' in content),
        ('图表图例', 'legend:' in content),
        ('坐标轴配置', 'scales:' in content),
        ('网格线样式', 'grid:' in content or 'ticks:' in content)
    ]

    print(f"\n🎨 图表样式检查:")
    style_score = 0
    for check_name, result in styling_features:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            style_score += 1

    # 8. 检查性能优化
    performance_features = [
        ('延迟初始化', 'window.pendingChartsInit' in content),
        ('图表缓存', 'latencyChart = new Chart' in content),
        ('内存管理', 'destroy()' in content),
        ('加载状态管理', 'chartJSLoaded' in content)
    ]

    print(f"\n⚡ 性能优化检查:")
    perf_score = 0
    for check_name, result in performance_features:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            perf_score += 1

    # 9. 检查错误处理
    error_handling = [
        ('Chart.js加载失败处理', 'onerror' in content and 'loadChartJS' in content),
        ('图表初始化异常处理', 'try {' in content and 'initCharts' in content),
        ('数据获取失败处理', 'catch' in content and 'updateCharts' in content),
        ('用户友好的错误提示', 'console.warn' in content or 'console.error' in content)
    ]

    print(f"\n🛡️ 图表错误处理检查:")
    error_score = 0
    for check_name, result in error_handling:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            error_score += 1

    # 计算总分
    total_score = (dependency_score + canvas_score + init_score + config_score +
                   data_score + responsive_score + style_score + perf_score + error_score)
    max_score = (len(chartjs_dependencies) + len(expected_charts) + len(init_functions) +
                 len(chart_configs) + len(data_sources) + len(responsive_features) +
                 len(styling_features) + len(performance_features) + len(error_handling))

    # 总结
    print(f"\n🎯 测试总结:")
    print(f"- Chart.js依赖: {dependency_score}/{len(chartjs_dependencies)}")
    print(f"- Canvas元素: {canvas_score}/{len(expected_charts)}")
    print(f"- 初始化函数: {init_score}/{len(init_functions)}")
    print(f"- 图表配置: {config_score}/{len(chart_configs)}")
    print(f"- 数据源: {data_score}/{len(data_sources)}")
    print(f"- 响应式设计: {responsive_score}/{len(responsive_features)}")
    print(f"- 图表样式: {style_score}/{len(styling_features)}")
    print(f"- 性能优化: {perf_score}/{len(performance_features)}")
    print(f"- 错误处理: {error_score}/{len(error_handling)}")
    print(f"- 总分: {total_score}/{max_score} ({total_score/max_score*100:.1f}%)")

    # 判断是否通过
    essential_checks = [
        dependency_score >= 3,    # Chart.js依赖至少3项通过
        canvas_score >= 4,        # 至少4个canvas元素
        init_score >= 3,          # 初始化函数至少3项通过
        config_score >= 3,        # 图表配置至少3项通过
        data_score >= 4,          # 数据源至少4项通过
        error_score >= 2          # 错误处理至少2项通过
    ]

    success = all(essential_checks)

    if success:
        print("\n🎉 图表初始化功能测试通过！")
        return True
    else:
        print("\n⚠️ 图表初始化功能测试失败！")
        return False

if __name__ == "__main__":
    import sys
    success = test_charts_initialization()
    sys.exit(0 if success else 1)
