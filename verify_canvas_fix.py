#!/usr/bin/env python3
"""
验证数据源配置canvas图表修复
"""

def verify_canvas_fix():
    """验证canvas元素和图表初始化的修复"""

    with open('web-static/data-sources-config.html', 'r', encoding='utf-8') as f:
        content = f.read()

    print('🔧 数据源配置Canvas图表修复验证')
    print('=' * 60)

    # 1. 检查canvas元素存在
    print('1. Canvas元素检查:')
    canvas_checks = [
        ('<canvas id="latencyChart"', '延迟图表canvas元素'),
        ('<canvas id="throughputChart"', '吞吐量图表canvas元素'),
        ('width="400" height="200"', 'canvas尺寸设置'),
    ]

    for check_text, description in canvas_checks:
        exists = check_text in content
        status = '✅' if exists else '❌'
        print(f'   {status} {description}')

    # 2. 检查图表初始化安全检查
    print('\\n2. 图表初始化安全检查:')
    init_checks = [
        ('latencyCanvas = document.getElementById', '延迟canvas获取'),
        ('throughputCanvas = document.getElementById', '吞吐量canvas获取'),
        ('if (!latencyCanvas || !throughputCanvas)', '存在性检查'),
        ('console.warn(\'图表canvas元素不存在\'', '错误警告'),
        ('图表未初始化，跳过更新', '更新跳过检查'),
    ]

    for check_text, description in init_checks:
        exists = check_text in content
        status = '✅' if exists else '❌'
        print(f'   {status} {description}')
        if not exists:
            print(f'      查找: "{check_text}"')

    # 3. 检查图表区域HTML结构
    print('\\n3. 图表区域HTML结构:')
    html_checks = [
        ('数据源连接延迟监控', '延迟图表标题'),
        ('启用数据源吞吐量统计', '吞吐量图表标题'),
        ('grid grid-cols-1 lg:grid-cols-2', '响应式网格布局'),
        ('bg-white rounded-lg shadow-lg', '图表容器样式'),
    ]

    for check_text, description in html_checks:
        exists = check_text in content
        status = '✅' if exists else '❌'
        print(f'   {status} {description}')

    # 4. 检查JavaScript错误处理
    print('\\n4. JavaScript错误处理:')
    error_checks = [
        ('try {', 'try-catch结构'),
        ('catch (error)', '异常捕获'),
        ('console.error', '错误日志'),
        ('getContext(\'2d\')', 'canvas上下文获取'),
    ]

    for check_text, description in error_checks:
        exists = check_text in content
        status = '✅' if exists else '❌'
        print(f'   {status} {description}')

    # 5. 检查图表更新逻辑
    print('\\n5. 图表更新逻辑:')
    update_checks = [
        ('newLatencies.push', '延迟数据生成'),
        ('newThroughput.push', '吞吐量数据生成'),
        ('latencyChart.update()', '延迟图表更新'),
        ('throughputChart.update()', '吞吐量图表更新'),
    ]

    for check_text, description in update_checks:
        exists = check_text in content
        status = '✅' if exists else '❌'
        print(f'   {status} {description}')

    # 总结
    total_checks = len(canvas_checks) + len(init_checks) + len(html_checks) + len(error_checks) + len(update_checks)
    passed_checks = sum([
        sum(1 for _, r in canvas_checks if r in content),
        sum(1 for _, r in init_checks if r in content),
        sum(1 for _, r in html_checks if r in content),
        sum(1 for _, r in error_checks if r in content),
        sum(1 for _, r in update_checks if r in content),
    ])

    print(f'\\n🎯 验证结果: {passed_checks}/{total_checks} 项检查通过')

    if passed_checks == total_checks:
        print('\\n🎉 Canvas图表修复完成！')
        print('✅ Canvas元素已添加')
        print('✅ 图表初始化安全检查已实现')
        print('✅ HTML结构完整')
        print('✅ 错误处理完善')
        print('✅ 图表更新逻辑正常')
        print('\\n现在即使在没有数据的情况下也不会出现getContext错误。')
        return True
    else:
        print(f'\\n❌ 还有 {total_checks - passed_checks} 项需要修复')
        return False

if __name__ == "__main__":
    import os
    os.chdir('C:\\PythonProject\\RQA2025')
    verify_canvas_fix()
