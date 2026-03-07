#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re

def test_data_loading():
    """测试数据源列表加载功能"""
    print("📋 开始数据源列表加载功能测试...")

    try:
        with open('web-static/data-sources-config.html', 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")
        return False

    # 1. 检查表格结构
    table_checks = [
        ('data-sources-table', 'id="data-sources-table"' in content),
        ('表头行', '<thead' in content and '<th' in content),
        ('数据行容器', '<tbody' in content),
        ('加载提示行', 'loading-row' in content),
        ('数据行模板', 'data-source-row' in content)
    ]

    print("\n📊 表格结构检查:")
    table_score = 0
    for check_name, result in table_checks:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            table_score += 1

    # 2. 检查数据加载逻辑
    loading_logic = [
        ('loadDataSources函数', 'function loadDataSources' in content),
        ('初始化调用', 'initDataSourceLoading' in content),
        ('DOM加载事件', 'DOMContentLoaded' in content),
        ('强制刷新功能', 'forceRefreshDataSources' in content),
        ('数据渲染函数', 'renderDataSources' in content)
    ]

    print(f"\n🔄 数据加载逻辑检查:")
    loading_score = 0
    for check_name, result in loading_logic:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            loading_score += 1

    # 3. 检查数据源显示字段
    display_fields = [
        ('数据源名称', 'source.name' in content),
        ('数据源类型', 'source.type' in content),
        ('连接状态', 'connectionStatus' in content),
        ('启用状态', 'source.enabled' in content),
        ('最后测试时间', 'source.last_test' in content),
        ('频率限制', 'source.rate_limit' in content)
    ]

    print(f"\n👁️ 数据源显示字段检查:")
    display_score = 0
    for check_name, result in display_fields:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            display_score += 1

    # 4. 检查数据过滤和排序
    filtering_features = [
        ('禁用数据源过滤', 'showDisabledToggle' in content),
        ('类型徽章样式', 'getTypeBadgeClass' in content),
        ('状态指示器', 'status-indicator' in content),
        ('数据过滤逻辑', 'filter(' in content and 'source' in content),
        ('可见计数更新', 'updateVisibleCount' in content)
    ]

    print(f"\n🔍 数据过滤功能检查:")
    filter_score = 0
    for check_name, result in filtering_features:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            filter_score += 1

    # 5. 检查数据源状态管理
    status_management = [
        ('状态文本生成', 'connectionStatus' in content),
        ('启用状态切换', 'toggleDataSource' in content),
        ('状态样式类', 'enabled-source' in content and 'disabled-source' in content),
        ('状态更新逻辑', 'updateDataSourceRowStatus' in content),
        ('批量状态操作', 'batchEnableSources' in content or 'batchDisableSources' in content)
    ]

    print(f"\n🔄 数据源状态管理检查:")
    status_score = 0
    for check_name, result in status_management:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            status_score += 1

    # 6. 检查数据持久化
    persistence_features = [
        ('数据缓存', 'dataCache' in content),
        ('缓存清理', 'clearCache' in content),
        ('本地存储', 'localStorage' in content),
        ('会话存储', 'sessionStorage' in content),
        ('缓存有效性检查', 'isCacheValid' in content)
    ]

    print(f"\n💾 数据持久化检查:")
    persistence_score = 0
    for check_name, result in persistence_features:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            persistence_score += 1

    # 7. 检查错误处理和用户反馈
    user_feedback = [
        ('加载状态指示', '正在加载数据源' in content),
        ('错误消息显示', '无法加载数据源配置' in content),
        ('重试按钮', '重试' in content),
        ('成功通知', 'showNotification' in content),
        ('用户友好的错误信息', '请检查浏览器开发者工具' in content)
    ]

    print(f"\n💬 用户反馈机制检查:")
    feedback_score = 0
    for check_name, result in user_feedback:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            feedback_score += 1

    # 8. 检查性能优化
    performance_features = [
        ('加载状态防重复', 'isLoadingDataSources' in content),
        ('请求超时控制', 'AbortController' in content),
        ('延迟初始化', 'setTimeout' in content and 'initCharts' in content),
        ('渐进式加载', 'initDataSourceLoading' in content),
        ('资源预加载', 'loadChartJS' in content)
    ]

    print(f"\n⚡ 性能优化检查:")
    performance_score = 0
    for check_name, result in performance_features:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            performance_score += 1

    # 计算总分
    total_score = (table_score + loading_score + display_score + filter_score +
                   status_score + persistence_score + feedback_score + performance_score)
    max_score = (len(table_checks) + len(loading_logic) + len(display_fields) +
                 len(filtering_features) + len(status_management) + len(persistence_features) +
                 len(user_feedback) + len(performance_features))

    # 总结
    print(f"\n🎯 测试总结:")
    print(f"- 表格结构: {table_score}/{len(table_checks)}")
    print(f"- 数据加载: {loading_score}/{len(loading_logic)}")
    print(f"- 显示字段: {display_score}/{len(display_fields)}")
    print(f"- 数据过滤: {filter_score}/{len(filtering_features)}")
    print(f"- 状态管理: {status_score}/{len(status_management)}")
    print(f"- 数据持久化: {persistence_score}/{len(persistence_features)}")
    print(f"- 用户反馈: {feedback_score}/{len(user_feedback)}")
    print(f"- 性能优化: {performance_score}/{len(performance_features)}")
    print(f"- 总分: {total_score}/{max_score} ({total_score/max_score*100:.1f}%)")

    # 判断是否通过
    essential_checks = [
        table_score >= 4,      # 表格结构至少4项通过
        loading_score >= 4,    # 数据加载至少4项通过
        display_score >= 5,    # 显示字段至少5项通过
        filter_score >= 3,     # 数据过滤至少3项通过
        status_score >= 3,     # 状态管理至少3项通过
        feedback_score >= 3,   # 用户反馈至少3项通过
    ]

    success = all(essential_checks)

    if success:
        print("\n🎉 数据源列表加载功能测试通过！")
        return True
    else:
        print("\n⚠️ 数据源列表加载功能测试失败！")
        return False

if __name__ == "__main__":
    import sys
    success = test_data_loading()
    sys.exit(0 if success else 1)
