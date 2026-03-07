#!/usr/bin/env python3
"""
验证数据源删除功能修复
"""

def verify_deletion_fix():
    """验证删除功能的修复"""

    # 检查后端API
    with open('src/gateway/web/api.py', 'r', encoding='utf-8') as f:
        api_content = f.read()

    api_checks = [
        ('DELETE装饰器', '@app.delete("/api/v1/data/sources/{source_id}")' in api_content),
        ('删除函数', 'async def delete_data_source(source_id: str):' in api_content),
        ('删除逻辑', 'sources.pop(i)' in api_content),
        ('保存配置', 'save_data_sources(sources)' in api_content),
        ('成功响应', '"success": True' in api_content),
        ('错误处理', '数据源 {source_id} 不存在' in api_content)
    ]

    print('后端API检查:')
    for check_name, result in api_checks:
        status = '✅' if result else '❌'
        print(f'{status} {check_name}')

    # 检查前端JavaScript
    with open('web-static/data-sources-config.html', 'r', encoding='utf-8') as f:
        html_content = f.read()

    frontend_checks = [
        ('API调用', 'fetch(`/api/v1/data/sources/${sourceId}`' in html_content),
        ('DELETE方法', "method: 'DELETE'" in html_content),
        ('确认对话框', '确定要删除这个数据源吗？此操作不可撤销。' in html_content),
        ('界面刷新', 'await loadDataSources()' in html_content),
        ('统计更新', 'updateStats()' in html_content),
        ('图表更新', 'updateCharts()' in html_content),
        ('加载状态', '删除中...' in html_content),
        ('错误处理', '删除数据源失败' in html_content)
    ]

    print('\n前端JavaScript检查:')
    for check_name, result in frontend_checks:
        status = '✅' if result else '❌'
        print(f'{status} {check_name}')

    # 统计结果
    api_passed = sum(1 for _, result in api_checks if result)
    frontend_passed = sum(1 for _, result in frontend_checks if result)

    print(f'\n后端API: {api_passed}/{len(api_checks)} 通过')
    print(f'前端界面: {frontend_passed}/{len(frontend_checks)} 通过')

    total_passed = api_passed + frontend_passed
    total_checks = len(api_checks) + len(frontend_checks)

    if total_passed == total_checks:
        print('🎉 数据源删除功能修复完整！')
        return True
    else:
        print('⚠️ 数据源删除功能需要进一步完善')
        return False

if __name__ == "__main__":
    import os
    os.chdir('C:\\PythonProject\\RQA2025')
    verify_deletion_fix()
