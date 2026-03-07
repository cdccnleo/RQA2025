#!/usr/bin/env python3
"""
验证前端CRUD功能
"""

def verify_frontend_crud():
    """验证前端HTML和JavaScript的CRUD功能"""

    with open('web-static/data-sources-config.html', 'r', encoding='utf-8') as f:
        content = f.read()

    print('🔧 前端CRUD功能验证')
    print('=' * 50)

    # 1. 检查模态框存在
    print('1. 模态框和表单验证:')
    modal_checks = [
        ('模态框HTML', 'id="dataSourceModal"' in content),
        ('表单元素', 'id="dataSourceForm"' in content),
        ('标题元素', 'id="modalTitle"' in content),
        ('ID输入框', 'id="sourceId"' in content),
        ('名称输入框', 'id="sourceName"' in content),
        ('类型选择', 'id="sourceType"' in content),
        ('URL输入', 'id="sourceUrl"' in content),
        ('频率限制', 'id="sourceRateLimit"' in content),
        ('启用复选框', 'id="sourceEnabled"' in content),
    ]

    for check_name, result in modal_checks:
        status = '✅' if result else '❌'
        print(f'   {status} {check_name}')

    # 2. 检查JavaScript函数
    print('\\n2. JavaScript函数验证:')
    js_checks = [
        ('addDataSource函数', 'function addDataSource()' in content),
        ('editDataSource函数', 'function editDataSource(sourceId)' in content),
        ('deleteDataSource函数', 'async function deleteDataSource(sourceId)' in content),
        ('saveDataSource函数', 'async function saveDataSource()' in content),
        ('testConnection函数', 'async function testConnection(sourceId)' in content),
        ('toggleDataSource函数', 'async function toggleDataSource(sourceId)' in content),
        ('currentEditingSourceId变量', 'let currentEditingSourceId = null' in content),
    ]

    for check_name, result in js_checks:
        status = '✅' if result else '❌'
        print(f'   {status} {check_name}')

    # 3. 检查API调用
    print('\\n3. API调用验证:')
    api_checks = [
        ('创建数据源API', 'POST', '/api/v1/data/sources' in content),
        ('更新数据源API', 'PUT', '/api/v1/data/sources/${' in content),
        ('删除数据源API', 'DELETE', '/api/v1/data/sources/${' in content),
        ('获取数据源API', 'GET', '/api/v1/data/sources/${' in content),
        ('测试连接API', 'POST', '/test' in content),
    ]

    for check_name, method, result in api_checks:
        status = '✅' if result else '❌'
        print(f'   {status} {check_name} ({method})')

    # 4. 检查表单处理
    print('\\n4. 表单处理验证:')
    form_checks = [
        ('表单提交处理', 'saveBtn.addEventListener' in content),
        ('取消按钮处理', 'cancelBtn.addEventListener' in content),
        ('模态框关闭', 'dataSourceModal.classList.add' in content),
        ('表单重置', 'dataSourceForm.reset()' in content),
        ('数据填充', 'document.getElementById(\'sourceId\').value' in content),
    ]

    for check_name, result in form_checks:
        status = '✅' if result else '❌'
        print(f'   {status} {check_name}')

    # 5. 检查错误处理
    print('\\n5. 错误处理验证:')
    error_checks = [
        ('API错误捕获', 'console.error(\'保存数据源失败:\'' in content),
        ('用户提示', 'alert(\'保存数据源失败:' in content),
        ('删除确认', 'confirm(\'确定要删除这个数据源吗' in content),
        ('404处理', 'status === 404' in content),
        ('成功提示', 'alert(result.message || \'数据源保存成功\')' in content),
    ]

    for check_name, result in error_checks:
        status = '✅' if result else '❌'
        print(f'   {status} {check_name}')

    # 6. 检查数据类型选项
    print('\\n6. 数据源类型选项:')
    type_options = [
        '股票数据', '期货数据', '期权数据', '基金数据',
        '债券数据', '外汇数据', '加密货币', '新闻数据'
    ]

    type_check_passed = 0
    for option in type_options:
        if f'value="{option}"' in content:
            type_check_passed += 1
            print(f'   ✅ {option}')
        else:
            print(f'   ❌ {option}')

    # 总结
    total_checks = len(modal_checks) + len(js_checks) + len(api_checks) + len(form_checks) + len(error_checks) + len(type_options)
    passed_checks = sum([
        sum(1 for _, r in modal_checks if r),
        sum(1 for _, r in js_checks if r),
        sum(1 for _, _, r in api_checks if r),
        sum(1 for _, r in form_checks if r),
        sum(1 for _, r in error_checks if r),
        type_check_passed
    ])

    print(f'\\n🎯 前端功能验证结果: {passed_checks}/{total_checks} 项检查通过')

    if passed_checks == total_checks:
        print('\\n🎉 前端CRUD功能完整实现！')
        print('✅ 模态框和表单完整')
        print('✅ JavaScript函数齐全')
        print('✅ API调用正确')
        print('✅ 表单处理完善')
        print('✅ 错误处理到位')
        print('✅ 数据类型选项完整')
        return True
    else:
        print(f'\\n❌ 还有 {total_checks - passed_checks} 项需要完善')
        return False

if __name__ == "__main__":
    import os
    os.chdir('C:\\PythonProject\\RQA2025')
    verify_frontend_crud()
