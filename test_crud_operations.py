#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re

def test_crud_operations():
    """测试数据源CRUD操作功能"""
    print("🔧 开始数据源CRUD操作功能测试...")

    try:
        with open('web-static/data-sources-config.html', 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")
        return False

    # 1. 检查创建功能 (Create)
    create_features = [
        ('添加按钮', 'addDataSource' in content),
        ('模态框表单', 'dataSourceModal' in content),
        ('表单验证', 'required' in content),
        ('数据提交', 'saveDataSource' in content),
        ('成功反馈', '数据源.*创建成功' in content)
    ]

    print("\n➕ 创建功能检查:")
    create_score = 0
    for check_name, result in create_features:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            create_score += 1

    # 2. 检查读取功能 (Read) - 已经在数据加载测试中验证
    read_features = [
        ('数据列表显示', 'renderDataSources' in content),
        ('详细信息查看', 'viewDataSample' in content),
        ('数据过滤', 'filter(' in content and 'source' in content)
    ]

    print(f"\n👁️ 读取功能检查:")
    read_score = 0
    for check_name, result in read_features:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            read_score += 1

    # 3. 检查更新功能 (Update)
    update_features = [
        ('编辑按钮', 'editDataSource' in content),
        ('表单预填充', 'currentEditingSourceId' in content),
        ('数据更新', 'PUT' in content and 'updateDataSource'),
        ('状态切换', 'toggleDataSource' in content),
        ('更新反馈', '数据源.*已.*更新' in content or '已.*启用' in content or '已.*禁用' in content)
    ]

    print(f"\n✏️ 更新功能检查:")
    update_score = 0
    for check_name, result in update_features:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            update_score += 1

    # 4. 检查删除功能 (Delete)
    delete_features = [
        ('删除按钮', 'deleteDataSource' in content),
        ('确认对话框', 'confirm(' in content),
        ('删除请求', 'DELETE' in content),
        ('删除反馈', '数据源.*已删除' in content),
        ('DOM更新', 'remove()' in content)
    ]

    print(f"\n🗑️ 删除功能检查:")
    delete_score = 0
    for check_name, result in delete_features:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            delete_score += 1

    # 5. 检查表单处理
    form_handling = [
        ('动态表单生成', 'renderConfigItems' in content),
        ('配置项收集', 'collectConfigData' in content),
        ('表单验证', 'checkPageReady' in content),
        ('表单重置', 'reset()' in content or 'clear' in content)
    ]

    print(f"\n📝 表单处理检查:")
    form_score = 0
    for check_name, result in form_handling:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            form_score += 1

    # 6. 检查状态管理
    state_management = [
        ('启用状态处理', 'enabled' in content and 'source.enabled' in content),
        ('状态同步', 'updateDataSourceRowStatus' in content),
        ('视觉反馈', 'status-indicator' in content),
        ('状态持久化', 'last_test' in content or 'status' in content)
    ]

    print(f"\n🔄 状态管理检查:")
    state_score = 0
    for check_name, result in state_management:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            state_score += 1

    # 计算总分
    total_score = create_score + read_score + update_score + delete_score + form_score + state_score
    max_score = len(create_features) + len(read_features) + len(update_features) + len(delete_features) + len(form_handling) + len(state_management)

    # 总结
    print(f"\n🎯 测试总结:")
    print(f"- 创建功能: {create_score}/{len(create_features)}")
    print(f"- 读取功能: {read_score}/{len(read_features)}")
    print(f"- 更新功能: {update_score}/{len(update_features)}")
    print(f"- 删除功能: {delete_score}/{len(delete_features)}")
    print(f"- 表单处理: {form_score}/{len(form_handling)}")
    print(f"- 状态管理: {state_score}/{len(state_management)}")
    print(f"- 总分: {total_score}/{max_score} ({total_score/max_score*100:.1f}%)")

    # 判断是否通过
    essential_checks = [
        create_score >= 3,    # 创建功能至少3项通过
        read_score >= 2,      # 读取功能至少2项通过
        update_score >= 3,    # 更新功能至少3项通过
        delete_score >= 3,    # 删除功能至少3项通过
        form_score >= 2,      # 表单处理至少2项通过
    ]

    success = all(essential_checks)

    if success:
        print("\n🎉 数据源CRUD操作功能测试通过！")
        return True
    else:
        print("\n⚠️ 数据源CRUD操作功能测试失败！")
        return False

if __name__ == "__main__":
    import sys
    success = test_crud_operations()
    sys.exit(0 if success else 1)