#!/usr/bin/env python3
"""
API模块重构状态验证脚本

验证内容:
1. 旧文件是否已移动到deprecated目录
2. 重构文件是否存在且可导入
3. 配置类是否可正常导入
4. 组件类是否可正常导入
5. 文件保存状态检查
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def check_deprecated_files():
    """检查旧文件是否已移动"""
    print_section("1. 检查旧文件移动情况")
    
    api_dir = project_root / "src" / "infrastructure" / "api"
    deprecated_dir = api_dir / "deprecated"
    
    old_files = [
        "api_documentation_enhancer.py",
        "api_documentation_search.py",
        "api_flow_diagram_generator.py",
        "api_test_case_generator.py",
        "openapi_generator.py",
    ]
    
    all_moved = True
    for filename in old_files:
        in_main = (api_dir / filename).exists()
        in_deprecated = (deprecated_dir / filename).exists()
        
        if in_deprecated:
            print(f"  ✅ {filename:45} → deprecated/")
        elif in_main:
            print(f"  ❌ {filename:45} 仍在主目录")
            all_moved = False
        else:
            print(f"  ⚠️  {filename:45} 未找到")
    
    if all_moved:
        print(f"\n  🎉 所有旧文件已成功移动到deprecated目录！")
    else:
        print(f"\n  ⚠️  部分文件仍在主目录，需要继续清理")
    
    return all_moved


def check_refactored_files():
    """检查重构文件是否存在"""
    print_section("2. 检查重构文件存在性")
    
    api_dir = project_root / "src" / "infrastructure" / "api"
    
    refactored_files = [
        "api_documentation_enhancer_refactored.py",
        "api_documentation_search_refactored.py",
        "api_flow_diagram_generator_refactored.py",
        "api_test_case_generator_refactored.py",
        "openapi_generator_refactored.py",
    ]
    
    all_exist = True
    for filename in refactored_files:
        file_path = api_dir / filename
        if file_path.exists():
            # 检查文件大小
            size_kb = file_path.stat().st_size / 1024
            lines = len(file_path.read_text(encoding='utf-8').splitlines())
            print(f"  ✅ {filename:50} ({lines:4}行, {size_kb:6.1f}KB)")
        else:
            print(f"  ❌ {filename:50} 不存在")
            all_exist = False
    
    if all_exist:
        print(f"\n  🎉 所有重构文件存在！")
    else:
        print(f"\n  ❌ 部分重构文件缺失")
    
    return all_exist


def check_component_files():
    """检查组件文件是否存在"""
    print_section("3. 检查组件文件组织")
    
    api_dir = project_root / "src" / "infrastructure" / "api"
    
    # 检查目录结构
    dirs_to_check = [
        "configs",
        "documentation_enhancement",
        "documentation_search",
        "flow_generation/strategies",
        "openapi_generation/builders",
        "test_generation",
    ]
    
    print("\n  📁 目录结构检查:")
    all_dirs_exist = True
    for dir_path in dirs_to_check:
        full_path = api_dir / dir_path
        if full_path.exists():
            py_files = list(full_path.glob("*.py"))
            py_count = len([f for f in py_files if f.name != '__init__.py'])
            print(f"  ✅ {dir_path:40} ({py_count}个组件)")
        else:
            print(f"  ❌ {dir_path:40} 不存在")
            all_dirs_exist = False
    
    return all_dirs_exist


def check_imports():
    """检查导入是否正常（谨慎测试，可能失败）"""
    print_section("4. 检查模块导入（谨慎）")
    
    print("\n  ⚠️  注意: 导入测试可能失败（如果文件未保存）")
    print("  请在编辑器中保存所有文件后再运行此测试\n")
    
    # 测试配置类导入
    print("  测试配置类导入...")
    try:
        from src.infrastructure.api.configs import BaseAPIConfig
        print("  ✅ BaseAPIConfig 导入成功")
    except Exception as e:
        print(f"  ❌ BaseAPIConfig 导入失败: {str(e)[:80]}")
    
    # 测试重构类导入（可能失败）
    print("\n  测试重构类导入...")
    refactored_classes = [
        ("RQAApiDocumentationGenerator", "openapi_generator_refactored"),
        ("APIFlowDiagramGenerator", "api_flow_diagram_generator_refactored"),
        ("APITestCaseGenerator", "api_test_case_generator_refactored"),
        ("APIDocumentationEnhancer", "api_documentation_enhancer_refactored"),
        ("APIDocumentationSearch", "api_documentation_search_refactored"),
    ]
    
    success_count = 0
    for class_name, module_name in refactored_classes:
        try:
            module = __import__(
                f"src.infrastructure.api.{module_name}",
                fromlist=[class_name]
            )
            cls = getattr(module, class_name)
            print(f"  ✅ {class_name:35} 导入成功")
            success_count += 1
        except Exception as e:
            error_msg = str(e)[:100]
            print(f"  ❌ {class_name:35} 导入失败")
            print(f"     错误: {error_msg}")
    
    print(f"\n  导入成功率: {success_count}/{len(refactored_classes)} ({success_count/len(refactored_classes)*100:.0f}%)")
    
    return success_count == len(refactored_classes)


def generate_report():
    """生成验证报告"""
    print_section("📊 API模块重构状态验证报告")
    print(f"\n  验证时间: 2025年10月24日")
    print(f"  验证范围: src/infrastructure/api")
    print(f"  项目根目录: {project_root}\n")
    
    # 执行各项检查
    check1 = check_deprecated_files()
    check2 = check_refactored_files()
    check3 = check_component_files()
    check4 = check_imports()
    
    # 总结
    print_section("📋 验证总结")
    
    checks = [
        ("旧文件移动", check1),
        ("重构文件存在", check2),
        ("组件文件组织", check3),
        ("模块导入测试", check4),
    ]
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    print(f"\n  验证项目: {total}项")
    print(f"  通过项目: {passed}项")
    print(f"  通过率: {passed/total*100:.0f}%\n")
    
    for check_name, result in checks:
        status = "✅ 通过" if result else "❌ 未通过"
        print(f"  {check_name:20} {status}")
    
    print(f"\n{'='*60}\n")
    
    if passed == total:
        print("  🎊 所有验证项目通过！API模块重构完成！")
        return 0
    elif passed >= 3:
        print("  ⚠️  大部分验证通过，但仍有问题需要解决")
        print("  💡 建议：保存编辑器中的所有文件后重新运行此脚本")
        return 1
    else:
        print("  ❌ 多个验证项目失败，需要检查重构文件")
        return 2


if __name__ == "__main__":
    try:
        exit_code = generate_report()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)

