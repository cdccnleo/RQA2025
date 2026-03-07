#!/usr/bin/env python3
"""
修复生成的组件文件中的变量作用域问题
"""

from pathlib import Path


def fix_component_file(file_path):
    """修复单个组件文件"""
    print(f"🔧 修复文件: {file_path.name}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复component_type变量问题
    content = content.replace(
        'component_type_name = component_type.replace(\'_templates\', \'\').title()',
        'component_type_name = "Cache"'
    )

    # 修复其他可能的变量问题
    content = content.replace(
        'f"{component_type_name}_Component_{component_id}"',
        'f"{component_type_name}_Component_{{component_id}}"'
    )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"   ✅ 已修复: {file_path.name}")


def main():
    """主函数"""
    cache_dir = Path("src/infrastructure/cache")

    component_files = [
        "cache_components.py",
        "client_components.py",
        "strategy_components.py",
        "optimizer_components.py"
    ]

    print("🚀 开始修复组件文件...")
    print("="*50)

    for filename in component_files:
        file_path = cache_dir / filename
        if file_path.exists():
            fix_component_file(file_path)
        else:
            print(f"   ⚠️  文件不存在: {filename}")

    print("\n" + "="*50)
    print("✅ 组件文件修复完成！")


if __name__ == "__main__":
    main()
