#!/usr/bin/env python3
"""
语法错误检查脚本
检查各层级Python文件的语法错误
"""

import os
import ast
from pathlib import Path


def check_syntax_errors(root_dir):
    """检查目录下的语法错误"""
    syntax_errors = []

    root_path = Path(root_dir)
    if not root_path.exists():
        return syntax_errors

    # 遍历所有Python文件
    for py_file in root_path.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()

            # 尝试解析语法
            ast.parse(source, filename=str(py_file))

        except SyntaxError as e:
            syntax_errors.append({
                'file': str(py_file),
                'line': e.lineno,
                'error': str(e),
                'type': 'SyntaxError'
            })
        except Exception as e:
            syntax_errors.append({
                'file': str(py_file),
                'line': 0,
                'error': str(e),
                'type': 'OtherError'
            })

    return syntax_errors


def main():
    """主函数"""
    print("🔍 RQA2025项目语法错误检查")
    print("=" * 50)

    # 检查各个层级
    layers = ['infrastructure', 'features', 'ml', 'trading', 'risk', 'core']

    total_errors = 0
    layer_results = {}

    for layer in layers:
        layer_dir = f'src/{layer}'
        print(f"\n📁 检查 {layer} 层:")

        if os.path.exists(layer_dir):
            errors = check_syntax_errors(layer_dir)
            layer_results[layer] = errors

            if errors:
                print(f"   ❌ 发现 {len(errors)} 个语法错误")
                total_errors += len(errors)

                # 显示前3个错误
                for i, error in enumerate(errors[:3]):
                    print(f"   {i+1}. {Path(error['file']).name}: {error['error']}")

                if len(errors) > 3:
                    print(f"   ... 还有 {len(errors) - 3} 个错误")

            else:
                print("   ✅ 无语法错误")
        else:
            print(f"   ⚠️ 目录不存在: {layer_dir}")

    # 总结
    print("\n" + "=" * 50)
    print("📊 语法检查总结:")
    print(f"总语法错误数: {total_errors}")

    if total_errors > 0:
        print("\n各层级错误统计:")
        for layer, errors in layer_results.items():
            if errors:
                print(f"  {layer}: {len(errors)} 个错误")

        print("\n🔧 下一步建议:")
        print("1. 按照优先级修复语法错误")
        print("2. 从错误数量少的层级开始")
        print("3. 逐个文件修复语法问题")

        # 找出错误最少的层级
        error_counts = {layer: len(errors) for layer, errors in layer_results.items() if errors}
        if error_counts:
            min_layer = min(error_counts, key=error_counts.get)
            print(f"4. 建议优先修复 {min_layer} 层 (错误数量最少)")
    else:
        print("🎉 所有层级语法检查通过!")


if __name__ == "__main__":
    main()
