"""
修复API测试中的patch路径错误

将错误的.generate patch改为正确的.create_flow
"""

import re
from pathlib import Path


def fix_api_patch_in_file(file_path):
    """修复API测试文件中的patch错误"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return False
    
    original_content = content
    
    # 修复策略类的patch路径
    # 错误: .generate  正确: .create_flow
    replacements = [
        (r"(\w+FlowStrategy)'\)\.generate", r"\1').create_flow"),
        (r"(TradingFlowStrategy.*?)\.generate", r"\1.create_flow"),
        (r"(DataServiceFlowStrategy.*?)\.generate", r"\1.create_flow"),
        (r"(FeatureFlowStrategy.*?)\.generate", r"\1.create_flow"),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            return False
    
    return False


def batch_fix_api_tests():
    """批量修复API测试"""
    
    api_test_dir = Path('tests/unit/infrastructure/api')
    
    if not api_test_dir.exists():
        print("目录不存在")
        return []
    
    fixed_files = []
    for test_file in api_test_dir.rglob('test_*.py'):
        if fix_api_patch_in_file(test_file):
            fixed_files.append(test_file.name)
            print(f"✅ 修复: {test_file.name}")
    
    print(f"\n修复完成！总计: {len(fixed_files)}个文件")
    return fixed_files


if __name__ == '__main__':
    print("修复API测试patch路径...")
    print("=" * 70)
    fixed = batch_fix_api_tests()


