"""
修复Cache测试中的错误patch路径

分析并修复tests/unit/infrastructure/cache中错误的patch语句
"""

import os
import re
from pathlib import Path


def analyze_patch_errors(file_path):
    """分析文件中的patch错误"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找所有patch语句
    patch_pattern = r"patch\(['\"]([^'\"]+)['\"]\)"
    patches = re.findall(patch_pattern, content)
    
    # 常见的错误patch和修复
    error_patches = {
        '_init_components': None,  # 不存在的方法
        '_setup_multi_level': None,
        '_setup_distributed': None,
        '_setup_monitoring': None,
    }
    
    found_errors = []
    for patch_path in patches:
        method_name = patch_path.split('.')[-1]
        if method_name in error_patches:
            found_errors.append((patch_path, method_name))
    
    return found_errors


def fix_test_patches(file_path):
    """修复测试文件中的patch错误"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return False
    
    original_content = content
    
    # 移除错误的patch装饰器
    # 策略：将整个patch上下文管理器注释掉或删除
    
    # 查找并注释掉错误的patch
    patterns_to_remove = [
        r"with patch\('src\.infrastructure\.cache\.core\.cache_manager\.UnifiedCacheManager\._init_components'\).*?,\s*\\\n",
        r"patch\('src\.infrastructure\.cache\.core\.cache_manager\.UnifiedCacheManager\._init_components'\),\s*\\\n",
        r"with patch\('.*\._init_components'\).*?,\s*\\\n",
    ]
    
    for pattern in patterns_to_remove:
        content = re.sub(pattern, '', content)
    
    # 如果内容改变，写回
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"写入失败: {e}")
            return False
    
    return False


def batch_fix_cache_patches():
    """批量修复cache测试的patch错误"""
    
    cache_test_dir = Path('tests/unit/infrastructure/cache')
    
    if not cache_test_dir.exists():
        print("目录不存在")
        return
    
    analyzed_files = []
    fixed_files = []
    
    for test_file in cache_test_dir.rglob('test_*.py'):
        # 先分析
        errors = analyze_patch_errors(test_file)
        if errors:
            print(f"\n📝 {test_file.name}")
            for patch_path, method in errors:
                print(f"  ⚠️ 错误patch: {method}")
            analyzed_files.append(test_file.name)
            
            # 尝试修复
            if fix_test_patches(test_file):
                fixed_files.append(test_file.name)
                print(f"  ✅ 已修复")
    
    print(f"\n总结：")
    print(f"  分析文件：{len(analyzed_files)}")
    print(f"  修复文件：{len(fixed_files)}")
    
    return fixed_files


if __name__ == '__main__':
    print("分析和修复Cache测试patch错误...")
    print("=" * 70)
    batch_fix_cache_patches()


