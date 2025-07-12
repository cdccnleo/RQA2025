import os
import sys

def check_test_structure():
    """验证测试目录结构是否符合规范"""
    import os
    import sys

    def print_directory_structure(startpath):
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print('{}{}'.format(subindent, f))

    def validate_test_structure():
        # 强制检查当前目录下的tests目录
        current_dir = os.path.abspath(os.getcwd())
        test_dir = os.path.join(current_dir, 'tests')
        
        print(f"\n当前工作目录: {current_dir}")
        print("\n完整目录结构:")
        print_directory_structure(current_dir)
        
        print(f"\n详细检查tests目录: {test_dir}")
        if not os.path.exists(test_dir):
            print("\n❌ 错误: 未找到tests目录")
            print_directory_structure(current_dir)
            sys.exit(1)
            
        print("\ntests目录结构:")
        print_directory_structure(test_dir)
        
        # 检查tests目录是否存在
        if not os.path.exists(test_dir):
            print(f"❌ Error: Missing 'tests' directory in {current_dir}")
            sys.exit(1)
        
        # 必需目录列表
        required_dirs = [
            ('unit/infrastructure/cache', '单元测试缓存目录'),
            ('integration', '集成测试目录'),
            ('performance', '性能测试目录')
        ]
        
        missing_dirs = []
        
        # 检查每个必需目录
        for rel_path, desc in required_dirs:
            full_path = os.path.join(test_dir, rel_path)
            if not os.path.exists(full_path):
                missing_dirs.append((rel_path, desc, full_path))
        
        # 报告缺失目录
        if missing_dirs:
            print("\n❌ Missing required test directories:")
            for rel_path, desc, full_path in missing_dirs:
                print(f" - {desc} ({rel_path})")
                print(f"   Expected at: {full_path}")
            sys.exit(1)
        
        print("\n✅ Test directory structure validation passed")
        sys.exit(0)

    if __name__ == "__main__":
        validate_test_structure()
    required_dirs = [
        'unit/infrastructure/cache',
        'integration',
        'performance'
    ]

    missing = []
    for rel_path in required_dirs:
        path = os.path.join(base_path, rel_path)
        if not os.path.exists(path):
            missing.append(rel_path)

    if missing:
        print(f"错误：缺少以下目录: {', '.join(missing)}")
        return False

    print("测试目录结构验证通过")
    return True

if __name__ == '__main__':
    sys.exit(0 if check_test_structure() else 1)
