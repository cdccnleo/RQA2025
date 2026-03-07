import sys
import subprocess
from pathlib import Path

# 需要查找的目录
search_dirs = [
    'tests/unit/infrastructure',
    'tests/unit/data',
    'tests/unit/features',
    'tests/unit/trading'
]

# 收集所有test_*.py文件
all_test_files = []
for d in search_dirs:
    d_path = Path(d)
    if d_path.exists():
        for f in d_path.rglob('test_*.py'):
            all_test_files.append(str(f))

if not all_test_files:
    print('未找到主流程相关的自动生成测试文件！')
    sys.exit(1)

print('即将运行以下测试文件：')
for f in all_test_files:
    print(f)

# 执行pytest
cmd = [sys.executable, '-m', 'pytest', '--maxfail=20', '--disable-warnings', '-q'] + all_test_files
print('\n开始批量执行...')
result = subprocess.run(cmd)
sys.exit(result.returncode)
