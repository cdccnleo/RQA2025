with open('src/gateway/web/postgresql_persistence.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 取文件的第 1500-1510 行
lines = content.split('\n')
lines_1500_1510 = '\n'.join(lines[1499:1510])

# 写入临时文件
with open('temp_lines_1500_1510.py', 'w', encoding='utf-8') as f:
    f.write(lines_1500_1510)

print('已创建临时文件 temp_lines_1500_1510.py，包含文件的第 1500-1510 行')
print('文件内容:')
print(lines_1500_1510)
print('\n开始测试语法...')

import ast
try:
    # 添加必要的导入和上下文
    test_content = 'from typing import List, Dict, Any\n' + lines_1500_1510
    tree = ast.parse(test_content)
    print('语法检查通过！')
except SyntaxError as e:
    print(f'语法错误: {e}')
    print(f'错误位置: 第 {e.lineno} 行, 第 {e.offset} 列')
except Exception as e:
    print(f'其他错误: {e}')
