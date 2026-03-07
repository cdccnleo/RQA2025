with open('src/gateway/web/postgresql_persistence.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 只取文件的前 1505 行
lines = content.split('\n')
first_1505_lines = '\n'.join(lines[:1505])

# 写入临时文件
with open('temp_first_1505_lines.py', 'w', encoding='utf-8') as f:
    f.write(first_1505_lines)

print('已创建临时文件 temp_first_1505_lines.py，包含文件的前 1505 行')
print('开始测试语法...')

import ast
try:
    tree = ast.parse(first_1505_lines)
    print('前 1505 行语法检查通过！')
except SyntaxError as e:
    print(f'前 1505 行语法错误: {e}')
    print(f'错误位置: 第 {e.lineno} 行, 第 {e.offset} 列')
except Exception as e:
    print(f'前 1505 行其他错误: {e}')
