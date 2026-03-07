import re

# 读取正确的函数定义
with open('correct_function_final.py', 'r', encoding='utf-8') as f:
    correct_function = f.read()

# 读取原文件内容
with open('src/gateway/web/postgresql_persistence.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 使用正则表达式匹配整个 get_stocks_by_industry 函数
# 匹配函数定义开始
pattern = r'def get_stocks_by_industry\(.*?\):[\s\S]*?(?=def |$)'

# 替换函数定义
new_content = re.sub(pattern, correct_function, content, flags=re.DOTALL)

# 写入新内容
with open('src/gateway/web/postgresql_persistence.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("成功替换 get_stocks_by_industry 函数")

# 检查语法
import ast
try:
    ast.parse(new_content)
    print("文件语法正确！")
except SyntaxError as e:
    print(f"语法错误：{e}")
    print(f"错误位置：行 {e.lineno}, 列 {e.offset}")
