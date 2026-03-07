import re

# 读取原文件
with open('src/gateway/web/postgresql_persistence.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 移除所有不可见的特殊字符（除了换行符和制表符）
cleaned_content = re.sub(r'[\x00-\x09\x0b-\x1f\x7f]', '', content)

# 写入新文件
with open('cleaned_postgresql_persistence.py', 'w', encoding='utf-8') as f:
    f.write(cleaned_content)

print("文件已清理并保存为 cleaned_postgresql_persistence.py")

# 检查新文件的语法
import ast
try:
    ast.parse(cleaned_content)
    print("清理后的文件语法正确！")
except SyntaxError as e:
    print(f"清理后的文件仍有语法错误：{e}")
    print(f"错误位置：行 {e.lineno}, 列 {e.offset}")
