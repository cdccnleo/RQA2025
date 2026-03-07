# 修复文件结构，移除多余的空行

# 读取原文件内容
with open('src/gateway/web/postgresql_persistence.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 处理行，移除多余的空行，保持代码结构
processed_lines = []

for line in lines:
    stripped_line = line.strip()
    # 只添加非空行
    if stripped_line:
        processed_lines.append(line)

# 写入新内容
with open('src/gateway/web/postgresql_persistence.py', 'w', encoding='utf-8') as f:
    f.writelines(processed_lines)

print("成功修复文件结构，移除了所有多余的空行")

# 检查语法
import ast
try:
    with open('src/gateway/web/postgresql_persistence.py', 'r', encoding='utf-8') as f:
        content = f.read()
    ast.parse(content)
    print("文件语法正确！")
except SyntaxError as e:
    print(f"语法错误：{e}")
    print(f"错误位置：行 {e.lineno}, 列 {e.offset}")
