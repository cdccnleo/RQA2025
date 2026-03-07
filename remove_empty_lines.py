# 删除文件中多余的空行

# 读取文件内容
with open('src/gateway/web/postgresql_persistence.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 处理行，删除多余的空行
processed_lines = []
empty_line_count = 0

for line in lines:
    stripped_line = line.strip()
    if not stripped_line:
        empty_line_count += 1
        # 最多保留两个连续的空行
        if empty_line_count <= 2:
            processed_lines.append(line)
    else:
        empty_line_count = 0
        processed_lines.append(line)

# 写入新内容
with open('src/gateway/web/postgresql_persistence.py', 'w', encoding='utf-8') as f:
    f.writelines(processed_lines)

print("成功删除多余的空行")

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
