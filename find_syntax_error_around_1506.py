import ast

# 读取文件内容
with open('src/gateway/web/postgresql_persistence.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"文件共有 {len(lines)} 行")
print("开始检查第 1495-1515 行附近的语法...")

# 从第 1495 行开始检查
start_line = 1494  # 索引从 0 开始
for i in range(start_line, min(start_line + 20, len(lines))):
    try:
        # 检查从文件开始到当前行的内容
        content = ''.join(lines[:i+1])
        ast.parse(content)
        print(f"第 {i+1} 行语法正确")
    except SyntaxError as e:
        print(f"\n语法错误出现在第 {i+1} 行")
        print(f"错误信息：{e}")
        print(f"错误位置：行 {e.lineno}, 列 {e.offset}")
        # 打印错误行附近的代码
        if e.lineno > 0 and e.lineno <= len(lines):
            print(f"错误行内容：{repr(lines[e.lineno-1])}")
            if e.lineno > 1:
                print(f"上一行内容：{repr(lines[e.lineno-2])}")
            if e.lineno < len(lines):
                print(f"下一行内容：{repr(lines[e.lineno])}")
        break

print("检查完成")
