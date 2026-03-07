import ast

# 读取文件内容
with open('src/gateway/web/postgresql_persistence.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"文件共有 {len(lines)} 行")
print("开始逐行检查语法...")

# 逐行检查
for i in range(1, len(lines) + 1):
    try:
        content = ''.join(lines[:i])
        ast.parse(content)
        if i % 100 == 0:
            print(f"已检查 {i} 行，无语法错误")
    except SyntaxError as e:
        print(f"\n语法错误出现在第 {i} 行")
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
