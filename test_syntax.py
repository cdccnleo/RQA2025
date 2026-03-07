import ast

# 读取文件的前 1510 行
with open('src/gateway/web/postgresql_persistence.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    content = ''.join(lines[:1510])

# 尝试解析代码
print("尝试解析文件的前 1510 行...")
try:
    ast.parse(content)
    print("前 1510 行语法正确！")
except SyntaxError as e:
    print(f"语法错误：{e}")
    print(f"错误位置：行 {e.lineno}, 列 {e.offset}")
    # 打印错误行附近的代码
    if e.lineno > 0 and e.lineno <= len(lines):
        print(f"错误行内容：{repr(lines[e.lineno-1])}")
        if e.lineno > 1:
            print(f"上一行内容：{repr(lines[e.lineno-2])}")
        if e.lineno < len(lines):
            print(f"下一行内容：{repr(lines[e.lineno])}")
