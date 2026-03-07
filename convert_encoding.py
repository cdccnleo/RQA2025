# 转换文件编码为 UTF-8

# 读取文件内容
with open('src/gateway/web/postgresql_persistence.py', 'rb') as f:
    raw_content = f.read()

# 尝试解码为 UTF-8
content = raw_content.decode('utf-8', errors='ignore')

# 写入新文件
with open('src/gateway/web/postgresql_persistence.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("文件编码已转换为 UTF-8")

# 检查语法
import ast
try:
    ast.parse(content)
    print("文件语法正确！")
except SyntaxError as e:
    print(f"语法错误：{e}")
    print(f"错误位置：行 {e.lineno}, 列 {e.offset}")
    # 打印错误行附近的代码
    lines = content.split('\n')
    if e.lineno > 0 and e.lineno <= len(lines):
        print(f"错误行内容：{repr(lines[e.lineno-1])}")
        if e.lineno > 1:
            print(f"上一行内容：{repr(lines[e.lineno-2])}")
        if e.lineno < len(lines):
            print(f"下一行内容：{repr(lines[e.lineno])}")
