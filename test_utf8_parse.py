import ast

print("开始使用 UTF-8 编码解析文件...")
try:
    with open('src/gateway/web/postgresql_persistence.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用 ast 模块解析文件
    tree = ast.parse(content)
    print("语法检查通过！")
except SyntaxError as e:
    print(f"语法错误: {e}")
    print(f"错误位置: 第 {e.lineno} 行, 第 {e.offset} 列")
    # 打印错误行及其前后几行
    lines = content.split('\n')
    start_line = max(0, e.lineno - 3)
    end_line = min(len(lines), e.lineno + 2)
    for i in range(start_line, end_line):
        line_num = i + 1
        marker = '->' if line_num == e.lineno else '  '
        print(f"{marker} {line_num}: {lines[i]}")
except Exception as e:
    print(f"其他错误: {e}")
