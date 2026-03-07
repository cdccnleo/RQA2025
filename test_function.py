# 测试 get_stocks_by_industry 函数的语法
import ast

# 读取文件内容
with open('src/gateway/web/postgresql_persistence.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 提取 get_stocks_by_industry 函数
start_idx = content.find('def get_stocks_by_industry')
if start_idx != -1:
    # 找到函数定义的开始
    # 简单提取函数内容（可能需要更精确的处理）
    # 这里我们使用 AST 来检查整个文件的语法
    try:
        ast.parse(content)
        print("文件语法正确，没有语法错误")
    except SyntaxError as e:
        print(f"语法错误：{e}")
        print(f"错误位置：行 {e.lineno}, 列 {e.offset}")
        # 打印错误行附近的代码
        lines = content.split('\n')
        if e.lineno > 0 and e.lineno <= len(lines):
            print(f"错误行内容：{lines[e.lineno-1]}")
            if e.lineno > 1:
                print(f"上一行内容：{lines[e.lineno-2]}")
            if e.lineno < len(lines):
                print(f"下一行内容：{lines[e.lineno]}")
else:
    print("未找到 get_stocks_by_industry 函数")
