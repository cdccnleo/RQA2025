import ast

# 读取原文件
with open('src/gateway/web/postgresql_persistence.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 提取 get_stocks_by_industry 函数
start_marker = 'def get_stocks_by_industry('

start_idx = content.find(start_marker)
if start_idx != -1:
    # 找到函数定义的开始
    # 计算函数的缩进级别
    line_start = content.rfind('\n', 0, start_idx) + 1
    indent = len(content[line_start:start_idx])
    
    # 从函数定义开始，逐行读取，直到找到与函数定义相同缩进级别的下一个 def 或文件末尾
    lines = content[start_idx:].split('\n')
    function_lines = []
    brace_count = 0
    in_function = True
    
    for line in lines:
        stripped_line = line.lstrip()
        
        # 检查是否是新的函数定义，且缩进级别与当前函数相同
        if stripped_line.startswith('def ') and len(line) - len(stripped_line) == indent:
            if function_lines:
                break
        
        function_lines.append(line)
    
    function_content = '\n'.join(function_lines)
    
    print(f"提取的函数长度: {len(function_content)} 字符")
    print(f"提取的函数行数: {len(function_lines)}")
    
    # 尝试解析函数内容
    try:
        ast.parse(function_content)
        print("\n函数内容语法正确！")
    except SyntaxError as e:
        print(f"\n函数内容语法错误：{e}")
        print(f"错误位置：行 {e.lineno}, 列 {e.offset}")
        # 打印错误行附近的代码
        error_line = e.lineno - 1
        if 0 <= error_line < len(function_lines):
            print(f"错误行内容：{repr(function_lines[error_line])}")
            if error_line > 0:
                print(f"上一行内容：{repr(function_lines[error_line-1])}")
            if error_line < len(function_lines) - 1:
                print(f"下一行内容：{repr(function_lines[error_line+1])}")
else:
    print("未找到 get_stocks_by_industry 函数")
