# 替换 get_stocks_by_industry 函数

# 读取正确的函数定义
with open('correct_function.py', 'r', encoding='utf-8') as f:
    correct_function = f.read()

# 读取原文件内容
with open('src/gateway/web/postgresql_persistence.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到 get_stocks_by_industry 函数的开始和结束位置
start_marker = 'def get_stocks_by_industry('
end_marker = 'def '

start_idx = content.find(start_marker)
if start_idx != -1:
    # 找到函数定义的开始
    # 查找下一个函数定义的开始，作为当前函数的结束
    next_func_idx = content.find(end_marker, start_idx + len(start_marker))
    if next_func_idx != -1:
        # 替换函数定义
        new_content = content[:start_idx] + correct_function + '\n\n' + content[next_func_idx:]
    else:
        # 如果没有下一个函数定义，就替换到文件末尾
        new_content = content[:start_idx] + correct_function
    
    # 写入新内容
    with open('src/gateway/web/postgresql_persistence.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("成功替换 get_stocks_by_industry 函数")
else:
    print("未找到 get_stocks_by_industry 函数")
