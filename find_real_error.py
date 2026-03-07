import ast

# 读取文件内容
with open('src/gateway/web/postgresql_persistence.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"文件共有 {len(lines)} 行")
print("开始逐段检查语法...")

# 从文件开头开始，每次增加 10 行，直到找到错误
chunk_size = 10
for i in range(0, len(lines), chunk_size):
    end_idx = min(i + chunk_size, len(lines))
    content = ''.join(lines[:end_idx])
    
    try:
        ast.parse(content)
        if i % 100 == 0:
            print(f"已检查 {end_idx} 行，无语法错误")
    except SyntaxError as e:
        print(f"\n语法错误出现在第 {end_idx} 行附近")
        print(f"错误信息：{e}")
        print(f"错误位置：行 {e.lineno}, 列 {e.offset}")
        
        # 从错误行的前 20 行开始，逐行检查
        start_check = max(0, e.lineno - 20)
        for j in range(start_check, e.lineno + 10):
            if j >= len(lines):
                break
            
            try:
                partial_content = ''.join(lines[:j])
                ast.parse(partial_content)
            except SyntaxError as e2:
                print(f"\n精确错误出现在第 {j} 行")
                print(f"错误信息：{e2}")
                print(f"错误位置：行 {e2.lineno}, 列 {e2.offset}")
                if e2.lineno > 0 and e2.lineno <= len(lines):
                    print(f"错误行内容：{repr(lines[e2.lineno-1])}")
                    if e2.lineno > 1:
                        print(f"上一行内容：{repr(lines[e2.lineno-2])}")
                    if e2.lineno < len(lines):
                        print(f"下一行内容：{repr(lines[e2.lineno])}")
                break
        break

print("检查完成")
