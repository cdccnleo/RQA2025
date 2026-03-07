# 检查文件的原始字节
with open('src/gateway/web/postgresql_persistence.py', 'rb') as f:
    content = f.read()

# 找到第 1500 行的开始位置
lines = content.split(b'\n')
start_line = 1499  # 索引从 0 开始
end_line = 1514    # 索引从 0 开始

print(f"检查第 {start_line+1}-{end_line+1} 行的原始字节")
print("-" * 80)

for i in range(start_line, min(end_line+1, len(lines))):
    line = lines[i]
    print(f"第 {i+1} 行: {line!r}")
    
    # 检查是否有非 ASCII 字符
    try:
        decoded_line = line.decode('ascii')
        print(f"  ASCII 解码: {decoded_line}")
    except UnicodeDecodeError:
        try:
            decoded_line = line.decode('utf-8')
            print(f"  UTF-8 解码: {decoded_line}")
        except UnicodeDecodeError:
            print(f"  无法解码")
    
    print()
