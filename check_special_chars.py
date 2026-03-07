import sys

print("开始检查文件中的特殊字符...")
try:
    with open('src/gateway/web/postgresql_persistence.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查文件的前几个字符，看看是否有 BOM 标记
    print("文件前 10 个字符的 Unicode 码点:")
    for i, char in enumerate(content[:10]):
        print(f"  位置 {i}: {repr(char)} (Unicode: U+{ord(char):04X})")
    
    # 检查第 1506 行附近的字符
    lines = content.split('\n')
    if len(lines) >= 1506:
        line_1505 = lines[1504]  # 第 1505 行
        line_1506 = lines[1505]  # 第 1506 行
        line_1507 = lines[1506]  # 第 1507 行
        
        print("\n第 1505 行的字符:")
        for i, char in enumerate(line_1505):
            print(f"  位置 {i}: {repr(char)} (Unicode: U+{ord(char):04X})")
        
        print("\n第 1506 行的字符:")
        for i, char in enumerate(line_1506):
            print(f"  位置 {i}: {repr(char)} (Unicode: U+{ord(char):04X})")
        
        print("\n第 1507 行的字符:")
        for i, char in enumerate(line_1507):
            print(f"  位置 {i}: {repr(char)} (Unicode: U+{ord(char):04X})")
    
    print("\n检查完成！")
except Exception as e:
    print(f"错误: {e}")
    sys.exit(1)
