# 检查文件的缩进问题
with open('src/gateway/web/postgresql_persistence.py.container', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"文件共有 {len(lines)} 行")
print("开始检查缩进...")

# 检查第 1220-1240 行的缩进
start_line = 1219  # 索引从 0 开始
end_line = 1239    # 索引从 0 开始

for i in range(start_line, min(end_line + 1, len(lines))):
    line = lines[i]
    indent = len(line) - len(line.lstrip())
    stripped_line = line.lstrip()
    start_char = stripped_line[0] if stripped_line else 'empty'
    print(f"Line {i+1}: {indent} spaces/tabs, starts with '{start_char}'")
    print(f"  内容: {repr(line)}")

print("检查完成")
