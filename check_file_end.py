with open('src/gateway/web/postgresql_persistence.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
print('文件总行数:', len(lines))
print('文件最后 20 行:')
for i, line in enumerate(lines[-20:], len(lines)-19):
    print(f'{i}: {line.rstrip()}')
