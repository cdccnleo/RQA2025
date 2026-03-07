import chardet

# 检查文件编码
with open('src/gateway/web/postgresql_persistence.py', 'rb') as f:
    result = chardet.detect(f.read())
    print(f"文件编码: {result['encoding']}")
    print(f"置信度: {result['confidence']}")
