# 检查文件编码
with open('src/gateway/web/postgresql_persistence.py', 'rb') as f:
    # 读取文件的前 1000 个字节
    raw_data = f.read(1000)
    
    # 检查 BOM
    if raw_data.startswith(b'\xef\xbb\xbf'):
        print('UTF-8 with BOM')
    elif raw_data.startswith(b'\xff\xfe'):
        print('UTF-16 LE')
    elif raw_data.startswith(b'\xfe\xff'):
        print('UTF-16 BE')
    else:
        # 尝试使用 UTF-8 解码
        try:
            raw_data.decode('utf-8')
            print('UTF-8')
        except UnicodeDecodeError:
            # 尝试使用 GBK 解码
            try:
                raw_data.decode('gbk')
                print('GBK')
            except UnicodeDecodeError:
                print('Unknown encoding')
