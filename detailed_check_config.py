import json
import os

print("Current working directory:", os.getcwd())
print("Checking config file...")

# 检查配置文件是否存在
config_path = "/app/data/data_sources_config.json"
if os.path.exists(config_path):
    print(f"Config file found at: {config_path}")
    print(f"File size: {os.path.getsize(config_path)} bytes")
    
    # 读取配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Total data sources: {len(data)}")
    
    # 查找 akshare_stock_basic 数据源
    for i, source in enumerate(data):
        print(f"Source {i+1}: {source['id']}")
        if source["id"] == "akshare_stock_basic":
            print("\\nFound akshare_stock_basic:")
            print(f"ID: {source['id']}")
            print(f"Name: {source['name']}")
            print(f"URL: {source['url']}")
            print(f"Type: {source['type']}")
            print(f"Status: {source['status']}")
            print(f"Last test: {source['last_test']}")
            print(f"Enabled: {source['enabled']}")
            print(f"Rate limit: {source['rate_limit']}")
            print("\\nConfig:")
            for key, value in source['config'].items():
                if key == 'akshare_function':
                    print(f"  {key}: {value} <-- THIS IS THE FUNCTION WE CARE ABOUT")
                else:
                    print(f"  {key}: {value}")
            break
else:
    print(f"Config file not found at: {config_path}")
    print("Listing directory contents:")
    if os.path.exists("/app/data"):
        print(os.listdir("/app/data"))
    else:
        print("/app/data directory does not exist")
