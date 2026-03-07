import json
with open("/app/data/data_sources_config.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for source in data:
    if source["id"] == "akshare_stock_basic":
        print("ID:", source["id"])
        print("Function:", source["config"]["akshare_function"])
        break
