import json
with open("/app/data/data_sources_config.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for source in data:
    if source["id"] == "akshare_stock_basic":
        print("Current function:", source["config"]["akshare_function"])
        source["config"]["akshare_function"] = "stock_zh_a_spot"
        print("Updated function:", source["config"]["akshare_function"])
        break

with open("/app/data/data_sources_config.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Configuration updated successfully")
