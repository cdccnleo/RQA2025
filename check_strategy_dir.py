#!/usr/bin/env python3
"""
检查策略构思目录
"""
import os
import json

STRATEGY_CONCEPTION_DIR = "data/strategy_conceptions"

# 检查目录是否存在
if not os.path.exists(STRATEGY_CONCEPTION_DIR):
    print(f"目录不存在: {STRATEGY_CONCEPTION_DIR}")
    os.makedirs(STRATEGY_CONCEPTION_DIR, exist_ok=True)
    print(f"已创建目录: {STRATEGY_CONCEPTION_DIR}")
else:
    print(f"目录已存在: {STRATEGY_CONCEPTION_DIR}")

# 列出目录中的文件
files = [f for f in os.listdir(STRATEGY_CONCEPTION_DIR) if f.endswith('.json')]
print(f"JSON文件数量: {len(files)}")

if not files:
    print("目录为空，创建示例策略...")
    
    # 创建示例策略1：动量策略
    momentum_strategy = {
        "id": "momentum_strategy_001",
        "name": "动量策略",
        "description": "基于价格动量的交易策略",
        "type": "momentum",
        "parameters": {
            "lookback_period": {
                "name": "回看周期",
                "type": "integer",
                "default": 20,
                "min": 5,
                "max": 60,
                "description": "计算动量的回看周期"
            },
            "threshold": {
                "name": "动量阈值",
                "type": "float",
                "default": 0.05,
                "min": 0.01,
                "max": 0.2,
                "description": "触发交易的动量阈值"
            }
        },
        "created_at": "2026-02-17T00:00:00",
        "updated_at": "2026-02-17T00:00:00"
    }
    
    # 创建示例策略2：均值回归策略
    mean_reversion_strategy = {
        "id": "mean_reversion_strategy_001",
        "name": "均值回归策略",
        "description": "基于价格均值回归的交易策略",
        "type": "mean_reversion",
        "parameters": {
            "ma_period": {
                "name": "均线周期",
                "type": "integer",
                "default": 20,
                "min": 5,
                "max": 60,
                "description": "移动平均线的周期"
            },
            "std_multiplier": {
                "name": "标准差倍数",
                "type": "float",
                "default": 2.0,
                "min": 1.0,
                "max": 3.0,
                "description": "布林带标准差倍数"
            }
        },
        "created_at": "2026-02-17T00:00:00",
        "updated_at": "2026-02-17T00:00:00"
    }
    
    # 创建示例策略3：机器学习策略
    ml_strategy = {
        "id": "ml_strategy_001",
        "name": "机器学习策略",
        "description": "基于机器学习的智能交易策略",
        "type": "ml",
        "parameters": {
            "model_type": {
                "name": "模型类型",
                "type": "string",
                "default": "random_forest",
                "options": ["random_forest", "xgboost", "lightgbm"],
                "description": "机器学习模型类型"
            },
            "feature_window": {
                "name": "特征窗口",
                "type": "integer",
                "default": 10,
                "min": 5,
                "max": 30,
                "description": "特征提取的时间窗口"
            }
        },
        "created_at": "2026-02-17T00:00:00",
        "updated_at": "2026-02-17T00:00:00"
    }
    
    # 保存示例策略
    strategies = [momentum_strategy, mean_reversion_strategy, ml_strategy]
    for strategy in strategies:
        filepath = os.path.join(STRATEGY_CONCEPTION_DIR, f"{strategy['id']}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(strategy, f, ensure_ascii=False, indent=2)
        print(f"已创建策略: {strategy['name']} ({strategy['id']})")

print("\n目录中的策略文件:")
for filename in os.listdir(STRATEGY_CONCEPTION_DIR):
    if filename.endswith('.json'):
        filepath = os.path.join(STRATEGY_CONCEPTION_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"  - {data.get('name', filename)} ({data.get('id', 'unknown')})")
