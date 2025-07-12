"""龙虎榜增量更新系统"""
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pymongo import UpdateOne
from src.data.db_client import get_db_client

class DragonBoardUpdater:
    """龙虎榜数据增量更新处理器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db = get_db_client().get_database("quant_data")
        self.collection = self.db.dragon_board
        self.cache = {}  # 内存缓存最新数据

        # 初始化缓存
        self._init_cache()

    def _init_cache(self):
        """初始化内存缓存"""
        latest_records = self.collection.find().sort("timestamp", -1).limit(500)
        for record in latest_records:
            symbol = record["symbol"]
            self.cache[symbol] = record["timestamp"]
        self.logger.info(f"已加载{len(self.cache)}条龙虎榜缓存记录")

    def _get_data_hash(self, data: Dict) -> str:
        """计算数据哈希值用于去重"""
        hash_str = f"{data['symbol']}_{data['buyer']}_{data['seller']}_{data['amount']}"
        return hashlib.md5(hash_str.encode()).hexdigest()

    def fetch_new_data(self, broker_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        获取增量数据
        返回: (新增数据列表, 更新数据列表)
        """
        new_data = []
        updated_data = []

        for item in broker_data:
            symbol = item["symbol"]
            item_hash = self._get_data_hash(item)
            item["data_hash"] = item_hash

            # 检查是否为全新数据
            if symbol not in self.cache:
                new_data.append(item)
                continue

            # 检查是否为更新数据
            if item["timestamp"] > self.cache[symbol]:
                if self._is_data_updated(item):
                    updated_data.append(item)

        self.logger.info(f"发现{len(new_data)}条新增数据, {len(updated_data)}条更新数据")
        return new_data, updated_data

    def _is_data_updated(self, new_item: Dict) -> bool:
        """检查数据是否真正更新"""
        old_record = self.collection.find_one(
            {"symbol": new_item["symbol"]},
            sort=[("timestamp", -1)]
        )

        if not old_record:
            return True

        # 对比关键字段变化
        key_fields = ["buyer", "seller", "amount", "direction"]
        for field in key_fields:
            if new_item.get(field) != old_record.get(field):
                return True

        return False

    def batch_update(self, new_data: List[Dict], updated_data: List[Dict]) -> Dict:
        """批量更新数据库"""
        # 准备批量操作
        operations = []

        # 新增数据操作
        for item in new_data:
            operations.append(
                UpdateOne(
                    {"data_hash": item["data_hash"]},
                    {"$setOnInsert": item},
                    upsert=True
                )
            )

        # 更新数据操作
        for item in updated_data:
            operations.append(
                UpdateOne(
                    {"symbol": item["symbol"], "timestamp": {"$lt": item["timestamp"]}},
                    {"$set": item}
                )
            )

        # 执行批量操作
        if operations:
            result = self.collection.bulk_write(operations)
            self._update_cache(new_data + updated_data)
            return {
                "inserted": result.upserted_count,
                "modified": result.modified_count
            }
        return {"inserted": 0, "modified": 0}

    def _update_cache(self, items: List[Dict]):
        """更新内存缓存"""
        for item in items:
            symbol = item["symbol"]
            if symbol in self.cache:
                if item["timestamp"] > self.cache[symbol]:
                    self.cache[symbol] = item["timestamp"]
            else:
                self.cache[symbol] = item["timestamp"]

    def get_institutional_patterns(self, days: int = 3) -> Dict[str, List]:
        """
        识别机构交易模式
        返回: {
            "buy_patterns": [机构买入模式列表],
            "sell_patterns": [机构卖出模式列表]
        }
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        query = {
            "timestamp": {"$gte": start_date, "$lte": end_date},
            "institutional": True
        }

        records = list(self.collection.find(query))

        buy_patterns = self._analyze_buy_patterns(records)
        sell_patterns = self._analyze_sell_patterns(records)

        return {
            "buy_patterns": buy_patterns,
            "sell_patterns": sell_patterns
        }

    def _analyze_buy_patterns(self, records: List[Dict]) -> List[Dict]:
        """分析机构买入模式"""
        # 实现实际的模式识别逻辑
        buy_records = [r for r in records if r["direction"] == "buy"]

        patterns = []
        if len(buy_records) > 10:
            patterns.append({
                "name": "机构集中买入",
                "stocks": list(set([r["symbol"] for r in buy_records])),
                "strength": len(buy_records) / 10  # 简单强度指标
            })

        return patterns

    def _analyze_sell_patterns(self, records: List[Dict]) -> List[Dict]:
        """分析机构卖出模式"""
        # 实现实际的模式识别逻辑
        sell_records = [r for r in records if r["direction"] == "sell"]

        patterns = []
        if len(sell_records) > 10:
            patterns.append({
                "name": "机构集中卖出",
                "stocks": list(set([r["symbol"] for r in sell_records])),
                "strength": len(sell_records) / 10  # 简单强度指标
            })

        return patterns

    def trigger_feature_update(self):
        """触发特征层更新"""
        from src.features import FeatureEngineer
        FeatureEngineer().update_dragon_board_features()
        self.logger.info("已触发龙虎榜特征更新")

class DragonBoardAPI:
    """龙虎榜数据API接口"""

    @staticmethod
    def get_latest_updates(broker: str, last_update: datetime) -> List[Dict]:
        """从券商API获取最新龙虎榜数据"""
        # 实现实际的API调用逻辑
        return []
