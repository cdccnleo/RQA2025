"""
策略持久化实现
Strategy Persistence Implementation

负责策略数据的存储、加载和管理
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from ..interfaces.strategy_interfaces import IStrategyPersistence


class StrategyPersistence(IStrategyPersistence):
    """策略持久化实现"""

    def __init__(self, storage_path: str = None):
        """初始化持久化管理器"""
        if storage_path is None:
            # 默认存储路径
            self.storage_path = Path.home() / ".rqa2025" / "strategies"
        else:
            self.storage_path = Path(storage_path)

        # 创建存储目录
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 策略数据存储
        self._strategies_file = self.storage_path / "strategies.json"
        self._configs_file = self.storage_path / "configs.json"

        # 内存缓存
        self._strategy_cache: Dict[str, Dict[str, Any]] = {}
        self._config_cache: Dict[str, Dict[str, Any]] = {}

        # 加载现有数据
        self._load_data()

    def _load_data(self):
        """加载现有数据"""
        try:
            if self._strategies_file.exists():
                with open(self._strategies_file, 'r', encoding='utf-8') as f:
                    self._strategy_cache = json.load(f)
        except Exception:
            self._strategy_cache = {}

        try:
            if self._configs_file.exists():
                with open(self._configs_file, 'r', encoding='utf-8') as f:
                    self._config_cache = json.load(f)
        except Exception:
            self._config_cache = {}

    def _save_data(self):
        """保存数据到文件"""
        try:
            with open(self._strategies_file, 'w', encoding='utf-8') as f:
                json.dump(self._strategy_cache, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        try:
            with open(self._configs_file, 'w', encoding='utf-8') as f:
                json.dump(self._config_cache, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def save_strategy(self, strategy_id: str, strategy_data: Dict[str, Any]) -> bool:
        """保存策略"""
        try:
            # 添加时间戳
            strategy_data['_saved_at'] = datetime.now().isoformat()

            self._strategy_cache[strategy_id] = strategy_data
            self._save_data()
            return True
        except Exception:
            return False

    def load_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """加载策略"""
        return self._strategy_cache.get(strategy_id)

    def delete_strategy(self, strategy_id: str) -> bool:
        """删除策略"""
        try:
            if strategy_id in self._strategy_cache:
                del self._strategy_cache[strategy_id]
                # 同时删除配置
                if strategy_id in self._config_cache:
                    del self._config_cache[strategy_id]
                self._save_data()
                return True
            return False
        except Exception:
            return False

    def list_strategies(self) -> List[str]:
        """列出所有策略"""
        return list(self._strategy_cache.keys())

    def save_strategy_config(self, strategy_id: str, config: Dict[str, Any]) -> bool:
        """保存策略配置"""
        try:
            # 添加时间戳
            config['_saved_at'] = datetime.now().isoformat()

            self._config_cache[strategy_id] = config
            self._save_data()
            return True
        except Exception:
            return False

    def load_strategy_config(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """加载策略配置"""
        return self._config_cache.get(strategy_id)


# 全局实例
_strategy_persistence = None

def get_strategy_persistence() -> StrategyPersistence:
    """获取策略持久化实例"""
    global _strategy_persistence
    if _strategy_persistence is None:
        _strategy_persistence = StrategyPersistence()
    return _strategy_persistence
