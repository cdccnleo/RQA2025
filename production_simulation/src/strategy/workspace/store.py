import logging
"""
策略存储组件

from src.engine.logging.unified_logger import get_unified_logger
提供策略版本管理和持久化功能，包括：
- 策略版本控制
- 策略元数据管理
- 策略配置存储
- 策略性能历史
"""

import json
import pickle
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from .visual_editor import VisualStrategyEditor
from .simulator import SimulationResult

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetadata:

    """策略元数据"""
    strategy_id: str
    name: str
    description: str
    author: str
    created_at: datetime
    updated_at: datetime
    version: str
    tags: List[str]
    market_type: str
    risk_level: str
    status: str  # draft, active, archived


@dataclass
class StrategyVersion:

    """策略版本"""
    version_id: str
    strategy_id: str
    created_at: datetime
    config: Dict
    parameters: Dict
    performance_metrics: Optional[Dict] = None
    notes: str = ""


class StrategyStore:

    """策略存储组件"""

    def __init__(self, storage_path: str = "data / strategies"):

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.storage_path / "metadata").mkdir(exist_ok=True)
        (self.storage_path / "configs").mkdir(exist_ok=True)
        (self.storage_path / "results").mkdir(exist_ok=True)
        (self.storage_path / "backups").mkdir(exist_ok=True)

        self.metadata_file = self.storage_path / "metadata" / "strategies.json"
        self._load_metadata()

    def _load_metadata(self):
        """加载策略元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf - 8') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"加载策略元数据失败: {e}")
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        """保存策略元数据"""
        try:
            with open(self.metadata_file, 'w', encoding='utf - 8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"保存策略元数据失败: {e}")

    def create_strategy(self, name: str, description: str, author: str,


                        market_type: str, risk_level: str, tags: List[str] = None) -> str:
        """创建新策略

        Args:
            name: 策略名称
            description: 策略描述
            author: 作者
            market_type: 市场类型
            risk_level: 风险等级
            tags: 标签列表

        Returns:
            str: 策略ID
        """
        try:
            strategy_id = f"strategy_{datetime.now().strftime('%Y % m % d_ % H % M % S')}"

            metadata = StrategyMetadata(
                strategy_id=strategy_id,
                name=name,
                description=description,
                author=author,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                version="1.0.0",
                tags=tags or [],
                market_type=market_type,
                risk_level=risk_level,
                status="draft"
            )

            # 保存元数据
            self.metadata[strategy_id] = asdict(metadata)
            self._save_metadata()

            logger.info(f"创建策略成功: {strategy_id}")
            return strategy_id

        except Exception as e:
            logger.error(f"创建策略失败: {e}")
            raise

    def save_strategy(self, strategy_id: str, strategy: VisualStrategyEditor,


                      parameters: Dict, version_notes: str = "") -> str:
        """保存策略

        Args:
            strategy_id: 策略ID
            parameters: 策略参数
            version_notes: 版本说明

        Returns:
            str: 版本ID
        """
        try:
            if strategy_id not in self.metadata:
                raise ValueError(f"策略不存在: {strategy_id}")

            # 生成版本ID
            version_id = f"v{datetime.now().strftime('%Y % m % d_ % H % M % S')}"

            # 创建版本记录
            version = StrategyVersion(
                version_id=version_id,
                strategy_id=strategy_id,
                created_at=datetime.now(),
                config=strategy.export_strategy(),
                parameters=parameters,
                notes=version_notes
            )

            # 保存版本配置
            config_file = self.storage_path / "configs" / f"{strategy_id}_{version_id}.json"
            with open(config_file, 'w', encoding='utf - 8') as f:
                json.dump(asdict(version), f, indent=2, ensure_ascii=False, default=str)

            # 更新元数据
            self.metadata[strategy_id]["updated_at"] = datetime.now().isoformat()
            self.metadata[strategy_id]["version"] = version_id
            self._save_metadata()

            logger.info(f"保存策略版本成功: {strategy_id} {version_id}")
            return version_id

        except Exception as e:
            logger.error(f"保存策略失败: {e}")
            raise

    def load_strategy(self, strategy_id: str, version_id: Optional[str] = None) -> Tuple[VisualStrategyEditor, Dict]:
        """加载策略

        Args:
            strategy_id: 策略ID
            version_id: 版本ID，如果为None则加载最新版本

        Returns:
            Tuple[VisualStrategyEditor, Dict]: 策略编辑器和参数
        """
        try:
            if strategy_id not in self.metadata:
                raise ValueError(f"策略不存在: {strategy_id}")

            # 确定版本ID
            if version_id is None:
                version_id = self.metadata[strategy_id]["version"]

            # 加载版本配置
            config_file = self.storage_path / "configs" / f"{strategy_id}_{version_id}.json"
            if not config_file.exists():
                raise ValueError(f"策略版本不存在: {strategy_id} {version_id}")

            with open(config_file, 'r', encoding='utf - 8') as f:
                version_data = json.load(f)

            # 创建策略编辑器
            strategy = VisualStrategyEditor()
            strategy.import_strategy(version_data["config"])

            logger.info(f"加载策略成功: {strategy_id} {version_id}")
            return strategy, version_data["parameters"]

        except Exception as e:
            logger.error(f"加载策略失败: {e}")
            raise

    def save_simulation_result(self, strategy_id: str, version_id: str,


                               result: SimulationResult) -> str:
        """保存模拟结果

        Args:
            strategy_id: 策略ID
            version_id: 版本ID
            result: 模拟结果

        Returns:
            str: 结果ID
        """
        try:
            result_id = f"result_{datetime.now().strftime('%Y % m % d_ % H % M % S')}"

            # 保存结果
            result_file = self.storage_path / "results" / \
                f"{strategy_id}_{version_id}_{result_id}.pkl"
            with open(result_file, 'wb') as f:
                pickle.dump(result, f)

            # 更新版本元数据
            config_file = self.storage_path / "configs" / f"{strategy_id}_{version_id}.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf - 8') as f:
                    version_data = json.load(f)

                version_data["performance_metrics"] = {
                    "total_return": result.total_return,
                    "annualized_return": result.annualized_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor
                }

                with open(config_file, 'w', encoding='utf - 8') as f:
                    json.dump(version_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"保存模拟结果成功: {result_id}")
            return result_id

        except Exception as e:
            logger.error(f"保存模拟结果失败: {e}")
            raise

    def load_simulation_result(self, strategy_id: str, version_id: str,


                               result_id: str) -> SimulationResult:
        """加载模拟结果

        Args:
            strategy_id: 策略ID
            version_id: 版本ID
            result_id: 结果ID

        Returns:
            SimulationResult: 模拟结果
        """
        try:
            result_file = self.storage_path / "results" / \
                f"{strategy_id}_{version_id}_{result_id}.pkl"
            if not result_file.exists():
                raise ValueError(f"模拟结果不存在: {result_id}")

            with open(result_file, 'rb') as f:
                result = pickle.load(f)

            logger.info(f"加载模拟结果成功: {result_id}")
            return result

        except Exception as e:
            logger.error(f"加载模拟结果失败: {e}")
            raise

    def list_strategies(self) -> List[Dict]:
        """列出所有策略

        Returns:
            List[Dict]: 策略列表
        """
        return list(self.metadata.values())

    def list_versions(self, strategy_id: str) -> List[Dict]:
        """列出策略的所有版本

        Args:
            strategy_id: 策略ID

        Returns:
            List[Dict]: 版本列表
        """
        try:
            versions = []
            config_dir = self.storage_path / "configs"

            for config_file in config_dir.glob(f"{strategy_id}_v*.json"):
                with open(config_file, 'r', encoding='utf - 8') as f:
                    version_data = json.load(f)
                versions.append(version_data)

            # 按创建时间排序
            versions.sort(key=lambda x: x["created_at"], reverse=True)
            return versions

        except Exception as e:
            logger.error(f"列出策略版本失败: {e}")
            return []

    def delete_strategy(self, strategy_id: str):
        """删除策略

        Args:
            strategy_id: 策略ID
        """
        try:
            if strategy_id not in self.metadata:
                raise ValueError(f"策略不存在: {strategy_id}")

            # 删除所有相关文件
            config_dir = self.storage_path / "configs"
            result_dir = self.storage_path / "results"

            # 删除配置文件
            for config_file in config_dir.glob(f"{strategy_id}_*.json"):
                config_file.unlink()

            # 删除结果文件
            for result_file in result_dir.glob(f"{strategy_id}_*.pkl"):
                result_file.unlink()

            # 删除元数据
            del self.metadata[strategy_id]
            self._save_metadata()

            logger.info(f"删除策略成功: {strategy_id}")

        except Exception as e:
            logger.error(f"删除策略失败: {e}")
            raise

    def backup_strategy(self, strategy_id: str) -> str:
        """备份策略

        Args:
            strategy_id: 策略ID

        Returns:
            str: 备份文件路径
        """
        try:
            if strategy_id not in self.metadata:
                raise ValueError(f"策略不存在: {strategy_id}")

            backup_file = self.storage_path / "backups" / \
                f"{strategy_id}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}.json"

            backup_data = {
                "metadata": self.metadata[strategy_id],
                "versions": self.list_versions(strategy_id)
            }

            with open(backup_file, 'w', encoding='utf - 8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"备份策略成功: {backup_file}")
            return str(backup_file)

        except Exception as e:
            logger.error(f"备份策略失败: {e}")
            raise

    def restore_strategy(self, backup_file: str) -> str:
        """恢复策略

        Args:
            backup_file: 备份文件路径

        Returns:
            str: 恢复的策略ID
        """
        try:
            with open(backup_file, 'r', encoding='utf - 8') as f:
                backup_data = json.load(f)

            metadata = backup_data["metadata"]
            versions = backup_data["versions"]

            strategy_id = metadata["strategy_id"]

            # 恢复元数据
            self.metadata[strategy_id] = metadata
            self._save_metadata()

            # 恢复版本配置
            for version in versions:
                config_file = self.storage_path / "configs" / \
                    f"{strategy_id}_{version['version_id']}.json"
                with open(config_file, 'w', encoding='utf - 8') as f:
                    json.dump(version, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"恢复策略成功: {strategy_id}")
            return strategy_id

        except Exception as e:
            logger.error(f"恢复策略失败: {e}")
            raise

    def search_strategies(self, query: str, tags: List[str] = None,


                          market_type: str = None, risk_level: str = None) -> List[Dict]:
        """搜索策略

        Args:
            query: 搜索查询
            tags: 标签过滤
            market_type: 市场类型过滤
            risk_level: 风险等级过滤

        Returns:
            List[Dict]: 搜索结果
        """
        try:
            results = []

            for strategy in self.metadata.values():
                # 文本搜索
                if query and query.lower() not in strategy["name"].lower() and query.lower() not in strategy["description"].lower():
                    continue

                # 标签过滤
                if tags and not any(tag in strategy["tags"] for tag in tags):
                    continue

                # 市场类型过滤
                if market_type and strategy["market_type"] != market_type:
                    continue

                # 风险等级过滤
                if risk_level and strategy["risk_level"] != risk_level:
                    continue

                results.append(strategy)

            return results

        except Exception as e:
            logger.error(f"搜索策略失败: {e}")
            return []

    def get_strategy_statistics(self) -> Dict:
        """获取策略统计信息

        Returns:
            Dict: 统计信息
        """
        try:
            stats = {
                "total_strategies": len(self.metadata),
                "by_status": {},
                "by_market_type": {},
                "by_risk_level": {},
                "by_author": {},
                "recent_activity": []
            }

            for strategy in self.metadata.values():
                # 按状态统计
                status = strategy["status"]
                stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

                # 按市场类型统计
                market_type = strategy["market_type"]
                stats["by_market_type"][market_type] = stats["by_market_type"].get(
                    market_type, 0) + 1

                # 按风险等级统计
                risk_level = strategy["risk_level"]
                stats["by_risk_level"][risk_level] = stats["by_risk_level"].get(risk_level, 0) + 1

                # 按作者统计
                author = strategy["author"]
                stats["by_author"][author] = stats["by_author"].get(author, 0) + 1

            # 最近活动
            sorted_strategies = sorted(self.metadata.values(),
                                       key=lambda x: x["updated_at"], reverse=True)
            stats["recent_activity"] = sorted_strategies[:10]

            return stats

        except Exception as e:
            logger.error(f"获取策略统计信息失败: {e}")
            return {}
