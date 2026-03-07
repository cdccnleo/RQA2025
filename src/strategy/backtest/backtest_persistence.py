#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
回测持久化实现
Backtest Persistence Implementation

提供回测结果和配置的持久化存储功能。
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import logging
from strategy.interfaces.backtest_interfaces import (
    IBacktestPersistence, BacktestConfig, BacktestResult
)
from core.integration.business_adapters import get_unified_adapter_factory

logger = logging.getLogger(__name__)


class BacktestPersistence(IBacktestPersistence):

    """
    回测持久化
    Backtest Persistence

    负责回测配置和结果的持久化存储。
    """

    def __init__(self, storage_path: str = "./data / backtest"):
        """
        初始化持久化服务

        Args:
            storage_path: 存储路径
        """
        self.storage_path = Path(storage_path)
        self.adapter_factory = get_unified_adapter_factory()

        # 确保存储目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        self.configs_path = self.storage_path / "configs"
        self.results_path = self.storage_path / "results"
        self.reports_path = self.storage_path / "reports"

        self.configs_path.mkdir(exist_ok=True)
        self.results_path.mkdir(exist_ok=True)
        self.reports_path.mkdir(exist_ok=True)

        logger.info(f"回测持久化服务初始化完成，存储路径: {self.storage_path}")

    def save_backtest_result(self, result: BacktestResult) -> bool:
        """
        保存回测结果

        Args:
            result: 回测结果

        Returns:
            bool: 保存是否成功
        """
        try:
            # 转换为字典格式
            result_dict = self._backtest_result_to_dict(result)

            # 保存到文件
            file_path = self.results_path / f"{result.backtest_id}.json"
            with open(file_path, 'w', encoding='utf - 8') as f:
                json.dump(result_dict, f, indent=2, default=str)

            logger.info(f"回测结果已保存: {result.backtest_id}")
            return True

        except Exception as e:
            logger.error(f"回测结果保存失败: {e}")
            return False

    def load_backtest_result(self, backtest_id: str) -> Optional[BacktestResult]:
        """
        加载回测结果

        Args:
            backtest_id: 回测ID

        Returns:
            Optional[BacktestResult]: 回测结果
        """
        try:
            file_path = self.results_path / f"{backtest_id}.json"

            if not file_path.exists():
                return None

            with open(file_path, 'r', encoding='utf - 8') as f:
                result_dict = json.load(f)

            # 转换为BacktestResult对象
            result = self._dict_to_backtest_result(result_dict)

            logger.info(f"回测结果已加载: {backtest_id}")
            return result

        except Exception as e:
            logger.error(f"回测结果加载失败: {e}")
            return None

    def save_backtest_config(self, config: BacktestConfig) -> bool:
        """
        保存回测配置

        Args:
            config: 回测配置

        Returns:
            bool: 保存是否成功
        """
        try:
            # 转换为字典格式
            config_dict = self._backtest_config_to_dict(config)

            # 保存到文件
            file_path = self.configs_path / f"{config.backtest_id}.json"
            with open(file_path, 'w', encoding='utf - 8') as f:
                json.dump(config_dict, f, indent=2, default=str)

            logger.info(f"回测配置已保存: {config.backtest_id}")
            return True

        except Exception as e:
            logger.error(f"回测配置保存失败: {e}")
            return False

    def load_backtest_config(self, backtest_id: str) -> Optional[BacktestConfig]:
        """
        加载回测配置

        Args:
            backtest_id: 回测ID

        Returns:
            Optional[BacktestConfig]: 回测配置
        """
        try:
            file_path = self.configs_path / f"{backtest_id}.json"

            if not file_path.exists():
                return None

            with open(file_path, 'r', encoding='utf - 8') as f:
                config_dict = json.load(f)

            # 转换为BacktestConfig对象
            config = self._dict_to_backtest_config(config_dict)

            logger.info(f"回测配置已加载: {backtest_id}")
            return config

        except Exception as e:
            logger.error(f"回测配置加载失败: {e}")
            return None

    def delete_backtest_data(self, backtest_id: str) -> bool:
        """
        删除回测数据

        Args:
            backtest_id: 回测ID

        Returns:
            bool: 删除是否成功
        """
        try:
            # 删除结果文件
            result_file = self.results_path / f"{backtest_id}.json"
            if result_file.exists():
                result_file.unlink()

            # 删除配置文件
            config_file = self.configs_path / f"{backtest_id}.json"
            if config_file.exists():
                config_file.unlink()

            # 删除报告文件
            report_file = self.reports_path / f"{backtest_id}.json"
            if report_file.exists():
                report_file.unlink()

            logger.info(f"回测数据已删除: {backtest_id}")
            return True

        except Exception as e:
            logger.error(f"回测数据删除失败: {e}")
            return False

    def list_backtest_ids(self) -> List[str]:
        """
        列出所有回测ID

        Returns:
            List[str]: 回测ID列表
        """
        try:
            result_files = self.results_path.glob("*.json")
            backtest_ids = [f.stem for f in result_files]
            return backtest_ids

        except Exception as e:
            logger.error(f"获取回测ID列表失败: {e}")
            return []

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息

        Returns:
            Dict[str, Any]: 存储统计
        """
        try:
            stats = {
                'total_configs': len(list(self.configs_path.glob("*.json"))),
                'total_results': len(list(self.results_path.glob("*.json"))),
                'total_reports': len(list(self.reports_path.glob("*.json"))),
                'storage_path': str(self.storage_path),
                'total_size_mb': self._calculate_storage_size() / (1024 * 1024)
            }

            return stats

        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
            return {}

    def _calculate_storage_size(self) -> int:
        """
        计算存储大小

        Returns:
            int: 存储大小（字节）
        """
        total_size = 0

        for path in [self.configs_path, self.results_path, self.reports_path]:
            for file_path in path.glob("**/*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size

        return total_size

    def _backtest_result_to_dict(self, result: BacktestResult) -> Dict[str, Any]:
        """
        将BacktestResult转换为字典

        Args:
            result: 回测结果

        Returns:
            Dict[str, Any]: 字典格式
        """
        return {
            'backtest_id': result.backtest_id,
            'strategy_id': result.strategy_id,
            'returns': result.returns.to_dict() if not result.returns.empty else {},
            'positions': result.positions.to_dict('records') if not result.positions.empty else [],
            'trades': result.trades.to_dict('records') if not result.trades.empty else [],
            'metrics': result.metrics,
            'risk_metrics': result.risk_metrics,
            'status': result.status.value,
            'execution_time': result.execution_time,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat(),
            'error_message': result.error_message,
            'metadata': result.metadata
        }

    def _dict_to_backtest_result(self, data: Dict[str, Any]) -> BacktestResult:
        """
        将字典转换为BacktestResult

        Args:
            data: 字典数据

        Returns:
            BacktestResult: 回测结果对象
        """
        from strategy.interfaces.backtest_interfaces import BacktestStatus

        # 转换Series和DataFrame
        returns = pd.Series(data.get('returns', {}))
        positions = pd.DataFrame(data.get('positions', []))
        trades = pd.DataFrame(data.get('trades', []))

        return BacktestResult(
            backtest_id=data['backtest_id'],
            strategy_id=data['strategy_id'],
            returns=returns,
            positions=positions,
            trades=trades,
            metrics=data.get('metrics', {}),
            risk_metrics=data.get('risk_metrics', {}),
            status=BacktestStatus(data['status']),
            execution_time=data['execution_time'],
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']),
            error_message=data.get('error_message'),
            metadata=data.get('metadata', {})
        )

    def _backtest_config_to_dict(self, config: BacktestConfig) -> Dict[str, Any]:
        """
        将BacktestConfig转换为字典

        Args:
            config: 回测配置

        Returns:
            Dict[str, Any]: 字典格式
        """

        return {
            'backtest_id': config.backtest_id,
            'strategy_id': config.strategy_id,
            'start_date': config.start_date.isoformat(),
            'end_date': config.end_date.isoformat(),
            'initial_capital': config.initial_capital,
            'commission': config.commission,
            'slippage': config.slippage,
            'benchmark_symbol': config.benchmark_symbol,
            'data_frequency': config.data_frequency,
            'mode': config.mode.value,
            'parameters': config.parameters,
            'risk_limits': config.risk_limits,
            'created_at': config.created_at.isoformat()
        }

    def _dict_to_backtest_config(self, data: Dict[str, Any]) -> BacktestConfig:
        """
        将字典转换为BacktestConfig

        Args:
            data: 字典数据

        Returns:
            BacktestConfig: 回测配置对象
        """
        from strategy.interfaces.backtest_interfaces import BacktestMode

        return BacktestConfig(
            backtest_id=data['backtest_id'],
            strategy_id=data['strategy_id'],
            start_date=datetime.fromisoformat(data['start_date']),
            end_date=datetime.fromisoformat(data['end_date']),
            initial_capital=data['initial_capital'],
            commission=data.get('commission', 0.0003),
            slippage=data.get('slippage', 0.0001),
            benchmark_symbol=data.get('benchmark_symbol'),
            data_frequency=data.get('data_frequency', '1d'),
            mode=BacktestMode(data['mode']),
            parameters=data.get('parameters', {}),
            risk_limits=data.get('risk_limits', {}),
            created_at=datetime.fromisoformat(data['created_at'])
        )

    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """
        清理旧数据

        Args:
            days_to_keep: 保留天数

        Returns:
            int: 删除的文件数量
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0

            # 清理所有目录中的旧文件
            for path in [self.configs_path, self.results_path, self.reports_path]:
                for file_path in path.glob("*.json"):
                    if file_path.stat().st_mtime < cutoff_date.timestamp():
                        file_path.unlink()
                        deleted_count += 1

            logger.info(f"已清理 {deleted_count} 个旧文件")
            return deleted_count

        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")
            return 0


# 导出类
__all__ = [
    'BacktestPersistence'
]

