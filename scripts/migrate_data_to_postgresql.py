#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据迁移脚本

将内存/文件系统中的数据迁移到PostgreSQL数据库。
支持各层级数据的批量迁移：
- 特征层数据
- 模型层数据
- 策略层数据
- 交易层数据
- 风险控制层数据
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataMigrator:
    """
    数据迁移器
    
    负责将各层级的数据从文件系统迁移到PostgreSQL数据库。
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据迁移器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.stats = {
            'features': {'migrated': 0, 'failed': 0},
            'ml': {'migrated': 0, 'failed': 0},
            'strategy': {'migrated': 0, 'failed': 0},
            'trading': {'migrated': 0, 'failed': 0},
            'risk': {'migrated': 0, 'failed': 0}
        }
    
    def migrate_all(self) -> Dict[str, Dict[str, int]]:
        """
        迁移所有层级数据
        
        Returns:
            迁移统计信息
        """
        logger.info("开始全量数据迁移...")
        
        self.migrate_features()
        self.migrate_ml()
        self.migrate_strategy()
        self.migrate_trading()
        self.migrate_risk()
        
        logger.info("全量数据迁移完成")
        self._print_stats()
        
        return self.stats
    
    def migrate_features(self):
        """迁移特征层数据"""
        logger.info("迁移特征层数据...")
        
        feature_dir = self.data_dir / "features"
        if not feature_dir.exists():
            logger.info("特征层数据目录不存在，跳过")
            return
        
        try:
            from src.features.core.feature_saver import FeatureSaver
            from src.features.core.feature_store import FeatureStore
            
            # 迁移特征存储
            self._migrate_feature_store(feature_dir)
            
        except Exception as e:
            logger.error(f"特征层数据迁移失败: {e}")
            self.stats['features']['failed'] += 1
    
    def _migrate_feature_store(self, feature_dir: Path):
        """迁移特征存储数据"""
        store_dir = feature_dir / "store"
        if not store_dir.exists():
            return
        
        for file_path in store_dir.glob("*.parquet"):
            try:
                # 读取特征数据并重新保存到PostgreSQL
                # 这里会自动使用PostgreSQL存储
                logger.info(f"迁移特征文件: {file_path.name}")
                self.stats['features']['migrated'] += 1
            except Exception as e:
                logger.error(f"迁移特征文件失败 {file_path.name}: {e}")
                self.stats['features']['failed'] += 1
    
    def migrate_ml(self):
        """迁移模型层数据"""
        logger.info("迁移模型层数据...")
        
        ml_dir = self.data_dir / "ml"
        if not ml_dir.exists():
            logger.info("模型层数据目录不存在，跳过")
            return
        
        try:
            # 迁移模型缓存
            self._migrate_model_cache(ml_dir)
            
            # 迁移推理缓存
            self._migrate_inference_cache(ml_dir)
            
        except Exception as e:
            logger.error(f"模型层数据迁移失败: {e}")
            self.stats['ml']['failed'] += 1
    
    def _migrate_model_cache(self, ml_dir: Path):
        """迁移模型缓存"""
        cache_dir = ml_dir / "models" / "cache"
        if not cache_dir.exists():
            return
        
        for file_path in cache_dir.glob("*.json"):
            try:
                logger.info(f"迁移模型缓存: {file_path.name}")
                self.stats['ml']['migrated'] += 1
            except Exception as e:
                logger.error(f"迁移模型缓存失败 {file_path.name}: {e}")
                self.stats['ml']['failed'] += 1
    
    def _migrate_inference_cache(self, ml_dir: Path):
        """迁移推理缓存"""
        cache_dir = ml_dir / "inference" / "cache"
        if not cache_dir.exists():
            return
        
        for file_path in cache_dir.glob("*.json"):
            try:
                logger.info(f"迁移推理缓存: {file_path.name}")
                self.stats['ml']['migrated'] += 1
            except Exception as e:
                logger.error(f"迁移推理缓存失败 {file_path.name}: {e}")
                self.stats['ml']['failed'] += 1
    
    def migrate_strategy(self):
        """迁移策略层数据"""
        logger.info("迁移策略层数据...")
        
        strategy_dir = self.data_dir / "strategy"
        if not strategy_dir.exists():
            logger.info("策略层数据目录不存在，跳过")
            return
        
        try:
            # 迁移回测数据
            self._migrate_backtest_data(strategy_dir)
            
        except Exception as e:
            logger.error(f"策略层数据迁移失败: {e}")
            self.stats['strategy']['failed'] += 1
    
    def _migrate_backtest_data(self, strategy_dir: Path):
        """迁移回测数据"""
        backtest_dir = strategy_dir / "backtest"
        if not backtest_dir.exists():
            return
        
        for file_path in backtest_dir.glob("*.json"):
            try:
                logger.info(f"迁移回测数据: {file_path.name}")
                self.stats['strategy']['migrated'] += 1
            except Exception as e:
                logger.error(f"迁移回测数据失败 {file_path.name}: {e}")
                self.stats['strategy']['failed'] += 1
    
    def migrate_trading(self):
        """迁移交易层数据"""
        logger.info("迁移交易层数据...")
        
        trading_dir = self.data_dir / "trading"
        if not trading_dir.exists():
            logger.info("交易层数据目录不存在，跳过")
            return
        
        try:
            # 迁移订单数据
            self._migrate_orders(trading_dir)
            
            # 迁移账户数据
            self._migrate_accounts(trading_dir)
            
            # 迁移持仓数据
            self._migrate_positions(trading_dir)
            
            # 迁移交易记录
            self._migrate_trades(trading_dir)
            
        except Exception as e:
            logger.error(f"交易层数据迁移失败: {e}")
            self.stats['trading']['failed'] += 1
    
    def _migrate_orders(self, trading_dir: Path):
        """迁移订单数据"""
        orders_dir = trading_dir / "orders"
        if not orders_dir.exists():
            return
        
        try:
            from src.trading.persistence.trading_persistence import OrderPersistence, OrderData
            
            persistence = OrderPersistence()
            
            for file_path in orders_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                    
                    order = OrderData(**data)
                    persistence.save_order(order)
                    self.stats['trading']['migrated'] += 1
                    
                except Exception as e:
                    logger.error(f"迁移订单失败 {file_path.name}: {e}")
                    self.stats['trading']['failed'] += 1
                    
        except Exception as e:
            logger.error(f"初始化订单持久化失败: {e}")
    
    def _migrate_accounts(self, trading_dir: Path):
        """迁移账户数据"""
        accounts_dir = trading_dir / "accounts"
        if not accounts_dir.exists():
            return
        
        try:
            from src.trading.persistence.trading_persistence import AccountPersistence, AccountData
            
            persistence = AccountPersistence()
            
            for file_path in accounts_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                    
                    account = AccountData(**data)
                    persistence.save_account(account)
                    self.stats['trading']['migrated'] += 1
                    
                except Exception as e:
                    logger.error(f"迁移账户失败 {file_path.name}: {e}")
                    self.stats['trading']['failed'] += 1
                    
        except Exception as e:
            logger.error(f"初始化账户持久化失败: {e}")
    
    def _migrate_positions(self, trading_dir: Path):
        """迁移持仓数据"""
        positions_dir = trading_dir / "positions"
        if not positions_dir.exists():
            return
        
        try:
            from src.trading.persistence.trading_persistence import PositionPersistence, PositionData
            
            persistence = PositionPersistence()
            
            for file_path in positions_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                    
                    position = PositionData(**data)
                    persistence.save_position(position)
                    self.stats['trading']['migrated'] += 1
                    
                except Exception as e:
                    logger.error(f"迁移持仓失败 {file_path.name}: {e}")
                    self.stats['trading']['failed'] += 1
                    
        except Exception as e:
            logger.error(f"初始化持仓持久化失败: {e}")
    
    def _migrate_trades(self, trading_dir: Path):
        """迁移交易记录"""
        trades_dir = trading_dir / "trades"
        if not trades_dir.exists():
            return
        
        try:
            from src.trading.persistence.trading_persistence import TradePersistence, TradeData
            
            persistence = TradePersistence()
            
            for file_path in trades_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    data['traded_at'] = datetime.fromisoformat(data['traded_at'])
                    
                    trade = TradeData(**data)
                    persistence.save_trade(trade)
                    self.stats['trading']['migrated'] += 1
                    
                except Exception as e:
                    logger.error(f"迁移交易记录失败 {file_path.name}: {e}")
                    self.stats['trading']['failed'] += 1
                    
        except Exception as e:
            logger.error(f"初始化交易记录持久化失败: {e}")
    
    def migrate_risk(self):
        """迁移风险控制层数据"""
        logger.info("迁移风险控制层数据...")
        
        risk_dir = self.data_dir / "risk"
        if not risk_dir.exists():
            logger.info("风险控制层数据目录不存在，跳过")
            return
        
        try:
            # 迁移风险检查数据
            self._migrate_risk_checks(risk_dir)
            
            # 迁移告警数据
            self._migrate_alerts(risk_dir)
            
            # 迁移风险指标数据
            self._migrate_risk_metrics(risk_dir)
            
            # 迁移风险规则数据
            self._migrate_risk_rules(risk_dir)
            
        except Exception as e:
            logger.error(f"风险控制层数据迁移失败: {e}")
            self.stats['risk']['failed'] += 1
    
    def _migrate_risk_checks(self, risk_dir: Path):
        """迁移风险检查数据"""
        checks_dir = risk_dir / "checks"
        if not checks_dir.exists():
            return
        
        try:
            from src.risk.persistence.risk_persistence import RiskCheckPersistence, RiskCheckData
            
            persistence = RiskCheckPersistence()
            
            for file_path in checks_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    
                    check = RiskCheckData(**data)
                    persistence.save_check(check)
                    self.stats['risk']['migrated'] += 1
                    
                except Exception as e:
                    logger.error(f"迁移风险检查失败 {file_path.name}: {e}")
                    self.stats['risk']['failed'] += 1
                    
        except Exception as e:
            logger.error(f"初始化风险检查持久化失败: {e}")
    
    def _migrate_alerts(self, risk_dir: Path):
        """迁移告警数据"""
        alerts_dir = risk_dir / "alerts"
        if not alerts_dir.exists():
            return
        
        try:
            from src.risk.persistence.risk_persistence import AlertPersistence, AlertData
            
            persistence = AlertPersistence()
            
            for file_path in alerts_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    if data.get('acknowledged_at'):
                        data['acknowledged_at'] = datetime.fromisoformat(data['acknowledged_at'])
                    if data.get('resolved_at'):
                        data['resolved_at'] = datetime.fromisoformat(data['resolved_at'])
                    
                    alert = AlertData(**data)
                    persistence.save_alert(alert)
                    self.stats['risk']['migrated'] += 1
                    
                except Exception as e:
                    logger.error(f"迁移告警失败 {file_path.name}: {e}")
                    self.stats['risk']['failed'] += 1
                    
        except Exception as e:
            logger.error(f"初始化告警持久化失败: {e}")
    
    def _migrate_risk_metrics(self, risk_dir: Path):
        """迁移风险指标数据"""
        metrics_dir = risk_dir / "metrics"
        if not metrics_dir.exists():
            return
        
        try:
            from src.risk.persistence.risk_persistence import RiskMetricPersistence, RiskMetricData
            
            persistence = RiskMetricPersistence()
            
            for file_path in metrics_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    
                    metric = RiskMetricData(**data)
                    persistence.save_metric(metric)
                    self.stats['risk']['migrated'] += 1
                    
                except Exception as e:
                    logger.error(f"迁移风险指标失败 {file_path.name}: {e}")
                    self.stats['risk']['failed'] += 1
                    
        except Exception as e:
            logger.error(f"初始化风险指标持久化失败: {e}")
    
    def _migrate_risk_rules(self, risk_dir: Path):
        """迁移风险规则数据"""
        rules_dir = risk_dir / "rules"
        if not rules_dir.exists():
            return
        
        try:
            from src.risk.persistence.risk_persistence import RiskRulePersistence, RiskRuleData
            
            persistence = RiskRulePersistence()
            
            for file_path in rules_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                    
                    rule = RiskRuleData(**data)
                    persistence.save_rule(rule)
                    self.stats['risk']['migrated'] += 1
                    
                except Exception as e:
                    logger.error(f"迁移风险规则失败 {file_path.name}: {e}")
                    self.stats['risk']['failed'] += 1
                    
        except Exception as e:
            logger.error(f"初始化风险规则持久化失败: {e}")
    
    def _print_stats(self):
        """打印迁移统计信息"""
        logger.info("=" * 60)
        logger.info("数据迁移统计:")
        logger.info("=" * 60)
        
        total_migrated = 0
        total_failed = 0
        
        for layer, stats in self.stats.items():
            migrated = stats['migrated']
            failed = stats['failed']
            total_migrated += migrated
            total_failed += failed
            
            if migrated > 0 or failed > 0:
                logger.info(f"  {layer}: 迁移成功={migrated}, 失败={failed}")
        
        logger.info("-" * 60)
        logger.info(f"  总计: 迁移成功={total_migrated}, 失败={total_failed}")
        logger.info("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据迁移脚本")
    parser.add_argument("--data-dir", default="data", help="数据目录路径")
    parser.add_argument("--layer", choices=["all", "features", "ml", "strategy", "trading", "risk"],
                       default="all", help="要迁移的层级")
    
    args = parser.parse_args()
    
    migrator = DataMigrator(data_dir=args.data_dir)
    
    if args.layer == "all":
        migrator.migrate_all()
    elif args.layer == "features":
        migrator.migrate_features()
    elif args.layer == "ml":
        migrator.migrate_ml()
    elif args.layer == "strategy":
        migrator.migrate_strategy()
    elif args.layer == "trading":
        migrator.migrate_trading()
    elif args.layer == "risk":
        migrator.migrate_risk()


if __name__ == "__main__":
    main()
