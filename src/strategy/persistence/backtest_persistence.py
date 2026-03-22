#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测持久化实现
Backtest Persistence Implementation

负责回测结果和配置的持久化存储
实现PostgreSQL优先存储策略：
- 主存储: PostgreSQL 数据库
- 降级存储: 文件系统 (JSON/Pickle)

作者: RQA2025 Team
日期: 2026-03-22
"""

import json
import os
import time
import pickle
import logging
import threading
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

import pandas as pd
import numpy as np

from ..interfaces.backtest_interfaces import (
    IBacktestPersistence,
    BacktestConfig,
    BacktestResult,
    BacktestMetrics,
    BacktestTrade,
    BacktestStatus,
    BacktestMode
)

logger = logging.getLogger(__name__)


class BacktestPersistence(IBacktestPersistence):
    """
    回测持久化实现
    
    实现PostgreSQL优先存储，数据库连接失败时自动降级到文件系统。
    """
    
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 1.0
    
    def __init__(self, storage_path: str = None, enable_postgresql: bool = True):
        """
        初始化回测持久化管理器
        
        Args:
            storage_path: 文件系统存储路径
            enable_postgresql: 是否启用PostgreSQL存储
        """
        if storage_path is None:
            self.storage_path = Path.home() / ".rqa2025" / "backtests"
        else:
            self.storage_path = Path(storage_path)
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._results_cache: Dict[str, BacktestResult] = {}
        self._configs_cache: Dict[str, BacktestConfig] = {}
        
        self._lock = threading.Lock()
        
        self._pg_config = None
        self._pg_available = False
        
        if enable_postgresql:
            self._pg_config = self._get_postgresql_config()
            self._pg_available = self._test_postgresql_connection()
            if self._pg_available:
                self._ensure_tables_exist()
                logger.info("✅ BacktestPersistence: PostgreSQL 存储已启用")
            else:
                logger.warning("⚠️ BacktestPersistence: PostgreSQL 不可用，使用文件系统存储")
        
        self.stats = {
            'total_saves': 0,
            'total_loads': 0,
            'postgresql_saves': 0,
            'filesystem_saves': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

    def _get_postgresql_config(self) -> Optional[Dict[str, str]]:
        """获取PostgreSQL配置（使用统一配置模块）"""
        try:
            from src.infrastructure.persistence.database_config import get_db_config
            config = get_db_config()
            return config.to_dict()
        except ImportError:
            logger.debug("统一数据库配置模块导入失败，尝试从环境变量获取")
            return {
                "host": os.getenv("POSTGRES_HOST", "postgres"),
                "port": os.getenv("POSTGRES_PORT", "5432"),
                "database": os.getenv("POSTGRES_DB", "rqa2025_prod"),
                "user": os.getenv("POSTGRES_USER", "rqa2025_admin"),
                "password": os.getenv("POSTGRES_PASSWORD", "SecurePass123!")
            }
        except Exception as e:
            logger.error(f"获取PostgreSQL配置失败: {e}")
            return None

    def _get_db_connection(self):
        """获取数据库连接（带重试机制）"""
        if not self._pg_config:
            return None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                import psycopg2
                conn = psycopg2.connect(
                    host=self._pg_config["host"],
                    port=self._pg_config["port"],
                    database=self._pg_config["database"],
                    user=self._pg_config["user"],
                    password=self._pg_config["password"],
                    connect_timeout=5
                )
                return conn
            except Exception as e:
                logger.debug(f"PostgreSQL连接失败 (尝试 {attempt + 1}/{self.MAX_RETRIES}): {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY_BASE * (2 ** attempt))
        
        return None

    def _test_postgresql_connection(self) -> bool:
        """测试PostgreSQL连接是否可用"""
        conn = self._get_db_connection()
        if conn:
            conn.close()
            return True
        return False

    def _ensure_tables_exist(self):
        """确保数据库表存在"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return
            
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_configs (
                        id SERIAL PRIMARY KEY,
                        backtest_id VARCHAR(128) NOT NULL UNIQUE,
                        strategy_id VARCHAR(128) NOT NULL,
                        start_date TIMESTAMP NOT NULL,
                        end_date TIMESTAMP NOT NULL,
                        initial_capital FLOAT NOT NULL,
                        commission FLOAT DEFAULT 0.0003,
                        slippage FLOAT DEFAULT 0.0001,
                        benchmark_symbol VARCHAR(32),
                        data_frequency VARCHAR(16) DEFAULT '1d',
                        mode VARCHAR(32) DEFAULT 'single',
                        parameters JSONB DEFAULT '{}',
                        risk_limits JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_backtest_configs_strategy 
                    ON backtest_configs(strategy_id)
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_results (
                        id SERIAL PRIMARY KEY,
                        backtest_id VARCHAR(128) NOT NULL UNIQUE,
                        strategy_id VARCHAR(128) NOT NULL,
                        status VARCHAR(32) DEFAULT 'completed',
                        execution_time FLOAT,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        returns_data BYTEA,
                        positions_data BYTEA,
                        trades_data BYTEA,
                        metrics JSONB DEFAULT '{}',
                        risk_metrics JSONB DEFAULT '{}',
                        error_message TEXT,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy 
                    ON backtest_results(strategy_id)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_backtest_results_status 
                    ON backtest_results(status)
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_metrics (
                        id SERIAL PRIMARY KEY,
                        backtest_id VARCHAR(128) NOT NULL,
                        total_return FLOAT,
                        annual_return FLOAT,
                        volatility FLOAT,
                        sharpe_ratio FLOAT,
                        max_drawdown FLOAT,
                        win_rate FLOAT,
                        profit_factor FLOAT,
                        calmar_ratio FLOAT,
                        sortino_ratio FLOAT,
                        alpha FLOAT,
                        beta FLOAT,
                        information_ratio FLOAT,
                        var_95 FLOAT,
                        expected_shortfall FLOAT,
                        recovery_time INTEGER,
                        consecutive_wins INTEGER,
                        consecutive_losses INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_backtest_metrics_backtest 
                    ON backtest_metrics(backtest_id)
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_trades (
                        id SERIAL PRIMARY KEY,
                        trade_id VARCHAR(128) NOT NULL,
                        backtest_id VARCHAR(128) NOT NULL,
                        strategy_id VARCHAR(128) NOT NULL,
                        symbol VARCHAR(32) NOT NULL,
                        side VARCHAR(16) NOT NULL,
                        quantity FLOAT NOT NULL,
                        price FLOAT NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        commission FLOAT,
                        slippage FLOAT,
                        pnl FLOAT,
                        pnl_pct FLOAT,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_backtest_trades_backtest 
                    ON backtest_trades(backtest_id)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_backtest_trades_symbol 
                    ON backtest_trades(symbol, timestamp)
                """)
            
            conn.commit()
            logger.debug("回测存储表已确保存在")
            
        except Exception as e:
            logger.error(f"创建回测存储表失败: {e}")
        finally:
            if conn:
                conn.close()

    def save_backtest_config(self, config: BacktestConfig) -> bool:
        """
        保存回测配置
        
        Args:
            config: 回测配置
            
        Returns:
            保存是否成功
        """
        with self._lock:
            try:
                if self._pg_available:
                    self._save_config_to_postgresql(config)
                
                self._configs_cache[config.backtest_id] = config
                self._save_config_to_filesystem(config)
                
                return True
                
            except Exception as e:
                logger.error(f"保存回测配置失败: {e}")
                return False

    def _save_config_to_postgresql(self, config: BacktestConfig) -> bool:
        """保存配置到PostgreSQL"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return False
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO backtest_configs (
                        backtest_id, strategy_id, start_date, end_date,
                        initial_capital, commission, slippage, benchmark_symbol,
                        data_frequency, mode, parameters, risk_limits, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (backtest_id) DO UPDATE SET
                        strategy_id = EXCLUDED.strategy_id,
                        start_date = EXCLUDED.start_date,
                        end_date = EXCLUDED.end_date,
                        initial_capital = EXCLUDED.initial_capital,
                        commission = EXCLUDED.commission,
                        slippage = EXCLUDED.slippage,
                        benchmark_symbol = EXCLUDED.benchmark_symbol,
                        data_frequency = EXCLUDED.data_frequency,
                        mode = EXCLUDED.mode,
                        parameters = EXCLUDED.parameters,
                        risk_limits = EXCLUDED.risk_limits
                """, (
                    config.backtest_id,
                    config.strategy_id,
                    config.start_date,
                    config.end_date,
                    config.initial_capital,
                    config.commission,
                    config.slippage,
                    config.benchmark_symbol,
                    config.data_frequency,
                    config.mode.value if hasattr(config.mode, 'value') else str(config.mode),
                    json.dumps(config.parameters or {}),
                    json.dumps(config.risk_limits or {}),
                    config.created_at or datetime.now()
                ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"保存配置到PostgreSQL失败: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def _save_config_to_filesystem(self, config: BacktestConfig):
        """保存配置到文件系统"""
        try:
            config_file = self.storage_path / f"{config.backtest_id}_config.json"
            config_data = {
                'backtest_id': config.backtest_id,
                'strategy_id': config.strategy_id,
                'start_date': config.start_date.isoformat() if config.start_date else None,
                'end_date': config.end_date.isoformat() if config.end_date else None,
                'initial_capital': config.initial_capital,
                'commission': config.commission,
                'slippage': config.slippage,
                'benchmark_symbol': config.benchmark_symbol,
                'data_frequency': config.data_frequency,
                'mode': config.mode.value if hasattr(config.mode, 'value') else str(config.mode),
                'parameters': config.parameters,
                'risk_limits': config.risk_limits,
                'created_at': config.created_at.isoformat() if config.created_at else None
            }
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"保存配置到文件系统失败: {e}")

    def load_backtest_config(self, backtest_id: str) -> Optional[BacktestConfig]:
        """
        加载回测配置
        
        Args:
            backtest_id: 回测ID
            
        Returns:
            回测配置
        """
        if backtest_id in self._configs_cache:
            return self._configs_cache[backtest_id]
        
        if self._pg_available:
            config = self._load_config_from_postgresql(backtest_id)
            if config:
                self._configs_cache[backtest_id] = config
                return config
        
        return self._load_config_from_filesystem(backtest_id)

    def _load_config_from_postgresql(self, backtest_id: str) -> Optional[BacktestConfig]:
        """从PostgreSQL加载配置"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return None
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT backtest_id, strategy_id, start_date, end_date,
                           initial_capital, commission, slippage, benchmark_symbol,
                           data_frequency, mode, parameters, risk_limits, created_at
                    FROM backtest_configs
                    WHERE backtest_id = %s AND is_active = TRUE
                """, (backtest_id,))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                return BacktestConfig(
                    backtest_id=row[0],
                    strategy_id=row[1],
                    start_date=row[2],
                    end_date=row[3],
                    initial_capital=row[4],
                    commission=row[5],
                    slippage=row[6],
                    benchmark_symbol=row[7],
                    data_frequency=row[8],
                    mode=BacktestMode(row[9]) if row[9] else BacktestMode.SINGLE,
                    parameters=row[10] or {},
                    risk_limits=row[11] or {},
                    created_at=row[12]
                )
            
        except Exception as e:
            logger.error(f"从PostgreSQL加载配置失败: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def _load_config_from_filesystem(self, backtest_id: str) -> Optional[BacktestConfig]:
        """从文件系统加载配置"""
        try:
            config_file = self.storage_path / f"{backtest_id}_config.json"
            if not config_file.exists():
                return None
            
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return BacktestConfig(
                backtest_id=data['backtest_id'],
                strategy_id=data['strategy_id'],
                start_date=datetime.fromisoformat(data['start_date']) if data.get('start_date') else None,
                end_date=datetime.fromisoformat(data['end_date']) if data.get('end_date') else None,
                initial_capital=data['initial_capital'],
                commission=data.get('commission', 0.0003),
                slippage=data.get('slippage', 0.0001),
                benchmark_symbol=data.get('benchmark_symbol'),
                data_frequency=data.get('data_frequency', '1d'),
                mode=BacktestMode(data.get('mode', 'single')),
                parameters=data.get('parameters', {}),
                risk_limits=data.get('risk_limits', {}),
                created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None
            )
            
        except Exception as e:
            logger.error(f"从文件系统加载配置失败: {e}")
            return None

    def save_backtest_result(self, result: BacktestResult) -> bool:
        """
        保存回测结果（PostgreSQL优先，降级到文件系统）
        
        Args:
            result: 回测结果
            
        Returns:
            保存是否成功
        """
        with self._lock:
            try:
                storage_type = "memory"
                
                if self._pg_available:
                    if self._save_result_to_postgresql(result):
                        storage_type = "postgresql"
                        self.stats['postgresql_saves'] += 1
                    else:
                        logger.warning("PostgreSQL保存失败，降级到文件系统")
                
                if storage_type == "memory":
                    self._save_result_to_filesystem(result)
                    storage_type = "filesystem"
                    self.stats['filesystem_saves'] += 1
                
                self._results_cache[result.backtest_id] = result
                
                self.stats['total_saves'] += 1
                logger.info(f"✅ 回测结果已保存: {result.backtest_id} ({storage_type})")
                
                return True
                
            except Exception as e:
                logger.error(f"保存回测结果失败: {e}")
                return False

    def _save_result_to_postgresql(self, result: BacktestResult) -> bool:
        """保存结果到PostgreSQL"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return False
            
            returns_data = pickle.dumps(result.returns) if result.returns is not None else None
            positions_data = pickle.dumps(result.positions) if result.positions is not None else None
            trades_data = pickle.dumps(result.trades) if result.trades is not None else None
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO backtest_results (
                        backtest_id, strategy_id, status, execution_time,
                        start_time, end_time, returns_data, positions_data,
                        trades_data, metrics, risk_metrics, error_message, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (backtest_id) DO UPDATE SET
                        strategy_id = EXCLUDED.strategy_id,
                        status = EXCLUDED.status,
                        execution_time = EXCLUDED.execution_time,
                        start_time = EXCLUDED.start_time,
                        end_time = EXCLUDED.end_time,
                        returns_data = EXCLUDED.returns_data,
                        positions_data = EXCLUDED.positions_data,
                        trades_data = EXCLUDED.trades_data,
                        metrics = EXCLUDED.metrics,
                        risk_metrics = EXCLUDED.risk_metrics,
                        error_message = EXCLUDED.error_message,
                        metadata = EXCLUDED.metadata
                """, (
                    result.backtest_id,
                    result.strategy_id,
                    result.status.value if hasattr(result.status, 'value') else str(result.status),
                    result.execution_time,
                    result.start_time,
                    result.end_time,
                    returns_data,
                    positions_data,
                    trades_data,
                    json.dumps(result.metrics or {}),
                    json.dumps(result.risk_metrics or {}),
                    result.error_message,
                    json.dumps(result.metadata or {})
                ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"保存结果到PostgreSQL失败: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def _save_result_to_filesystem(self, result: BacktestResult):
        """保存结果到文件系统"""
        try:
            result_file = self.storage_path / f"{result.backtest_id}_result.pkl"
            with open(result_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.error(f"保存结果到文件系统失败: {e}")

    def load_backtest_result(self, backtest_id: str) -> Optional[BacktestResult]:
        """
        加载回测结果（优先从缓存，然后PostgreSQL，最后文件系统）
        
        Args:
            backtest_id: 回测ID
            
        Returns:
            回测结果
        """
        self.stats['total_loads'] += 1
        
        if backtest_id in self._results_cache:
            self.stats['cache_hits'] += 1
            return self._results_cache[backtest_id]
        
        self.stats['cache_misses'] += 1
        
        if self._pg_available:
            result = self._load_result_from_postgresql(backtest_id)
            if result:
                self._results_cache[backtest_id] = result
                return result
        
        result = self._load_result_from_filesystem(backtest_id)
        if result:
            self._results_cache[backtest_id] = result
        
        return result

    def _load_result_from_postgresql(self, backtest_id: str) -> Optional[BacktestResult]:
        """从PostgreSQL加载结果"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return None
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT backtest_id, strategy_id, status, execution_time,
                           start_time, end_time, returns_data, positions_data,
                           trades_data, metrics, risk_metrics, error_message, metadata
                    FROM backtest_results
                    WHERE backtest_id = %s
                """, (backtest_id,))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                returns = pickle.loads(row[6]) if row[6] else pd.Series()
                positions = pickle.loads(row[7]) if row[7] else pd.DataFrame()
                trades = pickle.loads(row[8]) if row[8] else pd.DataFrame()
                
                return BacktestResult(
                    backtest_id=row[0],
                    strategy_id=row[1],
                    status=BacktestStatus(row[2]) if row[2] else BacktestStatus.COMPLETED,
                    execution_time=row[3] or 0.0,
                    start_time=row[4],
                    end_time=row[5],
                    returns=returns,
                    positions=positions,
                    trades=trades,
                    metrics=row[9] or {},
                    risk_metrics=row[10] or {},
                    error_message=row[11],
                    metadata=row[12] or {}
                )
            
        except Exception as e:
            logger.error(f"从PostgreSQL加载结果失败: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def _load_result_from_filesystem(self, backtest_id: str) -> Optional[BacktestResult]:
        """从文件系统加载结果"""
        try:
            result_file = self.storage_path / f"{backtest_id}_result.pkl"
            if not result_file.exists():
                return None
            
            with open(result_file, 'rb') as f:
                return pickle.load(f)
            
        except Exception as e:
            logger.error(f"从文件系统加载结果失败: {e}")
            return None

    def delete_backtest(self, backtest_id: str) -> bool:
        """
        删除回测数据
        
        Args:
            backtest_id: 回测ID
            
        Returns:
            删除是否成功
        """
        return self.delete_backtest_data(backtest_id)

    def delete_backtest_data(self, backtest_id: str) -> bool:
        """
        删除回测数据（接口方法实现）
        
        Args:
            backtest_id: 回测ID
            
        Returns:
            删除是否成功
        """
        with self._lock:
            try:
                deleted = False
                
                if self._pg_available:
                    conn = None
                    try:
                        conn = self._get_db_connection()
                        if conn:
                            with conn.cursor() as cur:
                                cur.execute("DELETE FROM backtest_results WHERE backtest_id = %s", (backtest_id,))
                                cur.execute("UPDATE backtest_configs SET is_active = FALSE WHERE backtest_id = %s", (backtest_id,))
                            conn.commit()
                            deleted = True
                    except Exception:
                        pass
                    finally:
                        if conn:
                            conn.close()
                
                result_file = self.storage_path / f"{backtest_id}_result.pkl"
                if result_file.exists():
                    result_file.unlink()
                    deleted = True
                
                config_file = self.storage_path / f"{backtest_id}_config.json"
                if config_file.exists():
                    config_file.unlink()
                    deleted = True
                
                self._results_cache.pop(backtest_id, None)
                self._configs_cache.pop(backtest_id, None)
                
                if deleted:
                    logger.info(f"回测数据已删除: {backtest_id}")
                
                return deleted
                
            except Exception as e:
                logger.error(f"删除回测数据失败: {e}")
                return False

    def list_backtests(
        self, 
        strategy_id: Optional[str] = None,
        status: Optional[BacktestStatus] = None
    ) -> List[str]:
        """
        列出回测ID
        
        Args:
            strategy_id: 策略ID过滤
            status: 状态过滤
            
        Returns:
            回测ID列表
        """
        backtest_ids = []
        
        if self._pg_available:
            conn = None
            try:
                conn = self._get_db_connection()
                if conn:
                    with conn.cursor() as cur:
                        query = "SELECT backtest_id FROM backtest_configs WHERE is_active = TRUE"
                        params = []
                        
                        if strategy_id:
                            query += " AND strategy_id = %s"
                            params.append(strategy_id)
                        
                        cur.execute(query, params)
                        backtest_ids = [row[0] for row in cur.fetchall()]
            except Exception:
                pass
            finally:
                if conn:
                    conn.close()
        
        if not backtest_ids:
            for config_file in self.storage_path.glob("*_config.json"):
                backtest_id = config_file.stem.replace("_config", "")
                backtest_ids.append(backtest_id)
        
        return backtest_ids

    def save_backtest_metrics(self, metrics: BacktestMetrics) -> bool:
        """
        保存回测指标
        
        Args:
            metrics: 回测指标
            
        Returns:
            保存是否成功
        """
        if not self._pg_available:
            return False
        
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return False
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO backtest_metrics (
                        backtest_id, total_return, annual_return, volatility,
                        sharpe_ratio, max_drawdown, win_rate, profit_factor,
                        calmar_ratio, sortino_ratio, alpha, beta,
                        information_ratio, var_95, expected_shortfall,
                        recovery_time, consecutive_wins, consecutive_losses
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (backtest_id) DO UPDATE SET
                        total_return = EXCLUDED.total_return,
                        annual_return = EXCLUDED.annual_return,
                        volatility = EXCLUDED.volatility,
                        sharpe_ratio = EXCLUDED.sharpe_ratio,
                        max_drawdown = EXCLUDED.max_drawdown,
                        win_rate = EXCLUDED.win_rate,
                        profit_factor = EXCLUDED.profit_factor,
                        calmar_ratio = EXCLUDED.calmar_ratio,
                        sortino_ratio = EXCLUDED.sortino_ratio,
                        alpha = EXCLUDED.alpha,
                        beta = EXCLUDED.beta,
                        information_ratio = EXCLUDED.information_ratio,
                        var_95 = EXCLUDED.var_95,
                        expected_shortfall = EXCLUDED.expected_shortfall,
                        recovery_time = EXCLUDED.recovery_time,
                        consecutive_wins = EXCLUDED.consecutive_wins,
                        consecutive_losses = EXCLUDED.consecutive_losses
                """, (
                    metrics.backtest_id,
                    metrics.total_return,
                    metrics.annual_return,
                    metrics.volatility,
                    metrics.sharpe_ratio,
                    metrics.max_drawdown,
                    metrics.win_rate,
                    metrics.profit_factor,
                    metrics.calmar_ratio,
                    metrics.sortino_ratio,
                    metrics.alpha,
                    metrics.beta,
                    metrics.information_ratio,
                    metrics.var_95,
                    metrics.expected_shortfall,
                    metrics.recovery_time,
                    metrics.consecutive_wins,
                    metrics.consecutive_losses
                ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"保存回测指标失败: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        stats = self.stats.copy()
        stats['postgresql_available'] = self._pg_available
        stats['results_cache_size'] = len(self._results_cache)
        stats['configs_cache_size'] = len(self._configs_cache)
        
        total_requests = stats['cache_hits'] + stats['cache_misses']
        stats['cache_hit_rate'] = stats['cache_hits'] / total_requests if total_requests > 0 else 0.0
        
        return stats

    def sync_to_postgresql(self) -> int:
        """
        将文件系统中的数据同步到PostgreSQL
        
        Returns:
            同步的记录数量
        """
        if not self._pg_available:
            logger.warning("PostgreSQL不可用，无法同步")
            return 0
        
        synced = 0
        
        for result_file in self.storage_path.glob("*_result.pkl"):
            try:
                with open(result_file, 'rb') as f:
                    result = pickle.load(f)
                
                if self._save_result_to_postgresql(result):
                    synced += 1
            except Exception as e:
                logger.error(f"同步回测结果失败 {result_file}: {e}")
        
        logger.info(f"已同步 {synced} 个回测结果到PostgreSQL")
        return synced


_backtest_persistence = None


def get_backtest_persistence(**kwargs) -> BacktestPersistence:
    """
    获取回测持久化实例
    
    Args:
        **kwargs: BacktestPersistence初始化参数
        
    Returns:
        BacktestPersistence实例
    """
    global _backtest_persistence
    if _backtest_persistence is None:
        _backtest_persistence = BacktestPersistence(**kwargs)
    return _backtest_persistence
