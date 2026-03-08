"""
parallel_calculator.py

并行特征计算器模块

提供高性能并行特征计算功能，支持：
- 多进程并行计算
- 异步IO优化
- 数据库连接池管理
- 任务队列和调度
- 进度监控和错误处理

适用于A股市场大规模特征计算场景，显著提升计算性能。

作者: RQA2025 Team
日期: 2026-02-13
"""

import asyncio
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """并行计算配置"""
    max_workers: int = 4                    # 最大工作进程数
    batch_size: int = 50                    # 每批处理股票数
    use_processes: bool = True              # 使用进程池（True）或线程池（False）
    chunk_size: int = 10                    # 每个任务处理的股票数
    timeout_seconds: int = 300              # 任务超时时间
    retry_count: int = 3                    # 重试次数
    enable_async: bool = True               # 启用异步IO
    db_pool_size: int = 10                  # 数据库连接池大小


@dataclass
class ParallelResult:
    """并行计算结果"""
    symbol: str
    success: bool
    features: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    calc_time_ms: float = 0.0
    worker_id: int = 0


@dataclass
class BatchProgress:
    """批次进度"""
    batch_id: int
    total: int
    completed: int = 0
    failed: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    @property
    def progress_percent(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100
    
    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()


class ParallelFeatureCalculator:
    """
    并行特征计算器
    
    使用多进程/线程并行计算特征，显著提升大规模计算性能。
    
    Attributes:
        config: 并行计算配置
        _executor: 进程/线程池执行器
        _progress_callbacks: 进度回调函数列表
        
    Example:
        >>> config = ParallelConfig(max_workers=8, batch_size=100)
        >>> calculator = ParallelFeatureCalculator(config)
        >>> 
        >>> # 并行计算多只股票
        >>> symbols = ["002837", "688702", ...]  # 5000只股票
        >>> results = calculator.calculate_batch(symbols, "2026-02-13")
        >>> 
        >>> # 使用异步API
        >>> results = await calculator.calculate_batch_async(symbols, "2026-02-13")
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        初始化并行特征计算器
        
        Args:
            config: 并行计算配置
        """
        self.config = config or ParallelConfig()
        self._executor: Optional[Any] = None
        self._progress_callbacks: List[Callable] = []
        self._batch_progress: Dict[int, BatchProgress] = {}
        
        # 初始化执行器
        self._init_executor()
        
        logger.info(f"ParallelFeatureCalculator 初始化完成: "
                   f"workers={self.config.max_workers}, "
                   f"batch_size={self.config.batch_size}, "
                   f"use_processes={self.config.use_processes}")
    
    def _init_executor(self):
        """初始化执行器"""
        if self.config.use_processes:
            # 使用进程池（CPU密集型任务）
            self._executor = ProcessPoolExecutor(
                max_workers=self.config.max_workers,
                mp_context=mp.get_context('spawn')  # 使用spawn模式避免fork问题
            )
            logger.info(f"进程池初始化完成: {self.config.max_workers} 个进程")
        else:
            # 使用线程池（IO密集型任务）
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.max_workers
            )
            logger.info(f"线程池初始化完成: {self.config.max_workers} 个线程")
    
    def _calculate_single_stock(
        self,
        symbol: str,
        end_date: str,
        feature_type: str = "technical"
    ) -> ParallelResult:
        """
        计算单只股票特征（用于进程池）
        
        注意：此方法在子进程中执行，不能访问父进程的资源
        """
        import os
        worker_id = os.getpid()
        start_time = time.time()
        
        try:
            # 在子进程中重新初始化数据加载器
            from src.data_management.loaders import PostgreSQLDataLoader, DataLoaderConfig
            from src.gateway.web.feature_engineering_service import get_feature_engine
            
            config = DataLoaderConfig(source_type="postgresql")
            data_loader = PostgreSQLDataLoader(config)
            
            # 加载数据（使用较大的日期范围）
            start_date = "2020-01-01"
            load_result = data_loader.load_stock_data(symbol, start_date, end_date)
            
            if not load_result.success or load_result.data is None:
                return ParallelResult(
                    symbol=symbol,
                    success=False,
                    error=f"加载数据失败: {load_result.message}",
                    calc_time_ms=(time.time() - start_time) * 1000,
                    worker_id=worker_id
                )
            
            # 计算特征
            engine = get_feature_engine()
            if engine and hasattr(engine, 'process_features'):
                features = engine.process_features(load_result.data)
                calc_time_ms = (time.time() - start_time) * 1000
                
                # 关闭数据加载器
                data_loader.close()
                
                return ParallelResult(
                    symbol=symbol,
                    success=True,
                    features=features,
                    calc_time_ms=calc_time_ms,
                    worker_id=worker_id
                )
            else:
                return ParallelResult(
                    symbol=symbol,
                    success=False,
                    error="特征引擎不可用",
                    calc_time_ms=(time.time() - start_time) * 1000,
                    worker_id=worker_id
                )
        
        except Exception as e:
            return ParallelResult(
                symbol=symbol,
                success=False,
                error=str(e),
                calc_time_ms=(time.time() - start_time) * 1000,
                worker_id=worker_id
            )
    
    def calculate_batch(
        self,
        symbols: List[str],
        end_date: str,
        feature_type: str = "technical",
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        批量并行计算特征（同步接口）
        
        Args:
            symbols: 股票代码列表
            end_date: 数据截止日期
            feature_type: 特征类型
            progress_callback: 进度回调函数
            
        Returns:
            计算结果统计
        """
        if not self._executor:
            raise RuntimeError("执行器未初始化")
        
        total = len(symbols)
        batch_id = int(time.time())
        progress = BatchProgress(batch_id=batch_id, total=total)
        self._batch_progress[batch_id] = progress
        
        logger.info(f"开始批量并行计算: {total} 只股票, "
                   f"workers={self.config.max_workers}")
        
        results = []
        completed = 0
        failed = 0
        
        start_time = time.time()
        
        # 提交所有任务
        future_to_symbol = {
            self._executor.submit(
                self._calculate_single_stock,
                symbol,
                end_date,
                feature_type
            ): symbol
            for symbol in symbols
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result(timeout=self.config.timeout_seconds)
                results.append(result)
                
                if result.success:
                    completed += 1
                    progress.completed += 1
                else:
                    failed += 1
                    progress.failed += 1
                    logger.warning(f"{symbol} 计算失败: {result.error}")
                
                # 调用进度回调
                if progress_callback:
                    progress_callback(progress)
                
                # 每10个任务输出一次进度
                if (completed + failed) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (completed + failed) / elapsed if elapsed > 0 else 0
                    logger.info(f"进度: {completed + failed}/{total} "
                               f"(成功: {completed}, 失败: {failed}, "
                               f"速率: {rate:.2f} 只/秒)")
            
            except Exception as e:
                logger.error(f"处理 {symbol} 结果时出错: {e}")
                failed += 1
                progress.failed += 1
        
        progress.end_time = datetime.now()
        elapsed = time.time() - start_time
        
        # 统计结果
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        avg_calc_time = sum(r.calc_time_ms for r in successful_results) / len(successful_results) if successful_results else 0
        
        stats = {
            "batch_id": batch_id,
            "total": total,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total if total > 0 else 0,
            "elapsed_seconds": elapsed,
            "avg_rate": total / elapsed if elapsed > 0 else 0,
            "avg_calc_time_ms": avg_calc_time,
            "worker_count": self.config.max_workers,
            "failed_symbols": [r.symbol for r in failed_results]
        }
        
        logger.info(f"批量计算完成: 总计 {total}, 成功 {completed}, 失败 {failed}, "
                   f"耗时 {elapsed:.2f} 秒, 平均速率 {stats['avg_rate']:.2f} 只/秒")
        
        return {
            "stats": stats,
            "results": results
        }
    
    async def calculate_batch_async(
        self,
        symbols: List[str],
        end_date: str,
        feature_type: str = "technical",
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        批量并行计算特征（异步接口）
        
        Args:
            symbols: 股票代码列表
            end_date: 数据截止日期
            feature_type: 特征类型
            progress_callback: 进度回调函数
            
        Returns:
            计算结果统计
        """
        loop = asyncio.get_event_loop()
        
        # 在线程池中执行同步计算
        return await loop.run_in_executor(
            None,
            self.calculate_batch,
            symbols,
            end_date,
            feature_type,
            progress_callback
        )
    
    async def calculate_with_async_io(
        self,
        symbols: List[str],
        end_date: str,
        feature_type: str = "technical"
    ) -> Dict[str, Any]:
        """
        使用异步IO计算特征
        
        适用于IO密集型场景，通过异步并发提高性能。
        
        Args:
            symbols: 股票代码列表
            end_date: 数据截止日期
            feature_type: 特征类型
            
        Returns:
            计算结果统计
        """
        import aiohttp
        import asyncpg
        
        total = len(symbols)
        completed = 0
        failed = 0
        results = []
        
        start_time = time.time()
        
        # 创建数据库连接池
        try:
            pool = await asyncpg.create_pool(
                host=os.getenv('DB_HOST', 'localhost'),
                port=int(os.getenv('DB_PORT', '5432')),
                database=os.getenv('DB_NAME', 'rqa2025_prod'),
                user=os.getenv('DB_USER', 'rqa2025_admin'),
                password=os.getenv('DB_PASSWORD', 'CHANGE_ME_IN_PRODUCTION'),
                min_size=5,
                max_size=self.config.db_pool_size
            )
        except Exception as e:
            logger.error(f"创建数据库连接池失败: {e}")
            return {"error": str(e)}
        
        async def calculate_single_async(symbol: str) -> ParallelResult:
            """异步计算单只股票"""
            calc_start = time.time()
            
            try:
                # 异步查询数据
                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT date, open_price, high_price, low_price, close_price, volume
                        FROM akshare_stock_data
                        WHERE symbol = $1 AND date <= $2
                        ORDER BY date ASC
                        """,
                        symbol, end_date
                    )
                    
                    if not rows:
                        return ParallelResult(
                            symbol=symbol,
                            success=False,
                            error="无数据",
                            calc_time_ms=(time.time() - calc_start) * 1000
                        )
                    
                    # 转换为DataFrame
                    df = pd.DataFrame(rows)
                    
                    # 计算特征（同步操作，但数据已经在内存中）
                    from src.gateway.web.feature_engineering_service import get_feature_engine
                    engine = get_feature_engine()
                    
                    if engine and hasattr(engine, 'process_features'):
                        features = engine.process_features(df)
                        return ParallelResult(
                            symbol=symbol,
                            success=True,
                            features=features,
                            calc_time_ms=(time.time() - calc_start) * 1000
                        )
                    else:
                        return ParallelResult(
                            symbol=symbol,
                            success=False,
                            error="特征引擎不可用",
                            calc_time_ms=(time.time() - calc_start) * 1000
                        )
            
            except Exception as e:
                return ParallelResult(
                    symbol=symbol,
                    success=False,
                    error=str(e),
                    calc_time_ms=(time.time() - calc_start) * 1000
                )
        
        # 使用信号量限制并发数
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def bounded_calculate(symbol: str) -> ParallelResult:
            async with semaphore:
                return await calculate_single_async(symbol)
        
        # 并发执行所有任务
        tasks = [bounded_calculate(symbol) for symbol in symbols]
        
        # 等待所有任务完成
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)
            
            if result.success:
                completed += 1
            else:
                failed += 1
            
            # 每10个任务输出进度
            if (completed + failed) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (completed + failed) / elapsed if elapsed > 0 else 0
                logger.info(f"异步进度: {completed + failed}/{total} "
                           f"(成功: {completed}, 失败: {failed}, "
                           f"速率: {rate:.2f} 只/秒)")
        
        # 关闭连接池
        await pool.close()
        
        elapsed = time.time() - start_time
        
        stats = {
            "total": total,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total if total > 0 else 0,
            "elapsed_seconds": elapsed,
            "avg_rate": total / elapsed if elapsed > 0 else 0
        }
        
        logger.info(f"异步计算完成: 总计 {total}, 成功 {completed}, 失败 {failed}, "
                   f"耗时 {elapsed:.2f} 秒")
        
        return {
            "stats": stats,
            "results": results
        }
    
    def get_progress(self, batch_id: int) -> Optional[BatchProgress]:
        """获取批次进度"""
        return self._batch_progress.get(batch_id)
    
    def shutdown(self):
        """关闭执行器"""
        if self._executor:
            self._executor.shutdown(wait=True)
            logger.info("并行计算器已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()


# 全局计算器实例（单例模式）
_global_parallel_calculator: Optional[ParallelFeatureCalculator] = None


def get_parallel_calculator(config: Optional[ParallelConfig] = None) -> ParallelFeatureCalculator:
    """
    获取全局并行特征计算器实例
    
    Args:
        config: 并行计算配置
        
    Returns:
        并行特征计算器实例
    """
    global _global_parallel_calculator
    
    if _global_parallel_calculator is None:
        _global_parallel_calculator = ParallelFeatureCalculator(config)
    
    return _global_parallel_calculator


def close_parallel_calculator():
    """关闭全局并行特征计算器实例"""
    global _global_parallel_calculator
    
    if _global_parallel_calculator:
        _global_parallel_calculator.shutdown()
        _global_parallel_calculator = None


# 性能测试函数
def benchmark_parallel_calculation(
    symbols: List[str],
    end_date: str,
    workers_list: List[int] = [1, 2, 4, 8]
) -> Dict[str, Any]:
    """
    基准测试并行计算性能
    
    Args:
        symbols: 股票代码列表
        end_date: 数据截止日期
        workers_list: 测试的工作进程数列表
        
    Returns:
        基准测试结果
    """
    results = []
    
    for workers in workers_list:
        logger.info(f"测试 {workers} 个worker...")
        
        config = ParallelConfig(max_workers=workers)
        calculator = ParallelFeatureCalculator(config)
        
        start_time = time.time()
        result = calculator.calculate_batch(symbols, end_date)
        elapsed = time.time() - start_time
        
        stats = result["stats"]
        results.append({
            "workers": workers,
            "elapsed_seconds": elapsed,
            "avg_rate": stats["avg_rate"],
            "success_rate": stats["success_rate"]
        })
        
        calculator.shutdown()
    
    # 找出最优配置
    best = max(results, key=lambda x: x["avg_rate"])
    
    return {
        "benchmarks": results,
        "optimal_workers": best["workers"],
        "optimal_rate": best["avg_rate"]
    }
