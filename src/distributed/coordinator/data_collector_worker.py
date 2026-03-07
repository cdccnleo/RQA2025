"""
数据采集工作器
负责执行数据采集任务，从AKShare获取数据并写入数据库
包含防封禁机制：随机延迟、请求频率限制、错误退避
"""

import logging
import time
import threading
import random
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class DataCollectorWorker:
    """
    数据采集工作器
    
    职责：
    1. 从统一调度器获取数据采集任务
    2. 使用AKShare采集股票数据
    3. 将数据写入akshare_stock_data表
    4. 报告任务完成状态
    """
    
    def __init__(self, worker_id: Optional[str] = None):
        """
        初始化数据采集工作器
        
        Args:
            worker_id: 工作器ID，默认自动生成
        """
        self.worker_id = worker_id or f"data_collector_{int(time.time())}"
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_task: Optional[str] = None
        self._stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "data_points_collected": 0
        }
        
        # 初始化采集器
        self._collector = None
        
        # 防封禁机制配置
        self._anti_ban_config = {
            "min_delay": 1,      # 最小延迟1秒
            "max_delay": 5,      # 最大延迟5秒
            "error_backoff": 2,  # 错误退避倍数
            "max_backoff": 300,  # 最大退避300秒
            "consecutive_errors": 0,  # 连续错误计数
            "last_request_time": 0    # 上次请求时间
        }
        
        logger.info(f"数据采集工作器初始化完成: {self.worker_id}")
    
    def _apply_anti_ban_delay(self):
        """
        应用防封禁延迟
        
        添加随机延迟，模拟人工操作，防止被封禁
        """
        try:
            config = self._anti_ban_config
            
            # 计算基础延迟（随机1-5秒）
            base_delay = random.uniform(config["min_delay"], config["max_delay"])
            
            # 如果有连续错误，添加指数退避
            if config["consecutive_errors"] > 0:
                backoff = min(
                    config["error_backoff"] ** config["consecutive_errors"],
                    config["max_backoff"]
                )
                delay = base_delay + backoff
                logger.warning(f"⚠️ 连续错误 {config['consecutive_errors']} 次，应用退避延迟: {delay:.2f}秒")
            else:
                delay = base_delay
            
            # 确保请求间隔至少1秒
            time_since_last = time.time() - config["last_request_time"]
            if time_since_last < 1:
                delay += (1 - time_since_last)
            
            if delay > 0:
                logger.debug(f"⏱️ 应用防封禁延迟: {delay:.2f}秒")
                time.sleep(delay)
            
            # 更新上次请求时间
            config["last_request_time"] = time.time()
            
        except Exception as e:
            logger.error(f"应用防封禁延迟失败: {e}")
    
    def _record_success(self):
        """记录成功，重置连续错误计数"""
        if self._anti_ban_config["consecutive_errors"] > 0:
            logger.info(f"✅ 采集成功，重置连续错误计数 (之前: {self._anti_ban_config['consecutive_errors']})")
            self._anti_ban_config["consecutive_errors"] = 0
    
    def _record_error(self):
        """记录错误，增加连续错误计数"""
        self._anti_ban_config["consecutive_errors"] += 1
        logger.warning(f"⚠️ 采集错误，连续错误计数: {self._anti_ban_config['consecutive_errors']}")

    def _get_symbols_from_config(self, source_config: Dict[str, Any]) -> List[str]:
        """
        从数据源配置中获取股票列表（混合方案）

        优先级：
        1. 配置的 custom_stocks
        2. 配置的 symbols
        3. 根据 akshare_function 自动获取
        4. 使用默认测试股票

        Args:
            source_config: 数据源配置

        Returns:
            List[str]: 股票代码列表
        """
        symbols = []

        if source_config:
            # 1. 优先使用配置的 custom_stocks
            custom_stocks = source_config.get("custom_stocks", [])
            if custom_stocks:
                symbols = [stock.get("code") for stock in custom_stocks if stock.get("code")]
                if symbols:
                    logger.info(f"从 custom_stocks 获取到 {len(symbols)} 个股票")
                    return symbols

            # 2. 尝试使用配置的 symbols
            config_symbols = source_config.get("symbols", [])
            if config_symbols:
                if isinstance(config_symbols, list):
                    symbols = config_symbols
                elif isinstance(config_symbols, str):
                    symbols = [s.strip() for s in config_symbols.split(",") if s.strip()]
                if symbols:
                    logger.info(f"从 symbols 获取到 {len(symbols)} 个股票")
                    return symbols

            # 3. 根据 akshare_function 自动获取股票列表
            akshare_function = source_config.get("akshare_function")
            if akshare_function:
                auto_symbols = self._auto_get_symbols_by_function(akshare_function)
                if auto_symbols:
                    logger.info(f"根据 {akshare_function} 自动获取到 {len(auto_symbols)} 个股票")
                    return auto_symbols

        # 4. 使用默认测试股票并记录警告
        logger.warning("未配置股票列表，使用默认测试股票 000001")
        return ["000001"]

    def _auto_get_symbols_by_function(self, akshare_function: str) -> List[str]:
        """
        根据 AKShare 函数名自动获取股票列表

        Args:
            akshare_function: AKShare 函数名

        Returns:
            List[str]: 股票代码列表
        """
        try:
            import akshare as ak

            # A股相关函数
            if akshare_function in ["stock_zh_a_spot_em", "stock_zh_a_hist"]:
                # 获取A股所有股票列表
                df = ak.stock_zh_a_spot_em()
                if df is not None and not df.empty:
                    # 限制数量，避免采集过多
                    symbols = df["代码"].tolist()[:50]  # 取前50只
                    logger.info(f"获取到A股列表: {len(symbols)} 只")
                    return symbols

            # 港股相关函数
            elif akshare_function in ["stock_hk_spot_em", "stock_hk_hist"]:
                df = ak.stock_hk_spot_em()
                if df is not None and not df.empty:
                    symbols = df["代码"].tolist()[:50]
                    logger.info(f"获取到港股列表: {len(symbols)} 只")
                    return symbols

            # 指数相关函数
            elif akshare_function in ["stock_zh_index_spot_em", "stock_zh_index_hist"]:
                df = ak.stock_zh_index_spot_em()
                if df is not None and not df.empty:
                    symbols = df["代码"].tolist()[:20]
                    logger.info(f"获取到指数列表: {len(symbols)} 个")
                    return symbols

            # 期货相关函数
            elif akshare_function in ["futures_zh_daily_sina", "futures_zh_spot"]:
                df = ak.futures_zh_spot()
                if df is not None and not df.empty:
                    symbols = df["代码"].tolist()[:30]
                    logger.info(f"获取到期货列表: {len(symbols)} 个")
                    return symbols

            # 外汇相关函数
            elif akshare_function in ["currency_boc_safe", "currency_boc_sina"]:
                # 外汇使用货币对代码
                return ["USDCNY", "EURCNY", "GBPCNY", "JPYCNY", "HKDCNY"]

            # 债券相关函数
            elif akshare_function in ["bond_zh_us_rate", "bond_zh_spot"]:
                df = ak.bond_zh_spot()
                if df is not None and not df.empty:
                    symbols = df["代码"].tolist()[:20]
                    logger.info(f"获取到债券列表: {len(symbols)} 个")
                    return symbols

            # 宏观经济数据（不需要股票代码）
            elif akshare_function in ["macro_china_gdp_yearly", "macro_usa_gdp_monthly"]:
                # 宏观经济数据不需要股票代码，返回空列表
                logger.info("宏观经济数据不需要股票代码")
                return []

            # 新闻数据（不需要股票代码）
            elif "news" in akshare_function:
                logger.info("新闻数据不需要股票代码")
                return []

            logger.warning(f"未找到 {akshare_function} 对应的股票列表获取方法")
            return []

        except Exception as e:
            logger.error(f"自动获取股票列表失败: {e}")
            return []

    def _get_collector(self):
        """获取数据采集器（延迟初始化）"""
        if self._collector is None:
            try:
                from src.data.collectors.akshare_collector import AKShareCollector
                self._collector = AKShareCollector()
                logger.info("AKShare采集器初始化成功")
            except Exception as e:
                logger.error(f"AKShare采集器初始化失败: {e}")
                raise
        return self._collector
    
    def start(self):
        """启动工作器"""
        if self._running:
            logger.warning(f"工作器 {self.worker_id} 已在运行中")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        
        # 注册到WorkerRegistry
        self._register_worker()
        
        logger.info(f"数据采集工作器已启动: {self.worker_id}")
    
    def stop(self):
        """停止工作器"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        
        # 从WorkerRegistry注销
        self._unregister_worker()
        
        logger.info(f"数据采集工作器已停止: {self.worker_id}")
    
    def _register_worker(self):
        """注册到统一工作器注册表"""
        try:
            from src.distributed.registry.unified_worker_registry import (
                get_unified_worker_registry, WorkerType, WorkerStatus
            )
            
            registry = get_unified_worker_registry()
            registry.register_worker(
                worker_id=self.worker_id,
                worker_type=WorkerType.DATA_COLLECTOR,
                capabilities={
                    "data_sources": ["akshare"],
                    "data_types": ["stock", "index"],
                    "max_concurrent_tasks": 1
                },
                metadata={
                    "started_at": datetime.now().isoformat(),
                    "version": "1.0.0"
                }
            )
            
            logger.info(f"工作器已注册到Registry: {self.worker_id}")
            
        except Exception as e:
            logger.error(f"注册工作器失败: {e}")
    
    def _unregister_worker(self):
        """从统一工作器注册表注销"""
        try:
            from src.distributed.registry.unified_worker_registry import (
                get_unified_worker_registry
            )
            
            registry = get_unified_worker_registry()
            registry.unregister_worker(self.worker_id)
            
            logger.info(f"工作器已从Registry注销: {self.worker_id}")
            
        except Exception as e:
            logger.error(f"注销工作器失败: {e}")
    
    def _worker_loop(self):
        """工作器主循环"""
        logger.info(f"工作器主循环已启动: {self.worker_id}")
        
        while self._running:
            try:
                # 获取任务
                task = self._get_task()
                if task:
                    self._process_task(task)
                else:
                    # 没有任务，休眠一段时间
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"工作器循环错误: {e}")
                time.sleep(5)
        
        logger.info(f"工作器主循环已停止: {self.worker_id}")
    
    def _get_task(self) -> Optional[Dict[str, Any]]:
        """从调度器获取任务"""
        try:
            from src.distributed.coordinator.unified_scheduler import (
                get_unified_scheduler
            )
            from src.distributed.registry.unified_worker_registry import (
                WorkerType
            )
            
            scheduler = get_unified_scheduler()
            
            # 更新Worker状态为忙碌
            self._update_worker_status("busy")
            
            # 获取任务
            task = scheduler.get_task(self.worker_id, WorkerType.DATA_COLLECTOR)
            
            if task:
                self._current_task = task.task_id
                logger.info(f"获取到任务: {task.task_id}")
                return {
                    "task_id": task.task_id,
                    "task_type": task.task_type.value,
                    "data": task.data,
                    "priority": task.priority.value,
                    "metadata": task.metadata or {}  # 添加元数据
                }
            
            # 没有任务，更新状态为空闲
            self._update_worker_status("idle")
            return None
            
        except Exception as e:
            logger.error(f"获取任务失败: {e}")
            return None
    
    def _update_worker_status(self, status: str):
        """更新Worker状态"""
        try:
            from src.distributed.registry.unified_worker_registry import (
                get_unified_worker_registry, WorkerStatus
            )
            
            registry = get_unified_worker_registry()
            
            status_map = {
                "idle": WorkerStatus.IDLE,
                "busy": WorkerStatus.BUSY,
                "error": WorkerStatus.ERROR
            }
            
            worker_status = status_map.get(status, WorkerStatus.IDLE)
            registry.update_worker_status(self.worker_id, worker_status)
            
        except Exception as e:
            logger.debug(f"更新Worker状态失败: {e}")
    
    def _process_task(self, task: Dict[str, Any]):
        """处理数据采集任务（包含防封禁机制）"""
        task_id = task.get("task_id", "unknown")
        task_data = task.get("data", {})

        logger.info(f"开始处理任务: {task_id}")

        # 获取任务历史管理器
        from src.gateway.web.task_history_manager import get_task_history_manager
        history_manager = get_task_history_manager()

        # 更新任务状态为运行中
        history_manager.update_task_started(task_id)
        history_manager.add_task_log(task_id, "INFO", f"工作器 {self.worker_id} 开始处理任务")

        try:
            # 应用防封禁延迟
            self._apply_anti_ban_delay()

            # 提取任务参数
            # 从任务数据中获取source_id和source_config
            source_id = task_data.get("source_id")
            source_config = task_data.get("source_config", {})

            # 从source_config中提取股票列表（使用混合方案）
            symbols = self._get_symbols_from_config(source_config)

            history_manager.add_task_log(task_id, "INFO", f"获取到 {len(symbols)} 个股票代码: {symbols[:5]}...")

            # 获取日期范围
            from datetime import datetime, timedelta

            # 优先使用任务指定的日期范围（用于历史数据补齐）
            collection_type = task_data.get("collection_type", "immediate")
            task_start_date = task_data.get("start_date")
            task_end_date = task_data.get("end_date")

            if task_start_date and task_end_date:
                # 使用任务指定的日期范围
                start_date = task_start_date.replace("-", "")  # 转换为YYYYMMDD格式
                end_date = task_end_date.replace("-", "")
                logger.info(f"使用任务指定的日期范围: {start_date} 到 {end_date}")
                history_manager.add_task_log(task_id, "INFO", f"采集模式: {collection_type}, 日期范围: {task_start_date} 到 {task_end_date}")
            else:
                # 使用默认日期范围
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

                # 检查是否有数据源特定的日期配置
                if source_config:
                    default_days = source_config.get("default_days", 30)
                    start_date = (datetime.now() - timedelta(days=default_days)).strftime("%Y%m%d")
            
            data_types = task_data.get("data_types", ["stock"])
            
            logger.info(f"任务参数: source_id={source_id}, symbols={symbols}, start_date={start_date}, end_date={end_date}")
            
            # 执行数据采集
            results = []
            has_error = False
            
            for symbol in symbols:
                try:
                    # 每个股票采集前应用防封禁延迟
                    if len(results) > 0:  # 第一个股票不需要额外延迟
                        self._apply_anti_ban_delay()

                    history_manager.add_task_log(task_id, "INFO", f"开始采集股票: {symbol}")
                    success = self._collect_and_save(symbol, start_date, end_date)
                    results.append({
                        "symbol": symbol,
                        "success": success
                    })

                    if success:
                        self._stats["tasks_completed"] += 1
                        self._record_success()  # 记录成功
                        history_manager.add_task_log(task_id, "INFO", f"股票 {symbol} 采集成功")
                    else:
                        self._stats["tasks_failed"] += 1
                        self._record_error()  # 记录错误
                        has_error = True
                        history_manager.add_task_log(task_id, "WARN", f"股票 {symbol} 采集失败")

                except Exception as e:
                    logger.error(f"采集股票数据失败 {symbol}: {e}")
                    results.append({
                        "symbol": symbol,
                        "success": False,
                        "error": str(e)
                    })
                    self._stats["tasks_failed"] += 1
                    self._record_error()  # 记录错误
                    has_error = True
                    history_manager.add_task_log(task_id, "ERROR", f"股票 {symbol} 采集异常: {str(e)}")
            
            # 完成任务 - 传递source_id用于更新最后采集时间
            # source_id 已在前面提取

            # 如果全部成功，传递结果；如果有错误，传递None
            final_result = results if not has_error else None
            error_msg = None if not has_error else "部分股票采集失败"

            # 更新任务历史状态
            if not has_error:
                # 计算采集的记录数（这里简化处理，实际应该从结果中统计）
                total_records = len([r for r in results if r['success']]) * 100  # 估算
                history_manager.update_task_completed(
                    task_id=task_id,
                    records_count=total_records,
                    data_size_mb=round(total_records * 0.001, 2),  # 估算数据大小
                    logs=[
                        {"timestamp": datetime.now().isoformat(), "level": "INFO", "message": f"成功采集 {len(results)} 个股票"}
                    ]
                )
                history_manager.add_task_log(task_id, "INFO", f"任务完成，成功采集 {len(results)} 个股票")
            else:
                history_manager.update_task_failed(
                    task_id=task_id,
                    error_message=error_msg or "部分股票采集失败",
                    logs=[
                        {"timestamp": datetime.now().isoformat(), "level": "ERROR", "message": error_msg or "部分股票采集失败"}
                    ]
                )
                history_manager.add_task_log(task_id, "ERROR", error_msg or "部分股票采集失败")

            self._complete_task(task_id, final_result, error_msg, source_id=source_id)

            logger.info(f"任务处理完成: {task_id}, 结果: {len(results)} 个股票, 成功: {sum(1 for r in results if r['success'])}")

        except Exception as e:
            logger.error(f"处理任务失败 {task_id}: {e}")
            self._record_error()  # 记录错误
            # 更新任务历史状态为失败
            history_manager.update_task_failed(
                task_id=task_id,
                error_message=str(e),
                logs=[
                    {"timestamp": datetime.now().isoformat(), "level": "ERROR", "message": f"任务执行异常: {str(e)}"}
                ]
            )
            history_manager.add_task_log(task_id, "ERROR", f"任务执行异常: {str(e)}")
            # source_id 已在前面提取，如果出错时还没提取到，则尝试从task_data获取
            if 'source_id' not in locals() or source_id is None:
                source_id = task_data.get("source_id") if 'task_data' in locals() else None
            self._complete_task(task_id, None, str(e), source_id=source_id)
    
    def _collect_and_save(self, symbol: str, start_date: str, end_date: str) -> bool:
        """
        采集并保存股票数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            是否成功
        """
        try:
            collector = self._get_collector()
            success = collector.collect_and_save(symbol, start_date, end_date)
            
            if success:
                self._stats["data_points_collected"] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"采集并保存数据失败 {symbol}: {e}")
            return False
    
    def _complete_task(self, task_id: str, result: Any = None, error: str = None, source_id: str = None):
        """完成任务并更新最后采集时间
        
        Args:
            task_id: 任务ID
            result: 任务结果
            error: 错误信息
            source_id: 数据源ID（用于更新最后采集时间）
        """
        try:
            from src.distributed.coordinator.unified_scheduler import (
                get_unified_scheduler
            )
            
            scheduler = get_unified_scheduler()
            scheduler.complete_task(task_id, result, error)
            
            # 更新数据源的 last_test 时间
            if result and not error and source_id:
                self._update_last_collection_time(source_id, result)
            
            self._current_task = None
            
            logger.info(f"任务已完成: {task_id}")
            
        except Exception as e:
            logger.error(f"完成任务失败 {task_id}: {e}")
    
    def _update_last_collection_time(self, source_id: str, result: Any):
        """
        更新数据源的最后采集时间
        
        Args:
            source_id: 数据源ID
            result: 任务结果
        """
        try:
            if not source_id:
                logger.debug("没有提供数据源ID，跳过更新最后采集时间")
                return
            
            # 更新数据源配置
            from src.gateway.web.data_source_config_manager import (
                get_data_source_config_manager
            )
            from src.gateway.web.config_manager import save_data_sources
            from datetime import datetime
            
            config_manager = get_data_source_config_manager()
            
            # 获取当前时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 更新数据源的 last_test
            config_manager.update_data_source(source_id, {
                "last_test": current_time,
                "status": "连接正常"
            })
            
            # 保存配置
            save_data_sources(config_manager.get_data_sources())
            
            logger.info(f"✅ 已更新数据源 {source_id} 的最后采集时间: {current_time}")
                
        except Exception as e:
            logger.error(f"更新最后采集时间失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取工作器统计信息"""
        return {
            "worker_id": self.worker_id,
            "is_running": self._running,
            "current_task": self._current_task,
            **self._stats
        }


# 全局工作器实例
_data_collector_worker: Optional[DataCollectorWorker] = None


def get_data_collector_worker() -> DataCollectorWorker:
    """获取全局数据采集工作器实例"""
    global _data_collector_worker
    if _data_collector_worker is None:
        _data_collector_worker = DataCollectorWorker()
    return _data_collector_worker


def start_data_collector_worker():
    """启动数据采集工作器"""
    worker = get_data_collector_worker()
    worker.start()
    return worker


def stop_data_collector_worker():
    """停止数据采集工作器"""
    global _data_collector_worker
    if _data_collector_worker:
        _data_collector_worker.stop()
        _data_collector_worker = None
