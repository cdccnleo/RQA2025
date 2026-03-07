"""
特征任务执行器
负责从调度器获取任务并执行特征提取，更新任务状态和进度

符合数据管理层架构设计：
- 使用 PostgreSQLDataLoader 从数据库加载真实股票数据
- 使用 FeatureEngine 进行特征计算
- 支持数据验证和质量检查
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)

# 导入统一调度器
try:
    from src.distributed.coordinator.unified_scheduler import get_unified_scheduler, TaskType
    from src.distributed.registry import WorkerType
    UNIFIED_SCHEDULER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入统一调度器: {e}")
    UNIFIED_SCHEDULER_AVAILABLE = False

# 导入特征引擎
try:
    from .feature_engineering_service import get_feature_engine
    FEATURE_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入特征引擎: {e}")
    FEATURE_ENGINE_AVAILABLE = False

# 导入PostgreSQL数据加载器（符合数据管理层架构设计）
try:
    from src.data.loader import PostgreSQLDataLoader, DataLoaderConfig
    DATA_LOADER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入数据加载器: {e}")
    DATA_LOADER_AVAILABLE = False


class FeatureTaskExecutor:
    """
    特征任务执行器
    
    负责从调度器获取任务并执行特征提取，更新任务状态和进度。
    符合数据管理层架构设计，使用 PostgreSQLDataLoader 从数据库加载真实数据。
    """
    
    def __init__(self):
        """初始化执行器"""
        self.scheduler = None
        self.engine = None
        self.data_loader = None
        self.running = False
        self.worker_id = f"feature_executor_{id(self)}"
        self._execution_task: Optional[asyncio.Task] = None
        
        # 初始化统一调度器和引擎
        if UNIFIED_SCHEDULER_AVAILABLE:
            self.scheduler = get_unified_scheduler()
            logger.info("统一调度器已初始化")
        if FEATURE_ENGINE_AVAILABLE:
            self.engine = get_feature_engine()
        
        # 初始化PostgreSQL数据加载器（符合数据管理层架构设计）
        if DATA_LOADER_AVAILABLE:
            try:
                config = DataLoaderConfig(
                    source_type="postgresql",
                    timeout=30,
                    retry_count=3
                )
                self.data_loader = PostgreSQLDataLoader(config)
                logger.info("PostgreSQL数据加载器初始化成功")
            except Exception as e:
                logger.error(f"初始化PostgreSQL数据加载器失败: {e}")
                self.data_loader = None
    
    async def start(self):
        """启动执行器"""
        if self.running:
            logger.warning("特征任务执行器已在运行")
            return
        
        if not self.scheduler:
            logger.error("统一调度器不可用，无法启动执行器")
            return
        
        self.running = True
        
        # 注册工作节点到统一调度器
        try:
            # 通过调度器的注册表注册工作节点
            if hasattr(self.scheduler, '_registry'):
                self.scheduler._registry.register_worker(
                    self.worker_id,
                    WorkerType.FEATURE_WORKER,
                    {"capabilities": ["feature_extraction"], "type": "feature_executor"}
                )
                logger.info(f"工作节点已注册到统一调度器: {self.worker_id}")
            else:
                logger.warning("调度器没有注册表，跳过工作节点注册")
        except Exception as e:
            logger.error(f"注册工作节点到统一调度器失败: {e}")
        
        # 启动执行循环
        try:
            self._execution_task = asyncio.create_task(self._execution_loop())
            logger.info("特征任务执行器已启动")
        except Exception as e:
            logger.error(f"启动执行循环失败: {e}")
            self.running = False
    
    async def stop(self):
        """停止执行器"""
        if not self.running:
            return
        
        logger.info("正在停止特征任务执行器...")
        self.running = False
        
        # 取消执行任务
        if self._execution_task:
            self._execution_task.cancel()
            try:
                await self._execution_task
            except asyncio.CancelledError:
                pass
        
        # 注销工作节点
        if self.scheduler:
            try:
                # 通过调度器的注册表注销工作节点
                if hasattr(self.scheduler, '_registry'):
                    self.scheduler._registry.unregister_worker(self.worker_id)
                    logger.info(f"工作节点已注销: {self.worker_id}")
            except Exception as e:
                logger.error(f"注销工作节点失败: {e}")
        
        logger.info("特征任务执行器已停止")
    
    async def _execution_loop(self):
        """执行循环"""
        logger.info("特征任务执行循环已启动")
        
        while self.running:
            try:
                # 从调度器获取任务
                if not self.scheduler:
                    await asyncio.sleep(5)
                    continue
                
                task = self.scheduler.get_task(self.worker_id, WorkerType.FEATURE_WORKER)
                
                if task:
                    logger.info(f"获取到任务: {task.task_id}")
                    await self._execute_task(task)
                else:
                    # 无任务时等待
                    await asyncio.sleep(1)
                    
            except asyncio.CancelledError:
                logger.info("执行循环被取消")
                break
            except Exception as e:
                logger.error(f"执行循环错误: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        logger.info("特征任务执行循环已停止")
    
    async def _execute_task(self, task):
        """执行任务"""
        task_id = task.task_id
        original_task_id = None
        
        # 从metadata中获取原始task_id（统一调度器的任务metadata中存储了原始特征任务ID）
        if hasattr(task, 'metadata') and task.metadata and "task_id" in task.metadata:
            original_task_id = task.metadata["task_id"]
            task_id = original_task_id  # 使用原始task_id进行状态更新
        
        # 处理统一调度器的TaskType枚举
        if hasattr(task.task_type, 'value'):
            task_type_str = task.task_type.value
        else:
            task_type_str = task.task_type
        
        logger.info(f"开始执行任务: {task_id}, 类型: {task_type_str}")
        
        try:
            # 更新任务状态为running
            await self._update_task_status(task_id, "running")
            
            # 执行特征提取
            result = await self._process_feature_task(task)
            
            # 更新任务状态为completed
            await self._update_task_status(
                task_id, "completed",
                progress=100,
                feature_count=result.get("count", 0)
            )
            
            # 保存特征到特征存储表
            features_list = result.get("features", [])
            logger.info(f"📝 准备保存特征到存储表，任务ID: {task_id}, 特征数量: {len(features_list)}")
            if features_list:
                try:
                    from .feature_task_persistence import save_features_to_store
                    
                    # 过滤基础价格特征
                    basic_price_features = {'open', 'high', 'low', 'close', 'volume', 'amount', 'date', 'datetime', 'timestamp'}
                    technical_features = [f for f in features_list if f.lower() not in basic_price_features]
                    filtered_count = len(features_list) - len(technical_features)
                    if filtered_count > 0:
                        logger.info(f"📝 已过滤 {filtered_count} 个基础价格特征")
                    
                    symbol = result.get("symbols", [None])[0] if result.get("symbols") else None

                    # 为特征生成差异化质量评分
                    from src.features.quality import calculate_quality_scores
                    quality_scores = calculate_quality_scores(technical_features)

                    logger.info(f"🚀 调用 save_features_to_store，任务ID: {task_id}, 股票代码: {symbol}, 技术特征数量: {len(technical_features)}")
                    save_result = save_features_to_store(
                        task_id=task_id,
                        features=technical_features,
                        symbol=symbol,
                        quality_scores=quality_scores
                    )
                    if save_result:
                        logger.info(f"✅ 已保存 {len(technical_features)} 个技术特征到特征存储表")
                    else:
                        logger.error(f"❌ 保存特征到存储表失败，返回False")
                except Exception as e:
                    logger.error(f"❌ 保存特征到存储表失败: {e}", exc_info=True)
            else:
                logger.warning(f"⚠️ 特征列表为空，跳过保存到存储表")
            
            # 通知调度器任务完成（使用调度器的task_id）
            if self.scheduler:
                self.scheduler.complete_task(task.task_id, result, None)
            
            logger.info(f"任务执行完成: {task_id}")
            
        except Exception as e:
            logger.error(f"执行任务失败: {task_id}, 错误: {e}", exc_info=True)
            
            # 更新任务状态为failed
            await self._update_task_status(
                task_id, "failed",
                error=str(e)
            )
            
            # 通知调度器任务失败
            if self.scheduler:
                self.scheduler.complete_task(task.task_id, None, str(e))
    
    async def _process_feature_task(self, task):
        """
        处理特征任务 - 使用真实数据
        
        符合数据管理层架构设计：
        1. 从PostgreSQL加载真实股票数据
        2. 使用FeatureEngine进行特征计算
        3. 返回真实的特征结果
        """
        # 处理统一调度器的TaskType枚举
        if hasattr(task.task_type, 'value'):
            task_type = task.task_type.value
        else:
            task_type = task.task_type
        config = task.data if isinstance(task.data, dict) else {}
        original_task_id = task.metadata.get("task_id") if task.metadata else None
        
        logger.info(f"处理特征任务: {task_type}, 配置: {config}")
        
        # 初始化结果
        result = {
            "count": 0,
            "features": [],
            "processing_time": 0,
            "data_source": "database",  # 标记数据来源
            "symbols": [],
            "date_range": {}
        }
        
        start_time = time.time()
        
        try:
            # 步骤1: 从配置获取参数
            symbol = config.get("symbol")
            start_date = config.get("start_date")
            end_date = config.get("end_date")
            indicators = config.get("indicators", ["SMA", "EMA", "RSI", "MACD"])
            
            if not symbol:
                raise ValueError("配置中缺少股票代码(symbol)")
            
            logger.info(f"任务参数: symbol={symbol}, date_range={start_date}~{end_date}, indicators={indicators}")
            
            # 步骤2: 从PostgreSQL加载真实数据（符合数据管理层架构设计）
            if not self.data_loader:
                raise RuntimeError("PostgreSQL数据加载器不可用")
            
            logger.info(f"从PostgreSQL加载股票数据: {symbol}")
            load_result = self.data_loader.load_stock_data(symbol, start_date, end_date)
            
            if not load_result.success:
                raise RuntimeError(f"加载股票数据失败: {load_result.message}")
            
            if load_result.data is None or load_result.data.empty:
                raise ValueError(f"未找到股票数据: {symbol}, 日期范围: {start_date}~{end_date}")
            
            df = load_result.data
            logger.info(f"成功加载 {load_result.row_count} 条股票数据记录")
            
            # 更新进度
            if original_task_id:
                await self._update_task_progress(original_task_id, 20)
            
            # 步骤3: 使用特征引擎处理真实数据
            if self.engine and hasattr(self.engine, 'process_features'):
                logger.info("使用特征引擎处理真实数据")
                
                # 列名映射：将数据库列名映射为标准列名
                column_mapping = {
                    'open_price': 'open',
                    'high_price': 'high',
                    'low_price': 'low',
                    'close_price': 'close',
                    'volume': 'volume'
                }
                
                # 重命名列
                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns and new_col not in df.columns:
                        df = df.rename(columns={old_col: new_col})
                        logger.debug(f"列名映射: {old_col} -> {new_col}")
                
                # 验证数据格式
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    logger.warning(f"数据缺少必要列: {missing_columns}, 可用列: {list(df.columns)}")
                    # 如果有 close 列，尝试使用 close 作为所有价格列
                    if 'close' in df.columns:
                        logger.info("使用 close 列填充缺失的价格列")
                        for col in ['open', 'high', 'low']:
                            if col not in df.columns:
                                df[col] = df['close']
                        missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        raise ValueError(f"数据缺少必要列: {missing_columns}")
                
                # 🚀 关键修复：确保所有价格列都是数值类型
                import pandas as pd
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    if col in df.columns:
                        # 转换为数值类型，无法转换的设为 NaN
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        logger.debug(f"列 {col} 转换为数值类型，类型: {df[col].dtype}")
                
                # 删除包含 NaN 的行
                original_count = len(df)
                df = df.dropna(subset=[col for col in numeric_columns if col in df.columns])
                dropped_count = original_count - len(df)
                if dropped_count > 0:
                    logger.warning(f"删除了 {dropped_count} 行包含 NaN 的数据")
                
                if df.empty:
                    raise ValueError("数据转换后为空，请检查数据质量")
                
                # 处理特征 - 传入 indicators 配置
                try:
                    from src.features.core.config import FeatureConfig, FeatureType
                    
                    # 创建特征配置，传入 indicators
                    feature_config = FeatureConfig(
                        feature_types=[FeatureType.TECHNICAL],
                        technical_indicators=[ind.lower() for ind in indicators] if indicators else ["sma", "ema", "rsi", "macd"],
                        enable_feature_selection=False,
                        enable_standardization=False
                    )
                    
                    logger.info(f"🚀 特征配置: indicators={feature_config.technical_indicators}")
                    
                    processed_df = self.engine.process_features(df, config=feature_config)
                    feature_count = len(processed_df.columns)
                    result["count"] = feature_count
                    result["features"] = list(processed_df.columns)
                    logger.info(f"✅ 特征计算完成: 生成 {feature_count} 个特征, 列表: {result['features'][:10]}...")
                except Exception as e:
                    logger.error(f"❌ 特征计算失败: {e}", exc_info=True)
                    # 降级处理：使用原始数据列作为特征
                    result["count"] = len(df.columns)
                    result["features"] = list(df.columns)
                    result["warning"] = f"特征计算失败，使用原始数据列: {str(e)}"
            else:
                # 特征引擎不可用，使用原始数据
                logger.warning("特征引擎不可用，使用原始数据列作为特征")
                result["count"] = len(df.columns)
                result["features"] = list(df.columns)
            
            # 更新进度
            if original_task_id:
                await self._update_task_progress(original_task_id, 80)
            
            # 填充结果元数据
            result["symbols"] = [symbol]
            result["date_range"] = {
                "start": str(df['date'].min()) if 'date' in df.columns else start_date,
                "end": str(df['date'].max()) if 'date' in df.columns else end_date
            }
            result["row_count"] = len(df)
            
            # 模拟剩余进度更新
            for progress in [90, 100]:
                await asyncio.sleep(0.1)
                if original_task_id:
                    await self._update_task_progress(original_task_id, progress)
            
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            
            logger.info(f"特征提取完成: 从 {load_result.row_count} 条真实数据提取了 {result['count']} 个特征，"
                       f"耗时 {processing_time:.2f} 秒")
            
        except Exception as e:
            logger.error(f"处理特征任务失败: {e}", exc_info=True)
            result["error"] = str(e)
            raise
        
        return result
    
    async def _update_task_status(self, task_id: str, status: str, **updates):
        """更新任务状态"""
        try:
            from .feature_task_persistence import update_feature_task
            
            update_data = {"status": status}
            update_data.update(updates)
            
            # 根据状态设置时间戳
            if status == "running":
                if "start_time" not in update_data:
                    update_data["start_time"] = int(datetime.now().timestamp())
            elif status in ["completed", "failed"]:
                if "end_time" not in update_data:
                    update_data["end_time"] = int(datetime.now().timestamp())
            
            # 更新持久化存储
            success = update_feature_task(task_id, update_data)
            
            if success:
                logger.debug(f"任务状态已更新: {task_id} -> {status}")
            else:
                logger.warning(f"更新任务状态失败: {task_id} -> {status}")
                
        except Exception as e:
            logger.error(f"更新任务状态异常: {task_id} -> {status}, 错误: {e}")
    
    async def _update_task_progress(self, task_id: str, progress: int):
        """更新任务进度"""
        try:
            from .feature_task_persistence import update_feature_task
            
            update_data = {"progress": progress}
            update_feature_task(task_id, update_data)
            
            logger.debug(f"任务进度已更新: {task_id} -> {progress}%")
            
        except Exception as e:
            logger.error(f"更新任务进度异常: {task_id} -> {progress}%, 错误: {e}")


# 全局执行器实例
_executor_instance: Optional[FeatureTaskExecutor] = None


def get_feature_task_executor() -> Optional[FeatureTaskExecutor]:
    """获取特征任务执行器实例"""
    return _executor_instance


async def start_feature_task_executor():
    """启动特征任务执行器（供应用启动时调用）"""
    global _executor_instance
    
    if _executor_instance is None:
        _executor_instance = FeatureTaskExecutor()
        await _executor_instance.start()
    
    return _executor_instance


async def stop_feature_task_executor():
    """停止特征任务执行器（供应用关闭时调用）"""
    global _executor_instance
    
    if _executor_instance:
        await _executor_instance.stop()
        _executor_instance = None

