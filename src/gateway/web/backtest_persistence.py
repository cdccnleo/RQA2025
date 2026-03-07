"""
回测结果持久化模块
存储回测结果到文件系统或PostgreSQL
符合架构设计：使用统一日志系统和统一持久化模块
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# 使用统一日志系统（符合架构设计：基础设施层统一日志接口）
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# 使用统一持久化模块
from .unified_persistence import get_backtest_persistence

# 全局回测结果持久化管理器
backtest_persistence = get_backtest_persistence()


def save_backtest_result(backtest: Dict[str, Any]) -> bool:
    """
    保存回测结果到持久化存储
    
    Args:
        backtest: 回测结果字典，必须包含backtest_id字段
    
    Returns:
        是否成功保存
    """
    try:
        backtest_id = backtest.get("backtest_id")
        if not backtest_id:
            logger.error("回测结果缺少backtest_id字段，无法保存")
            return False
        
        # 添加保存时间戳
        backtest_data = backtest.copy()
        backtest_data["saved_at"] = time.time()
        backtest_data["updated_at"] = time.time()
        
        # 如果created_at是字符串，保持原样；如果是datetime对象，转换为ISO格式字符串
        if "created_at" in backtest_data:
            if isinstance(backtest_data["created_at"], datetime):
                backtest_data["created_at"] = backtest_data["created_at"].isoformat()
            elif isinstance(backtest_data["created_at"], str):
                pass  # 已经是字符串
            else:
                backtest_data["created_at"] = datetime.now().isoformat()
        else:
            backtest_data["created_at"] = datetime.now().isoformat()
        
        # 使用统一持久化模块保存
        success = backtest_persistence.save(backtest_data, primary_key="backtest_id")
        if success:
            logger.info(f"回测结果已保存: {backtest_id}")
        return success
    except Exception as e:
        logger.error(f"保存回测结果失败: {e}")
        return False


def load_backtest_result(backtest_id: str) -> Optional[Dict[str, Any]]:
    """
    加载回测结果
    
    Args:
        backtest_id: 回测ID
    
    Returns:
        回测结果字典，如果不存在则返回None
    """
    try:
        # 使用统一持久化模块加载
        return backtest_persistence.load(backtest_id, primary_key="backtest_id")
    except Exception as e:
        logger.error(f"加载回测结果失败: {e}")
        return None


def list_backtest_results(strategy_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    列出回测结果
    
    Args:
        strategy_id: 策略ID过滤器（可选）
        limit: 返回的最大结果数
    
    Returns:
        回测结果列表
    """
    try:
        # 构建过滤条件
        filters = None
        if strategy_id:
            filters = {"strategy_id": strategy_id}
        
        # 使用统一持久化模块列出
        results = backtest_persistence.list(filters=filters, limit=limit)
        
        # 按创建时间排序
        results.sort(key=lambda x: x.get("created_at", x.get("saved_at", "")), reverse=True)
        
        return results[:limit]
    except Exception as e:
        logger.error(f"列出回测结果失败: {e}")
        return []


def update_backtest_result(backtest_id: str, updates: Dict[str, Any]) -> bool:
    """
    更新回测结果
    
    Args:
        backtest_id: 回测ID
        updates: 要更新的字段字典
    
    Returns:
        是否成功更新
    """
    try:
        # 加载现有回测结果
        result = load_backtest_result(backtest_id)
        if not result:
            logger.warning(f"回测结果不存在: {backtest_id}")
            return False
        
        # 更新字段
        result.update(updates)
        result["updated_at"] = time.time()
        
        # 保存更新后的回测结果
        return save_backtest_result(result)
    except Exception as e:
        logger.error(f"更新回测结果失败: {e}")
        return False


def delete_backtest_result(backtest_id: str) -> bool:
    """
    删除回测结果及其关联的交易记录

    删除回测结果时，会同时删除：
    - 回测结果主记录
    - 关联的交易记录（trades）
    - 资金曲线数据（equity_curve）
    - 其他关联的详细数据

    Args:
        backtest_id: 回测ID

    Returns:
        是否成功删除
    """
    try:
        # 首先获取回测结果，记录交易记录数量用于日志
        backtest = backtest_persistence.load(backtest_id, primary_key="backtest_id")
        trade_count = 0
        if backtest:
            trades = backtest.get("trades", [])
            if isinstance(trades, list):
                trade_count = len(trades)

        # 使用统一持久化模块删除回测结果
        # 这会删除整个回测记录，包括其中的交易记录
        success = backtest_persistence.delete(backtest_id, primary_key="backtest_id")

        if success:
            logger.info(f"回测结果已删除: {backtest_id}, 同时删除了 {trade_count} 条交易记录")

            # 清理可能存在的独立交易记录文件（兼容旧数据）
            _cleanup_trade_files(backtest_id)
        else:
            logger.warning(f"删除回测结果失败或记录不存在: {backtest_id}")

        return success
    except Exception as e:
        logger.error(f"删除回测结果失败: {e}", exc_info=True)
        return False


def _cleanup_trade_files(backtest_id: str) -> None:
    """
    清理可能存在的独立交易记录文件

    用于兼容旧版本数据，旧版本可能将交易记录单独存储。

    Args:
        backtest_id: 回测ID
    """
    try:
        import os
        import glob

        # 可能的交易记录文件路径模式
        patterns = [
            f"/app/data/trades_{backtest_id}*.json",
            f"/app/data/trades/{backtest_id}*.json",
            f"/app/cache/trades_{backtest_id}*.json",
        ]

        deleted_count = 0
        for pattern in patterns:
            files = glob.glob(pattern)
            for file_path in files:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    logger.debug(f"删除交易记录文件: {file_path}")
                except Exception as e:
                    logger.warning(f"删除交易记录文件失败 {file_path}: {e}")

        if deleted_count > 0:
            logger.info(f"清理了 {deleted_count} 个独立交易记录文件")

    except Exception as e:
        logger.warning(f"清理交易记录文件时出错: {e}")


def batch_save_backtest_results(backtests: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, Any]:
    """
    批量保存回测结果
    
    Args:
        backtests: 回测结果列表，每个元素必须包含backtest_id字段
        batch_size: 每批处理的数据量
    
    Returns:
        批量操作结果，包含成功、失败和跳过的数量
    """
    try:
        # 准备数据，添加时间戳
        prepared_backtests = []
        for backtest in backtests:
            backtest_data = backtest.copy()
            backtest_data["saved_at"] = time.time()
            backtest_data["updated_at"] = time.time()
            
            # 处理created_at字段
            if "created_at" in backtest_data:
                if isinstance(backtest_data["created_at"], datetime):
                    backtest_data["created_at"] = backtest_data["created_at"].isoformat()
                elif isinstance(backtest_data["created_at"], str):
                    pass  # 已经是字符串
                else:
                    backtest_data["created_at"] = datetime.now().isoformat()
            else:
                backtest_data["created_at"] = datetime.now().isoformat()
            
            prepared_backtests.append(backtest_data)
        
        # 使用统一持久化模块批量保存
        result = backtest_persistence.batch_save(prepared_backtests, primary_key="backtest_id", batch_size=batch_size)
        logger.info(f"批量保存回测结果完成: 处理 {result.get('total_processed', 0)} 条，成功 {result.get('success_count', 0)} 条，失败 {result.get('failed_count', 0)} 条")
        return result
    except Exception as e:
        logger.error(f"批量保存回测结果失败: {e}")
        return {
            "success": False,
            "total_processed": 0,
            "success_count": 0,
            "failed_count": len(backtests),
            "skipped_count": 0,
            "processing_time": 0
        }


def batch_delete_backtest_results(backtest_ids: List[str], batch_size: int = 100) -> Dict[str, Any]:
    """
    批量删除回测结果及其关联的交易记录

    批量删除回测结果时，会同时删除每个回测结果关联的交易记录。

    Args:
        backtest_ids: 回测ID列表
        batch_size: 每批处理的数据量

    Returns:
        批量操作结果，包含成功和失败的数量，以及删除的交易记录总数
    """
    try:
        # 首先统计所有回测结果中的交易记录数量
        total_trade_count = 0
        for backtest_id in backtest_ids:
            try:
                backtest = backtest_persistence.load(backtest_id, primary_key="backtest_id")
                if backtest:
                    trades = backtest.get("trades", [])
                    if isinstance(trades, list):
                        total_trade_count += len(trades)
            except Exception as e:
                logger.warning(f"获取回测结果 {backtest_id} 失败: {e}")

        # 使用统一持久化模块批量删除回测结果
        result = backtest_persistence.batch_delete(backtest_ids, primary_key="backtest_id", batch_size=batch_size)

        success_count = result.get('success_count', 0)
        failed_count = result.get('failed_count', 0)

        # 清理可能存在的独立交易记录文件
        for backtest_id in backtest_ids:
            try:
                _cleanup_trade_files(backtest_id)
            except Exception as e:
                logger.warning(f"清理交易记录文件失败 {backtest_id}: {e}")

        logger.info(f"批量删除回测结果完成: 处理 {result.get('total_processed', 0)} 条，"
                   f"成功 {success_count} 条，失败 {failed_count} 条，"
                   f"同时删除了 {total_trade_count} 条交易记录")

        # 返回增强的结果信息
        return {
            "success": True,
            "total_processed": result.get('total_processed', 0),
            "success_count": success_count,
            "failed_count": failed_count,
            "trade_count": total_trade_count,
            "processing_time": result.get('processing_time', 0)
        }
    except Exception as e:
        logger.error(f"批量删除回测结果失败: {e}", exc_info=True)
        return {
            "success": False,
            "total_processed": 0,
            "success_count": 0,
            "failed_count": len(backtest_ids),
            "trade_count": 0,
            "processing_time": 0,
            "error": str(e)
        }
