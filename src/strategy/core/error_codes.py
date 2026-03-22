#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略层错误码体系

提供标准化的错误码定义，支持：
1. 错误码分类管理
2. 错误信息标准化
3. 错误处理策略
4. 错误日志记录

Author: RQA2025 Development Team
Date: 2026-03-24
"""

from enum import Enum
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class StrategyErrorCode(Enum):
    """
    策略层错误码
    
    错误码格式: STRATEGY_XXXX
    - 1000-1999: 通用错误
    - 2000-2999: 策略管理错误
    - 3000-3999: 回测错误
    - 4000-4999: 优化错误
    - 5000-5999: 执行错误
    - 6000-6999: 数据错误
    - 7000-7999: 配置错误
    - 8000-8999: 系统错误
    """
    
    # 通用错误 (1000-1999)
    UNKNOWN_ERROR = ("STRATEGY_1000", "未知错误", "请检查系统日志或联系技术支持")
    INVALID_PARAMETER = ("STRATEGY_1001", "参数无效", "请检查输入参数是否符合要求")
    MISSING_PARAMETER = ("STRATEGY_1002", "缺少必需参数", "请提供所有必需参数")
    OPERATION_FAILED = ("STRATEGY_1003", "操作失败", "操作执行失败，请重试")
    TIMEOUT_ERROR = ("STRATEGY_1004", "操作超时", "操作执行时间超过限制，请检查系统负载")
    
    # 策略管理错误 (2000-2999)
    STRATEGY_NOT_FOUND = ("STRATEGY_2000", "策略不存在", "指定的策略ID不存在或已被删除")
    STRATEGY_ALREADY_EXISTS = ("STRATEGY_2001", "策略已存在", "策略ID已被使用，请使用其他ID")
    STRATEGY_CREATE_FAILED = ("STRATEGY_2002", "策略创建失败", "策略创建过程中发生错误")
    STRATEGY_UPDATE_FAILED = ("STRATEGY_2003", "策略更新失败", "策略更新过程中发生错误")
    STRATEGY_DELETE_FAILED = ("STRATEGY_2004", "策略删除失败", "策略删除过程中发生错误")
    STRATEGY_INVALID_STATUS = ("STRATEGY_2005", "策略状态无效", "当前策略状态不允许执行此操作")
    STRATEGY_INITIALIZATION_FAILED = ("STRATEGY_2006", "策略初始化失败", "策略初始化过程中发生错误")
    STRATEGY_VALIDATION_FAILED = ("STRATEGY_2007", "策略验证失败", "策略配置验证未通过")
    
    # 回测错误 (3000-3999)
    BACKTEST_NOT_FOUND = ("STRATEGY_3000", "回测不存在", "指定的回测ID不存在")
    BACKTEST_CREATE_FAILED = ("STRATEGY_3001", "回测创建失败", "回测创建过程中发生错误")
    BACKTEST_EXECUTION_FAILED = ("STRATEGY_3002", "回测执行失败", "回测执行过程中发生错误")
    BACKTEST_DATA_INSUFFICIENT = ("STRATEGY_3003", "回测数据不足", "历史数据不足以完成回测")
    BACKTEST_TIMEOUT = ("STRATEGY_3004", "回测超时", "回测执行时间超过限制")
    BACKTEST_INVALID_CONFIG = ("STRATEGY_3005", "回测配置无效", "回测配置参数不正确")
    
    # 优化错误 (4000-4999)
    OPTIMIZATION_NOT_FOUND = ("STRATEGY_4000", "优化任务不存在", "指定的优化任务ID不存在")
    OPTIMIZATION_CREATE_FAILED = ("STRATEGY_4001", "优化任务创建失败", "优化任务创建过程中发生错误")
    OPTIMIZATION_EXECUTION_FAILED = ("STRATEGY_4002", "优化执行失败", "参数优化过程中发生错误")
    OPTIMIZATION_CONVERGENCE_FAILED = ("STRATEGY_4003", "优化未收敛", "优化算法未能收敛到最优解")
    OPTIMIZATION_INVALID_BOUNDS = ("STRATEGY_4004", "优化边界无效", "参数边界设置不正确")
    
    # 执行错误 (5000-5999)
    EXECUTION_NOT_FOUND = ("STRATEGY_5000", "执行实例不存在", "指定的执行实例ID不存在")
    EXECUTION_CREATE_FAILED = ("STRATEGY_5001", "执行实例创建失败", "执行实例创建过程中发生错误")
    EXECUTION_START_FAILED = ("STRATEGY_5002", "执行启动失败", "策略执行启动过程中发生错误")
    EXECUTION_STOP_FAILED = ("STRATEGY_5003", "执行停止失败", "策略执行停止过程中发生错误")
    EXECUTION_ALREADY_RUNNING = ("STRATEGY_5004", "执行已在运行", "策略执行实例已在运行中")
    EXECUTION_NOT_RUNNING = ("STRATEGY_5005", "执行未运行", "策略执行实例未在运行")
    SIGNAL_GENERATION_FAILED = ("STRATEGY_5006", "信号生成失败", "交易信号生成过程中发生错误")
    
    # 数据错误 (6000-6999)
    DATA_NOT_FOUND = ("STRATEGY_6000", "数据不存在", "请求的数据不存在")
    DATA_LOAD_FAILED = ("STRATEGY_6001", "数据加载失败", "数据加载过程中发生错误")
    DATA_SAVE_FAILED = ("STRATEGY_6002", "数据保存失败", "数据保存过程中发生错误")
    DATA_VALIDATION_FAILED = ("STRATEGY_6003", "数据验证失败", "数据格式或内容验证未通过")
    DATA_CORRUPTED = ("STRATEGY_6004", "数据损坏", "数据文件损坏或格式错误")
    MARKET_DATA_UNAVAILABLE = ("STRATEGY_6005", "市场数据不可用", "无法获取所需的市场数据")
    
    # 配置错误 (7000-7999)
    CONFIG_NOT_FOUND = ("STRATEGY_7000", "配置不存在", "指定的配置不存在")
    CONFIG_INVALID = ("STRATEGY_7001", "配置无效", "配置内容不符合要求")
    CONFIG_LOAD_FAILED = ("STRATEGY_7002", "配置加载失败", "配置文件加载过程中发生错误")
    CONFIG_SAVE_FAILED = ("STRATEGY_7003", "配置保存失败", "配置文件保存过程中发生错误")
    CONFIG_VALIDATION_FAILED = ("STRATEGY_7004", "配置验证失败", "配置验证未通过")
    
    # 系统错误 (8000-8999)
    SYSTEM_ERROR = ("STRATEGY_8000", "系统错误", "系统内部错误")
    DATABASE_ERROR = ("STRATEGY_8001", "数据库错误", "数据库操作过程中发生错误")
    NETWORK_ERROR = ("STRATEGY_8002", "网络错误", "网络通信过程中发生错误")
    RESOURCE_EXHAUSTED = ("STRATEGY_8003", "资源耗尽", "系统资源不足")
    SERVICE_UNAVAILABLE = ("STRATEGY_8004", "服务不可用", "服务暂时不可用，请稍后重试")
    
    def __init__(self, code: str, message: str, suggestion: str):
        self.code = code
        self.message = message
        self.suggestion = suggestion
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典"""
        return {
            "error_code": self.code,
            "error_message": self.message,
            "suggestion": self.suggestion
        }
    
    def format_error(self, detail: str = None) -> str:
        """格式化错误信息"""
        parts = [f"[{self.code}] {self.message}"]
        if detail:
            parts.append(f"详情: {detail}")
        parts.append(f"建议: {self.suggestion}")
        return " | ".join(parts)


class StrategyException(Exception):
    """
    策略层基础异常
    
    所有策略层异常的基类
    """
    
    def __init__(
        self,
        error_code: StrategyErrorCode,
        detail: str = None,
        context: Dict[str, Any] = None,
        original_error: Exception = None
    ):
        self.error_code = error_code
        self.detail = detail
        self.context = context or {}
        self.original_error = original_error
        
        message = error_code.format_error(detail)
        super().__init__(message)
        
        # 记录错误日志
        logger.error(f"策略异常: {message}, 上下文: {self.context}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = self.error_code.to_dict()
        result["detail"] = self.detail
        result["context"] = self.context
        if self.original_error:
            result["original_error"] = str(self.original_error)
        return result


# 便捷异常类

class StrategyNotFoundException(StrategyException):
    """策略不存在异常"""
    def __init__(self, strategy_id: str, context: Dict[str, Any] = None):
        super().__init__(
            StrategyErrorCode.STRATEGY_NOT_FOUND,
            f"策略ID: {strategy_id}",
            context
        )


class StrategyAlreadyExistsException(StrategyException):
    """策略已存在异常"""
    def __init__(self, strategy_id: str, context: Dict[str, Any] = None):
        super().__init__(
            StrategyErrorCode.STRATEGY_ALREADY_EXISTS,
            f"策略ID: {strategy_id}",
            context
        )


class BacktestExecutionException(StrategyException):
    """回测执行异常"""
    def __init__(self, backtest_id: str, detail: str, context: Dict[str, Any] = None):
        super().__init__(
            StrategyErrorCode.BACKTEST_EXECUTION_FAILED,
            f"回测ID: {backtest_id}, {detail}",
            context
        )


class OptimizationException(StrategyException):
    """优化异常"""
    def __init__(self, optimization_id: str, detail: str, context: Dict[str, Any] = None):
        super().__init__(
            StrategyErrorCode.OPTIMIZATION_EXECUTION_FAILED,
            f"优化任务ID: {optimization_id}, {detail}",
            context
        )


class ExecutionException(StrategyException):
    """执行异常"""
    def __init__(self, execution_id: str, detail: str, context: Dict[str, Any] = None):
        super().__init__(
            StrategyErrorCode.EXECUTION_START_FAILED,
            f"执行实例ID: {execution_id}, {detail}",
            context
        )


class DataException(StrategyException):
    """数据异常"""
    def __init__(self, data_id: str, detail: str, context: Dict[str, Any] = None):
        super().__init__(
            StrategyErrorCode.DATA_LOAD_FAILED,
            f"数据ID: {data_id}, {detail}",
            context
        )


class ConfigException(StrategyException):
    """配置异常"""
    def __init__(self, config_name: str, detail: str, context: Dict[str, Any] = None):
        super().__init__(
            StrategyErrorCode.CONFIG_INVALID,
            f"配置名称: {config_name}, {detail}",
            context
        )


# 错误处理工具函数

def handle_error(
    error_code: StrategyErrorCode,
    detail: str = None,
    context: Dict[str, Any] = None,
    log_level: int = logging.ERROR
) -> Dict[str, Any]:
    """
    处理错误并返回标准化错误信息
    
    Args:
        error_code: 错误码
        detail: 错误详情
        context: 错误上下文
        log_level: 日志级别
        
    Returns:
        标准化错误信息字典
    """
    error_info = error_code.to_dict()
    error_info["detail"] = detail
    error_info["context"] = context
    
    # 记录日志
    message = error_code.format_error(detail)
    logger.log(log_level, message)
    
    return error_info


def get_error_by_code(code: str) -> Optional[StrategyErrorCode]:
    """
    根据错误码获取错误类型
    
    Args:
        code: 错误码字符串
        
    Returns:
        错误类型或None
    """
    for error in StrategyErrorCode:
        if error.code == code:
            return error
    return None


def is_retryable_error(error_code: StrategyErrorCode) -> bool:
    """
    判断错误是否可重试
    
    Args:
        error_code: 错误码
        
    Returns:
        是否可重试
    """
    retryable_codes = [
        StrategyErrorCode.TIMEOUT_ERROR,
        StrategyErrorCode.DATABASE_ERROR,
        StrategyErrorCode.NETWORK_ERROR,
        StrategyErrorCode.SERVICE_UNAVAILABLE,
        StrategyErrorCode.OPERATION_FAILED,
    ]
    return error_code in retryable_codes
