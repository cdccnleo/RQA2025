#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一数据源管理器

负责管理多个数据源（AKShare、BaoStock等），实现智能切换、
优先级管理、负载均衡和故障检测，提高数据获取的可靠性和稳定性。
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import time

# 导入数据源服务
from .akshare_service import get_akshare_service
from .baostock_service import get_baostock_service

# 延迟导入配置管理系统，避免循环导入
config_manager = None

def get_config_manager():
    """
    延迟获取配置管理器实例，避免循环导入
    
    Returns:
        配置管理器实例
    """
    global config_manager
    if config_manager is None:
        from src.gateway.web.data_source_config_manager import get_data_source_config_manager
        config_manager = get_data_source_config_manager()
    return config_manager

logger = logging.getLogger(__name__)


class DataSourceType:
    """数据源类型枚举"""
    AK_SHARE = "akshare"
    BAO_STOCK = "baostock"


class DataSourceStatus:
    """数据源状态枚举"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"


class DataSourceManager:
    """统一数据源管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据源管理器
        
        Args:
            config: 配置字典
        """
        # 从配置管理系统加载配置
        self.config = self._load_config_from_manager()
        
        # 如果提供了配置，合并配置
        if config:
            self._merge_config(self.config, config)
        
        # 初始化数据源状态
        self.data_source_status = {}
        for source_name in self.config["data_sources"]:
            self.data_source_status[source_name] = {
                "status": DataSourceStatus.AVAILABLE,
                "last_health_check": 0,
                "failure_count": 0,
                "last_failure_time": 0,
                "recovery_time": 0,
                "response_times": [],
                "success_rate": 1.0
            }
        
        # 初始化数据源服务
        self.data_services = {}
        for source_name in self.config["data_sources"]:
            self.data_services[source_name] = None
        
        # 缓存
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 切换状态
        self.last_switch_time = 0
        self.active_failures = {}
        
        # 初始化服务
        self._init_services()
        
        logger.info("✅ 数据源管理器初始化完成")
    
    def _load_config_from_manager(self) -> Dict[str, Any]:
        """
        从配置管理系统加载配置
        
        Returns:
            Dict[str, Any]: 加载的配置
        """
        try:
            # 获取配置管理器实例（延迟导入）
            config_manager = get_config_manager()
            
            # 获取所有数据源配置
            data_sources = config_manager.get_data_sources()
            
            # 构建数据源配置
            data_source_config = {}
            for source in data_sources:
                source_id = source.get('id')
                if source_id:
                    # 映射配置管理系统的配置到数据源管理器的配置
                    if 'akshare' in source_id.lower():
                        data_source_config['akshare'] = {
                            "priority": 1,
                            "enabled": source.get('enabled', True),
                            "timeout": 30,
                            "max_retries": 3,
                            "weight": 0.7,
                            "health_check_interval": 60
                        }
                    elif 'baostock' in source_id.lower():
                        data_source_config['baostock'] = {
                            "priority": 2,
                            "enabled": source.get('enabled', True),
                            "timeout": 60,
                            "max_retries": 3,
                            "weight": 0.3,
                            "health_check_interval": 120
                        }
            
            # 构建完整配置
            config = {
                "data_sources": data_source_config,
                "switching_strategy": {
                    "enable_auto_switch": True,
                    "failure_threshold": 3,
                    "recovery_wait_time": 300,
                    "min_switch_interval": 60
                },
                "load_balancing": {
                    "enabled": True,
                    "method": "weighted_round_robin"
                },
                "caching": {
                    "enabled": True,
                    "ttl_seconds": 300,
                    "max_size": 10000
                }
            }
            
            logger.info(f"✅ 从配置管理系统加载数据源配置成功: {list(data_source_config.keys())}")
            return config
        except Exception as e:
            logger.error(f"❌ 从配置管理系统加载配置失败: {e}")
            # 不再使用默认配置作为后备，而是返回空配置
            # 这样可以确保只有在配置管理系统中明确配置的数据源才会被使用
            return {
                "data_sources": {},
                "switching_strategy": {
                    "enable_auto_switch": True,
                    "failure_threshold": 3,
                    "recovery_wait_time": 300,
                    "min_switch_interval": 60
                },
                "load_balancing": {
                    "enabled": True,
                    "method": "weighted_round_robin"
                },
                "caching": {
                    "enabled": True,
                    "ttl_seconds": 300,
                    "max_size": 10000
                }
            }
    
    def _merge_config(self, target: Dict[str, Any], source: Dict[str, Any]):
        """递归合并配置"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config(target[key], value)
            else:
                target[key] = value
    
    def _init_services(self):
        """初始化数据源服务"""
        # 初始化AKShare服务
        if "akshare" in self.config["data_sources"]:
            try:
                self.data_services["akshare"] = get_akshare_service()
                logger.info("✅ AKShare服务初始化成功")
            except Exception as e:
                logger.error(f"❌ AKShare服务初始化失败: {e}")
                if "akshare" in self.data_source_status:
                    self.data_source_status["akshare"]["status"] = DataSourceStatus.UNAVAILABLE
        
        # 初始化BaoStock服务
        if "baostock" in self.config["data_sources"]:
            try:
                self.data_services["baostock"] = get_baostock_service()
                logger.info("✅ BaoStock服务初始化成功")
            except Exception as e:
                logger.error(f"❌ BaoStock服务初始化失败: {e}")
                if "baostock" in self.data_source_status:
                    self.data_source_status["baostock"]["status"] = DataSourceStatus.UNAVAILABLE
    
    async def get_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        data_type: str = "daily",
        adjust: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        """
        获取股票数据，智能选择数据源
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            data_type: 数据类型
            adjust: 复权方式
            
        Returns:
            股票数据DataFrame
        """
        # 生成缓存键
        cache_key = f"stock_data:{symbol}:{start_date}:{end_date}:{data_type}:{adjust}"
        
        # 检查缓存
        if self.config["caching"]["enabled"]:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"✅ 缓存命中: {cache_key}")
                self.cache_hits += 1
                return cached_data
            else:
                self.cache_misses += 1
        
        # 选择数据源
        selected_source = await self._select_data_source("stock_data")
        if not selected_source:
            logger.error("❌ 无可用数据源")
            return None
        
        logger.info(f"📊 选择数据源: {selected_source} 获取股票数据: {symbol}")
        
        # 尝试从选中的数据源获取数据
        result = await self._get_data_from_source(
            selected_source,
            "stock_data",
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )
        
        # 如果失败，尝试其他数据源
        if result is None and self.config["switching_strategy"]["enable_auto_switch"]:
            logger.warning(f"⚠️ 数据源 {selected_source} 获取数据失败，尝试其他数据源")
            
            # 标记失败
            self._record_failure(selected_source)
            
            # 选择备用数据源
            backup_source = await self._select_data_source("stock_data", exclude=[selected_source])
            if backup_source:
                logger.info(f"🔄 切换到备用数据源: {backup_source}")
                result = await self._get_data_from_source(
                    backup_source,
                    "stock_data",
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust
                )
        
        # 缓存结果
        if result is not None and self.config["caching"]["enabled"]:
            self._add_to_cache(cache_key, result)
        
        return result
    
    async def get_market_data(self) -> Optional[pd.DataFrame]:
        """
        获取市场数据
        
        Returns:
            市场数据DataFrame
        """
        # 生成缓存键
        cache_key = f"market_data:{datetime.now().strftime('%Y%m%d')}"
        
        # 检查缓存
        if self.config["caching"]["enabled"]:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"✅ 缓存命中: {cache_key}")
                self.cache_hits += 1
                return cached_data
            else:
                self.cache_misses += 1
        
        # 选择数据源
        selected_source = await self._select_data_source("market_data")
        if not selected_source:
            logger.error("❌ 无可用数据源")
            return None
        
        logger.info(f"📊 选择数据源: {selected_source} 获取市场数据")
        
        # 尝试获取数据
        result = await self._get_data_from_source(selected_source, "market_data")
        
        # 如果失败，尝试其他数据源
        if result is None and self.config["switching_strategy"]["enable_auto_switch"]:
            logger.warning(f"⚠️ 数据源 {selected_source} 获取市场数据失败，尝试其他数据源")
            
            # 标记失败
            self._record_failure(selected_source)
            
            # 选择备用数据源
            backup_source = await self._select_data_source("market_data", exclude=[selected_source])
            if backup_source:
                logger.info(f"🔄 切换到备用数据源: {backup_source}")
                result = await self._get_data_from_source(backup_source, "market_data")
        
        # 缓存结果
        if result is not None and self.config["caching"]["enabled"]:
            self._add_to_cache(cache_key, result)
        
        return result
    
    async def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取股票基本信息
        
        Args:
            symbol: 股票代码
            
        Returns:
            股票基本信息字典
        """
        # 生成缓存键
        cache_key = f"stock_info:{symbol}"
        
        # 检查缓存
        if self.config["caching"]["enabled"]:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"✅ 缓存命中: {cache_key}")
                self.cache_hits += 1
                return cached_data
            else:
                self.cache_misses += 1
        
        # 选择数据源
        selected_source = await self._select_data_source("stock_info")
        if not selected_source:
            logger.error("❌ 无可用数据源")
            return None
        
        logger.info(f"📋 选择数据源: {selected_source} 获取股票信息: {symbol}")
        
        # 尝试获取数据
        result = await self._get_data_from_source(selected_source, "stock_info", symbol=symbol)
        
        # 如果失败，尝试其他数据源
        if result is None and self.config["switching_strategy"]["enable_auto_switch"]:
            logger.warning(f"⚠️ 数据源 {selected_source} 获取股票信息失败，尝试其他数据源")
            
            # 标记失败
            self._record_failure(selected_source)
            
            # 选择备用数据源
            backup_source = await self._select_data_source("stock_info", exclude=[selected_source])
            if backup_source:
                logger.info(f"🔄 切换到备用数据源: {backup_source}")
                result = await self._get_data_from_source(backup_source, "stock_info", symbol=symbol)
        
        # 缓存结果
        if result is not None and self.config["caching"]["enabled"]:
            self._add_to_cache(cache_key, result)
        
        return result
    
    async def _select_data_source(self, data_type: str, exclude: List[str] = None) -> Optional[str]:
        """
        选择数据源
        
        Args:
            data_type: 数据类型
            exclude: 排除的数据源列表
            
        Returns:
            选中的数据源名称
        """
        # 过滤可用数据源
        available_sources = []
        current_time = time.time()
        
        for source_name, config in self.config["data_sources"].items():
            # 检查是否排除
            if exclude and source_name in exclude:
                continue
            
            # 检查是否启用
            if not config["enabled"]:
                continue
            
            # 检查状态
            status = self.data_source_status[source_name]["status"]
            if status == DataSourceStatus.UNAVAILABLE:
                # 检查是否可以恢复
                recovery_time = self.data_source_status[source_name]["recovery_time"]
                if current_time >= recovery_time:
                    # 尝试恢复
                    if await self._check_data_source_health(source_name):
                        logger.info(f"✅ 数据源 {source_name} 恢复可用")
                        self.data_source_status[source_name]["status"] = DataSourceStatus.AVAILABLE
                        self.data_source_status[source_name]["failure_count"] = 0
                    else:
                        # 更新恢复时间
                        self.data_source_status[source_name]["recovery_time"] = current_time + self.config["switching_strategy"]["recovery_wait_time"]
                        continue
            elif status == DataSourceStatus.DEGRADED:
                # 降级状态，仍然可用但优先级降低
                pass
            
            # 执行健康检查
            last_check = self.data_source_status[source_name]["last_health_check"]
            check_interval = config["health_check_interval"]
            if current_time - last_check > check_interval:
                await self._check_data_source_health(source_name)
            
            # 添加到可用列表
            available_sources.append((source_name, config["priority"]))
        
        if not available_sources:
            return None
        
        # 按优先级排序
        available_sources.sort(key=lambda x: x[1])
        
        # 应用负载均衡
        if self.config["load_balancing"]["enabled"]:
            return await self._apply_load_balancing([s[0] for s in available_sources])
        else:
            # 返回优先级最高的数据源
            return available_sources[0][0]
    
    async def _apply_load_balancing(self, available_sources: List[str]) -> str:
        """
        应用负载均衡策略
        
        Args:
            available_sources: 可用数据源列表
            
        Returns:
            选中的数据源
        """
        method = self.config["load_balancing"]["method"]
        
        if method == "weighted_round_robin":
            # 加权轮询
            weights = []
            for source in available_sources:
                weight = self.config["data_sources"][source]["weight"]
                # 考虑健康状态调整权重
                if self.data_source_status[source]["status"] == DataSourceStatus.DEGRADED:
                    weight *= 0.5
                weights.append((source, weight))
            
            # 简单加权随机选择
            total_weight = sum(w[1] for w in weights)
            import random
            r = random.uniform(0, total_weight)
            cumulative = 0
            for source, weight in weights:
                cumulative += weight
                if r <= cumulative:
                    return source
        
        # 默认返回第一个
        return available_sources[0]
    
    async def _get_data_from_source(
        self,
        source_name: str,
        data_type: str,
        **kwargs
    ) -> Optional[Any]:
        """
        从指定数据源获取数据
        
        Args:
            source_name: 数据源名称
            data_type: 数据类型
            **kwargs: 其他参数
            
        Returns:
            获取的数据
        """
        start_time = time.time()
        
        try:
            service = self.data_services[source_name]
            if not service:
                logger.error(f"❌ 数据源服务未初始化: {source_name}")
                return None
            
            # 根据数据类型调用对应方法
            if data_type == "stock_data":
                # 转换日期格式
                start_date = kwargs.get("start_date")
                end_date = kwargs.get("end_date")
                
                # 调整BaoStock的日期格式
                if source_name == "baostock":
                    # BaoStock需要YYYY-MM-DD格式
                    if len(start_date) == 8:
                        start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
                    if len(end_date) == 8:
                        end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
                    
                    # BaoStock复权参数: 1=不复权, 2=前复权(qfq), 3=后复权(hfq)
                    adjust_mapping = {"qfq": "2", "hfq": "3", "": "1"}
                    adjustflag = adjust_mapping.get(kwargs.get("adjust", "qfq"), "2")
                    
                    result = await service.get_stock_data(
                        symbol=kwargs.get("symbol"),
                        start_date=start_date,
                        end_date=end_date,
                        frequency="d",  # 日线
                        adjustflag=adjustflag
                    )
                else:  # akshare
                    result = await service.get_stock_data(
                        symbol=kwargs.get("symbol"),
                        start_date=kwargs.get("start_date"),
                        end_date=kwargs.get("end_date"),
                        adjust=kwargs.get("adjust"),
                        data_type="daily"  # 默认为日线数据
                    )
                    
            elif data_type == "market_data":
                result = await service.get_market_data()
                
            elif data_type == "stock_info":
                result = await service.get_stock_info(kwargs.get("symbol"))
                
            else:
                logger.error(f"❌ 不支持的数据类型: {data_type}")
                return None
            
            # 记录响应时间
            response_time = time.time() - start_time
            self.data_source_status[source_name]["response_times"].append(response_time)
            # 只保留最近10个响应时间
            if len(self.data_source_status[source_name]["response_times"]) > 10:
                self.data_source_status[source_name]["response_times"] = self.data_source_status[source_name]["response_times"][-10:]
            
            # 更新成功状态
            if result is not None:
                logger.info(f"✅ 数据源 {source_name} 获取 {data_type} 成功")
                self._record_success(source_name)
                
                # 对股票数据进行标准化处理
                if data_type == "stock_data" and isinstance(result, pd.DataFrame) and not result.empty:
                    logger.info(f"🔄 应用统一标准化处理到 {source_name} 数据源")
                    normalized_result = self._normalize_stock_data(result, source_name)
                    return normalized_result
                
                return result
            else:
                logger.warning(f"⚠️ 数据源 {source_name} 获取 {data_type} 返回空数据")
                self._record_failure(source_name)
                return None
                
        except Exception as e:
            logger.error(f"❌ 数据源 {source_name} 获取 {data_type} 异常: {e}")
            self._record_failure(source_name)
            return None
    
    async def _check_data_source_health(self, source_name: str) -> bool:
        """
        检查数据源健康状态
        
        Args:
            source_name: 数据源名称
            
        Returns:
            是否健康
        """
        logger.info(f"🔍 检查数据源健康状态: {source_name}")
        
        try:
            service = self.data_services[source_name]
            if not service:
                return False
            
            # 简单健康检查
            if source_name == "akshare":
                # 检查是否可用
                if hasattr(service, "is_available") and not service.is_available:
                    return False
                
                # 尝试获取简单数据
                try:
                    # 使用一个常用的股票代码进行测试
                    test_symbol = "000001"
                    test_start = (datetime.now() - pd.Timedelta(days=7)).strftime("%Y%m%d")
                    test_end = datetime.now().strftime("%Y%m%d")
                    
                    result = await service.get_stock_data(
                        symbol=test_symbol,
                        start_date=test_start,
                        end_date=test_end,
                        adjust="qfq",
                        data_type="daily"
                    )
                    
                    return result is not None
                except:
                    return False
                    
            elif source_name == "baostock":
                # 检查是否可用
                if hasattr(service, "is_available") and not service.is_available:
                    return False
                
                # 尝试连接
                try:
                    if hasattr(service, "_ensure_connected"):
                        return await service._ensure_connected()
                    else:
                        return False
                except:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"❌ 检查数据源健康状态异常: {e}")
            return False
        finally:
            # 更新健康检查时间
            self.data_source_status[source_name]["last_health_check"] = time.time()
    
    def _record_success(self, source_name: str):
        """
        记录成功
        
        Args:
            source_name: 数据源名称
        """
        self.data_source_status[source_name]["failure_count"] = max(0, self.data_source_status[source_name]["failure_count"] - 1)
        
        # 更新成功率
        success_rate = self.data_source_status[source_name]["success_rate"]
        self.data_source_status[source_name]["success_rate"] = success_rate * 0.9 + 0.1
    
    def _record_failure(self, source_name: str):
        """
        记录失败
        
        Args:
            source_name: 数据源名称
        """
        current_time = time.time()
        self.data_source_status[source_name]["failure_count"] += 1
        self.data_source_status[source_name]["last_failure_time"] = current_time
        
        # 更新成功率
        success_rate = self.data_source_status[source_name]["success_rate"]
        self.data_source_status[source_name]["success_rate"] = success_rate * 0.9
        
        # 检查是否达到失败阈值
        failure_threshold = self.config["switching_strategy"]["failure_threshold"]
        if self.data_source_status[source_name]["failure_count"] >= failure_threshold:
            logger.warning(f"⚠️ 数据源 {source_name} 达到失败阈值，标记为不可用")
            self.data_source_status[source_name]["status"] = DataSourceStatus.UNAVAILABLE
            # 设置恢复时间
            self.data_source_status[source_name]["recovery_time"] = current_time + self.config["switching_strategy"]["recovery_wait_time"]
        else:
            # 标记为降级
            self.data_source_status[source_name]["status"] = DataSourceStatus.DEGRADED
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """
        从缓存获取数据
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的数据
        """
        if key not in self.cache:
            return None
        
        cached_item = self.cache[key]
        current_time = time.time()
        
        # 检查是否过期
        if current_time > cached_item["expire_time"]:
            del self.cache[key]
            return None
        
        return cached_item["data"]
    
    def _add_to_cache(self, key: str, data: Any):
        """
        添加数据到缓存
        
        Args:
            key: 缓存键
            data: 要缓存的数据
        """
        # 检查缓存大小
        if len(self.cache) >= self.config["caching"]["max_size"]:
            # 删除最旧的缓存
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        
        # 添加到缓存
        expire_time = time.time() + self.config["caching"]["ttl_seconds"]
        self.cache[key] = {
            "data": data,
            "timestamp": time.time(),
            "expire_time": expire_time
        }
    
    def _normalize_stock_data(self, df: pd.DataFrame, source_name: str) -> List[Dict[str, Any]]:
        """
        统一处理股票数据，计算标准字段
        
        Args:
            df: 股票数据DataFrame
            source_name: 数据源名称
            
        Returns:
            标准化后的字典列表
        """
        normalized_data = []
        
        for _, row in df.iterrows():
            record = {
                'symbol': '',
                'date': '',
                'open': 0,
                'high': 0,
                'low': 0,
                'close': 0,
                'volume': 0,
                'amount': 0,
                'data_source': source_name,
                'timestamp': time.time()
            }
            
            # 填充字段 - 兼容不同数据源的字段名
            # 日期字段
            if '日期' in row:
                record['date'] = row['日期']
            elif 'date' in row:
                record['date'] = row['date']
            
            # 价格字段
            if '开盘' in row:
                record['open'] = row['开盘']
            elif 'open' in row:
                record['open'] = row['open']
            
            if '最高' in row:
                record['high'] = row['最高']
            elif 'high' in row:
                record['high'] = row['high']
            
            if '最低' in row:
                record['low'] = row['最低']
            elif 'low' in row:
                record['low'] = row['low']
            
            if '收盘' in row:
                record['close'] = row['收盘']
            elif 'close' in row:
                record['close'] = row['close']
            
            # 成交量和成交额
            if '成交量' in row:
                record['volume'] = row['成交量']
            elif 'volume' in row:
                record['volume'] = row['volume']
            
            if '成交额' in row:
                record['amount'] = row['成交额']
            elif 'amount' in row:
                record['amount'] = row['amount']
            
            # 统一计算字段
            # 涨跌额 (close - open)
            if record['open'] > 0 and record['close'] > 0:
                record['change'] = record['close'] - record['open']
            else:
                record['change'] = 0
            
            # 振幅 ((high - low) / open * 100)
            if record['open'] > 0 and record['high'] > 0 and record['low'] > 0:
                record['amplitude'] = (record['high'] - record['low']) / record['open'] * 100
            else:
                record['amplitude'] = 0
            
            # 涨跌幅 ((close - open) / open * 100)
            if record['open'] > 0 and record['close'] > 0:
                record['pct_change'] = (record['close'] - record['open']) / record['open'] * 100
            else:
                record['pct_change'] = 0
            
            # 换手率字段
            if '换手率' in row:
                record['turnover_rate'] = row['换手率']
            elif 'turnover_rate' in row:
                record['turnover_rate'] = row['turnover_rate']
            elif 'turn' in row:
                record['turnover_rate'] = row['turn']
            
            normalized_data.append(record)
        
        logger.info(f"✅ 标准化处理完成: {len(normalized_data)} 条记录, 数据源: {source_name}")
        return normalized_data
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息
        """
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "size": len(self.cache),
            "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }
    
    def get_data_source_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        获取数据源统计信息
        
        Returns:
            数据源统计信息
        """
        stats = {}
        for source_name, status in self.data_source_status.items():
            stats[source_name] = {
                "status": status["status"],
                "failure_count": status["failure_count"],
                "success_rate": status["success_rate"],
                "response_times": status["response_times"],
                "avg_response_time": sum(status["response_times"]) / len(status["response_times"]) if status["response_times"] else 0
            }
        return stats
    
    def reset(self):
        """
        重置数据源管理器
        """
        # 重置状态
        for source_name in self.data_source_status:
            self.data_source_status[source_name] = {
                "status": DataSourceStatus.AVAILABLE,
                "last_health_check": 0,
                "failure_count": 0,
                "last_failure_time": 0,
                "recovery_time": 0,
                "response_times": [],
                "success_rate": 1.0
            }
        
        # 清空缓存
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 重置服务
        self._init_services()
        
        logger.info("🔄 数据源管理器已重置")


# 全局数据源管理器实例
_data_source_manager_instance = None


def get_data_source_manager(config: Optional[Dict[str, Any]] = None) -> DataSourceManager:
    """
    获取全局数据源管理器实例
    
    Args:
        config: 配置参数
        
    Returns:
        DataSourceManager实例
    """
    global _data_source_manager_instance
    if _data_source_manager_instance is None:
        _data_source_manager_instance = DataSourceManager(config)
    elif config:
        # 更新配置
        _merge_config(_data_source_manager_instance.config, config)
    return _data_source_manager_instance


def _merge_config(target: Dict[str, Any], source: Dict[str, Any]):
    """
    递归合并配置
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _merge_config(target[key], value)
        else:
            target[key] = value


def reset_data_source_manager():
    """
    重置数据源管理器实例
    """
    global _data_source_manager_instance
    if _data_source_manager_instance:
        _data_source_manager_instance.reset()
    _data_source_manager_instance = None
    logger.info("🔄 数据源管理器实例已重置")