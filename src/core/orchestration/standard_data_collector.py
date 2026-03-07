#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准数据采集器

负责使用标准映射采集股票数据，集成统一的AKShare服务，
支持批量采集和增量采集，确保数据映射一致性。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
import pandas as pd

# 导入统一的数据源管理器
from src.core.integration.data_source_manager import get_data_source_manager

# 导入数据质量监控
from src.data.quality.unified_quality_monitor import UnifiedQualityMonitor

# 导入监控系统
from src.core.monitoring.historical_data_monitor import get_historical_data_monitor

logger = logging.getLogger(__name__)


class StandardDataCollector:
    """
    标准数据采集器
    
    使用标准映射采集股票数据，确保与日常增量采集和历史数据采集的一致性
    """
    
    def __init__(self):
        """初始化标准数据采集器"""
        # 集成统一的数据源管理器
        self.data_source_manager = get_data_source_manager()
        # 初始化质量监控器
        self.quality_monitor = UnifiedQualityMonitor()
        # 初始化监控器
        self.historical_monitor = get_historical_data_monitor()
        
        # 配置参数
        self.max_concurrent_tasks = 5
        self.batch_size = 10
        self.retry_attempts = 3
        self.retry_delay = 5
        
        # 标准配置
        self.standard_source_id = "akshare_stock_a"
        self.standard_data_types = ["daily", "weekly", "monthly"]
        
        logger.info("✅ 标准数据采集器初始化完成")
        logger.info(f"📊 数据源管理器状态: 已初始化")
    
    @property
    def is_available(self) -> bool:
        """检查采集器是否可用"""
        return True  # 数据源管理器会自动处理故障转移，所以总是返回可用
    
    async def collect_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        data_type: str = "daily",
        adjust: str = "qfq"
    ) -> Dict[str, Any]:
        """
        采集单个股票数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            data_type: 数据类型 (daily/weekly/monthly)
            adjust: 复权方式 (qfq/hfq)
            
        Returns:
            采集结果字典
        """
        if not self.is_available:
            logger.error("❌ AKShare服务不可用，无法采集数据")
            return {
                "success": False,
                "error": "AKShare服务不可用",
                "symbol": symbol
            }
        
        if data_type not in self.standard_data_types:
            logger.error(f"❌ 不支持的数据类型: {data_type}")
            return {
                "success": False,
                "error": f"不支持的数据类型: {data_type}",
                "symbol": symbol
            }
        
        logger.info(f"📈 开始采集股票数据: {symbol}")
        logger.info(f"日期范围: {start_date} ~ {end_date}")
        logger.info(f"数据类型: {data_type}, 复权方式: {adjust}")
        
        try:
            # 使用数据源管理器获取数据
            df = await self.data_source_manager.get_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
                data_type=data_type
            )
            
            if df is None or df.empty:
                logger.warning(f"⚠️  未获取到数据: {symbol}")
                return {
                    "success": False,
                    "error": "未获取到数据",
                    "symbol": symbol
                }
            
            logger.info(f"✅ 数据获取成功: {symbol}, {len(df)} 条记录")
            
            # 数据质量验证
            validation_result = self._validate_data_quality(df, symbol, data_type)
            
            # 转换为标准格式
            standard_records = self._convert_to_standard_records(df, symbol, data_type)
            
            # 标准映射验证（集成自动化验证）
            standard_validation_result = self._validate_standard_mapping(standard_records)
            
            # 综合验证结果
            combined_validation = {
                "basic_validation": validation_result,
                "standard_mapping_validation": standard_validation_result,
                "overall_quality_score": min(
                    validation_result.get("quality_score", 0),
                    standard_validation_result.get("quality_score", 0)
                )
            }
            
            return {
                "success": True,
                "symbol": symbol,
                "data_type": data_type,
                "records_count": len(standard_records),
                "validation": combined_validation,
                "data": standard_records,
                "source_id": self.standard_source_id
            }
            
        except Exception as e:
            logger.error(f"❌ 采集数据失败: {symbol}, 错误: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
            }
    
    async def batch_collect_stock_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        data_type: str = "daily",
        adjust: str = "qfq"
    ) -> List[Dict[str, Any]]:
        """
        批量采集股票数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            data_type: 数据类型 (daily/weekly/monthly)
            adjust: 复权方式 (qfq/hfq)
            
        Returns:
            采集结果列表
        """
        if not symbols:
            return []
        
        logger.info(f"📊 开始批量采集: {len(symbols)} 个股票")
        logger.info(f"日期范围: {start_date} ~ {end_date}")
        
        # 分批处理
        results = []
        for i in range(0, len(symbols), self.batch_size):
            batch_symbols = symbols[i:i + self.batch_size]
            logger.info(f"🔄 处理批次: {i//self.batch_size + 1}, 股票数: {len(batch_symbols)}")
            
            # 并发处理批次
            tasks = []
            for symbol in batch_symbols:
                task = self.collect_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    data_type=data_type,
                    adjust=adjust
                )
                tasks.append(task)
            
            # 限制并发
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            # 批次间延迟
            if i + self.batch_size < len(symbols):
                await asyncio.sleep(2)
        
        # 统计结果
        success_count = sum(1 for r in results if r.get("success"))
        fail_count = len(results) - success_count
        
        logger.info(f"🎉 批量采集完成")
        logger.info(f"✅ 成功: {success_count}, ❌ 失败: {fail_count}")
        
        return results
    
    async def incremental_collect(
        self,
        symbols: List[str],
        days: int = 7,
        data_type: str = "daily",
        adjust: str = "qfq"
    ) -> List[Dict[str, Any]]:
        """
        增量采集股票数据
        
        Args:
            symbols: 股票代码列表
            days: 采集天数
            data_type: 数据类型 (daily/weekly/monthly)
            adjust: 复权方式 (qfq/hfq)
            
        Returns:
            采集结果列表
        """
        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")
        
        logger.info(f"🔄 开始增量采集: {len(symbols)} 个股票")
        logger.info(f"增量范围: 最近 {days} 天")
        logger.info(f"日期范围: {start_date_str} ~ {end_date_str}")
        
        # 调用批量采集
        return await self.batch_collect_stock_data(
            symbols=symbols,
            start_date=start_date_str,
            end_date=end_date_str,
            data_type=data_type,
            adjust=adjust
        )
    
    def _validate_data_quality(
        self,
        df: pd.DataFrame,
        symbol: str,
        data_type: str
    ) -> Dict[str, Any]:
        """
        验证数据质量
        
        Args:
            df: 股票数据
            symbol: 股票代码
            data_type: 数据类型
            
        Returns:
            验证结果
        """
        logger.info(f"🔍 验证数据质量: {symbol}")
        
        try:
            # 使用统一质量监控器验证
            validation_result = self.quality_monitor.check_quality(df, "STOCK")
            
            # 记录质量监控
            quality_score = validation_result.get("overall_score", 0)
            logger.info(f"📊 数据质量得分: {quality_score:.2f}")
            
            # 检查关键字段
            required_fields = ["日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额"]
            missing_fields = [field for field in required_fields if field not in df.columns]
            
            if missing_fields:
                logger.warning(f"⚠️  缺少关键字段: {missing_fields}")
                validation_result["missing_fields"] = missing_fields
            else:
                logger.info("✅ 所有关键字段完整")
            
            return {
                "quality_score": quality_score,
                "missing_fields": missing_fields,
                "validation_result": validation_result
            }
            
        except Exception as e:
            logger.error(f"❌ 数据质量验证失败: {e}")
            return {
                "quality_score": 0,
                "error": str(e),
                "missing_fields": []
            }
    
    def _convert_to_standard_records(
        self,
        df: pd.DataFrame,
        symbol: str,
        data_type: str
    ) -> List[Dict[str, Any]]:
        """
        转换为标准格式记录
        
        Args:
            df: 股票数据
            symbol: 股票代码
            data_type: 数据类型
            
        Returns:
            标准格式记录列表
        """
        records = []
        
        for _, row in df.iterrows():
            record = {
                "source_id": self.standard_source_id,
                "symbol": symbol,
                "date": "",
                "data_type": data_type,
                "open_price": None,
                "high_price": None,
                "low_price": None,
                "close_price": None,
                "volume": None,
                "amount": None,
                "pct_change": None,
                "change": None,
                "turnover_rate": None,
                "amplitude": None,
                "data_source": "akshare",
                "collected_at": datetime.now().isoformat()
            }
            
            # 填充关键字段
            if "日期" in row:
                record["date"] = str(row["日期"])
            elif "date" in row:
                record["date"] = str(row["date"])
            
            if "开盘" in row:
                record["open_price"] = float(row["开盘"])
            elif "open" in row:
                record["open_price"] = float(row["open"])
            
            if "最高" in row:
                record["high_price"] = float(row["最高"])
            elif "high" in row:
                record["high_price"] = float(row["high"])
            
            if "最低" in row:
                record["low_price"] = float(row["最低"])
            elif "low" in row:
                record["low_price"] = float(row["low"])
            
            if "收盘" in row:
                record["close_price"] = float(row["收盘"])
            elif "close" in row:
                record["close_price"] = float(row["close"])
            
            if "成交量" in row:
                record["volume"] = int(row["成交量"])
            elif "volume" in row:
                record["volume"] = int(row["volume"])
            
            if "成交额" in row:
                record["amount"] = float(row["成交额"])
            elif "amount" in row:
                record["amount"] = float(row["amount"])
            
            if "换手率" in row:
                record["turnover_rate"] = float(row["换手率"])
            
            # 计算涨跌幅和涨跌额
            if record["open_price"] and record["close_price"]:
                try:
                    record["change"] = record["close_price"] - record["open_price"]
                    if record["open_price"] > 0:
                        record["pct_change"] = (record["change"] / record["open_price"]) * 100
                except:
                    pass
            
            # 计算振幅
            if record["high_price"] and record["low_price"] and record["open_price"]:
                try:
                    record["amplitude"] = ((record["high_price"] - record["low_price"]) / record["open_price"]) * 100
                except:
                    pass
            
            records.append(record)
        
        logger.info(f"🔄 转换完成: {len(records)} 条标准格式记录")
        return records
    
    async def collect_market_data(self) -> Dict[str, Any]:
        """
        采集市场数据
        
        Returns:
            采集结果字典
        """
        if not self.is_available:
            logger.error("❌ AKShare服务不可用，无法采集市场数据")
            return {
                "success": False,
                "error": "AKShare服务不可用"
            }
        
        logger.info("🌐 开始采集市场数据")
        
        try:
            # 使用数据源管理器获取市场数据
            df = await self.data_source_manager.get_market_data()
            
            if df is None or df.empty:
                logger.warning("⚠️  未获取到市场数据")
                return {
                    "success": False,
                    "error": "未获取到市场数据"
                }
            
            logger.info(f"✅ 市场数据获取成功: {len(df)} 条记录")
            
            # 数据质量验证
            validation_result = self._validate_data_quality(df, "market", "daily")
            
            return {
                "success": True,
                "records_count": len(df),
                "validation": validation_result,
                "source_id": self.standard_source_id
            }
            
        except Exception as e:
            logger.error(f"❌ 采集市场数据失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_standard_config(self) -> Dict[str, Any]:
        """
        获取标准配置
        
        Returns:
            标准配置字典
        """
        return {
            "standard_source_id": self.standard_source_id,
            "standard_data_types": self.standard_data_types,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "batch_size": self.batch_size,
            "retry_attempts": self.retry_attempts,
            "data_source_manager_available": self.is_available
        }
    
    def _validate_standard_mapping(self, standard_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        验证标准映射数据
        
        Args:
            standard_records: 标准格式记录列表
            
        Returns:
            验证结果字典
        """
        logger.info(f"🔍 验证标准映射数据: {len(standard_records)} 条记录")
        
        try:
            # 使用UnifiedQualityMonitor进行标准映射验证
            validation_result = self.quality_monitor.validator.validate(
                data=standard_records,
                data_type="standard_mapping"
            )
            
            # 分析验证结果
            issues = validation_result.get("issues", [])
            critical_issues = [i for i in issues if i.severity == "critical"]
            high_issues = [i for i in issues if i.severity == "high"]
            medium_issues = [i for i in issues if i.severity == "medium"]
            low_issues = [i for i in issues if i.severity == "low"]
            
            # 计算质量分数
            total_issues = len(issues)
            if total_issues == 0:
                quality_score = 1.0
            else:
                # 基于问题严重程度计算质量分数
                severity_weights = {
                    "critical": 0.0,
                    "high": 0.3,
                    "medium": 0.6,
                    "low": 0.8
                }
                
                weighted_score = sum(
                    severity_weights.get(issue.severity, 0.5) for issue in issues
                ) / total_issues if total_issues > 0 else 1.0
                quality_score = max(0, min(1, weighted_score))
            
            logger.info(f"📊 标准映射验证结果:")
            logger.info(f"✅ 验证通过: {validation_result.get('valid', False)}")
            logger.info(f"📋 质量分数: {quality_score:.2f}")
            logger.info(f"⚠️  问题统计: 严重={len(critical_issues)}, 高={len(high_issues)}, 中={len(medium_issues)}, 低={len(low_issues)}")
            
            # 生成验证报告
            validation_report = {
                "quality_score": quality_score,
                "valid": validation_result.get("valid", False),
                "total_issues": total_issues,
                "issue_details": {
                    "critical": len(critical_issues),
                    "high": len(high_issues),
                    "medium": len(medium_issues),
                    "low": len(low_issues)
                },
                "issues": [
                    {
                        "type": issue.issue_type,
                        "severity": issue.severity,
                        "description": issue.description,
                        "confidence": issue.confidence
                    }
                    for issue in issues
                ]
            }
            
            # 记录验证结果
            if critical_issues:
                logger.warning(f"🚨 发现严重问题: {len(critical_issues)} 个")
            elif high_issues:
                logger.warning(f"⚠️  发现高优先级问题: {len(high_issues)} 个")
            else:
                logger.info("✅ 标准映射验证通过")
            
            return validation_report
            
        except Exception as e:
            logger.error(f"❌ 标准映射验证失败: {e}")
            return {
                "quality_score": 0,
                "valid": False,
                "error": str(e),
                "total_issues": 0,
                "issue_details": {}
            }


# 全局标准数据采集器实例
_standard_collector_instance = None


def get_standard_data_collector() -> StandardDataCollector:
    """
    获取全局标准数据采集器实例
    
    Returns:
        StandardDataCollector实例
    """
    global _standard_collector_instance
    if _standard_collector_instance is None:
        _standard_collector_instance = StandardDataCollector()
    return _standard_collector_instance


def reset_standard_data_collector():
    """
    重置标准数据采集器实例
    """
    global _standard_collector_instance
    _standard_collector_instance = None
    logger.info("🔄 标准数据采集器实例已重置")
