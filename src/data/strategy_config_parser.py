#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略配置解析器

功能：
- 解析策略配置文件（YAML/JSON）
- 提取股票代码列表
- 支持动态配置更新

作者: AI Assistant
创建日期: 2026-02-21
"""

import logging
import json
import yaml
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import os

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """策略配置数据类"""
    strategy_id: str
    strategy_name: str
    symbols: List[str] = field(default_factory=list)
    description: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=lambda: ["akshare"])
    time_range: Dict[str, str] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    enabled: bool = True


class StrategyConfigParser:
    """
    策略配置解析器
    
    职责：
    1. 解析策略配置文件（YAML/JSON）
    2. 提取股票代码列表
    3. 支持动态配置更新
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化策略配置解析器
        
        Args:
            config_dir: 策略配置文件目录，默认为项目根目录下的 config/strategies
        """
        if config_dir is None:
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config" / "strategies"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 缓存已加载的配置
        self._config_cache: Dict[str, StrategyConfig] = {}
        
        logger.info(f"策略配置解析器初始化完成，配置目录: {self.config_dir}")
    
    def parse_config(self, config_path: Union[str, Path]) -> Optional[StrategyConfig]:
        """
        解析策略配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            策略配置对象
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            return None
        
        try:
            # 根据文件扩展名选择解析方式
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                logger.error(f"不支持的配置文件格式: {config_path.suffix}")
                return None
            
            # 解析配置数据
            config = self._parse_config_data(data, config_path.stem)
            
            # 缓存配置
            self._config_cache[config.strategy_id] = config
            
            logger.info(f"成功解析策略配置: {config.strategy_id}")
            return config
            
        except Exception as e:
            logger.error(f"解析配置文件失败 {config_path}: {e}")
            return None
    
    def _parse_config_data(self, data: Dict[str, Any], default_id: str) -> StrategyConfig:
        """
        解析配置数据字典
        
        Args:
            data: 配置数据字典
            default_id: 默认策略ID
            
        Returns:
            策略配置对象
        """
        # 提取股票代码
        symbols = self._extract_symbols(data)
        
        # 解析时间范围
        time_range = data.get('time_range', {})
        if isinstance(time_range, dict):
            time_range = {
                'start_date': time_range.get('start_date', ''),
                'end_date': time_range.get('end_date', '')
            }
        
        # 创建配置对象
        config = StrategyConfig(
            strategy_id=data.get('strategy_id', default_id),
            strategy_name=data.get('strategy_name', data.get('name', default_id)),
            symbols=symbols,
            description=data.get('description'),
            parameters=data.get('parameters', {}),
            data_sources=data.get('data_sources', ['akshare']),
            time_range=time_range,
            created_at=self._parse_datetime(data.get('created_at')),
            updated_at=self._parse_datetime(data.get('updated_at')),
            enabled=data.get('enabled', True)
        )
        
        return config
    
    def _extract_symbols(self, data: Dict[str, Any]) -> List[str]:
        """
        从配置数据中提取股票代码
        
        Args:
            data: 配置数据字典
            
        Returns:
            股票代码列表
        """
        symbols = []
        
        # 直接指定的股票代码
        if 'symbols' in data:
            symbols_data = data['symbols']
            if isinstance(symbols_data, list):
                symbols.extend(symbols_data)
            elif isinstance(symbols_data, str):
                # 逗号分隔的字符串
                symbols.extend([s.strip() for s in symbols_data.split(',')])
        
        # 从股票池中提取
        if 'stock_pool' in data:
            stock_pool = data['stock_pool']
            if isinstance(stock_pool, list):
                symbols.extend(stock_pool)
            elif isinstance(stock_pool, dict):
                # 支持按类别分组的股票池
                for category, stocks in stock_pool.items():
                    if isinstance(stocks, list):
                        symbols.extend(stocks)
                    elif isinstance(stocks, str):
                        symbols.extend([s.strip() for s in stocks.split(',')])
        
        # 从universe中提取
        if 'universe' in data:
            universe = data['universe']
            if isinstance(universe, list):
                symbols.extend(universe)
        
        # 去重并过滤空值
        symbols = list(set([s for s in symbols if s]))
        
        return symbols
    
    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """
        解析日期时间
        
        Args:
            value: 日期时间值
            
        Returns:
            datetime对象或None
        """
        if value is None:
            return None
        
        if isinstance(value, datetime):
            return value
        
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                try:
                    return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    return None
        
        return None
    
    def load_all_configs(self) -> Dict[str, StrategyConfig]:
        """
        加载所有策略配置
        
        Returns:
            策略ID到配置的映射
        """
        configs = {}
        
        # 遍历配置目录
        for config_file in self.config_dir.iterdir():
            if config_file.suffix.lower() in ['.yaml', '.yml', '.json']:
                config = self.parse_config(config_file)
                if config:
                    configs[config.strategy_id] = config
        
        logger.info(f"加载了 {len(configs)} 个策略配置")
        return configs
    
    def get_config(self, strategy_id: str) -> Optional[StrategyConfig]:
        """
        获取策略配置
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            策略配置对象
        """
        # 先从缓存获取
        if strategy_id in self._config_cache:
            return self._config_cache[strategy_id]
        
        # 尝试从文件加载
        for ext in ['.yaml', '.yml', '.json']:
            config_path = self.config_dir / f"{strategy_id}{ext}"
            if config_path.exists():
                return self.parse_config(config_path)
        
        logger.warning(f"未找到策略配置: {strategy_id}")
        return None
    
    def get_symbols_for_strategy(self, strategy_id: str) -> List[str]:
        """
        获取策略相关的股票代码
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            股票代码列表
        """
        config = self.get_config(strategy_id)
        if config:
            return config.symbols
        return []
    
    def reload_config(self, strategy_id: str) -> Optional[StrategyConfig]:
        """
        重新加载策略配置
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            策略配置对象
        """
        # 从缓存中移除
        if strategy_id in self._config_cache:
            del self._config_cache[strategy_id]
        
        # 重新加载
        return self.get_config(strategy_id)
    
    def create_default_config(self, strategy_id: str, symbols: List[str]) -> StrategyConfig:
        """
        创建默认策略配置
        
        Args:
            strategy_id: 策略ID
            symbols: 股票代码列表
            
        Returns:
            策略配置对象
        """
        config = StrategyConfig(
            strategy_id=strategy_id,
            strategy_name=f"Strategy {strategy_id}",
            symbols=symbols,
            description=f"Auto-generated config for strategy {strategy_id}",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # 保存到文件
        self.save_config(config)
        
        return config
    
    def save_config(self, config: StrategyConfig, format: str = 'yaml') -> bool:
        """
        保存策略配置到文件
        
        Args:
            config: 策略配置对象
            format: 文件格式（yaml 或 json）
            
        Returns:
            是否保存成功
        """
        try:
            # 转换为字典
            data = {
                'strategy_id': config.strategy_id,
                'strategy_name': config.strategy_name,
                'symbols': config.symbols,
                'description': config.description,
                'parameters': config.parameters,
                'data_sources': config.data_sources,
                'time_range': config.time_range,
                'created_at': config.created_at.isoformat() if config.created_at else None,
                'updated_at': datetime.now().isoformat(),
                'enabled': config.enabled
            }
            
            # 移除None值
            data = {k: v for k, v in data.items() if v is not None}
            
            # 确定文件路径
            ext = '.yaml' if format == 'yaml' else '.json'
            config_path = self.config_dir / f"{config.strategy_id}{ext}"
            
            # 保存文件
            with open(config_path, 'w', encoding='utf-8') as f:
                if format == 'yaml':
                    yaml.dump(data, f, allow_unicode=True, sort_keys=False)
                else:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"策略配置已保存: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存策略配置失败: {e}")
            return False
    
    def validate_config(self, config: StrategyConfig) -> tuple[bool, List[str]]:
        """
        验证策略配置
        
        Args:
            config: 策略配置对象
            
        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        if not config.strategy_id:
            errors.append("策略ID不能为空")
        
        if not config.strategy_name:
            errors.append("策略名称不能为空")
        
        if not config.symbols:
            errors.append("股票代码列表不能为空")
        
        # 验证股票代码格式
        for symbol in config.symbols:
            if not isinstance(symbol, str) or len(symbol) != 6:
                errors.append(f"股票代码格式错误: {symbol}")
        
        return len(errors) == 0, errors


# 单例实例
_parser: Optional[StrategyConfigParser] = None


def get_strategy_config_parser(config_dir: Optional[str] = None) -> StrategyConfigParser:
    """
    获取策略配置解析器单例
    
    Args:
        config_dir: 配置目录
        
    Returns:
        StrategyConfigParser实例
    """
    global _parser
    if _parser is None:
        _parser = StrategyConfigParser(config_dir=config_dir)
    return _parser
