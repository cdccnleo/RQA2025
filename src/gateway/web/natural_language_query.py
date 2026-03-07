"""
自然语言查询接口模块

功能：
- 自然语言理解(NLU)处理
- 查询意图识别
- 实体提取与解析
- SQL/查询语句生成
- 多轮对话支持
- 查询结果解释

技术栈：
- 正则表达式: 模式匹配
- jieba: 中文分词
- 模板匹配: 查询模板

作者: Claude
创建日期: 2026-02-21
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """查询意图类型"""
    MARKET_DATA = "market_data"           # 市场数据查询
    STOCK_INFO = "stock_info"             # 股票信息查询
    STRATEGY_PERFORMANCE = "strategy_performance"  # 策略表现查询
    PORTFOLIO_STATUS = "portfolio_status" # 投资组合状态
    RISK_ANALYSIS = "risk_analysis"       # 风险分析
    TRADE_HISTORY = "trade_history"       # 交易历史
    INDICATOR_CALCULATION = "indicator_calculation"  # 指标计算
    COMPARISON = "comparison"             # 对比分析
    TREND_ANALYSIS = "trend_analysis"     # 趋势分析
    UNKNOWN = "unknown"                   # 未知意图


class EntityType(Enum):
    """实体类型"""
    STOCK_CODE = "stock_code"             # 股票代码
    STOCK_NAME = "stock_name"             # 股票名称
    DATE = "date"                         # 日期
    DATE_RANGE = "date_range"             # 日期范围
    INDICATOR = "indicator"               # 技术指标
    STRATEGY = "strategy"                 # 策略名称
    NUMBER = "number"                     # 数字
    PERCENTAGE = "percentage"             # 百分比
    TIME_PERIOD = "time_period"           # 时间段


@dataclass
class Entity:
    """提取的实体"""
    type: EntityType
    value: str
    normalized_value: Any
    start_pos: int
    end_pos: int
    confidence: float = 1.0


@dataclass
class ParsedQuery:
    """解析后的查询"""
    original_text: str
    intent: QueryIntent
    entities: List[Entity]
    parameters: Dict[str, Any]
    sql_query: Optional[str] = None
    confidence: float = 0.0
    suggested_queries: List[str] = field(default_factory=list)


@dataclass
class QueryResult:
    """查询结果"""
    query: ParsedQuery
    data: Any
    explanation: str
    visualization_type: Optional[str] = None
    execution_time_ms: float = 0.0


class EntityExtractor:
    """
    实体提取器
    
    从自然语言文本中提取结构化实体
    """
    
    def __init__(self):
        """初始化实体提取器"""
        # 股票代码模式 (A股)
        self.stock_code_pattern = re.compile(r'\b(\d{6})\b')
        
        # 日期模式
        self.date_patterns = [
            (re.compile(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})'), 'YYYY-MM-DD'),
            (re.compile(r'(\d{4})(\d{2})(\d{2})'), 'YYYYMMDD'),
            (re.compile(r'(\d{1,2})[-/](\d{1,2})'), 'MM-DD'),
        ]
        
        # 数字和百分比
        self.number_pattern = re.compile(r'\b(\d+\.?\d*)\b')
        self.percentage_pattern = re.compile(r'(\d+\.?\d*)%')
        
        # 时间段关键词
        self.time_period_keywords = {
            '今天': 0,
            '昨天': -1,
            '前天': -2,
            '本周': -7,
            '上周': -14,
            '本月': -30,
            '上月': -60,
            '今年': -365,
            '去年': -730,
        }
        
        # 技术指标关键词
        self.indicator_keywords = {
            'MA': ['ma', '均线', '移动平均线'],
            'MACD': ['macd', '异同移动平均线'],
            'RSI': ['rsi', '相对强弱指标'],
            'KDJ': ['kdj', '随机指标'],
            'BOLL': ['boll', '布林带', '布林线'],
            'VOL': ['vol', '成交量', '量能'],
            'PE': ['pe', '市盈率'],
            'PB': ['pb', '市净率'],
        }
        
        # 常见股票名称映射
        self.stock_names = {
            '茅台': '600519',
            '腾讯': '00700',
            '阿里巴巴': 'BABA',
            '苹果': 'AAPL',
            '谷歌': 'GOOGL',
            '微软': 'MSFT',
            '特斯拉': 'TSLA',
            '亚马逊': 'AMZN',
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        提取所有实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        entities = []
        
        # 提取股票代码
        entities.extend(self._extract_stock_codes(text))
        
        # 提取股票名称
        entities.extend(self._extract_stock_names(text))
        
        # 提取日期
        entities.extend(self._extract_dates(text))
        
        # 提取时间段
        entities.extend(self._extract_time_periods(text))
        
        # 提取技术指标
        entities.extend(self._extract_indicators(text))
        
        # 提取数字和百分比
        entities.extend(self._extract_numbers(text))
        
        # 按位置排序
        entities.sort(key=lambda x: x.start_pos)
        
        return entities
    
    def _extract_stock_codes(self, text: str) -> List[Entity]:
        """提取股票代码"""
        entities = []
        for match in self.stock_code_pattern.finditer(text):
            entities.append(Entity(
                type=EntityType.STOCK_CODE,
                value=match.group(1),
                normalized_value=match.group(1),
                start_pos=match.start(),
                end_pos=match.end()
            ))
        return entities
    
    def _extract_stock_names(self, text: str) -> List[Entity]:
        """提取股票名称"""
        entities = []
        for name, code in self.stock_names.items():
            if name in text:
                pos = text.index(name)
                entities.append(Entity(
                    type=EntityType.STOCK_NAME,
                    value=name,
                    normalized_value=code,
                    start_pos=pos,
                    end_pos=pos + len(name)
                ))
        return entities
    
    def _extract_dates(self, text: str) -> List[Entity]:
        """提取日期"""
        entities = []
        
        for pattern, format_str in self.date_patterns:
            for match in pattern.finditer(text):
                try:
                    if format_str == 'YYYY-MM-DD':
                        year, month, day = match.groups()
                        date_obj = datetime(int(year), int(month), int(day))
                    elif format_str == 'YYYYMMDD':
                        year, month, day = match.groups()
                        date_obj = datetime(int(year), int(month), int(day))
                    elif format_str == 'MM-DD':
                        month, day = match.groups()
                        year = datetime.now().year
                        date_obj = datetime(year, int(month), int(day))
                    
                    entities.append(Entity(
                        type=EntityType.DATE,
                        value=match.group(0),
                        normalized_value=date_obj,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
                except ValueError:
                    continue
        
        return entities
    
    def _extract_time_periods(self, text: str) -> List[Entity]:
        """提取时间段"""
        entities = []
        
        for keyword, days in self.time_period_keywords.items():
            if keyword in text:
                pos = text.index(keyword)
                end_date = datetime.now()
                if days < 0:
                    start_date = end_date + timedelta(days=days)
                else:
                    start_date = end_date
                
                entities.append(Entity(
                    type=EntityType.TIME_PERIOD,
                    value=keyword,
                    normalized_value={
                        'start': start_date,
                        'end': end_date,
                        'days': abs(days)
                    },
                    start_pos=pos,
                    end_pos=pos + len(keyword)
                ))
        
        return entities
    
    def _extract_indicators(self, text: str) -> List[Entity]:
        """提取技术指标"""
        entities = []
        text_lower = text.lower()
        
        for indicator, keywords in self.indicator_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    pos = text_lower.index(keyword)
                    entities.append(Entity(
                        type=EntityType.INDICATOR,
                        value=keyword,
                        normalized_value=indicator,
                        start_pos=pos,
                        end_pos=pos + len(keyword)
                    ))
                    break
        
        return entities
    
    def _extract_numbers(self, text: str) -> List[Entity]:
        """提取数字和百分比"""
        entities = []
        
        # 提取百分比
        for match in self.percentage_pattern.finditer(text):
            entities.append(Entity(
                type=EntityType.PERCENTAGE,
                value=match.group(0),
                normalized_value=float(match.group(1)) / 100,
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        # 提取普通数字
        for match in self.number_pattern.finditer(text):
            # 检查是否已经是百分比的一部分
            is_percentage = any(
                e.start_pos <= match.start() < e.end_pos 
                for e in entities 
                if e.type == EntityType.PERCENTAGE
            )
            if not is_percentage:
                entities.append(Entity(
                    type=EntityType.NUMBER,
                    value=match.group(1),
                    normalized_value=float(match.group(1)),
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        return entities


class IntentClassifier:
    """
    意图分类器
    
    识别用户查询的意图类型
    """
    
    def __init__(self):
        """初始化意图分类器"""
        self.intent_patterns = {
            QueryIntent.MARKET_DATA: [
                r'.*行情.*',
                r'.*走势.*',
                r'.*价格.*',
                r'.*涨跌.*',
                r'.*大盘.*',
                r'.*指数.*',
            ],
            QueryIntent.STOCK_INFO: [
                r'.*股票.*',
                r'.*个股.*',
                r'.*代码.*',
                r'.*基本面.*',
                r'.*财务.*',
            ],
            QueryIntent.STRATEGY_PERFORMANCE: [
                r'.*策略.*',
                r'.*表现.*',
                r'.*收益.*',
                r'.*回测.*',
                r'.*绩效.*',
            ],
            QueryIntent.PORTFOLIO_STATUS: [
                r'.*组合.*',
                r'.*持仓.*',
                r'.*资产.*',
                r'.*仓位.*',
                r'.*配置.*',
            ],
            QueryIntent.RISK_ANALYSIS: [
                r'.*风险.*',
                r'.*回撤.*',
                r'.*波动.*',
                r'.*夏普.*',
                r'.*最大回撤.*',
            ],
            QueryIntent.TRADE_HISTORY: [
                r'.*交易.*',
                r'.*成交.*',
                r'.*买卖.*',
                r'.*委托.*',
                r'.*历史.*',
            ],
            QueryIntent.INDICATOR_CALCULATION: [
                r'.*指标.*',
                r'.*MACD.*',
                r'.*KDJ.*',
                r'.*RSI.*',
                r'.*均线.*',
                r'.*布林带.*',
            ],
            QueryIntent.COMPARISON: [
                r'.*对比.*',
                r'.*比较.*',
                r'.*vs.*',
                r'.*和.*相比.*',
                r'.*哪个.*',
            ],
            QueryIntent.TREND_ANALYSIS: [
                r'.*趋势.*',
                r'.*预测.*',
                r'.*分析.*',
                r'.*判断.*',
                r'.*方向.*',
            ],
        }
    
    def classify(self, text: str, entities: List[Entity]) -> Tuple[QueryIntent, float]:
        """
        分类查询意图
        
        Args:
            text: 输入文本
            entities: 提取的实体
            
        Returns:
            (意图, 置信度)
        """
        scores = defaultdict(float)
        
        # 基于模式匹配评分
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    scores[intent] += 1.0
        
        # 基于实体类型评分
        entity_types = [e.type for e in entities]
        
        if EntityType.STOCK_CODE in entity_types or EntityType.STOCK_NAME in entity_types:
            scores[QueryIntent.STOCK_INFO] += 1.5
            scores[QueryIntent.MARKET_DATA] += 1.0
        
        if EntityType.INDICATOR in entity_types:
            scores[QueryIntent.INDICATOR_CALCULATION] += 2.0
        
        if EntityType.DATE in entity_types or EntityType.TIME_PERIOD in entity_types:
            scores[QueryIntent.MARKET_DATA] += 0.5
            scores[QueryIntent.TRADE_HISTORY] += 0.5
        
        # 选择最高分的意图
        if scores:
            best_intent = max(scores, key=scores.get)
            max_score = scores[best_intent]
            # 归一化置信度
            confidence = min(max_score / 3.0, 1.0)
            return best_intent, confidence
        
        return QueryIntent.UNKNOWN, 0.0


class QueryBuilder:
    """
    查询构建器
    
    将解析后的查询转换为可执行的查询语句
    """
    
    def __init__(self):
        """初始化查询构建器"""
        self.query_templates = {
            QueryIntent.MARKET_DATA: self._build_market_data_query,
            QueryIntent.STOCK_INFO: self._build_stock_info_query,
            QueryIntent.STRATEGY_PERFORMANCE: self._build_strategy_query,
            QueryIntent.PORTFOLIO_STATUS: self._build_portfolio_query,
            QueryIntent.RISK_ANALYSIS: self._build_risk_query,
            QueryIntent.TRADE_HISTORY: self._build_trade_history_query,
            QueryIntent.INDICATOR_CALCULATION: self._build_indicator_query,
            QueryIntent.COMPARISON: self._build_comparison_query,
            QueryIntent.TREND_ANALYSIS: self._build_trend_query,
        }
    
    def build(self, parsed_query: ParsedQuery) -> Optional[str]:
        """
        构建查询
        
        Args:
            parsed_query: 解析后的查询
            
        Returns:
            SQL查询语句或None
        """
        builder_func = self.query_templates.get(parsed_query.intent)
        if builder_func:
            return builder_func(parsed_query)
        return None
    
    def _build_market_data_query(self, query: ParsedQuery) -> str:
        """构建市场数据查询"""
        stock_codes = [e.normalized_value for e in query.entities 
                      if e.type == EntityType.STOCK_CODE]
        
        sql = "SELECT * FROM market_data WHERE 1=1"
        
        if stock_codes:
            codes_str = ', '.join(f"'{code}'" for code in stock_codes)
            sql += f" AND stock_code IN ({codes_str})"
        
        # 添加日期范围
        date_entities = [e for e in query.entities if e.type == EntityType.TIME_PERIOD]
        if date_entities:
            date_range = date_entities[0].normalized_value
            sql += f" AND date >= '{date_range['start'].strftime('%Y-%m-%d')}'"
            sql += f" AND date <= '{date_range['end'].strftime('%Y-%m-%d')}'"
        
        sql += " ORDER BY date DESC"
        return sql
    
    def _build_stock_info_query(self, query: ParsedQuery) -> str:
        """构建股票信息查询"""
        stock_codes = [e.normalized_value for e in query.entities 
                      if e.type == EntityType.STOCK_CODE]
        stock_names = [e.normalized_value for e in query.entities 
                      if e.type == EntityType.STOCK_NAME]
        
        sql = "SELECT * FROM stock_info WHERE 1=1"
        
        if stock_codes:
            codes_str = ', '.join(f"'{code}'" for code in stock_codes)
            sql += f" AND stock_code IN ({codes_str})"
        
        if stock_names:
            names_str = ', '.join(f"'{name}'" for name in stock_names)
            sql += f" AND stock_code IN ({names_str})"
        
        return sql
    
    def _build_strategy_query(self, query: ParsedQuery) -> str:
        """构建策略表现查询"""
        return "SELECT * FROM strategy_performance ORDER BY date DESC LIMIT 30"
    
    def _build_portfolio_query(self, query: ParsedQuery) -> str:
        """构建投资组合查询"""
        return "SELECT * FROM portfolio_status WHERE date = (SELECT MAX(date) FROM portfolio_status)"
    
    def _build_risk_query(self, query: ParsedQuery) -> str:
        """构建风险分析查询"""
        return "SELECT * FROM risk_metrics ORDER BY date DESC LIMIT 30"
    
    def _build_trade_history_query(self, query: ParsedQuery) -> str:
        """构建交易历史查询"""
        sql = "SELECT * FROM trade_history WHERE 1=1"
        
        date_entities = [e for e in query.entities if e.type == EntityType.TIME_PERIOD]
        if date_entities:
            date_range = date_entities[0].normalized_value
            sql += f" AND trade_date >= '{date_range['start'].strftime('%Y-%m-%d')}'"
            sql += f" AND trade_date <= '{date_range['end'].strftime('%Y-%m-%d')}'"
        
        sql += " ORDER BY trade_date DESC"
        return sql
    
    def _build_indicator_query(self, query: ParsedQuery) -> str:
        """构建指标计算查询"""
        indicators = [e.normalized_value for e in query.entities 
                     if e.type == EntityType.INDICATOR]
        stock_codes = [e.normalized_value for e in query.entities 
                      if e.type == EntityType.STOCK_CODE]
        
        sql = "SELECT date, stock_code"
        
        for indicator in indicators:
            sql += f", {indicator.lower()}"
        
        sql += " FROM technical_indicators WHERE 1=1"
        
        if stock_codes:
            codes_str = ', '.join(f"'{code}'" for code in stock_codes)
            sql += f" AND stock_code IN ({codes_str})"
        
        sql += " ORDER BY date DESC LIMIT 30"
        return sql
    
    def _build_comparison_query(self, query: ParsedQuery) -> str:
        """构建对比分析查询"""
        stock_codes = [e.normalized_value for e in query.entities 
                      if e.type == EntityType.STOCK_CODE]
        
        if len(stock_codes) >= 2:
            codes_str = ', '.join(f"'{code}'" for code in stock_codes[:2])
            return f"SELECT * FROM stock_comparison WHERE stock_code IN ({codes_str})"
        
        return "SELECT * FROM stock_comparison LIMIT 10"
    
    def _build_trend_query(self, query: ParsedQuery) -> str:
        """构建趋势分析查询"""
        return "SELECT * FROM trend_analysis ORDER BY date DESC LIMIT 30"


class NaturalLanguageQueryEngine:
    """
    自然语言查询引擎
    
    主类：协调实体提取、意图分类和查询构建
    """
    
    def __init__(self):
        """初始化自然语言查询引擎"""
        self.entity_extractor = EntityExtractor()
        self.intent_classifier = IntentClassifier()
        self.query_builder = QueryBuilder()
        
        # 查询历史
        self.query_history: List[ParsedQuery] = []
        
        # 建议查询模板
        self.suggestion_templates = {
            QueryIntent.MARKET_DATA: [
                "{stock}最近一周的行情",
                "{stock}今天的走势如何",
                "大盘指数今天涨跌",
            ],
            QueryIntent.STOCK_INFO: [
                "{stock}的基本面信息",
                "{stock}的市盈率是多少",
                "查询{stock}的财务数据",
            ],
            QueryIntent.STRATEGY_PERFORMANCE: [
                "策略最近的表现如何",
                "查看策略收益曲线",
                "策略的回测结果",
            ],
            QueryIntent.RISK_ANALYSIS: [
                "组合的风险指标",
                "查看最大回撤",
                "夏普比率是多少",
            ],
        }
    
    def parse(self, text: str) -> ParsedQuery:
        """
        解析自然语言查询
        
        Args:
            text: 用户输入的自然语言文本
            
        Returns:
            解析后的查询对象
        """
        # 提取实体
        entities = self.entity_extractor.extract_entities(text)
        
        # 分类意图
        intent, confidence = self.intent_classifier.classify(text, entities)
        
        # 构建参数
        parameters = self._build_parameters(entities)
        
        # 创建解析结果
        parsed = ParsedQuery(
            original_text=text,
            intent=intent,
            entities=entities,
            parameters=parameters,
            confidence=confidence
        )
        
        # 构建SQL查询
        parsed.sql_query = self.query_builder.build(parsed)
        
        # 生成建议查询
        parsed.suggested_queries = self._generate_suggestions(intent, entities)
        
        # 保存到历史
        self.query_history.append(parsed)
        
        return parsed
    
    def _build_parameters(self, entities: List[Entity]) -> Dict[str, Any]:
        """构建查询参数"""
        params = {}
        
        for entity in entities:
            if entity.type == EntityType.STOCK_CODE:
                params['stock_code'] = entity.normalized_value
            elif entity.type == EntityType.STOCK_NAME:
                params['stock_code'] = entity.normalized_value
            elif entity.type == EntityType.DATE:
                params['date'] = entity.normalized_value
            elif entity.type == EntityType.TIME_PERIOD:
                params['date_range'] = entity.normalized_value
            elif entity.type == EntityType.INDICATOR:
                params['indicator'] = entity.normalized_value
            elif entity.type == EntityType.NUMBER:
                params['number'] = entity.normalized_value
            elif entity.type == EntityType.PERCENTAGE:
                params['percentage'] = entity.normalized_value
        
        return params
    
    def _generate_suggestions(self, intent: QueryIntent, 
                             entities: List[Entity]) -> List[str]:
        """生成建议查询"""
        suggestions = []
        
        # 获取股票代码
        stock_codes = [e.value for e in entities 
                      if e.type in (EntityType.STOCK_CODE, EntityType.STOCK_NAME)]
        stock = stock_codes[0] if stock_codes else "600519"
        
        # 根据意图生成建议
        templates = self.suggestion_templates.get(intent, [])
        for template in templates:
            suggestions.append(template.format(stock=stock))
        
        return suggestions[:3]
    
    def execute(self, text: str, 
               data_fetcher: Optional[Callable] = None) -> QueryResult:
        """
        执行自然语言查询
        
        Args:
            text: 自然语言查询文本
            data_fetcher: 数据获取函数
            
        Returns:
            查询结果
        """
        import time
        start_time = time.time()
        
        # 解析查询
        parsed = self.parse(text)
        
        # 获取数据
        data = None
        if data_fetcher and parsed.sql_query:
            try:
                data = data_fetcher(parsed.sql_query)
            except Exception as e:
                logger.error(f"数据获取失败: {e}")
        
        # 生成解释
        explanation = self._generate_explanation(parsed)
        
        # 确定可视化类型
        viz_type = self._determine_visualization(parsed)
        
        execution_time = (time.time() - start_time) * 1000
        
        return QueryResult(
            query=parsed,
            data=data,
            explanation=explanation,
            visualization_type=viz_type,
            execution_time_ms=execution_time
        )
    
    def _generate_explanation(self, parsed: ParsedQuery) -> str:
        """生成查询解释"""
        explanations = []
        
        # 意图解释
        intent_names = {
            QueryIntent.MARKET_DATA: "市场数据查询",
            QueryIntent.STOCK_INFO: "股票信息查询",
            QueryIntent.STRATEGY_PERFORMANCE: "策略表现查询",
            QueryIntent.PORTFOLIO_STATUS: "投资组合查询",
            QueryIntent.RISK_ANALYSIS: "风险分析",
            QueryIntent.TRADE_HISTORY: "交易历史查询",
            QueryIntent.INDICATOR_CALCULATION: "技术指标计算",
            QueryIntent.COMPARISON: "对比分析",
            QueryIntent.TREND_ANALYSIS: "趋势分析",
            QueryIntent.UNKNOWN: "未知查询",
        }
        
        explanations.append(f"查询类型: {intent_names.get(parsed.intent, '未知')}")
        
        # 实体解释
        if parsed.entities:
            explanations.append("识别到的信息:")
            for entity in parsed.entities:
                explanations.append(f"  - {entity.type.value}: {entity.value}")
        
        # SQL解释
        if parsed.sql_query:
            explanations.append(f"生成的查询: {parsed.sql_query}")
        
        return '\n'.join(explanations)
    
    def _determine_visualization(self, parsed: ParsedQuery) -> Optional[str]:
        """确定可视化类型"""
        viz_map = {
            QueryIntent.MARKET_DATA: "candlestick",
            QueryIntent.STRATEGY_PERFORMANCE: "line_chart",
            QueryIntent.PORTFOLIO_STATUS: "pie_chart",
            QueryIntent.RISK_ANALYSIS: "bar_chart",
            QueryIntent.TRADE_HISTORY: "table",
            QueryIntent.INDICATOR_CALCULATION: "multi_line_chart",
            QueryIntent.COMPARISON: "comparison_chart",
            QueryIntent.TREND_ANALYSIS: "trend_chart",
        }
        
        return viz_map.get(parsed.intent)
    
    def get_history(self, limit: int = 10) -> List[ParsedQuery]:
        """获取查询历史"""
        return self.query_history[-limit:]
    
    def clear_history(self) -> None:
        """清除查询历史"""
        self.query_history.clear()


# 便捷函数
def create_nlq_engine() -> NaturalLanguageQueryEngine:
    """创建自然语言查询引擎"""
    return NaturalLanguageQueryEngine()


# 单例实例
_nlq_engine_instance: Optional[NaturalLanguageQueryEngine] = None


def get_nlq_engine() -> NaturalLanguageQueryEngine:
    """
    获取自然语言查询引擎单例
    
    Returns:
        NaturalLanguageQueryEngine实例
    """
    global _nlq_engine_instance
    if _nlq_engine_instance is None:
        _nlq_engine_instance = NaturalLanguageQueryEngine()
    return _nlq_engine_instance
