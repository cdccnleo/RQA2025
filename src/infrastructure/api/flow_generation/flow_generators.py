"""
流程生成器

职责：为不同业务场景生成流程图
"""

from .models import APIFlowDiagram
from .node_builder import FlowNodeBuilder


class DataServiceFlowGenerator:
    """数据服务流程生成器"""
    
    def __init__(self, node_builder: FlowNodeBuilder):
        self.builder = node_builder
    
    def generate(self) -> APIFlowDiagram:
        """生成数据服务流程图"""
        self.builder.clear()
        
        # 创建节点
        self.builder.create_start_node(position={"x": 100, "y": 100})
        self.builder.create_decision_node("auth_check", "身份验证", position={"x": 250, "y": 100})
        self.builder.create_decision_node("rate_limit", "频率限制", position={"x": 400, "y": 100})
        self.builder.create_process_node("param_validation", "参数验证", position={"x": 550, "y": 100})
        self.builder.create_decision_node("cache_check", "缓存检查", position={"x": 700, "y": 100})
        self.builder.create_api_call_node("query_db", "查询数据库", position={"x": 850, "y": 200})
        self.builder.create_process_node("format_data", "格式化数据", position={"x": 1000, "y": 100})
        self.builder.create_end_node(position={"x": 1150, "y": 100})
        
        # 创建边
        self.builder.create_edge("start", "auth_check")
        self.builder.create_edge("auth_check", "rate_limit", "通过", "success")
        self.builder.create_edge("auth_check", "end", "失败", "error")
        self.builder.create_edge("rate_limit", "param_validation", "通过", "success")
        self.builder.create_edge("rate_limit", "end", "超限", "error")
        self.builder.create_edge("param_validation", "cache_check")
        self.builder.create_edge("cache_check", "format_data", "命中", "success")
        self.builder.create_edge("cache_check", "query_db", "未命中")
        self.builder.create_edge("query_db", "format_data")
        self.builder.create_edge("format_data", "end")
        
        return APIFlowDiagram(
            id="data_service_flow",
            title="数据服务API调用流程",
            description="RQA2025数据服务API的完整调用流程",
            nodes=self.builder.get_nodes(),
            edges=self.builder.get_edges(),
            layout="horizontal"
        )


class TradingFlowGenerator:
    """交易流程生成器"""
    
    def __init__(self, node_builder: FlowNodeBuilder):
        self.builder = node_builder
    
    def generate(self) -> APIFlowDiagram:
        """生成交易流程图"""
        self.builder.clear()
        
        # 创建节点
        self.builder.create_start_node(position={"x": 100, "y": 100})
        self.builder.create_decision_node("auth_check", "身份验证", position={"x": 250, "y": 100})
        self.builder.create_process_node("strategy_load", "加载策略", position={"x": 400, "y": 100})
        self.builder.create_process_node("backtest_init", "初始化回测", position={"x": 550, "y": 100})
        self.builder.create_api_call_node("get_data", "获取数据", position={"x": 700, "y": 100})
        self.builder.create_process_node("run_backtest", "执行回测", position={"x": 850, "y": 100})
        self.builder.create_process_node("calc_metrics", "计算指标", position={"x": 1000, "y": 100})
        self.builder.create_end_node(position={"x": 1150, "y": 100})
        
        # 创建边
        self.builder.create_edge("start", "auth_check")
        self.builder.create_edge("auth_check", "strategy_load", "通过", "success")
        self.builder.create_edge("auth_check", "end", "失败", "error")
        self.builder.create_edge("strategy_load", "backtest_init")
        self.builder.create_edge("backtest_init", "get_data")
        self.builder.create_edge("get_data", "run_backtest")
        self.builder.create_edge("run_backtest", "calc_metrics")
        self.builder.create_edge("calc_metrics", "end")
        
        return APIFlowDiagram(
            id="trading_flow",
            title="交易回测API调用流程",
            description="交易策略回测的完整流程",
            nodes=self.builder.get_nodes(),
            edges=self.builder.get_edges(),
            layout="horizontal"
        )


class FeatureEngineeringFlowGenerator:
    """特征工程流程生成器"""
    
    def __init__(self, node_builder: FlowNodeBuilder):
        self.builder = node_builder
    
    def generate(self) -> APIFlowDiagram:
        """生成特征工程流程图"""
        self.builder.clear()
        
        # 创建节点
        self.builder.create_start_node(position={"x": 100, "y": 100})
        self.builder.create_decision_node("auth_check", "身份验证", position={"x": 250, "y": 100})
        self.builder.create_process_node("load_data", "加载数据", position={"x": 400, "y": 100})
        self.builder.create_process_node("calc_indicators", "计算指标", position={"x": 550, "y": 100})
        self.builder.create_process_node("feature_selection", "特征选择", position={"x": 700, "y": 100})
        self.builder.create_process_node("feature_transform", "特征转换", position={"x": 850, "y": 100})
        self.builder.create_end_node(position={"x": 1000, "y": 100})
        
        # 创建边
        self.builder.create_edge("start", "auth_check")
        self.builder.create_edge("auth_check", "load_data", "通过", "success")
        self.builder.create_edge("auth_check", "end", "失败", "error")
        self.builder.create_edge("load_data", "calc_indicators")
        self.builder.create_edge("calc_indicators", "feature_selection")
        self.builder.create_edge("feature_selection", "feature_transform")
        self.builder.create_edge("feature_transform", "end")
        
        return APIFlowDiagram(
            id="feature_engineering_flow",
            title="特征工程API调用流程",
            description="特征工程计算的完整流程",
            nodes=self.builder.get_nodes(),
            edges=self.builder.get_edges(),
            layout="horizontal"
        )

