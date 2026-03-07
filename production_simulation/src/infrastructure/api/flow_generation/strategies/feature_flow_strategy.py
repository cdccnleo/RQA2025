"""
特征工程服务流程生成策略

替代原create_feature_engineering_flow方法(121行, 116参数)
"""

from .base_flow_strategy import BaseFlowStrategy, FlowDiagram


class FeatureFlowStrategy(BaseFlowStrategy):
    """
    特征工程流程生成策略
    
    原方法: create_feature_engineering_flow(121行, 116参数)
    新策略: FeatureFlowStrategy.generate_flow(~70行, 0参数)
    
    优化:
    - 参数: 116 → 0 (-100%)
    - 行数: 121 → ~70 (-42%)
    """
    
    def generate_flow(self) -> FlowDiagram:
        """生成特征工程服务流程图"""
        # 创建核心节点
        self._create_start_node("特征工程请求开始")
        self._create_data_nodes()
        self._create_feature_extraction_nodes()
        self._create_feature_calculation_nodes()
        self._create_storage_nodes()
        self._create_end_node("特征工程完成")
        
        # 连接节点
        self._connect_flow()
        
        return FlowDiagram(
            flow_id="feature_engineering_flow",
            title="特征工程服务流程",
            description="RQA2025特征工程服务的完整处理流程",
            nodes=self.nodes,
            edges=self.edges,
            layout="horizontal"
        )
    
    def _create_data_nodes(self):
        """创建数据节点"""
        self._create_api_call_node("load_data", "加载原始数据")
        self._create_process_node("validate_data", "数据验证")
    
    def _create_feature_extraction_nodes(self):
        """创建特征提取节点"""
        self._create_process_node("extract_technical", "技术指标提取")
        self._create_process_node("extract_statistical", "统计特征提取")
        self._create_process_node("extract_pattern", "模式特征提取")
    
    def _create_feature_calculation_nodes(self):
        """创建特征计算节点"""
        self._create_api_call_node("calculate_macd", "MACD计算")
        self._create_api_call_node("calculate_rsi", "RSI计算")
        self._create_api_call_node("calculate_boll", "BOLL计算")
        self._create_process_node("combine_features", "特征组合")
    
    def _create_storage_nodes(self):
        """创建存储节点"""
        self._create_api_call_node("save_features", "保存特征")
        self._create_process_node("build_response", "构建响应")
    
    def _connect_flow(self):
        """连接流程"""
        # 数据加载和验证
        self._connect_nodes("start", "load_data")
        self._connect_nodes("load_data", "validate_data")
        
        # 并行特征提取
        self._connect_nodes("validate_data", "extract_technical")
        self._connect_nodes("validate_data", "extract_statistical")
        self._connect_nodes("validate_data", "extract_pattern")
        
        # 技术指标计算
        self._connect_nodes("extract_technical", "calculate_macd")
        self._connect_nodes("extract_technical", "calculate_rsi")
        self._connect_nodes("extract_technical", "calculate_boll")
        
        # 特征组合
        self._connect_nodes("calculate_macd", "combine_features")
        self._connect_nodes("calculate_rsi", "combine_features")
        self._connect_nodes("calculate_boll", "combine_features")
        self._connect_nodes("extract_statistical", "combine_features")
        self._connect_nodes("extract_pattern", "combine_features")
        
        # 保存和响应
        self._connect_nodes("combine_features", "save_features")
        self._connect_nodes("save_features", "build_response")
        self._connect_nodes("build_response", "end")


def create_feature_engineering_flow() -> FlowDiagram:
    """
    创建特征工程流程图（向后兼容函数）
    
    原函数: create_feature_engineering_flow(121行, 116参数)
    新实现: 使用策略模式(~5行, 0参数)
    
    Returns:
        FlowDiagram: 特征工程流程图
    """
    strategy = FeatureFlowStrategy()
    return strategy.generate_flow()

