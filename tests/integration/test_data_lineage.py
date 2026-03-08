"""
数据血缘追踪集成测试

测试数据血缘追踪系统的核心功能，包括：
1. 血缘图谱构建
2. 血缘查询
3. 影响分析
4. 与数据管道集成
"""

import pytest
from datetime import datetime

# 导入血缘追踪组件
from src.data.lineage import (
    LineageGraph,
    DataAsset,
    DataAssetType,
    LineageNode,
    LineageEdge,
    LineageType,
    Transformation,
    LineageQuery
)


class TestLineageGraph:
    """血缘图谱集成测试"""
    
    @pytest.fixture
    def lineage_graph(self):
        """创建血缘图谱实例"""
        return LineageGraph()
        
    @pytest.fixture
    def sample_data_assets(self):
        """创建示例数据资产"""
        return {
            "raw_data": DataAsset(
                id="raw_001",
                name="raw_market_data",
                type=DataAssetType.TABLE,
                source="wind_api",
                schema={"symbol": "str", "price": "float", "volume": "int"}
            ),
            "cleaned_data": DataAsset(
                id="clean_001",
                name="cleaned_market_data",
                type=DataAssetType.TABLE,
                source="data_pipeline",
                schema={"symbol": "str", "price": "float", "volume": "int", "quality_score": "float"}
            ),
            "features": DataAsset(
                id="feat_001",
                name="technical_features",
                type=DataAssetType.TABLE,
                source="feature_engineering",
                schema={"symbol": "str", "ma5": "float", "ma10": "float", "rsi": "float"}
            ),
            "model": DataAsset(
                id="model_001",
                name="prediction_model",
                type=DataAssetType.MODEL,
                source="ml_training",
                schema={"model_type": "str", "version": "str"}
            ),
            "predictions": DataAsset(
                id="pred_001",
                name="model_predictions",
                type=DataAssetType.TABLE,
                source="model_inference",
                schema={"symbol": "str", "prediction": "float", "confidence": "float"}
            )
        }
        
    def test_build_lineage_graph(self, lineage_graph, sample_data_assets):
        """测试构建血缘图谱"""
        assets = sample_data_assets
        
        # 添加节点（使用资产的id作为节点id）
        for asset in assets.values():
            node = LineageNode(id=asset.id, asset=asset)
            lineage_graph.add_node(node)
            
        # 添加边（血缘关系）
        edges = [
            LineageEdge(
                id="edge_1",
                source_id="raw_001",
                target_id="clean_001",
                type=LineageType.TRANSFORMATION,
                transformation=Transformation(
                    id="trans_1",
                    name="data_cleaning",
                    type=LineageType.TRANSFORMATION,
                    description="Clean and validate raw market data"
                )
            ),
            LineageEdge(
                id="edge_2",
                source_id="clean_001",
                target_id="feat_001",
                type=LineageType.DERIVATION,
                transformation=Transformation(
                    id="trans_2",
                    name="feature_engineering",
                    type=LineageType.DERIVATION,
                    description="Generate technical indicators"
                )
            ),
            LineageEdge(
                id="edge_3",
                source_id="feat_001",
                target_id="model_001",
                type=LineageType.AGGREGATION,
                transformation=Transformation(
                    id="trans_3",
                    name="model_training",
                    type=LineageType.AGGREGATION,
                    description="Train prediction model"
                )
            ),
            LineageEdge(
                id="edge_4",
                source_id="model_001",
                target_id="pred_001",
                type=LineageType.TRANSFORMATION,
                transformation=Transformation(
                    id="trans_4",
                    name="model_inference",
                    type=LineageType.TRANSFORMATION,
                    description="Generate predictions"
                )
            )
        ]
        
        for edge in edges:
            lineage_graph.add_edge(edge)
            
        # 验证图谱结构
        assert len(lineage_graph.nodes) == 5
        assert len(lineage_graph.edges) == 4
        
        # 验证邻接关系
        assert "clean_001" in lineage_graph.adjacency["raw_001"]
        assert "feat_001" in lineage_graph.adjacency["clean_001"]
        assert "model_001" in lineage_graph.adjacency["feat_001"]
        assert "pred_001" in lineage_graph.adjacency["model_001"]
        
    def test_upstream_query(self, lineage_graph, sample_data_assets):
        """测试上游依赖查询"""
        # 构建图谱
        self._build_sample_graph(lineage_graph, sample_data_assets)
        
        # 查询predictions的上游
        upstream = lineage_graph.get_upstream("pred_001")
        
        # 验证结果
        upstream_ids = [asset.id for asset in upstream]
        assert "model_001" in upstream_ids
        assert "feat_001" in upstream_ids
        assert "clean_001" in upstream_ids
        assert "raw_001" in upstream_ids
        
    def test_downstream_query(self, lineage_graph, sample_data_assets):
        """测试下游影响查询"""
        # 构建图谱
        self._build_sample_graph(lineage_graph, sample_data_assets)
        
        # 查询raw_data的下游
        downstream = lineage_graph.get_downstream("raw_001")
        
        # 验证结果
        downstream_ids = [asset.id for asset in downstream]
        assert "clean_001" in downstream_ids
        assert "feat_001" in downstream_ids
        assert "model_001" in downstream_ids
        assert "pred_001" in downstream_ids
        
    def test_find_path(self, lineage_graph, sample_data_assets):
        """测试查找血缘路径"""
        # 构建图谱
        self._build_sample_graph(lineage_graph, sample_data_assets)
        
        # 查找从raw_data到predictions的路径
        path = lineage_graph.find_path("raw_001", "pred_001")
        
        # 验证路径
        assert path is not None
        assert path.distance == 4
        assert len(path.nodes) == 5
        assert len(path.edges) == 4
        
        # 验证路径顺序
        node_ids = [node.id for node in path.nodes]
        assert node_ids == ["raw_001", "clean_001", "feat_001", "model_001", "pred_001"]
        
    def test_impact_analysis(self, lineage_graph, sample_data_assets):
        """测试影响分析"""
        # 构建图谱
        self._build_sample_graph(lineage_graph, sample_data_assets)
        
        # 分析cleaned_data变更的影响
        impact = lineage_graph.analyze_impact("clean_001")
        
        # 验证影响范围
        assert impact.asset_id == "clean_001"
        assert len(impact.upstream) == 1  # raw_data
        assert len(impact.downstream) == 3  # features, model, predictions
        assert impact.total_affected == 4
        
    def test_query_with_direction(self, lineage_graph, sample_data_assets):
        """测试带方向的查询"""
        # 构建图谱
        self._build_sample_graph(lineage_graph, sample_data_assets)
        
        # 双向查询
        query = LineageQuery(
            asset_id="feat_001",
            direction="both",
            depth=-1
        )
        result = lineage_graph.query(query)
        
        # 验证结果
        assert "upstream" in result
        assert "downstream" in result
        assert len(result["upstream"]) == 2  # raw, cleaned
        assert len(result["downstream"]) == 2  # model, predictions
        
    def test_query_with_depth_limit(self, lineage_graph, sample_data_assets):
        """测试带深度限制的查询"""
        # 构建图谱
        self._build_sample_graph(lineage_graph, sample_data_assets)
        
        # 限制深度为1
        query = LineageQuery(
            asset_id="raw_001",
            direction="downstream",
            depth=1
        )
        result = lineage_graph.query(query)
        
        # 验证结果（只有直接下游）
        assert len(result["downstream"]) == 1
        assert result["downstream"][0].id == "clean_001"
        
    def test_graph_statistics(self, lineage_graph, sample_data_assets):
        """测试图谱统计信息"""
        # 构建图谱
        self._build_sample_graph(lineage_graph, sample_data_assets)
        
        # 获取统计信息
        stats = lineage_graph.get_statistics()
        
        # 验证统计
        assert stats["total_nodes"] == 5
        assert stats["total_edges"] == 4
        assert stats["avg_out_degree"] == 0.8  # 4 edges / 5 nodes
        assert stats["avg_in_degree"] == 0.8
        
    @staticmethod
    def _build_sample_graph(lineage_graph, sample_data_assets):
        """构建示例血缘图谱"""
        assets = sample_data_assets
        
        # 添加节点（使用资产的id作为节点id）
        for asset in assets.values():
            node = LineageNode(id=asset.id, asset=asset)
            lineage_graph.add_node(node)
            
        # 添加边
        edges = [
            LineageEdge(
                id="edge_1",
                source_id="raw_001",
                target_id="clean_001",
                type=LineageType.TRANSFORMATION
            ),
            LineageEdge(
                id="edge_2",
                source_id="clean_001",
                target_id="feat_001",
                type=LineageType.DERIVATION
            ),
            LineageEdge(
                id="edge_3",
                source_id="feat_001",
                target_id="model_001",
                type=LineageType.AGGREGATION
            ),
            LineageEdge(
                id="edge_4",
                source_id="model_001",
                target_id="pred_001",
                type=LineageType.TRANSFORMATION
            )
        ]
        
        for edge in edges:
            lineage_graph.add_edge(edge)


class TestLineageWithDataPipeline:
    """血缘追踪与数据管道集成测试"""
    
    def test_track_data_pipeline(self):
        """测试追踪数据管道血缘"""
        graph = LineageGraph()
        
        # 创建数据管道中的资产
        assets = {
            "source": DataAsset(
                id="source_001",
                name="wind_market_data",
                type=DataAssetType.STREAM,
                source="wind_api"
            ),
            "bronze": DataAsset(
                id="bronze_001",
                name="bronze_market_data",
                type=DataAssetType.TABLE,
                source="data_lake_bronze"
            ),
            "silver": DataAsset(
                id="silver_001",
                name="silver_market_data",
                type=DataAssetType.TABLE,
                source="data_lake_silver"
            ),
            "gold": DataAsset(
                id="gold_001",
                name="gold_features",
                type=DataAssetType.TABLE,
                source="data_lake_gold"
            ),
            "model_input": DataAsset(
                id="model_input_001",
                name="model_training_data",
                type=DataAssetType.TABLE,
                source="feature_store"
            )
        }
        
        # 添加节点（使用资产的id作为节点id）
        for asset in assets.values():
            node = LineageNode(id=asset.id, asset=asset)
            graph.add_node(node)
            
        # 添加数据管道边
        pipeline_edges = [
            ("source_001", "bronze_001", LineageType.COPY),
            ("bronze_001", "silver_001", LineageType.TRANSFORMATION),
            ("silver_001", "gold_001", LineageType.AGGREGATION),
            ("gold_001", "model_input_001", LineageType.FILTER)
        ]
        
        for i, (source, target, edge_type) in enumerate(pipeline_edges):
            edge = LineageEdge(
                id=f"pipeline_edge_{i}",
                source_id=source,
                target_id=target,
                type=edge_type
            )
            graph.add_edge(edge)
            
        # 验证数据管道血缘
        downstream = graph.get_downstream("source_001")
        assert len(downstream) == 4
        
        # 验证数据湖分层
        path = graph.find_path("source_001", "model_input_001")
        assert path is not None
        assert path.distance == 4
        
    def test_track_feature_lineage(self):
        """测试特征血缘追踪"""
        graph = LineageGraph()
        
        # 创建特征相关资产
        raw_data = DataAsset(
            id="raw_prices",
            name="raw_price_data",
            type=DataAssetType.TABLE
        )
        
        features = [
            DataAsset(id="feat_ma5", name="ma5_feature", type=DataAssetType.COLUMN),
            DataAsset(id="feat_ma10", name="ma10_feature", type=DataAssetType.COLUMN),
            DataAsset(id="feat_rsi", name="rsi_feature", type=DataAssetType.COLUMN),
            DataAsset(id="feat_macd", name="macd_feature", type=DataAssetType.COLUMN)
        ]
        
        feature_set = DataAsset(
            id="feature_set_001",
            name="technical_feature_set",
            type=DataAssetType.TABLE
        )
        
        # 添加节点
        graph.add_node(LineageNode(id="raw_prices", asset=raw_data))
        for feat in features:
            graph.add_node(LineageNode(id=feat.id, asset=feat))
        graph.add_node(LineageNode(id="feature_set_001", asset=feature_set))
        
        # 添加特征派生关系
        for feat in features:
            edge = LineageEdge(
                id=f"edge_{feat.id}",
                source_id="raw_prices",
                target_id=feat.id,
                type=LineageType.DERIVATION
            )
            graph.add_edge(edge)
            
            # 特征聚合到特征集
            edge = LineageEdge(
                id=f"edge_agg_{feat.id}",
                source_id=feat.id,
                target_id="feature_set_001",
                type=LineageType.AGGREGATION
            )
            graph.add_edge(edge)
            
        # 验证特征血缘
        impact = graph.analyze_impact("raw_prices")
        assert impact.total_affected == 5  # 4 features + 1 feature set
        

class TestLineageDataModels:
    """血缘数据模型测试"""
    
    def test_data_asset_serialization(self):
        """测试数据资产序列化"""
        asset = DataAsset(
            id="test_001",
            name="test_asset",
            type=DataAssetType.TABLE,
            source="test_source",
            schema={"col1": "int", "col2": "str"},
            metadata={"owner": "test_team"}
        )
        
        # 序列化
        data = asset.to_dict()
        
        # 验证
        assert data["id"] == "test_001"
        assert data["name"] == "test_asset"
        assert data["type"] == "table"
        assert data["source"] == "test_source"
        
        # 反序列化
        restored = DataAsset.from_dict(data)
        assert restored.id == asset.id
        assert restored.name == asset.name
        assert restored.type == asset.type
        
    def test_transformation_serialization(self):
        """测试转换序列化"""
        trans = Transformation(
            id="trans_001",
            name="test_transform",
            type=LineageType.TRANSFORMATION,
            sql="SELECT * FROM table WHERE condition",
            description="Test transformation"
        )
        
        # 序列化
        data = trans.to_dict()
        
        # 验证
        assert data["id"] == "trans_001"
        assert data["sql"] == "SELECT * FROM table WHERE condition"
        
        # 反序列化
        restored = Transformation.from_dict(data)
        assert restored.id == trans.id
        assert restored.sql == trans.sql


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
