"""
api_flow_diagram_generator 模块

提供 api_flow_diagram_generator 相关功能和接口。
"""

import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API业务流程图生成器
生成API调用流程的可视化图表
"""


@dataclass
class APIFlowNode:
    """API流程节点"""
    id: str
    label: str
    node_type: str  # start, process, decision, end, api_call, database, cache
    description: str = ""
    position: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIFlowEdge:
    """API流程边"""
    source: str
    target: str
    label: str = ""
    edge_type: str = "normal"  # normal, success, error, conditional
    condition: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIFlowDiagram:
    """API流程图"""
    id: str
    title: str
    description: str
    nodes: List[APIFlowNode] = field(default_factory=list)
    edges: List[APIFlowEdge] = field(default_factory=list)
    layout: str = "horizontal"  # horizontal, vertical, radial
    theme: str = "default"


class APIFlowDiagramGenerator:
    """API流程图生成器"""

    def __init__(self):
        self.diagrams: Dict[str, APIFlowDiagram] = {}

    def create_data_service_flow(self) -> APIFlowDiagram:
        """创建数据服务流程图"""
        diagram = APIFlowDiagram(
            id="data_service_flow",
            title="数据服务API调用流程",
            description="RQA2025数据服务API的完整调用流程",
            layout="horizontal"
        )

        # 定义节点
        nodes = [
            APIFlowNode(
                id="start",
                label="开始",
                node_type="start",
                description="API请求开始",
                position={"x": 100, "y": 100}
            ),
            APIFlowNode(
                id="auth_check",
                label="身份验证",
                node_type="decision",
                description="检查请求身份验证",
                position={"x": 250, "y": 100}
            ),
            APIFlowNode(
                id="rate_limit",
                label="频率限制检查",
                node_type="decision",
                description="检查请求频率是否超限",
                position={"x": 400, "y": 100}
            ),
            APIFlowNode(
                id="param_validation",
                label="参数验证",
                node_type="process",
                description="验证请求参数的有效性",
                position={"x": 550, "y": 100}
            ),
            APIFlowNode(
                id="cache_check",
                label="缓存检查",
                node_type="decision",
                description="检查数据是否在缓存中",
                position={"x": 700, "y": 100}
            ),
            APIFlowNode(
                id="cache_hit",
                label="缓存命中",
                node_type="process",
                description="从缓存获取数据",
                position={"x": 850, "y": 200}
            ),
            APIFlowNode(
                id="db_query",
                label="数据库查询",
                node_type="database",
                description="从数据库查询数据",
                position={"x": 850, "y": 100}
            ),
            APIFlowNode(
                id="data_processing",
                label="数据处理",
                node_type="process",
                description="处理和格式化数据",
                position={"x": 1000, "y": 100}
            ),
            APIFlowNode(
                id="cache_update",
                label="缓存更新",
                node_type="process",
                description="更新缓存数据",
                position={"x": 1150, "y": 100}
            ),
            APIFlowNode(
                id="response_format",
                label="响应格式化",
                node_type="process",
                description="格式化API响应",
                position={"x": 1300, "y": 100}
            ),
            APIFlowNode(
                id="success_response",
                label="成功响应",
                node_type="end",
                description="返回成功响应",
                position={"x": 1450, "y": 100}
            ),
            APIFlowNode(
                id="auth_error",
                label="认证错误",
                node_type="end",
                description="返回认证错误",
                position={"x": 250, "y": 250}
            ),
            APIFlowNode(
                id="rate_limit_error",
                label="频率限制错误",
                node_type="end",
                description="返回频率限制错误",
                position={"x": 400, "y": 250}
            ),
            APIFlowNode(
                id="validation_error",
                label="验证错误",
                node_type="end",
                description="返回参数验证错误",
                position={"x": 550, "y": 250}
            )
        ]

        # 定义边
        edges = [
            APIFlowEdge("start", "auth_check", "接收请求"),
            APIFlowEdge("auth_check", "rate_limit", "认证通过", "success"),
            APIFlowEdge("auth_check", "auth_error", "认证失败", "error"),
            APIFlowEdge("rate_limit", "param_validation", "频率正常", "success"),
            APIFlowEdge("rate_limit", "rate_limit_error", "频率超限", "error"),
            APIFlowEdge("param_validation", "cache_check", "参数有效"),
            APIFlowEdge("param_validation", "validation_error", "参数无效", "error"),
            APIFlowEdge("cache_check", "cache_hit", "缓存命中", "success"),
            APIFlowEdge("cache_check", "db_query", "缓存未命中", "normal"),
            APIFlowEdge("cache_hit", "data_processing", "获取缓存数据"),
            APIFlowEdge("db_query", "data_processing", "查询数据库"),
            APIFlowEdge("data_processing", "cache_update", "数据处理完成"),
            APIFlowEdge("cache_update", "response_format", "缓存更新完成"),
            APIFlowEdge("response_format", "success_response", "响应格式化完成")
        ]

        diagram.nodes = nodes
        diagram.edges = edges

        return diagram

    def create_trading_flow(self) -> APIFlowDiagram:
        """创建交易流程图"""
        diagram = APIFlowDiagram(
            id="trading_flow",
            title="交易策略执行流程",
            description="量化交易策略的执行流程",
            layout="vertical"
        )

        nodes = [
            APIFlowNode(
                id="strategy_request",
                label="策略执行请求",
                node_type="start",
                description="接收交易策略执行请求",
                position={"x": 100, "y": 100}
            ),
            APIFlowNode(
                id="auth_verify",
                label="权限验证",
                node_type="decision",
                description="验证用户交易权限",
                position={"x": 100, "y": 200}
            ),
            APIFlowNode(
                id="risk_check",
                label="风险评估",
                node_type="decision",
                description="评估交易风险",
                position={"x": 100, "y": 300}
            ),
            APIFlowNode(
                id="market_data",
                label="获取市场数据",
                node_type="api_call",
                description="获取最新市场数据",
                position={"x": 100, "y": 400}
            ),
            APIFlowNode(
                id="signal_generate",
                label="信号生成",
                node_type="process",
                description="根据策略生成交易信号",
                position={"x": 100, "y": 500}
            ),
            APIFlowNode(
                id="position_calc",
                label="仓位计算",
                node_type="process",
                description="计算交易仓位",
                position={"x": 100, "y": 600}
            ),
            APIFlowNode(
                id="order_validate",
                label="订单验证",
                node_type="decision",
                description="验证订单参数",
                position={"x": 100, "y": 700}
            ),
            APIFlowNode(
                id="order_submit",
                label="提交订单",
                node_type="api_call",
                description="向交易所提交订单",
                position={"x": 100, "y": 800}
            ),
            APIFlowNode(
                id="order_confirm",
                label="订单确认",
                node_type="process",
                description="确认订单执行结果",
                position={"x": 100, "y": 900}
            ),
            APIFlowNode(
                id="success",
                label="执行成功",
                node_type="end",
                description="交易策略执行成功",
                position={"x": 100, "y": 1000}
            ),
            APIFlowNode(
                id="auth_failure",
                label="权限不足",
                node_type="end",
                description="用户权限不足",
                position={"x": 300, "y": 200}
            ),
            APIFlowNode(
                id="risk_rejection",
                label="风险拒绝",
                node_type="end",
                description="交易风险过高被拒绝",
                position={"x": 300, "y": 300}
            ),
            APIFlowNode(
                id="order_failure",
                label="订单失败",
                node_type="end",
                description="订单提交失败",
                position={"x": 300, "y": 700}
            )
        ]

        edges = [
            APIFlowEdge("strategy_request", "auth_verify", "接收请求"),
            APIFlowEdge("auth_verify", "risk_check", "权限通过", "success"),
            APIFlowEdge("auth_verify", "auth_failure", "权限不足", "error"),
            APIFlowEdge("risk_check", "market_data", "风险通过", "success"),
            APIFlowEdge("risk_check", "risk_rejection", "风险过高", "error"),
            APIFlowEdge("market_data", "signal_generate", "获取数据"),
            APIFlowEdge("signal_generate", "position_calc", "生成信号"),
            APIFlowEdge("position_calc", "order_validate", "计算仓位"),
            APIFlowEdge("order_validate", "order_submit", "验证通过", "success"),
            APIFlowEdge("order_validate", "order_failure", "验证失败", "error"),
            APIFlowEdge("order_submit", "order_confirm", "提交订单"),
            APIFlowEdge("order_confirm", "success", "确认成功")
        ]

        diagram.nodes = nodes
        diagram.edges = edges

        return diagram

    def create_feature_engineering_flow(self) -> APIFlowDiagram:
        """创建特征工程流程图"""
        diagram = APIFlowDiagram(
            id="feature_engineering_flow",
            title="特征工程处理流程",
            description="技术指标和情感分析的处理流程",
            layout="horizontal"
        )

        nodes = [
            APIFlowNode(
                id="feature_request",
                label="特征计算请求",
                node_type="start",
                description="接收特征计算请求",
                position={"x": 100, "y": 100}
            ),
            APIFlowNode(
                id="input_validation",
                label="输入验证",
                node_type="process",
                description="验证输入参数",
                position={"x": 250, "y": 100}
            ),
            APIFlowNode(
                id="data_retrieval",
                label="数据获取",
                node_type="database",
                description="获取历史数据",
                position={"x": 400, "y": 100}
            ),
            APIFlowNode(
                id="technical_analysis",
                label="技术分析",
                node_type="process",
                description="计算技术指标",
                position={"x": 550, "y": 100}
            ),
            APIFlowNode(
                id="sentiment_analysis",
                label="情感分析",
                node_type="process",
                description="分析市场情绪",
                position={"x": 550, "y": 200}
            ),
            APIFlowNode(
                id="feature_combination",
                label="特征组合",
                node_type="process",
                description="组合多种特征",
                position={"x": 700, "y": 100}
            ),
            APIFlowNode(
                id="feature_validation",
                label="特征验证",
                node_type="decision",
                description="验证特征质量",
                position={"x": 850, "y": 100}
            ),
            APIFlowNode(
                id="feature_storage",
                label="特征存储",
                node_type="database",
                description="存储计算结果",
                position={"x": 1000, "y": 100}
            ),
            APIFlowNode(
                id="response_generation",
                label="响应生成",
                node_type="process",
                description="生成API响应",
                position={"x": 1150, "y": 100}
            ),
            APIFlowNode(
                id="feature_success",
                label="特征计算成功",
                node_type="end",
                description="返回计算结果",
                position={"x": 1300, "y": 100}
            ),
            APIFlowNode(
                id="validation_error",
                label="验证失败",
                node_type="end",
                description="输入验证失败",
                position={"x": 250, "y": 250}
            ),
            APIFlowNode(
                id="feature_quality_error",
                label="特征质量不足",
                node_type="end",
                description="特征质量不符合要求",
                position={"x": 850, "y": 250}
            )
        ]

        edges = [
            APIFlowEdge("feature_request", "input_validation", "接收请求"),
            APIFlowEdge("input_validation", "data_retrieval", "验证通过"),
            APIFlowEdge("input_validation", "validation_error", "验证失败", "error"),
            APIFlowEdge("data_retrieval", "technical_analysis", "获取数据"),
            APIFlowEdge("data_retrieval", "sentiment_analysis", "获取数据"),
            APIFlowEdge("technical_analysis", "feature_combination", "技术指标计算完成"),
            APIFlowEdge("sentiment_analysis", "feature_combination", "情感分析完成"),
            APIFlowEdge("feature_combination", "feature_validation", "特征组合完成"),
            APIFlowEdge("feature_validation", "feature_storage", "质量合格", "success"),
            APIFlowEdge("feature_validation", "feature_quality_error", "质量不足", "error"),
            APIFlowEdge("feature_storage", "response_generation", "存储完成"),
            APIFlowEdge("response_generation", "feature_success", "响应生成完成")
        ]

        diagram = APIFlowDiagram(
            id="feature_engineering_flow",
            title="特征工程处理流程",
            description="技术指标和情感分析的处理流程",
            layout="horizontal"
        )
        diagram.nodes = nodes
        diagram.edges = edges

        return diagram

    def generate_all_flows(self) -> Dict[str, APIFlowDiagram]:
        """生成所有流程图"""
        flows = {
            "data_service": self.create_data_service_flow(),
            "trading": self.create_trading_flow(),
            "feature_engineering": self.create_feature_engineering_flow()
        }

        self.diagrams = flows
        return flows

    def export_to_mermaid(self, output_dir: str = "docs/api/flows"):
        """导出为Mermaid格式"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        flows = self.generate_all_flows()

        for flow_id, diagram in flows.items():
            mermaid_content = self._generate_mermaid(diagram)
            output_file = output_path / f"{flow_id}_flow.md"

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# {diagram.title}\n\n")
                f.write(f"{diagram.description}\n\n")
                f.write("```mermaid\n")
                f.write(mermaid_content)
                f.write("\n```\n")

            print(f"流程图已导出: {output_file}")

    def _generate_mermaid(self, diagram: APIFlowDiagram) -> str:
        """生成Mermaid图表代码"""
        lines = []

        # 图表方向
        if diagram.layout == "horizontal":
            lines.append("graph TD")
        elif diagram.layout == "vertical":
            lines.append("graph TB")
        else:
            lines.append("graph TD")

        lines.append("")

        # 定义节点
        for node in diagram.nodes:
            if node.node_type == "start":
                lines.append(f"    {node.id}([{node.label}])")
            elif node.node_type == "end":
                lines.append(f"    {node.id}([{node.label}])")
            elif node.node_type == "decision":
                lines.append(f"    {node.id}{{{node.label}}}")
            elif node.node_type == "process":
                lines.append(f"    {node.id}[{node.label}]")
            elif node.node_type == "database":
                lines.append(f"    {node.id}[({node.label})]")
            elif node.node_type == "api_call":
                lines.append(f"    {node.id}[/{node.label}/]")
            else:
                lines.append(f"    {node.id}[{node.label}]")

        lines.append("")

        # 定义边
        for edge in diagram.edges:
            if edge.edge_type == "success":
                lines.append(f"    {edge.source} -->|{edge.label}| {edge.target}")
            elif edge.edge_type == "error":
                lines.append(f"    {edge.source} -.->|{edge.label}| {edge.target}")
            elif edge.edge_type == "conditional":
                lines.append(f"    {edge.source} ==>|{edge.label}| {edge.target}")
            else:
                lines.append(f"    {edge.source} -->|{edge.label}| {edge.target}")

        return "\n".join(lines)

    def export_to_json(self, output_dir: str = "docs/api/flows"):
        """导出为JSON格式"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        flows = self.generate_all_flows()

        for flow_id, diagram in flows.items():
            flow_data = {
                "id": diagram.id,
                "title": diagram.title,
                "description": diagram.description,
                "layout": diagram.layout,
                "theme": diagram.theme,
                "nodes": [
                    {
                        "id": node.id,
                        "label": node.label,
                        "type": node.node_type,
                        "description": node.description,
                        "position": node.position,
                        "metadata": node.metadata
                    }
                    for node in diagram.nodes
                ],
                "edges": [
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "label": edge.label,
                        "type": edge.edge_type,
                        "condition": edge.condition,
                        "metadata": edge.metadata
                    }
                    for edge in diagram.edges
                ]
            }

            output_file = output_path / f"{flow_id}_flow.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(flow_data, f, indent=2, ensure_ascii=False)

            print(f"流程图JSON已导出: {output_file}")

    def get_flow_statistics(self) -> Dict[str, Any]:
        """获取流程图统计信息"""
        flows = self.generate_all_flows()

        stats = {
            "total_flows": len(flows),
            "total_nodes": 0,
            "total_edges": 0,
            "node_types": {},
            "edge_types": {},
            "flows_info": []
        }

        for flow_id, diagram in flows.items():
            flow_info = {
                "id": flow_id,
                "title": diagram.title,
                "nodes_count": len(diagram.nodes),
                "edges_count": len(diagram.edges),
                "layout": diagram.layout
            }
            stats["flows_info"].append(flow_info)

            stats["total_nodes"] += len(diagram.nodes)
            stats["total_edges"] += len(diagram.edges)

            # 统计节点类型
            for node in diagram.nodes:
                node_type = node.node_type
                stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1

            # 统计边类型
            for edge in diagram.edges:
                edge_type = edge.edge_type
                stats["edge_types"][edge_type] = stats["edge_types"].get(edge_type, 0) + 1

        return stats


if __name__ == "__main__":
    # 生成API业务流程图
    print("初始化API流程图生成器...")

    generator = APIFlowDiagramGenerator()

    # 生成所有流程图
    print("生成API业务流程图...")
    flows = generator.generate_all_flows()

    print(f"生成了 {len(flows)} 个流程图")

    # 导出为Mermaid格式
    print("导出Mermaid格式...")
    generator.export_to_mermaid()

    # 导出为JSON格式
    print("导出JSON格式...")
    generator.export_to_json()

    # 获取统计信息
    stats = generator.get_flow_statistics()

    print("\n📊 流程图统计:")
    print(f"   📈 总流程数: {stats['total_flows']} 个")
    print(f"   🔵 总节点数: {stats['total_nodes']} 个")
    print(f"   ➡️ 总边数: {stats['total_edges']} 个")
    print(f"   📋 节点类型分布: {stats['node_types']}")
    print(f"   🔗 边类型分布: {stats['edge_types']}")

    print("\n📋 流程图信息:")
    for flow_info in stats['flows_info']:
        print(
            f"   • {flow_info['title']} ({flow_info['nodes_count']} 节点, {flow_info['edges_count']} 边)")

    print("\\n🎉 API业务流程图生成完成！")
