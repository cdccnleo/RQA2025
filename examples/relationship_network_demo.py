"""
关系网络分析模块使用示例

展示如何使用RelationshipNetwork进行策略关系网络分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.trading.advanced_analysis.relationship_network import RelationshipNetwork


def create_sample_strategies():
    """创建示例策略数据"""
    strategies = []

    # 生成时间序列
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

    # 策略1：趋势跟踪策略
    np.random.seed(42)
    returns1 = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
    strategies.append({
        'id': 'trend_following',
        'returns': returns1,
        'risk_metrics': {
            'volatility': 0.02,
            'max_drawdown': 0.05,
            'sharpe_ratio': 1.1,
            'var_95': -0.03
        },
        'trades': pd.DataFrame({
            'size': np.random.randint(100, 1000, 50),
            'pnl': np.random.normal(10, 50, 50),
            'holding_time': np.random.randint(1, 10, 50)
        })
    })

    # 策略2：均值回归策略
    np.random.seed(43)
    returns2 = pd.Series(np.random.normal(0.0008, 0.015, len(dates)), index=dates)
    strategies.append({
        'id': 'mean_reversion',
        'returns': returns2,
        'risk_metrics': {
            'volatility': 0.015,
            'max_drawdown': 0.03,
            'sharpe_ratio': 1.3,
            'var_95': -0.025
        },
        'trades': pd.DataFrame({
            'size': np.random.randint(50, 500, 40),
            'pnl': np.random.normal(5, 30, 40),
            'holding_time': np.random.randint(1, 5, 40)
        })
    })

    # 策略3：动量策略
    np.random.seed(44)
    returns3 = pd.Series(np.random.normal(0.0012, 0.025, len(dates)), index=dates)
    strategies.append({
        'id': 'momentum',
        'returns': returns3,
        'risk_metrics': {
            'volatility': 0.025,
            'max_drawdown': 0.08,
            'sharpe_ratio': 0.9,
            'var_95': -0.04
        },
        'trades': pd.DataFrame({
            'size': np.random.randint(200, 1500, 60),
            'pnl': np.random.normal(15, 80, 60),
            'holding_time': np.random.randint(2, 15, 60)
        })
    })

    # 策略4：套利策略
    np.random.seed(45)
    returns4 = pd.Series(np.random.normal(0.0005, 0.008, len(dates)), index=dates)
    strategies.append({
        'id': 'arbitrage',
        'returns': returns4,
        'risk_metrics': {
            'volatility': 0.008,
            'max_drawdown': 0.01,
            'sharpe_ratio': 1.5,
            'var_95': -0.012
        },
        'trades': pd.DataFrame({
            'size': np.random.randint(1000, 5000, 30),
            'pnl': np.random.normal(2, 10, 30),
            'holding_time': np.random.randint(1, 3, 30)
        })
    })

    # 策略5：高频策略
    np.random.seed(46)
    returns5 = pd.Series(np.random.normal(0.0003, 0.005, len(dates)), index=dates)
    strategies.append({
        'id': 'high_frequency',
        'returns': returns5,
        'risk_metrics': {
            'volatility': 0.005,
            'max_drawdown': 0.005,
            'sharpe_ratio': 2.0,
            'var_95': -0.008
        },
        'trades': pd.DataFrame({
            'size': np.random.randint(500, 2000, 100),
            'pnl': np.random.normal(1, 5, 100),
            'holding_time': np.random.randint(1, 2, 100)
        })
    })

    return strategies


def main():
    """主函数"""
    print("=== 关系网络分析示例 ===\n")

    # 1. 创建示例策略数据
    print("1. 创建示例策略数据...")
    strategies = create_sample_strategies()
    print(f"   创建了 {len(strategies)} 个策略")

    # 2. 初始化关系网络分析器
    print("\n2. 初始化关系网络分析器...")
    network_analyzer = RelationshipNetwork(
        similarity_threshold=0.2,  # 较低的阈值以显示更多连接
        network_type='weighted',
        layout_method='spring',
        min_connections=1
    )

    # 3. 构建关系网络
    print("\n3. 构建关系网络...")
    success = network_analyzer.build_network(strategies)
    if success:
        print(
            f"   成功构建网络，包含 {network_analyzer.graph.number_of_nodes()} 个节点和 {network_analyzer.graph.number_of_edges()} 条边")
    else:
        print("   网络构建失败")
        return

    # 4. 获取网络指标
    print("\n4. 分析网络指标...")
    metrics = network_analyzer.get_network_metrics()
    print(f"   网络密度: {metrics['density']:.3f}")
    print(f"   平均聚类系数: {metrics['average_clustering']:.3f}")
    print(f"   连通分量数: {metrics['connected_components']}")

    # 5. 查找中心策略
    print("\n5. 查找中心策略...")
    central_strategies = network_analyzer.find_central_strategies(top_k=3)
    print("   中心策略排名:")
    for i, (strategy_id, centrality) in enumerate(central_strategies, 1):
        print(f"   {i}. {strategy_id}: {centrality:.3f}")

    # 6. 检测社区
    print("\n6. 检测社区结构...")
    communities = network_analyzer.detect_communities(method='louvain')
    print(f"   检测到 {len(communities)} 个社区:")
    for community_id, members in communities.items():
        print(f"   {community_id}: {members}")

    # 7. 查找相似策略
    print("\n7. 查找相似策略...")
    target_strategy = 'trend_following'
    similar_strategies = network_analyzer.find_similar_strategies(target_strategy, top_k=2)
    print(f"   与 {target_strategy} 最相似的策略:")
    for strategy_id, similarity in similar_strategies:
        print(f"   {strategy_id}: {similarity:.3f}")

    # 8. 获取网络摘要
    print("\n8. 网络摘要信息...")
    summary = network_analyzer.get_network_summary()
    print(f"   网络信息: {summary['network_info']}")
    print(f"   中心性信息: {summary['centrality_info']}")
    print(f"   聚类信息: {summary['clustering_info']}")

    # 9. 导出网络数据
    print("\n9. 导出网络数据...")
    export_success = network_analyzer.export_network_data("relationship_network_demo")
    if export_success:
        print("   网络数据已导出到 relationship_network_demo_nodes.csv 和 relationship_network_demo_edges.csv")

    # 10. 可视化网络（可选）
    print("\n10. 网络可视化...")
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 可视化网络
        network_analyzer.visualize_network(
            save_path="relationship_network_visualization.png",
            figsize=(10, 8),
            node_size=500,
            font_size=10
        )
        print("   网络图已保存为 relationship_network_visualization.png")
    except Exception as e:
        print(f"   可视化失败: {e}")

    print("\n=== 示例完成 ===")


if __name__ == "__main__":
    main()
