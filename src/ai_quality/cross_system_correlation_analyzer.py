"""
跨系统质量相关性分析系统

分析不同系统、模块间的质量指标相关性和影响关系：
1. 相关性网络构建 - 构建系统间的质量相关性网络
2. 影响传播分析 - 分析质量问题的传播路径
3. 根因定位 - 多系统协作的根本原因分析
4. 协同优化建议 - 基于相关性的系统协同优化
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mutual_info_score
from scipy.stats import pearsonr, spearmanr
from scipy.sparse.csgraph import connected_components
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CorrelationNetwork:
    """相关性网络"""

    def __init__(self):
        self.nodes = {}  # 系统节点
        self.edges = {}  # 相关性边
        self.correlation_matrix = None
        self.network_graph = None

    def add_system(self, system_id: str, system_info: Dict[str, Any]):
        """添加系统节点"""
        self.nodes[system_id] = {
            'info': system_info,
            'metrics': {},
            'connections': set()
        }

    def add_correlation(self, system_a: str, system_b: str,
                       correlation_data: Dict[str, Any]):
        """添加系统间相关性"""
        edge_key = tuple(sorted([system_a, system_b]))

        self.edges[edge_key] = {
            'systems': {system_a, system_b},
            'correlation_strength': correlation_data.get('strength', 0),
            'correlation_type': correlation_data.get('type', 'unknown'),
            'influence_direction': correlation_data.get('direction', 'bidirectional'),
            'metrics': correlation_data.get('metrics', {}),
            'confidence': correlation_data.get('confidence', 0)
        }

        # 更新节点连接
        self.nodes[system_a]['connections'].add(system_b)
        self.nodes[system_b]['connections'].add(system_a)

    def build_correlation_matrix(self) -> np.ndarray:
        """构建相关性矩阵"""
        n_systems = len(self.nodes)
        system_ids = list(self.nodes.keys())

        matrix = np.zeros((n_systems, n_systems))

        for i, sys_a in enumerate(system_ids):
            for j, sys_b in enumerate(system_ids):
                if i != j:
                    edge_key = tuple(sorted([sys_a, sys_b]))
                    if edge_key in self.edges:
                        matrix[i, j] = self.edges[edge_key]['correlation_strength']

        self.correlation_matrix = matrix
        return matrix

    def find_connected_components(self) -> List[List[str]]:
        """寻找连通组件"""
        if self.correlation_matrix is None:
            self.build_correlation_matrix()

        # 将相关性矩阵转换为邻接矩阵（阈值=0.3）
        adjacency = (np.abs(self.correlation_matrix) > 0.3).astype(int)

        # 寻找连通组件
        n_components, labels = connected_components(adjacency)

        system_ids = list(self.nodes.keys())
        components = []

        for i in range(n_components):
            component = [system_ids[j] for j in range(len(system_ids)) if labels[j] == i]
            if len(component) > 1:  # 只保留包含多个系统的组件
                components.append(component)

        return components

    def get_system_centrality(self) -> Dict[str, float]:
        """计算系统中心性"""
        if not self.network_graph:
            self._build_network_graph()

        # 计算度中心性
        degree_centrality = nx.degree_centrality(self.network_graph)

        # 计算介数中心性
        betweenness_centrality = nx.betweenness_centrality(self.network_graph)

        # 计算特征向量中心性
        try:
            eigenvector_centrality = nx.eigenvector_centrality(self.network_graph)
        except:
            eigenvector_centrality = {node: 0 for node in self.network_graph.nodes()}

        # 综合中心性评分
        centrality_scores = {}
        for node in self.network_graph.nodes():
            score = (
                degree_centrality.get(node, 0) * 0.4 +
                betweenness_centrality.get(node, 0) * 0.4 +
                eigenvector_centrality.get(node, 0) * 0.2
            )
            centrality_scores[node] = score

        return centrality_scores

    def _build_network_graph(self):
        """构建网络图"""
        self.network_graph = nx.Graph()

        # 添加节点
        for system_id, system_data in self.nodes.items():
            self.network_graph.add_node(system_id, **system_data['info'])

        # 添加边
        for edge_key, edge_data in self.edges.items():
            sys_a, sys_b = edge_key
            self.network_graph.add_edge(
                sys_a, sys_b,
                weight=edge_data['correlation_strength'],
                **edge_data
            )


class CrossSystemCorrelationAnalyzer:
    """跨系统质量相关性分析器"""

    def __init__(self, model_path: str = "models/cross_system_analyzer"):
        self.model_path = model_path
        self.correlation_network = CorrelationNetwork()
        self.system_metrics_history = {}
        self.correlation_cache = {}
        self.analysis_cache = {}
        self.correlation_threshold = 0.3  # 相关性阈值

        # 相关性分析方法
        self.correlation_methods = {
            'pearson': self._pearson_correlation,
            'spearman': self._spearman_correlation,
            'mutual_info': self._mutual_information,
            'granger': self._granger_causality,
            'cross_correlation': self._cross_correlation
        }

    def add_system(self, system_id: str, system_info: Dict[str, Any],
                  metrics_history: pd.DataFrame):
        """
        添加系统及其指标历史

        Args:
            system_id: 系统ID
            system_info: 系统信息
            metrics_history: 指标历史数据
        """
        try:
            # 添加到相关性网络
            self.correlation_network.add_system(system_id, system_info)

            # 存储指标历史
            self.system_metrics_history[system_id] = metrics_history.copy()

            # 清理缓存
            self.correlation_cache = {}
            self.analysis_cache = {}

            logger.info(f"成功添加系统: {system_id}")

        except Exception as e:
            logger.error(f"添加系统失败 {system_id}: {e}")

    def analyze_cross_system_correlations(self, analysis_period: timedelta = timedelta(days=30),
                                        correlation_methods: List[str] = None) -> Dict[str, Any]:
        """
        分析跨系统相关性

        Args:
            analysis_period: 分析时间周期
            correlation_methods: 使用的相关性分析方法

        Returns:
            相关性分析结果
        """
        try:
            if not correlation_methods:
                correlation_methods = ['pearson', 'spearman', 'mutual_info']

            # 确定分析时间范围
            end_time = datetime.now()
            start_time = end_time - analysis_period

            correlation_results = {}

            # 获取所有系统对
            system_pairs = self._get_system_pairs()

            for sys_a, sys_b in system_pairs:
                pair_key = f"{sys_a}_{sys_b}"

                # 分析系统对的相关性
                pair_correlations = self._analyze_system_pair_correlation(
                    sys_a, sys_b, start_time, end_time, correlation_methods
                )

                if pair_correlations:
                    correlation_results[pair_key] = {
                        'systems': [sys_a, sys_b],
                        'correlations': pair_correlations,
                        'overall_strength': self._calculate_overall_correlation_strength(pair_correlations),
                        'dominant_method': self._get_dominant_correlation_method(pair_correlations),
                        'confidence': self._calculate_correlation_confidence(pair_correlations)
                    }

                    # 添加到相关性网络
                    self.correlation_network.add_correlation(
                        sys_a, sys_b,
                        {
                            'strength': correlation_results[pair_key]['overall_strength'],
                            'type': correlation_results[pair_key]['dominant_method'],
                            'direction': self._determine_correlation_direction(pair_correlations),
                            'metrics': pair_correlations,
                            'confidence': correlation_results[pair_key]['confidence']
                        }
                    )

            # 构建相关性网络
            correlation_matrix = self.correlation_network.build_correlation_matrix()

            # 分析网络结构
            network_analysis = self._analyze_correlation_network()

            # 生成分析报告
            analysis_report = {
                'analysis_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration_days': analysis_period.days
                },
                'correlation_results': correlation_results,
                'correlation_matrix': correlation_matrix.tolist(),
                'network_analysis': network_analysis,
                'correlation_methods_used': correlation_methods,
                'significant_correlations': self._identify_significant_correlations(correlation_results),
                'timestamp': datetime.now()
            }

            # 缓存分析结果
            cache_key = f"{start_time}_{end_time}_{'_'.join(correlation_methods)}"
            self.analysis_cache[cache_key] = analysis_report

            return analysis_report

        except Exception as e:
            logger.error(f"跨系统相关性分析失败: {e}")
            return {'error': str(e)}

    def analyze_influence_propagation(self, initial_system: str, initial_issue: Dict[str, Any],
                                    max_depth: int = 3) -> Dict[str, Any]:
        """
        分析影响传播路径

        Args:
            initial_system: 初始受影响的系统
            initial_issue: 初始问题描述
            max_depth: 最大传播深度

        Returns:
            影响传播分析结果
        """
        try:
            propagation_paths = []
            affected_systems = set([initial_system])
            current_depth = 0

            # 广度优先搜索影响传播
            while current_depth < max_depth:
                new_affected = set()

                for system in affected_systems:
                    # 查找与当前系统高度相关的其他系统
                    related_systems = self._find_highly_related_systems(system)

                    for related_system in related_systems:
                        if related_system not in affected_systems:
                            # 计算传播概率和影响程度
                            propagation_info = self._calculate_propagation_probability(
                                initial_system, system, related_system, initial_issue
                            )

                            if propagation_info['probability'] > 0.3:  # 传播概率阈值
                                propagation_paths.append({
                                    'from_system': system,
                                    'to_system': related_system,
                                    'depth': current_depth + 1,
                                    'propagation_probability': propagation_info['probability'],
                                    'influence_strength': propagation_info['influence_strength'],
                                    'estimated_impact': propagation_info['estimated_impact'],
                                    'time_to_impact': propagation_info['time_to_impact']
                                })

                                new_affected.add(related_system)

                if not new_affected:
                    break  # 没有新的受影响系统

                affected_systems.update(new_affected)
                current_depth += 1

            # 分析整体影响
            overall_impact = self._analyze_overall_propagation_impact(
                initial_system, affected_systems, propagation_paths
            )

            return {
                'initial_system': initial_system,
                'initial_issue': initial_issue,
                'propagation_paths': propagation_paths,
                'affected_systems': list(affected_systems),
                'max_depth_reached': current_depth,
                'overall_impact': overall_impact,
                'containment_suggestions': self._generate_containment_suggestions(
                    initial_system, affected_systems, propagation_paths
                )
            }

        except Exception as e:
            logger.error(f"影响传播分析失败: {e}")
            return {'error': str(e)}

    def generate_collaborative_optimization_plan(self, optimization_goal: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成协同优化计划

        Args:
            optimization_goal: 优化目标

        Returns:
            协同优化计划
        """
        try:
            # 识别需要优化的系统
            target_systems = optimization_goal.get('target_systems', [])
            optimization_criteria = optimization_goal.get('criteria', {})

            # 分析系统间依赖关系
            system_dependencies = self._analyze_system_dependencies(target_systems)

            # 生成优化序列
            optimization_sequence = self._generate_optimization_sequence(
                target_systems, system_dependencies, optimization_criteria
            )

            # 计算协同效益
            collaborative_benefits = self._calculate_collaborative_benefits(
                optimization_sequence, optimization_criteria
            )

            # 生成详细计划
            optimization_plan = {
                'optimization_goal': optimization_goal,
                'target_systems': target_systems,
                'system_dependencies': system_dependencies,
                'optimization_sequence': optimization_sequence,
                'collaborative_benefits': collaborative_benefits,
                'implementation_plan': self._create_implementation_plan(optimization_sequence),
                'monitoring_plan': self._create_monitoring_plan(optimization_sequence),
                'rollback_plan': self._create_rollback_plan(optimization_sequence),
                'timestamp': datetime.now()
            }

            return optimization_plan

        except Exception as e:
            logger.error(f"生成协同优化计划失败: {e}")
            return {'error': str(e)}

    def _get_system_pairs(self) -> List[Tuple[str, str]]:
        """获取所有系统对"""
        system_ids = list(self.system_metrics_history.keys())
        pairs = []

        for i in range(len(system_ids)):
            for j in range(i + 1, len(system_ids)):
                pairs.append((system_ids[i], system_ids[j]))

        return pairs

    def _analyze_system_pair_correlation(self, sys_a: str, sys_b: str,
                                       start_time: datetime, end_time: datetime,
                                       methods: List[str]) -> Dict[str, Any]:
        """分析系统对的相关性"""
        try:
            # 获取两个系统在时间范围内的指标数据
            data_a = self._get_system_metrics_in_range(sys_a, start_time, end_time)
            data_b = self._get_system_metrics_in_range(sys_b, start_time, end_time)

            if data_a.empty or data_b.empty:
                return {}

            correlations = {}

            # 使用指定的方法计算相关性
            for method in methods:
                if method in self.correlation_methods:
                    correlation_result = self.correlation_methods[method](data_a, data_b)
                    if correlation_result:
                        correlations[method] = correlation_result

            return correlations

        except Exception as e:
            logger.error(f"系统对相关性分析失败 {sys_a}-{sys_b}: {e}")
            return {}

    def _get_system_metrics_in_range(self, system_id: str, start_time: datetime,
                                   end_time: datetime) -> pd.DataFrame:
        """获取系统在时间范围内的指标数据"""
        if system_id not in self.system_metrics_history:
            return pd.DataFrame()

        data = self.system_metrics_history[system_id]

        # 过滤时间范围
        if 'timestamp' in data.columns:
            data = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]

        return data

    def _pearson_correlation(self, data_a: pd.DataFrame, data_b: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """皮尔逊相关系数"""
        try:
            # 找到共同的指标
            common_columns = set(data_a.columns) & set(data_b.columns)
            common_columns.discard('timestamp')  # 排除时间戳

            if not common_columns:
                return None

            correlations = {}
            for col in common_columns:
                if col in data_a.columns and col in data_b.columns:
                    # 对齐数据
                    merged_data = pd.merge(data_a[['timestamp', col]],
                                         data_b[['timestamp', col]],
                                         on='timestamp', suffixes=('_a', '_b'))

                    if len(merged_data) > 10:  # 至少10个数据点
                        corr, p_value = pearsonr(merged_data[f'{col}_a'], merged_data[f'{col}_b'])
                        correlations[col] = {
                            'coefficient': float(corr),
                            'p_value': float(p_value),
                            'significance': 'significant' if p_value < 0.05 else 'not_significant'
                        }

            if correlations:
                # 计算平均相关性强度
                coeffs = [c['coefficient'] for c in correlations.values()]
                avg_strength = np.mean(np.abs(coeffs))

                return {
                    'method': 'pearson',
                    'correlations': correlations,
                    'average_strength': float(avg_strength),
                    'strongest_correlation': max(correlations.keys(),
                                                key=lambda k: abs(correlations[k]['coefficient']))
                }

            return None

        except Exception as e:
            logger.error(f"皮尔逊相关性计算失败: {e}")
            return None

    def _spearman_correlation(self, data_a: pd.DataFrame, data_b: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """斯皮尔曼等级相关"""
        try:
            # 实现类似皮尔逊相关的方法，但使用斯皮尔曼相关
            common_columns = set(data_a.columns) & set(data_b.columns)
            common_columns.discard('timestamp')

            if not common_columns:
                return None

            correlations = {}
            for col in common_columns:
                if col in data_a.columns and col in data_b.columns:
                    merged_data = pd.merge(data_a[['timestamp', col]],
                                         data_b[['timestamp', col]],
                                         on='timestamp', suffixes=('_a', '_b'))

                    if len(merged_data) > 10:
                        corr, p_value = spearmanr(merged_data[f'{col}_a'], merged_data[f'{col}_b'])
                        correlations[col] = {
                            'coefficient': float(corr),
                            'p_value': float(p_value),
                            'significance': 'significant' if p_value < 0.05 else 'not_significant'
                        }

            if correlations:
                coeffs = [c['coefficient'] for c in correlations.values()]
                avg_strength = np.mean(np.abs(coeffs))

                return {
                    'method': 'spearman',
                    'correlations': correlations,
                    'average_strength': float(avg_strength),
                    'strongest_correlation': max(correlations.keys(),
                                                key=lambda k: abs(correlations[k]['coefficient']))
                }

            return None

        except Exception as e:
            logger.error(f"斯皮尔曼相关性计算失败: {e}")
            return None

    def _mutual_information(self, data_a: pd.DataFrame, data_b: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """互信息"""
        try:
            common_columns = set(data_a.columns) & set(data_b.columns)
            common_columns.discard('timestamp')

            if not common_columns:
                return None

            correlations = {}
            for col in common_columns:
                if col in data_a.columns and col in data_b.columns:
                    merged_data = pd.merge(data_a[['timestamp', col]],
                                         data_b[['timestamp', col]],
                                         on='timestamp', suffixes=('_a', '_b'))

                    if len(merged_data) > 10:
                        # 离散化数据以计算互信息
                        data_a_discrete = pd.cut(merged_data[f'{col}_a'], bins=10, labels=False)
                        data_b_discrete = pd.cut(merged_data[f'{col}_b'], bins=10, labels=False)

                        mi_score = mutual_info_score(data_a_discrete, data_b_discrete)
                        correlations[col] = {
                            'mutual_information': float(mi_score),
                            'normalized_score': float(mi_score / np.log(10))  # 归一化
                        }

            if correlations:
                scores = [c['mutual_information'] for c in correlations.values()]
                avg_strength = np.mean(scores)

                return {
                    'method': 'mutual_info',
                    'correlations': correlations,
                    'average_strength': float(avg_strength),
                    'strongest_correlation': max(correlations.keys(),
                                                key=lambda k: correlations[k]['mutual_information'])
                }

            return None

        except Exception as e:
            logger.error(f"互信息计算失败: {e}")
            return None

    def _granger_causality(self, data_a: pd.DataFrame, data_b: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """格兰杰因果检验"""
        # 这里简化实现，实际应该使用statsmodels的grangercausalitytests
        return None  # 暂时返回None

    def _cross_correlation(self, data_a: pd.DataFrame, data_b: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """交叉相关分析"""
        try:
            common_columns = set(data_a.columns) & set(data_b.columns)
            common_columns.discard('timestamp')

            if not common_columns:
                return None

            correlations = {}
            for col in common_columns:
                if col in data_a.columns and col in data_b.columns:
                    merged_data = pd.merge(data_a[['timestamp', col]],
                                         data_b[['timestamp', col]],
                                         on='timestamp', suffixes=('_a', '_b'))

                    if len(merged_data) > 20:  # 需要足够的数据点
                        # 计算交叉相关
                        series_a = merged_data[f'{col}_a'].values
                        series_b = merged_data[f'{col}_b'].values

                        # 计算不同滞后的相关性
                        max_lag = min(10, len(series_a) // 4)
                        cross_corr = []

                        for lag in range(-max_lag, max_lag + 1):
                            if lag < 0:
                                corr = np.corrcoef(series_a[-lag:], series_b[:lag])[0, 1]
                            elif lag > 0:
                                corr = np.corrcoef(series_a[:-lag], series_b[lag:])[0, 1]
                            else:
                                corr = np.corrcoef(series_a, series_b)[0, 1]

                            cross_corr.append({
                                'lag': lag,
                                'correlation': float(corr)
                            })

                        # 找到最大相关性和对应的滞后
                        max_corr_info = max(cross_corr, key=lambda x: abs(x['correlation']))

                        correlations[col] = {
                            'max_correlation': max_corr_info['correlation'],
                            'optimal_lag': max_corr_info['lag'],
                            'cross_correlation': cross_corr
                        }

            if correlations:
                max_corrs = [c['max_correlation'] for c in correlations.values()]
                avg_strength = np.mean(np.abs(max_corrs))

                return {
                    'method': 'cross_correlation',
                    'correlations': correlations,
                    'average_strength': float(avg_strength),
                    'strongest_correlation': max(correlations.keys(),
                                                key=lambda k: abs(correlations[k]['max_correlation']))
                }

            return None

        except Exception as e:
            logger.error(f"交叉相关分析失败: {e}")
            return None

    def _calculate_overall_correlation_strength(self, correlations: Dict[str, Any]) -> float:
        """计算整体相关性强度"""
        try:
            strengths = []

            for method_result in correlations.values():
                if 'average_strength' in method_result:
                    strengths.append(method_result['average_strength'])
                elif 'correlations' in method_result:
                    # 计算各个指标的平均强度
                    method_corrs = method_result['correlations']
                    if method_corrs:
                        if 'coefficient' in list(method_corrs.values())[0]:
                            coeffs = [c['coefficient'] for c in method_corrs.values()]
                        elif 'mutual_information' in list(method_corrs.values())[0]:
                            coeffs = [c['mutual_information'] for c in method_corrs.values()]
                        elif 'max_correlation' in list(method_corrs.values())[0]:
                            coeffs = [c['max_correlation'] for c in method_corrs.values()]
                        else:
                            continue

                        strengths.append(np.mean(np.abs(coeffs)))

            return float(np.mean(strengths)) if strengths else 0.0

        except Exception:
            return 0.0

    def _get_dominant_correlation_method(self, correlations: Dict[str, Any]) -> str:
        """获取主导相关性方法"""
        try:
            method_strengths = {}

            for method, result in correlations.items():
                if 'average_strength' in result:
                    method_strengths[method] = result['average_strength']
                elif 'correlations' in result and result['correlations']:
                    # 计算方法强度
                    method_corrs = result['correlations']
                    if method_corrs:
                        values = []
                        for corr in method_corrs.values():
                            if 'coefficient' in corr:
                                values.append(abs(corr['coefficient']))
                            elif 'mutual_information' in corr:
                                values.append(corr['mutual_information'])
                            elif 'max_correlation' in corr:
                                values.append(abs(corr['max_correlation']))

                        if values:
                            method_strengths[method] = np.mean(values)

            if method_strengths:
                return max(method_strengths.keys(), key=lambda k: method_strengths[k])

            return 'unknown'

        except Exception:
            return 'unknown'

    def _calculate_correlation_confidence(self, correlations: Dict[str, Any]) -> float:
        """计算相关性置信度"""
        try:
            confidences = []

            for method_result in correlations.values():
                if 'correlations' in method_result:
                    method_corrs = method_result['correlations']

                    for corr in method_corrs.values():
                        if 'p_value' in corr:
                            # 基于p值计算置信度
                            p_value = corr['p_value']
                            confidence = max(0, 1 - p_value)  # p值越小，置信度越高
                            confidences.append(confidence)
                        elif 'significance' in corr:
                            # 基于显著性计算置信度
                            if corr['significance'] == 'significant':
                                confidences.append(0.8)
                            else:
                                confidences.append(0.4)

            return float(np.mean(confidences)) if confidences else 0.5

        except Exception:
            return 0.5

    def _analyze_correlation_network(self) -> Dict[str, Any]:
        """分析相关性网络"""
        try:
            # 寻找连通组件
            components = self.correlation_network.find_connected_components()

            # 计算系统中心性
            centrality_scores = self.correlation_network.get_system_centrality()

            # 分析网络结构
            network_structure = {
                'num_systems': len(self.correlation_network.nodes),
                'num_connections': len(self.correlation_network.edges),
                'connected_components': components,
                'system_centrality': centrality_scores,
                'most_central_system': max(centrality_scores.keys(),
                                          key=lambda k: centrality_scores[k]) if centrality_scores else None,
                'network_density': len(self.correlation_network.edges) / (
                    len(self.correlation_network.nodes) * (len(self.correlation_network.nodes) - 1) / 2
                ) if len(self.correlation_network.nodes) > 1 else 0
            }

            return network_structure

        except Exception as e:
            logger.error(f"相关性网络分析失败: {e}")
            return {'error': str(e)}

    def _identify_significant_correlations(self, correlation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别显著相关性"""
        try:
            significant_correlations = []

            for pair_key, result in correlation_results.items():
                strength = result.get('overall_strength', 0)
                confidence = result.get('confidence', 0)

                if strength > self.correlation_threshold and confidence > 0.6:
                    significant_correlations.append({
                        'system_pair': pair_key,
                        'correlation_strength': strength,
                        'confidence': confidence,
                        'dominant_method': result.get('dominant_method', 'unknown'),
                        'systems': result.get('systems', [])
                    })

            # 按强度排序
            significant_correlations.sort(key=lambda x: x['correlation_strength'], reverse=True)

            return significant_correlations

        except Exception as e:
            logger.error(f"显著相关性识别失败: {e}")
            return []

    def _find_highly_related_systems(self, system_id: str) -> List[str]:
        """查找高度相关的系统"""
        try:
            related_systems = []

            for edge_key, edge_data in self.correlation_network.edges.items():
                if system_id in edge_data['systems']:
                    other_system = list(edge_data['systems'] - {system_id})[0]

                    if edge_data['correlation_strength'] > 0.5:  # 高相关性阈值
                        related_systems.append(other_system)

            return related_systems

        except Exception:
            return []

    def _calculate_propagation_probability(self, initial_system: str, from_system: str,
                                         to_system: str, initial_issue: Dict[str, Any]) -> Dict[str, Any]:
        """计算传播概率"""
        try:
            # 基于相关性强度和问题类型计算传播概率
            edge_key = tuple(sorted([from_system, to_system]))

            if edge_key in self.correlation_network.edges:
                correlation_strength = self.correlation_network.edges[edge_key]['correlation_strength']
            else:
                correlation_strength = 0.0

            # 根据问题类型调整传播概率
            issue_type = initial_issue.get('type', 'general')
            type_multipliers = {
                'performance': 1.2,  # 性能问题更容易传播
                'reliability': 1.0,
                'security': 0.8,     # 安全问题传播较慢
                'general': 1.0
            }

            base_probability = correlation_strength * type_multipliers.get(issue_type, 1.0)
            propagation_probability = min(0.95, max(0.0, base_probability))

            # 计算影响强度和时间
            influence_strength = propagation_probability * 0.8
            time_to_impact = timedelta(hours=int(24 * (1 - propagation_probability)))  # 传播越可能，时间越短

            estimated_impact = initial_issue.get('severity', 'medium')

            return {
                'probability': float(propagation_probability),
                'influence_strength': float(influence_strength),
                'estimated_impact': estimated_impact,
                'time_to_impact': time_to_impact
            }

        except Exception as e:
            logger.error(f"传播概率计算失败: {e}")
            return {
                'probability': 0.0,
                'influence_strength': 0.0,
                'estimated_impact': 'unknown',
                'time_to_impact': timedelta(hours=24)
            }

    def _analyze_overall_propagation_impact(self, initial_system: str, affected_systems: Set[str],
                                          propagation_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析整体传播影响"""
        try:
            total_systems_affected = len(affected_systems)
            max_propagation_depth = max([p['depth'] for p in propagation_paths]) if propagation_paths else 0

            # 计算平均传播概率
            avg_probability = np.mean([p['propagation_probability'] for p in propagation_paths]) if propagation_paths else 0

            # 评估整体影响等级
            if total_systems_affected >= 5 or max_propagation_depth >= 3:
                impact_level = 'critical'
            elif total_systems_affected >= 3 or max_propagation_depth >= 2:
                impact_level = 'high'
            elif total_systems_affected >= 2:
                impact_level = 'medium'
            else:
                impact_level = 'low'

            # 估算恢复时间
            estimated_recovery_time = timedelta(
                hours=max_propagation_depth * 4 + total_systems_affected * 2
            )

            return {
                'total_systems_affected': total_systems_affected,
                'max_propagation_depth': max_propagation_depth,
                'avg_propagation_probability': float(avg_probability),
                'impact_level': impact_level,
                'estimated_recovery_time': estimated_recovery_time,
                'containment_priority': 'high' if impact_level in ['critical', 'high'] else 'medium'
            }

        except Exception as e:
            logger.error(f"整体传播影响分析失败: {e}")
            return {'error': str(e)}

    def _generate_containment_suggestions(self, initial_system: str, affected_systems: Set[str],
                                        propagation_paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成遏制建议"""
        try:
            suggestions = []

            # 识别关键传播路径
            critical_paths = [p for p in propagation_paths if p['propagation_probability'] > 0.7]

            if critical_paths:
                suggestions.append({
                    'priority': 'critical',
                    'action': '隔离关键系统',
                    'description': f'立即隔离传播概率>70%的系统连接',
                    'target_systems': list(set([p['to_system'] for p in critical_paths])),
                    'expected_impact': '阻止进一步传播'
                })

            # 基于影响深度的建议
            max_depth = max([p['depth'] for p in propagation_paths]) if propagation_paths else 0

            if max_depth >= 2:
                suggestions.append({
                    'priority': 'high',
                    'action': '实施级联保护',
                    'description': '启用多层防护机制防止级联故障',
                    'target_systems': list(affected_systems),
                    'expected_impact': '减少级联影响范围'
                })

            # 一般性建议
            suggestions.extend([
                {
                    'priority': 'medium',
                    'action': '增强监控频率',
                    'description': '增加受影响系统的监控频率',
                    'target_systems': list(affected_systems),
                    'expected_impact': '及早发现问题恶化'
                },
                {
                    'priority': 'low',
                    'action': '准备回滚计划',
                    'description': '制定系统回滚和恢复计划',
                    'target_systems': [initial_system],
                    'expected_impact': '确保快速恢复能力'
                }
            ])

            return suggestions

        except Exception as e:
            logger.error(f"遏制建议生成失败: {e}")
            return []

    def _analyze_system_dependencies(self, target_systems: List[str]) -> Dict[str, List[str]]:
        """分析系统依赖关系"""
        try:
            dependencies = {}

            for system in target_systems:
                if system in self.correlation_network.nodes:
                    connections = list(self.correlation_network.nodes[system]['connections'])
                    dependencies[system] = connections

            return dependencies

        except Exception as e:
            logger.error(f"系统依赖关系分析失败: {e}")
            return {}

    def _generate_optimization_sequence(self, target_systems: List[str],
                                      dependencies: Dict[str, List[str]],
                                      criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成优化序列"""
        try:
            # 基于依赖关系和优先级生成优化序列
            optimization_sequence = []

            # 计算系统优先级（基于中心性和当前状态）
            system_priorities = {}
            centrality_scores = self.correlation_network.get_system_centrality()

            for system in target_systems:
                # 优先级 = 中心性分数 + 依赖数量 + 优化需求权重
                centrality = centrality_scores.get(system, 0)
                dependency_count = len(dependencies.get(system, []))
                priority_score = centrality * 0.4 + dependency_count * 0.3

                # 添加优化标准权重
                if criteria.get('prioritize_critical'):
                    # 这里可以根据具体标准调整
                    pass

                system_priorities[system] = priority_score

            # 按优先级排序
            sorted_systems = sorted(system_priorities.keys(),
                                  key=lambda x: system_priorities[x],
                                  reverse=True)

            # 生成优化序列
            for i, system in enumerate(sorted_systems):
                optimization_sequence.append({
                    'sequence_order': i + 1,
                    'system': system,
                    'priority_score': system_priorities[system],
                    'dependencies': dependencies.get(system, []),
                    'estimated_effort': 'medium',  # 可以基于系统复杂度计算
                    'expected_benefit': system_priorities[system] * 10  # 简化的效益计算
                })

            return optimization_sequence

        except Exception as e:
            logger.error(f"优化序列生成失败: {e}")
            return []

    def _calculate_collaborative_benefits(self, optimization_sequence: List[Dict[str, Any]],
                                        criteria: Dict[str, Any]) -> Dict[str, Any]:
        """计算协同效益"""
        try:
            total_benefit = sum(item['expected_benefit'] for item in optimization_sequence)

            # 计算协同效应（当系统相关时，优化效益会放大）
            synergy_factor = 1.0
            for item in optimization_sequence:
                dependencies = item['dependencies']
                if dependencies:
                    # 如果存在依赖关系，增加协同因子
                    synergy_factor += len(dependencies) * 0.1

            collaborative_benefit = total_benefit * synergy_factor

            return {
                'individual_benefit': float(total_benefit),
                'synergy_factor': float(synergy_factor),
                'collaborative_benefit': float(collaborative_benefit),
                'benefit_increase_percent': float((synergy_factor - 1) * 100)
            }

        except Exception as e:
            logger.error(f"协同效益计算失败: {e}")
            return {'error': str(e)}

    def _create_implementation_plan(self, optimization_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建实施计划"""
        try:
            phases = []
            current_phase = []
            phase_effort = 0
            max_phase_effort = 100  # 每个阶段的最大工作量

            for item in optimization_sequence:
                effort = {'high': 50, 'medium': 30, 'low': 10}.get(item.get('estimated_effort', 'medium'), 30)

                if phase_effort + effort > max_phase_effort and current_phase:
                    phases.append({
                        'phase_number': len(phases) + 1,
                        'systems': current_phase.copy(),
                        'estimated_effort': phase_effort,
                        'duration_weeks': phase_effort // 20 + 1
                    })
                    current_phase = []
                    phase_effort = 0

                current_phase.append(item['system'])
                phase_effort += effort

            # 添加最后一个阶段
            if current_phase:
                phases.append({
                    'phase_number': len(phases) + 1,
                    'systems': current_phase,
                    'estimated_effort': phase_effort,
                    'duration_weeks': phase_effort // 20 + 1
                })

            return {
                'total_phases': len(phases),
                'phases': phases,
                'total_duration_weeks': sum(p['duration_weeks'] for p in phases),
                'parallel_opportunities': self._identify_parallel_opportunities(phases)
            }

        except Exception as e:
            logger.error(f"实施计划创建失败: {e}")
            return {'error': str(e)}

    def _create_monitoring_plan(self, optimization_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建监控计划"""
        try:
            monitoring_plan = {
                'baseline_metrics': ['cpu_usage', 'memory_usage', 'response_time', 'error_rate'],
                'monitoring_frequency': 'continuous',
                'alert_thresholds': {
                    'cpu_usage': 80,
                    'memory_usage': 85,
                    'response_time': 3.0,
                    'error_rate': 0.05
                },
                'key_indicators': [],
                'reporting_schedule': 'daily'
            }

            # 为每个优化系统添加关键指标
            for item in optimization_sequence:
                system = item['system']
                monitoring_plan['key_indicators'].extend([
                    f'{system}_performance_score',
                    f'{system}_error_rate',
                    f'{system}_response_time'
                ])

            return monitoring_plan

        except Exception as e:
            logger.error(f"监控计划创建失败: {e}")
            return {'error': str(e)}

    def _create_rollback_plan(self, optimization_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建回滚计划"""
        try:
            rollback_plan = {
                'rollback_triggers': [
                    '系统性能下降>20%',
                    '错误率增加>50%',
                    '关键功能不可用>5分钟'
                ],
                'rollback_levels': ['full_rollback', 'partial_rollback', 'config_rollback'],
                'system_backups': {},
                'rollback_procedures': {}
            }

            # 为每个系统创建回滚计划
            for item in optimization_sequence:
                system = item['system']
                rollback_plan['system_backups'][system] = {
                    'backup_frequency': 'before_optimization',
                    'backup_type': 'full_system_backup',
                    'retention_period': '30_days'
                }

                rollback_plan['rollback_procedures'][system] = [
                    '停止系统服务',
                    '恢复系统备份',
                    '验证系统完整性',
                    '重新启动服务',
                    '执行回归测试'
                ]

            return rollback_plan

        except Exception as e:
            logger.error(f"回滚计划创建失败: {e}")
            return {'error': str(e)}

    def _identify_parallel_opportunities(self, phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别并行机会"""
        try:
            parallel_opportunities = []

            for phase in phases:
                systems = phase['systems']

                # 检查系统间是否存在依赖关系
                independent_systems = []
                for system in systems:
                    dependencies = self.correlation_network.nodes.get(system, {}).get('connections', set())
                    # 如果系统中没有其他系统，则可以并行执行
                    other_systems_in_phase = set(systems) - {system}
                    if not (dependencies & other_systems_in_phase):
                        independent_systems.append(system)

                if len(independent_systems) > 1:
                    parallel_opportunities.append({
                        'phase': phase['phase_number'],
                        'parallel_systems': independent_systems,
                        'time_saving_potential': f"{len(independent_systems) * 20}%"
                    })

            return parallel_opportunities

        except Exception as e:
            logger.error(f"并行机会识别失败: {e}")
            return []

    def _determine_correlation_direction(self, correlations: Dict[str, Any]) -> str:
        """确定相关性方向"""
        try:
            # 简化的方向判断逻辑
            # 实际应该基于格兰杰因果检验等方法
            return 'bidirectional'  # 默认双向

        except Exception:
            return 'unknown'
