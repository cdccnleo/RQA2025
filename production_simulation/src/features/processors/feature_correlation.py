import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征相关性自动分析

实现特征间相关性分析和多重共线性检测
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)


class FeatureCorrelationAnalyzer:

    """特征相关性分析器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {
            'correlation_threshold': 0.8,
            'vif_threshold': 10.0,
            'pca_variance_threshold': 0.95,
            'max_features': 50,
            'random_state': 42
        }
        self.scaler = StandardScaler()
        self.correlation_matrix = None
        self.vif_scores = {}
        self.multicollinearity_groups = []

    def analyze_feature_correlation(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        分析特征相关性

        Args:
            features: 特征数据

        Returns:
            特征相关性分析结果
        """
        logger.info(f"开始特征相关性分析，特征数量: {len(features.columns)}")

        # 数据预处理
        features_processed = self._preprocess_features(features)

        # 多种相关性分析方法
        results = {
            'correlation_matrix': self._calculate_correlation_matrix(features_processed),
            'vif_analysis': self._calculate_vif_scores(features_processed),
            'pca_analysis': self._perform_pca_analysis(features_processed),
            'feature_selection_analysis': self._perform_feature_selection_analysis(features_processed),
            'multicollinearity_detection': self._detect_multicollinearity(features_processed)
        }

        # 生成分析报告
        analysis_report = self._generate_correlation_report(results, features_processed)

        # 保存结果
        self.correlation_matrix = results['correlation_matrix']
        self.vif_scores = results['vif_analysis']
        self.multicollinearity_groups = results['multicollinearity_detection']['groups']

        logger.info(f"特征相关性分析完成，检测到 {len(self.multicollinearity_groups)} 个多重共线性组")

        return {
            'analysis_results': results,
            'analysis_report': analysis_report
        }

    def _preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """预处理特征数据"""
        # 处理缺失值
        features_processed = features.copy()
        features_processed = features_processed.fillna(features_processed.mean())

        # 标准化
        features_scaled = self.scaler.fit_transform(features_processed)
        features_processed = pd.DataFrame(features_scaled,
                                          columns=features_processed.columns,
                                          index=features_processed.index)

        return features_processed

    def _calculate_correlation_matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        """计算相关性矩阵"""
        correlation_matrix = features.corr()
        return correlation_matrix

    def _calculate_vif_scores(self, features: pd.DataFrame) -> Dict[str, float]:
        """计算方差膨胀因子(VIF)"""
        vif_scores = {}

        for i, col in enumerate(features.columns):
            # 将当前特征作为目标变量，其他特征作为预测变量
            y = features[col]
            X = features.drop(columns=[col])

            if len(X.columns) == 0:
                vif_scores[col] = 1.0
                continue

            try:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X, y)

                # 计算R²
                y_pred = model.predict(X)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                # 计算VIF
                vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
                vif_scores[col] = vif

            except Exception as e:
                logger.warning(f"计算特征 {col} 的VIF失败: {e}")
                vif_scores[col] = float('inf')

        return vif_scores

    def _perform_pca_analysis(self, features: pd.DataFrame) -> Dict[str, Any]:
        """执行PCA分析"""
        try:
            pca = PCA(random_state=self.config['random_state'])
            pca.fit(features)

            # 计算解释方差比例
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

            # 找到达到方差阈值的组件数
            n_components_threshold = np.argmax(
                cumulative_variance_ratio >= self.config['pca_variance_threshold']) + 1

            pca_results = {
                'n_components': len(features.columns),
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_variance_ratio': cumulative_variance_ratio,
                'n_components_threshold': n_components_threshold,
                'reduction_ratio': 1 - (n_components_threshold / len(features.columns))
            }

            return pca_results

        except Exception as e:
            logger.warning(f"PCA分析失败: {e}")
            return {
                'n_components': len(features.columns),
                'explained_variance_ratio': [],
                'cumulative_variance_ratio': [],
                'n_components_threshold': len(features.columns),
                'reduction_ratio': 0.0
            }

    def _perform_feature_selection_analysis(self, features: pd.DataFrame) -> Dict[str, Any]:
        """执行特征选择分析"""
        # 创建虚拟目标变量用于特征选择
        dummy_target = np.random.randn(len(features))

        try:
            # F检验特征选择
            selector = SelectKBest(score_func=f_regression, k=min(
                self.config['max_features'], len(features.columns)))
            selector.fit(features, dummy_target)

            feature_scores = dict(zip(features.columns, selector.scores_))
            feature_pvalues = dict(zip(features.columns, selector.pvalues_))

            # 排序特征
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

            return {
                'feature_scores': feature_scores,
                'feature_pvalues': feature_pvalues,
                'sorted_features': sorted_features,
                'selected_features': features.columns[selector.get_support()].tolist()
            }

        except Exception as e:
            logger.warning(f"特征选择分析失败: {e}")
            return {
                'feature_scores': {},
                'feature_pvalues': {},
                'sorted_features': [],
                'selected_features': []
            }

    def _detect_multicollinearity(self, features: pd.DataFrame) -> Dict[str, Any]:
        """检测多重共线性"""
        correlation_matrix = self._calculate_correlation_matrix(features)
        threshold = self.config['correlation_threshold']

        # 找到高相关性的特征对
        high_correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = abs(correlation_matrix.iloc[i, j])
                if corr_value >= threshold:
                    high_correlation_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })

        # 分组高相关性特征
        groups = self._group_correlated_features(high_correlation_pairs)

        # 基于VIF的共线性检测
        high_vif_features = [feature for feature, vif in self.vif_scores.items()
                             if vif > self.config['vif_threshold']]

        return {
            'high_correlation_pairs': high_correlation_pairs,
            'groups': groups,
            'high_vif_features': high_vif_features,
            'correlation_threshold': threshold,
            'vif_threshold': self.config['vif_threshold']
        }

    def _group_correlated_features(self, correlation_pairs: List[Dict[str, Any]]) -> List[List[str]]:
        """将相关特征分组"""
        if not correlation_pairs:
            return []

        # 创建图结构
        feature_graph = {}
        for pair in correlation_pairs:
            f1, f2 = pair['feature1'], pair['feature2']
            if f1 not in feature_graph:
                feature_graph[f1] = set()
            if f2 not in feature_graph:
                feature_graph[f2] = set()
            feature_graph[f1].add(f2)
            feature_graph[f2].add(f1)

        # 使用深度优先搜索找到连通分量
        visited = set()
        groups = []

        def dfs(feature, group):

            visited.add(feature)
            group.append(feature)
            for neighbor in feature_graph.get(feature, set()):
                if neighbor not in visited:
                    dfs(neighbor, group)

        for feature in feature_graph:
            if feature not in visited:
                group = []
                dfs(feature, group)
                if len(group) > 1:  # 只保留有多个特征的组
                    groups.append(group)

        return groups

    def _generate_correlation_report(self, results: Dict[str, Any], features: pd.DataFrame) -> Dict[str, Any]:
        """生成相关性分析报告"""
        correlation_matrix = results['correlation_matrix']
        vif_analysis = results['vif_analysis']
        pca_analysis = results['pca_analysis']
        multicollinearity = results['multicollinearity_detection']

        report = {
            'summary': {
                'total_features': len(features.columns),
                'high_correlation_pairs': len(multicollinearity['high_correlation_pairs']),
                'multicollinearity_groups': len(multicollinearity['groups']),
                'high_vif_features': len(multicollinearity['high_vif_features']),
                'pca_reduction_ratio': pca_analysis['reduction_ratio']
            },
            'recommendations': [],
            'feature_groups': multicollinearity['groups'],
            'high_vif_features': multicollinearity['high_vif_features']
        }

        # 生成建议
        if len(multicollinearity['high_correlation_pairs']) > len(features.columns) * 0.1:
            report['recommendations'].append("检测到较多高相关性特征对，建议进行特征选择")

        if len(multicollinearity['groups']) > 0:
            report['recommendations'].append("检测到多重共线性组，建议从每组中选择一个代表性特征")

        if len(multicollinearity['high_vif_features']) > 0:
            report['recommendations'].append(
                f"检测到 {len(multicollinearity['high_vif_features'])} 个高VIF特征，建议移除或合并")

        if pca_analysis['reduction_ratio'] > 0.5:
            report['recommendations'].append("PCA分析显示可以显著减少特征数量，建议考虑降维")

        return report

    def get_feature_recommendations(self) -> Dict[str, List[str]]:
        """获取特征建议"""
        if not self.vif_scores:
            return {'keep': [], 'remove': [], 'merge': []}

        recommendations = {
            'keep': [],
            'remove': [],
            'merge': []
        }

        # 基于VIF的建议
        for feature, vif in self.vif_scores.items():
            if vif > self.config['vif_threshold']:
                recommendations['remove'].append(feature)
            elif vif < self.config['vif_threshold'] / 2:
                recommendations['keep'].append(feature)
            else:
                recommendations['merge'].append(feature)

        # 基于多重共线性组的建议
        for group in self.multicollinearity_groups:
            if len(group) > 1:
                # 保留第一个特征，移除其他
                recommendations['keep'].append(group[0])
                recommendations['remove'].extend(group[1:])

        # 去重
        for key in recommendations:
            recommendations[key] = list(set(recommendations[key]))

        return recommendations

    def plot_correlation_heatmap(self, save_path: Optional[str] = None):
        """绘制相关性热力图"""
        if self.correlation_matrix is None:
            logger.warning("没有相关性矩阵数据，无法绘制热力图")
            return

        plt.figure(figsize=(12, 10))
        sns.heatmap(self.correlation_matrix,
                    annot=True,
                    cmap='coolwarm',
                    center=0,
                    square=True,
                    fmt='.2f')
        plt.title('特征相关性热力图')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"相关性热力图已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

    def export_correlation_report(self, filepath: str):
        """导出相关性报告"""
        if not self.correlation_matrix is not None:
            logger.warning("没有相关性分析数据，无法导出报告")
            return

        report_data = {
            'correlation_matrix': self.correlation_matrix.to_dict(),
            'vif_scores': self.vif_scores,
            'multicollinearity_groups': self.multicollinearity_groups,
            'recommendations': self.get_feature_recommendations()
        }

        # 保存为JSON格式
        import json
        with open(filepath, 'w', encoding='utf - 8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        logger.info(f"特征相关性报告已导出到: {filepath}")
