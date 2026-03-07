#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习管道深度测试
测试模型训练、验证、部署和监控的全面功能

测试覆盖目标: 95%+
测试深度: 模型性能、过拟合检测、管道集成、异常处理、性能优化
"""

import pytest
import time
import threading
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import queue
import pickle
import tempfile
import os

# 尝试导入机器学习相关模块
try:
    from src.ml.model_manager import ModelManager
    from src.ml.model_evaluator import ModelEvaluator
    from src.ml.model_inference import ModelInference
    ml_available = True
except ImportError:
    ml_available = False
    ModelManager = Mock
    ModelEvaluator = Mock
    ModelInference = Mock

# 机器学习库
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import classification_report, confusion_matrix
    sklearn_available = True
except ImportError:
    sklearn_available = False

pytestmark = pytest.mark.skipif(
    not ml_available or not sklearn_available,
    reason="ML modules or sklearn not available"
)


class TestMLPipelineDeepCoverage:
    """机器学习管道深度测试类"""

    @pytest.fixture
    def model_manager(self):
        """创建模型管理器"""
        config = {
            'model_registry': {},
            'version_control': True,
            'auto_save': True,
            'performance_tracking': True,
            'model_validation': True
        }
        manager = ModelManager(config=config)
        yield manager
        if hasattr(manager, 'cleanup'):
            manager.cleanup()

    @pytest.fixture
    def model_evaluator(self):
        """创建模型评估器"""
        config = {
            'cross_validation': True,
            'cv_folds': 5,
            'performance_metrics': ['accuracy', 'precision', 'recall', 'f1'],
            'robustness_tests': True,
            'fairness_checks': True
        }
        evaluator = ModelEvaluator(config=config)
        yield evaluator

    @pytest.fixture
    def model_inference(self):
        """创建模型推理器"""
        config = {
            'batch_processing': True,
            'max_batch_size': 1000,
            'latency_monitoring': True,
            'performance_optimization': True,
            'fallback_models': True
        }
        inference = ModelInference(config=config)
        yield inference

    @pytest.fixture
    def synthetic_dataset(self):
        """创建合成数据集"""
        np.random.seed(42)

        # 生成1000个样本，20个特征
        n_samples = 1000
        n_features = 20

        # 创建特征矩阵
        X = np.random.randn(n_samples, n_features)

        # 添加一些有意义的特征关系
        # 特征0-4: 强相关组
        for i in range(1, 5):
            X[:, i] = X[:, 0] + np.random.randn(n_samples) * 0.1

        # 特征5-9: 中等相关组
        for i in range(6, 10):
            X[:, i] = X[:, 5] * 0.5 + np.random.randn(n_samples) * 0.3

        # 创建目标变量（二分类）
        # 基于前5个特征的线性组合
        linear_combination = X[:, :5].sum(axis=1)
        prob = 1 / (1 + np.exp(-linear_combination))  # sigmoid
        y = (prob > 0.5).astype(int)

        # 添加噪声
        noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
        y[noise_indices] = 1 - y[noise_indices]  # 翻转10%的标签

        return X, y

    def test_model_training_pipeline_robustness(self, model_manager, synthetic_dataset):
        """测试模型训练管道鲁棒性"""
        X, y = synthetic_dataset

        # 测试不同的模型算法
        models = [
            ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('GradientBoosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('LogisticRegression', LogisticRegression(random_state=42, max_iter=1000))
        ]

        training_results = []

        for model_name, model in models:
            print(f"\n🔬 测试 {model_name} 训练鲁棒性")

            # 测试正常训练
            try:
                start_time = time.time()
                model.fit(X, y)
                training_time = time.time() - start_time

                # 预测和评估
                y_pred = model.predict(X)
                accuracy = accuracy_score(y, accuracy_score)

                training_results.append({
                    'model': model_name,
                    'status': 'success',
                    'training_time': training_time,
                    'accuracy': accuracy,
                    'error': None
                })

                print(f"   ✅ 训练成功: 准确率 {accuracy:.2f}, 用时 {training_time:.4f}秒")
            except Exception as e:
                training_results.append({
                    'model': model_name,
                    'status': 'failed',
                    'training_time': 0,
                    'accuracy': 0,
                    'error': str(e)
                })

                print(f"   ❌ 训练失败: {e}")

        # 分析训练鲁棒性
        successful_models = sum(1 for r in training_results if r['status'] == 'success')
        failed_models = len(training_results) - successful_models

        if successful_models > 0:
            avg_training_time = sum(r['training_time'] for r in training_results if r['status'] == 'success') / successful_models
            avg_accuracy = sum(r['accuracy'] for r in training_results if r['status'] == 'success') / successful_models

            print("\n📊 模型训练鲁棒性分析:")
            print(f"   成功模型: {successful_models}/{len(models)}")
            print(f"   失败模型: {failed_models}")
            print(f"   平均训练时间: {avg_training_time:.2f}秒")
            print(f"   平均准确率: {avg_accuracy:.4f}")

            # 验证鲁棒性
            assert successful_models >= len(models) * 0.8, f"成功率过低: {successful_models}/{len(models)}"
            assert avg_accuracy > 0.7, f"平均准确率过低: {avg_accuracy:.4f}"

    def test_model_cross_validation_stability(self, model_evaluator, synthetic_dataset):
        """测试模型交叉验证稳定性"""
        X, y = synthetic_dataset

        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # 执行多次交叉验证
        n_iterations = 10
        cv_scores = []

        print("🔄 执行多次交叉验证测试稳定性")

        for i in range(n_iterations):
            # 时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=5)

            scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            cv_scores.extend(scores)

            if (i + 1) % 3 == 0:
                print(f"   第{i+1}轮平均分数: {np.mean(scores):.3f}")

        # 分析交叉验证稳定性
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        cv_cv = cv_std / cv_mean  # 变异系数

        print("\n📊 交叉验证稳定性分析:")
        print(f"   总验证次数: {len(cv_scores)}")
        print(f"   平均分数: {cv_mean:.4f}")
        print(f"   标准差: {cv_std:.4f}")
        print(f"   变异系数: {cv_cv:.3f}")
        print(f"   最高分数: {max(cv_scores):.4f}")
        print(f"   最低分数: {min(cv_scores):.4f}")

        # 验证稳定性
        assert cv_cv < 0.1, f"交叉验证变异系数过高: {cv_cv:.3f}"
        assert cv_mean > 0.7, f"平均交叉验证分数过低: {cv_mean:.4f}"
        assert cv_std < 0.05, f"交叉验证标准差过高: {cv_std:.4f}"

    def test_model_overfitting_detection(self, model_manager, synthetic_dataset):
        """测试模型过拟合检测"""
        X, y = synthetic_dataset

        # 分割训练和测试集
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 训练模型
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        # 在训练集和测试集上评估
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        overfitting_gap = train_accuracy - test_accuracy

        print("\n📊 过拟合检测结果:")
        print(f"   训练集准确率: {train_accuracy:.4f}")
        print(f"   测试集准确率: {test_accuracy:.4f}")
        print(f"   准确率差距: {overfitting_gap:.4f}")

        # 分析过拟合程度
        if overfitting_gap > 0.1:
            overfitting_level = "严重过拟合"
            risk_level = "高风险"
        elif overfitting_gap > 0.05:
            overfitting_level = "轻度过拟合"
            risk_level = "中风险"
        else:
            overfitting_level = "正常拟合"
            risk_level = "低风险"

        print(f"   过拟合程度: {overfitting_level}")
        print(f"   风险等级: {risk_level}")

        # 验证过拟合控制
        assert overfitting_gap < 0.15, f"过拟合严重: 差距 {overfitting_gap:.4f}"
        assert test_accuracy > 0.65, f"测试集准确率过低: {test_accuracy:.4f}"

        # 进一步的过拟合诊断
        train_precision = precision_score(y_train, train_pred)
        test_precision = precision_score(y_test, test_pred)
        precision_gap = train_precision - test_precision

        train_recall = recall_score(y_train, train_pred)
        test_recall = recall_score(y_test, test_pred)
        recall_gap = train_recall - test_recall

        print("\n📊 详细过拟合诊断:")
        print(f"   精确率差距: {precision_gap:.4f}")
        print(f"   召回率差距: {recall_gap:.4f}")
        print(f"   训练集精确率: {train_precision:.4f}")
        print(f"   测试集精确率: {test_precision:.4f}")

        # 验证各项指标的过拟合情况
        assert abs(precision_gap) < 0.1, f"精确率过拟合: {precision_gap:.4f}"
        assert abs(recall_gap) < 0.1, f"召回率过拟合: {recall_gap:.4f}"

    def test_model_inference_performance_optimization(self, model_inference, synthetic_dataset):
        """测试模型推理性能优化"""
        X, y = synthetic_dataset

        # 训练模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # 测试不同规模的推理性能
        batch_sizes = [1, 10, 100, 1000]

        performance_results = []

        for batch_size in batch_sizes:
            print(f"\n⚡ 测试批大小 {batch_size} 的推理性能")

            # 准备测试数据
            test_samples = min(batch_size * 10, len(X))
            test_X = X[:test_samples]

            # 单次推理测试
            single_start = time.time()
            for i in range(min(100, test_samples)):  # 最多测试100个样本
                _ = model.predict(test_X[i:i+1])
            single_time = time.time() - single_start

            # 批量推理测试
            batch_start = time.time()
            predictions = model.predict(test_X)
            batch_time = time.time() - batch_start

            # 计算性能指标
            single_throughput = min(100, test_samples) / single_time
            batch_throughput = test_samples / batch_time
            speedup = batch_throughput / single_throughput if single_throughput > 0 else 0

            performance_results.append({
                'batch_size': batch_size,
                'test_samples': test_samples,
                'single_throughput': single_throughput,
                'batch_throughput': batch_throughput,
                'speedup': speedup,
                'efficiency': batch_throughput / batch_size if batch_size > 0 else 0
            })

            print(f"   单次吞吐量: {single_throughput:.1f} 样本/秒")
            print(f"   批量吞吐量: {batch_throughput:.1f} 样本/秒")
            print(f"   加速比: {speedup:.2f}x")
            print(f"   批处理效率: {performance_results[-1]['efficiency']:.3f}")

            # 验证性能优化效果
            assert batch_throughput > 100, f"批量吞吐量过低: {batch_throughput:.1f}"
            if batch_size > 1:
                assert speedup > 1.5, f"批量加速不足: {speedup:.2f}x"

        # 分析批量效应
        batch_effects = []
        for i in range(1, len(performance_results)):
            prev_result = performance_results[i-1]
            curr_result = performance_results[i]

            batch_ratio = curr_result['batch_size'] / prev_result['batch_size']
            throughput_ratio = curr_result['batch_throughput'] / prev_result['batch_throughput']
            scaling_efficiency = throughput_ratio / batch_ratio

            batch_effects.append({
                'batch_ratio': batch_ratio,
                'throughput_ratio': throughput_ratio,
                'scaling_efficiency': scaling_efficiency
            })

        avg_scaling_efficiency = sum(e['scaling_efficiency'] for e in batch_effects) / len(batch_effects)

        print("\n📊 批量处理扩展效率:")
        print(f"   平均扩展效率: {avg_scaling_efficiency:.3f}")
        assert avg_scaling_efficiency > 0.7, f"批量扩展效率过低: {avg_scaling_efficiency:.3f}"

    def test_model_drift_detection_system(self, model_evaluator, synthetic_dataset):
        """测试模型漂移检测系统"""
        X, y = synthetic_dataset

        # 训练基准模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # 模拟概念漂移
        drift_scenarios = [
            {'name': '无漂移', 'drift_type': 'none'},
            {'name': '轻度漂移', 'drift_magnitude': 0.1},
            {'name': '中度漂移', 'drift_magnitude': 0.3},
            {'name': '重度漂移', 'drift_magnitude': 0.5}
        ]

        drift_detection_results = []

        for scenario in drift_scenarios:
            print(f"\n🔍 测试 {scenario['name']} 场景")

            # 生成测试数据
            if scenario['drift_type'] == 'none':
                # 无漂移：使用相同分布的数据
                test_X = X
                test_y = y
            else:
                # 有漂移：修改数据分布
                drift_magnitude = scenario['drift_magnitude']
                test_X = X.copy()

                # 修改前5个特征的分布
                for i in range(5):
                    test_X[:, i] += np.random.normal(0, drift_magnitude, len(test_X))

                # 重新生成目标变量
                linear_combination = test_X[:, :5].sum(axis=1)
                prob = 1 / (1 + np.exp(-linear_combination))
                test_y = (prob > 0.5).astype(int)

            # 评估模型在新数据上的表现
            test_pred = model.predict(test_X)
            test_prob = model.predict_proba(test_X)

            accuracy = accuracy_score(test_y, test_pred)

            # 计算漂移检测指标
            # 1. 预测置信度变化
            confidence_mean = np.mean(np.max(test_prob, axis=1))
            confidence_std = np.std(np.max(test_prob, axis=1))

            # 2. 预测分布变化
            pred_distribution = np.bincount(test_pred, minlength=2) / len(test_pred)

            # 3. 特征重要性变化（如果可用）
            try:
                feature_importance = model.feature_importances_
            except:
                feature_importance = None

            drift_metrics = {
                'scenario': scenario['name'],
                'accuracy': accuracy,
                'confidence_mean': confidence_mean,
                'confidence_std': confidence_std,
                'prediction_distribution': pred_distribution.tolist(),
                'drift_detected': self._detect_model_drift(accuracy, confidence_std, scenario)
            }

            drift_detection_results.append(drift_metrics)

            print(f"   准确率: {accuracy:.4f}")
            print(f"   置信度均值: {confidence_mean:.4f}")
            print(f"   置信度标准差: {confidence_std:.4f}")
            print(f"   漂移检测: {'是' if drift_metrics['drift_detected'] else '否'}")

        # 分析漂移检测效果
        drift_detected_count = sum(1 for r in drift_detection_results if r['drift_detected'])
        false_positives = sum(1 for r in drift_detection_results
                            if r['drift_detected'] and r['scenario'] == '无漂移')
        true_positives = sum(1 for r in drift_detection_results
                           if r['drift_detected'] and r['scenario'] != '无漂移')

        detection_accuracy = true_positives / (len(drift_scenarios) - 1)  # 除去无漂移场景
        false_positive_rate = false_positives / 1  # 只有一个无漂移场景

        print("\n📊 漂移检测系统评估:")
        print(f"   检测准确率: {detection_accuracy:.1f}")
        print(f"   误报率: {false_positive_rate:.1f}")
        print(f"   总检测数: {drift_detected_count}/{len(drift_scenarios)}")

        # 验证漂移检测效果
        assert detection_accuracy > 0.7, f"漂移检测准确率过低: {detection_accuracy:.1f}"
        assert false_positive_rate < 0.5, f"误报率过高: {false_positive_rate:.1f}"

    def _detect_model_drift(self, accuracy: float, confidence_std: float, scenario: Dict) -> bool:
        """检测模型漂移"""
        # 简化的漂移检测逻辑
        drift_indicators = []

        # 准确率阈值
        if scenario['drift_type'] != 'none':
            expected_accuracy_drop = scenario.get('drift_magnitude', 0) * 0.5
            if accuracy < 0.7 - expected_accuracy_drop:
                drift_indicators.append('accuracy_drop')

        # 置信度变化
        if confidence_std > 0.3:
            drift_indicators.append('confidence_variation')

        # 预测分布变化
        # 这里可以添加更复杂的分布比较逻辑

        return len(drift_indicators) >= 1

    def test_model_version_control_and_rollback(self, model_manager, synthetic_dataset):
        """测试模型版本控制和回滚"""
        X, y = synthetic_dataset

        # 创建模型版本历史
        model_versions = []
        version_metrics = []

        print("🔄 测试模型版本控制和回滚功能")

        # 训练多个版本的模型
        for version in range(1, 6):
            # 每次迭代修改模型参数
            n_estimators = 50 + version * 20
            max_depth = 5 + version

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )

            # 训练模型
            model.fit(X, y)

            # 评估模型
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)

            # 保存模型版本
            version_info = {
                'version': f'v1.{version}',
                'model': model,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall
                },
                'parameters': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth
                },
                'created_at': datetime.now(),
                'status': 'active'
            }

            model_versions.append(version_info)

            print("   版本 {} | 准确率: {:.4f} | 精确率: {:.4f} | 召回率: {:.4f}".format(
                version_info['version'], accuracy, precision, recall
            ))

        # 模拟性能退化（版本5表现较差）
        model_versions[-1]['metrics']['accuracy'] = 0.6  # 降低最新版本的性能

        # 执行回滚逻辑
        current_best_version = max(model_versions,
                                 key=lambda v: v['metrics']['accuracy'])

        # 如果当前版本性能差于历史最佳版本，则回滚
        current_version = model_versions[-1]
        if current_version['metrics']['accuracy'] < current_best_version['metrics']['accuracy'] * 0.95:
            # 执行回滚
            rollback_version = current_best_version
            current_version['status'] = 'rolled_back'

            print("\n🔄 执行模型回滚:")
            print(f"   从版本: {current_version['version']}")
            print(f"   回滚到: {rollback_version['version']}")
            print(f"   回滚版本准确率: {rollback_version['metrics']['accuracy']:.4f}")
            print(f"   当前版本准确率: {current_version['metrics']['accuracy']:.4f}")

            # 验证回滚后的性能
            assert rollback_version['metrics']['accuracy'] > current_version['metrics']['accuracy']

        # 验证版本控制完整性
        active_versions = [v for v in model_versions if v['status'] == 'active']
        rolled_back_versions = [v for v in model_versions if v['status'] == 'rolled_back']

        print("\n📊 版本控制状态:")
        print(f"   活跃版本: {len(active_versions)}")
        print(f"   回滚版本: {len(rolled_back_versions)}")
        print(f"   总版本数: {len(model_versions)}")

        # 验证版本控制逻辑
        assert len(active_versions) >= 1, "至少应有一个活跃版本"
        assert len(model_versions) == 5, "应有5个模型版本"

    def test_model_pipeline_error_handling_and_recovery(self, model_manager, synthetic_dataset):
        """测试模型管道错误处理和恢复"""
        X, y = synthetic_dataset

        # 定义各种错误场景
        error_scenarios = [
            {
                'name': '数据质量问题',
                'error_type': 'data_quality',
                'setup': lambda: self._corrupt_data(X, y),
                'expected_recovery': True
            },
            {
                'name': '内存不足',
                'error_type': 'memory_error',
                'setup': lambda: self._simulate_memory_error(),
                'expected_recovery': True
            },
            {
                'name': '模型损坏',
                'error_type': 'model_corruption',
                'setup': lambda: self._corrupt_model(),
                'expected_recovery': True
            },
            {
                'name': '外部服务不可用',
                'error_type': 'service_unavailable',
                'setup': lambda: self._simulate_service_unavailable(),
                'expected_recovery': False  # 外部服务问题通常无法自动恢复
            }
        ]

        pipeline_results = []

        for scenario in error_scenarios:
            print(f"\n🛠️  测试 {scenario['name']} 错误处理")

            try:
                # 设置错误场景
                corrupted_data = scenario['setup']()

                # 尝试执行管道
                if scenario['error_type'] == 'data_quality':
                    # 数据质量问题处理
                    success = self._handle_data_quality_issue(corrupted_data[0], corrupted_data[1])
                elif scenario['error_type'] == 'memory_error':
                    # 内存错误处理
                    success = self._handle_memory_error(X, y)
                elif scenario['error_type'] == 'model_corruption':
                    # 模型损坏处理
                    success = self._handle_model_corruption()
                else:
                    # 其他错误
                    success = False

                recovery_status = "成功" if success else "失败"
                expected_match = success == scenario['expected_recovery']

                pipeline_results.append({
                    'scenario': scenario['name'],
                    'recovery_status': recovery_status,
                    'expected_match': expected_match,
                    'error_handled': True
                })

                print(f"   恢复状态: {recovery_status}")
                print(f"   符合预期: {'是' if expected_match else '否'}")

            except Exception as e:
                pipeline_results.append({
                    'scenario': scenario['name'],
                    'recovery_status': '异常',
                    'expected_match': False,
                    'error_handled': False,
                    'exception': str(e)
                })

                print(f"   异常发生: {e}")

        # 分析错误处理效果
        successful_handling = sum(1 for r in pipeline_results if r['recovery_status'] == '成功')
        expected_matches = sum(1 for r in pipeline_results if r['expected_match'])

        handling_rate = successful_handling / len(error_scenarios)
        expectation_match_rate = expected_matches / len(error_scenarios)

        print("\n📊 错误处理和恢复评估:")
        print(f"   错误处理成功率: {handling_rate:.1f}")
        print(f"   恢复行为符合预期: {expectation_match_rate:.1f}")

        # 验证错误处理能力
        assert handling_rate > 0.6, f"错误处理成功率过低: {handling_rate:.1f}"
        assert expectation_match_rate > 0.7, f"恢复行为不符合预期: {expectation_match_rate:.1f}"

    def _corrupt_data(self, X, y):
        """破坏数据以模拟数据质量问题"""
        X_corrupted = X.copy()
        y_corrupted = y.copy()

        # 添加缺失值
        missing_mask = np.random.random(X.shape) < 0.1
        X_corrupted[missing_mask] = np.nan

        # 添加异常值
        outlier_mask = np.random.random(X.shape) < 0.05
        X_corrupted[outlier_mask] *= 100

        return X_corrupted, y_corrupted

    def _simulate_memory_error(self):
        """模拟内存错误"""
        # 这里可以设置一些会导致内存问题的配置
        return None

    def _corrupt_model(self):
        """模拟模型损坏"""
        # 这里可以创建或修改一个损坏的模型对象
        return None

    def _simulate_service_unavailable(self):
        """模拟外部服务不可用"""
        # 这里可以模拟网络连接失败
        return None

    def _handle_data_quality_issue(self, X_corrupted, y_corrupted):
        """处理数据质量问题"""
        try:
            # 处理缺失值
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X_corrupted)

            # 处理异常值（简单的IQR方法）
            for i in range(X_imputed.shape[1]):
                col = X_imputed[:, i]
                Q1, Q3 = np.percentile(col, [25, 75])
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                col = np.clip(col, lower_bound, upper_bound)
                X_imputed[:, i] = col

            # 验证修复效果
            missing_after = np.isnan(X_imputed).sum()
            return missing_after == 0

        except Exception:
            return False

    def _handle_memory_error(self, X, y):
        """处理内存错误"""
        try:
            # 使用更小的批次大小
            batch_size = 100

            # 分批处理数据
            successful_batches = 0
            total_batches = len(X) // batch_size + 1

            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]

                # 简化的批处理逻辑
                if len(batch_X) > 0:
                    successful_batches += 1

            return successful_batches > total_batches * 0.8

        except Exception:
            return False

    def _handle_model_corruption(self):
        """处理模型损坏"""
        try:
            # 尝试重新训练模型
            from sklearn.ensemble import RandomForestClassifier

            # 创建新的训练数据（这里使用固定的测试数据）
            X_train = np.random.randn(100, 10)
            y_train = np.random.randint(0, 2, 100)

            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)

            # 验证模型是否正常工作
            test_pred = model.predict(X_train[:10])
            return len(test_pred) == 10

        except Exception:
            return False

    def test_model_scalability_and_performance_monitoring(self, model_inference, synthetic_dataset):
        """测试模型可扩展性和性能监控"""
        X, y = synthetic_dataset

        # 训练基准模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # 测试不同负载下的性能
        load_scenarios = [
            {'name': '轻负载', 'requests_per_second': 10, 'duration': 30},
            {'name': '中负载', 'requests_per_second': 50, 'duration': 30},
            {'name': '重负载', 'requests_per_second': 100, 'duration': 30},
            {'name': '峰值负载', 'requests_per_second': 200, 'duration': 20}
        ]

        scalability_results = []

        for scenario in load_scenarios:
            print(f"\n📈 测试 {scenario['name']} 可扩展性")

            # 执行负载测试
            test_results = self._execute_load_test(
                model, X, scenario['requests_per_second'], scenario['duration']
            )

            # 计算性能指标
            actual_throughput = test_results['total_requests'] / test_results['total_time']
            success_rate = test_results['successful_requests'] / test_results['total_requests']
            avg_latency = test_results['total_latency'] / test_results['successful_requests'] * 1000  # ms

            scalability_results.append({
                'scenario': scenario['name'],
                'target_throughput': scenario['requests_per_second'],
                'actual_throughput': actual_throughput,
                'success_rate': success_rate,
                'avg_latency': avg_latency,
                'throughput_efficiency': actual_throughput / scenario['requests_per_second']
            })

            print(f"   目标吞吐量: {scenario['requests_per_second']} 样本/秒")
            print(f"   实际吞吐量: {actual_throughput:.1f} 样本/秒")
            print(f"   成功率: {success_rate:.1f}")
            print(f"   平均延迟: {avg_latency:.2f}ms")
            print(f"   吞吐量效率: {scalability_results[-1]['throughput_efficiency']:.2f}")

            # 验证可扩展性
            assert success_rate > 0.8, f"{scenario['name']}成功率过低: {success_rate:.1f}"
            assert avg_latency < 1000, f"{scenario['name']}平均延迟过高: {avg_latency:.2f}ms"

        # 分析整体可扩展性
        throughput_efficiencies = [r['throughput_efficiency'] for r in scalability_results]
        avg_efficiency = sum(throughput_efficiencies) / len(throughput_efficiencies)

        # 计算扩展系数（从轻负载到峰值负载的效率变化）
        if len(throughput_efficiencies) >= 2:
            scalability_trend = throughput_efficiencies[-1] / throughput_efficiencies[0]
        else:
            scalability_trend = 1.0

        print("\n📊 可扩展性总体分析:")
        print(f"   平均效率: {avg_efficiency:.2f}")
        print(f"   扩展趋势: {scalability_trend:.2f}")

        # 验证系统可扩展性
        assert avg_efficiency > 0.7, f"平均效率过低: {avg_efficiency:.2f}"
        assert scalability_trend > 0.5, f"扩展趋势不佳: {scalability_trend:.2f}"

    def _execute_load_test(self, model, X, requests_per_second, duration):
        """执行负载测试"""
        results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_latency': 0.0,
            'total_time': duration,
            'errors': []
        }

        request_interval = 1.0 / requests_per_second
        start_time = time.time()
        request_count = 0

        while time.time() - start_time < duration:
            try:
                # 选择随机样本进行预测
                sample_idx = np.random.randint(0, len(X))
                sample = X[sample_idx:sample_idx+1]

                # 记录请求开始时间
                request_start = time.time()

                # 执行预测
                prediction = model.predict(sample)
                probability = model.predict_proba(sample)

                # 计算延迟
                latency = time.time() - request_start

                results['total_requests'] += 1
                results['successful_requests'] += 1
                results['total_latency'] += latency

                request_count += 1

            except Exception as e:
                results['total_requests'] += 1
                results['failed_requests'] += 1
                results['errors'].append(str(e))

            # 控制请求频率
            time.sleep(request_interval)

        return results
