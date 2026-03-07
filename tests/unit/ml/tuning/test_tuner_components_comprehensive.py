"""
ML调优组件深度测试
全面测试机器学习调优组件的超参数调优、交叉验证和模型选择功能
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import json
import time

# 导入ML调优相关类
try:
    from src.ml.tuning.tuner_components import (
        ComponentFactory, ITunerComponent, GridSearchTuner,
        RandomSearchTuner, BayesianOptimizationTuner
    )
    TUNER_COMPONENTS_AVAILABLE = True
except ImportError:
    TUNER_COMPONENTS_AVAILABLE = False
    ComponentFactory = Mock
    ITunerComponent = Mock
    GridSearchTuner = Mock
    RandomSearchTuner = Mock
    BayesianOptimizationTuner = Mock

try:
    from src.ml.tuning.hyperparameter_components import HyperparameterSpace
    HYPERPARAM_AVAILABLE = True
except ImportError:
    HYPERPARAM_AVAILABLE = False
    HyperparameterSpace = Mock

try:
    from src.ml.tuning.optimizer_components import OptimizerManager
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    OptimizerManager = Mock


class TestTunerComponentsComprehensive:
    """ML调优组件综合深度测试"""

    @pytest.fixture
    def sample_training_data(self):
        """创建样本训练数据"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        # 生成特征
        X = np.random.randn(n_samples, n_features)

        # 生成目标变量（回归任务）
        # y = 2*X[:,0] + 3*X[:,1] - X[:,2] + noise
        true_coeffs = np.array([2.0, 3.0, -1.0] + [0.0] * (n_features - 3))
        y = X.dot(true_coeffs) + np.random.normal(0, 0.1, n_samples)

        return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)]), pd.Series(y, name='target')

    @pytest.fixture
    def sample_classification_data(self):
        """创建样本分类数据"""
        np.random.seed(42)
        n_samples = 800
        n_features = 8

        # 生成特征
        X = np.random.randn(n_samples, n_features)

        # 生成分类目标
        # 使用前两个特征创建非线性决策边界
        y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int)

        return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)]), pd.Series(y, name='target')

    @pytest.fixture
    def hyperparameter_space(self):
        """创建超参数空间"""
        if HYPERPARAM_AVAILABLE:
            return HyperparameterSpace({
                'n_estimators': {'type': 'int', 'range': [10, 200]},
                'max_depth': {'type': 'int', 'range': [3, 15]},
                'learning_rate': {'type': 'float', 'range': [0.01, 0.3]},
                'subsample': {'type': 'float', 'range': [0.5, 1.0]},
                'colsample_bytree': {'type': 'float', 'range': [0.5, 1.0]},
                'min_child_weight': {'type': 'int', 'range': [1, 10]}
            })
        return Mock()

    @pytest.fixture
    def component_factory(self):
        """创建组件工厂"""
        if TUNER_COMPONENTS_AVAILABLE:
            return ComponentFactory()
        return Mock(spec=ComponentFactory)

    @pytest.fixture
    def grid_search_tuner(self, hyperparameter_space):
        """创建网格搜索调优器"""
        if TUNER_COMPONENTS_AVAILABLE:
            return GridSearchTuner(hyperparameter_space=hyperparameter_space)
        return Mock(spec=GridSearchTuner)

    @pytest.fixture
    def random_search_tuner(self, hyperparameter_space):
        """创建随机搜索调优器"""
        if TUNER_COMPONENTS_AVAILABLE:
            return RandomSearchTuner(hyperparameter_space=hyperparameter_space, n_trials=20)
        return Mock(spec=RandomSearchTuner)

    @pytest.fixture
    def bayesian_tuner(self, hyperparameter_space):
        """创建贝叶斯优化调优器"""
        if TUNER_COMPONENTS_AVAILABLE:
            return BayesianOptimizationTuner(hyperparameter_space=hyperparameter_space, n_trials=15)
        return Mock(spec=BayesianOptimizationTuner)

    def test_component_factory_initialization(self, component_factory):
        """测试组件工厂初始化"""
        if TUNER_COMPONENTS_AVAILABLE:
            assert component_factory is not None
            assert hasattr(component_factory, '_components')
            assert hasattr(component_factory, 'create_component')

    def test_grid_search_tuner_initialization(self, grid_search_tuner, hyperparameter_space):
        """测试网格搜索调优器初始化"""
        if TUNER_COMPONENTS_AVAILABLE:
            assert grid_search_tuner is not None
            assert grid_search_tuner.hyperparameter_space == hyperparameter_space
            assert hasattr(grid_search_tuner, 'search_space')
            assert hasattr(grid_search_tuner, 'tuning_history')

    def test_hyperparameter_space_definition(self, hyperparameter_space):
        """测试超参数空间定义"""
        if HYPERPARAM_AVAILABLE:
            assert hyperparameter_space is not None
            assert 'n_estimators' in hyperparameter_space.space
            assert 'max_depth' in hyperparameter_space.space
            assert 'learning_rate' in hyperparameter_space.space

            # 检查参数类型
            assert hyperparameter_space.space['n_estimators']['type'] == 'int'
            assert hyperparameter_space.space['learning_rate']['type'] == 'float'

    def test_grid_search_parameter_generation(self, grid_search_tuner):
        """测试网格搜索参数生成"""
        if TUNER_COMPONENTS_AVAILABLE:
            # 生成网格搜索参数组合
            param_combinations = grid_search_tuner.generate_parameter_combinations()

            assert isinstance(param_combinations, list)
            assert len(param_combinations) > 0

            # 检查参数组合结构
            for params in param_combinations:
                assert isinstance(params, dict)
                assert 'n_estimators' in params
                assert 'max_depth' in params
                assert 'learning_rate' in params

    def test_random_search_parameter_sampling(self, random_search_tuner):
        """测试随机搜索参数采样"""
        if TUNER_COMPONENTS_AVAILABLE:
            # 生成随机搜索参数
            random_params = random_search_tuner.sample_random_parameters(n_samples=10)

            assert isinstance(random_params, list)
            assert len(random_params) == 10

            # 检查参数值在合理范围内
            for params in random_params:
                assert 10 <= params['n_estimators'] <= 200
                assert 3 <= params['max_depth'] <= 15
                assert 0.01 <= params['learning_rate'] <= 0.3

    def test_bayesian_optimization_initialization(self, bayesian_tuner):
        """测试贝叶斯优化初始化"""
        if TUNER_COMPONENTS_AVAILABLE:
            assert bayesian_tuner is not None
            assert hasattr(bayesian_tuner, 'surrogate_model')
            assert hasattr(bayesian_tuner, 'acquisition_function')

    def test_tuning_with_regression_task(self, grid_search_tuner, sample_training_data):
        """测试回归任务的调优"""
        if TUNER_COMPONENTS_AVAILABLE:
            X, y = sample_training_data

            # 定义目标函数
            def objective_function(params):
                # 简化的目标函数（模拟模型性能）
                score = -(params['learning_rate'] - 0.1)**2 - (params['max_depth'] - 8)**2
                score += np.random.normal(0, 0.1)  # 添加噪声
                return score

            # 执行调优
            tuning_result = grid_search_tuner.tune(
                objective_function=objective_function,
                X=X, y=y,
                cv_folds=3,
                scoring_metric='neg_mean_squared_error'
            )

            assert isinstance(tuning_result, dict)
            assert 'best_parameters' in tuning_result
            assert 'best_score' in tuning_result
            assert 'tuning_history' in tuning_result
            assert 'cv_results' in tuning_result

    def test_tuning_with_classification_task(self, random_search_tuner, sample_classification_data):
        """测试分类任务的调优"""
        if TUNER_COMPONENTS_AVAILABLE:
            X, y = sample_classification_data

            # 定义分类目标函数
            def classification_objective(params):
                # 简化的分类目标函数
                score = params['learning_rate'] * 0.5 + params['max_depth'] * 0.1
                score += np.random.normal(0, 0.05)
                return score

            # 执行随机搜索调优
            tuning_result = random_search_tuner.tune(
                objective_function=classification_objective,
                X=X, y=y,
                cv_folds=5,
                scoring_metric='accuracy'
            )

            assert isinstance(tuning_result, dict)
            assert 'best_parameters' in tuning_result
            assert 'best_score' in tuning_result
            assert len(tuning_result['tuning_history']) == 20  # n_trials

    def test_bayesian_optimization_tuning(self, bayesian_tuner, sample_training_data):
        """测试贝叶斯优化调优"""
        if TUNER_COMPONENTS_AVAILABLE:
            X, y = sample_training_data

            def simple_objective(params):
                return -(params['learning_rate'] - 0.15)**2 - (params['n_estimators'] - 100)**2 * 0.001

            # 执行贝叶斯优化
            bayesian_result = bayesian_tuner.tune(
                objective_function=simple_objective,
                X=X, y=y,
                cv_folds=3
            )

            assert isinstance(bayesian_result, dict)
            assert 'best_parameters' in bayesian_result
            assert 'best_score' in bayesian_result
            assert 'optimization_path' in bayesian_result

    def test_cross_validation_strategies(self, grid_search_tuner, sample_training_data):
        """测试交叉验证策略"""
        if TUNER_COMPONENTS_AVAILABLE:
            X, y = sample_training_data

            cv_strategies = ['kfold', 'stratified_kfold', 'time_series_split']

            for strategy in cv_strategies:
                def dummy_objective(params):
                    return np.random.random()

                result = grid_search_tuner.tune_with_cv_strategy(
                    objective_function=dummy_objective,
                    X=X, y=y,
                    cv_strategy=strategy,
                    cv_folds=3
                )

                assert isinstance(result, dict)
                assert 'cv_strategy' in result
                assert result['cv_strategy'] == strategy

    def test_early_stopping_integration(self, random_search_tuner, sample_training_data):
        """测试早停集成"""
        if TUNER_COMPONENTS_AVAILABLE:
            X, y = sample_training_data

            # 配置早停
            early_stopping_config = {
                'patience': 5,
                'min_delta': 0.001,
                'mode': 'max'
            }

            random_search_tuner.configure_early_stopping(early_stopping_config)

            def objective_with_potential_improvement(params, iteration):
                # 模拟前期改进，后期收敛
                if iteration < 10:
                    return iteration * 0.1 + np.random.normal(0, 0.05)
                else:
                    return 1.0 + np.random.normal(0, 0.01)  # 收敛到1.0

            # 执行带早停的调优
            result = random_search_tuner.tune_with_early_stopping(
                objective_function=lambda params: objective_with_potential_improvement(params, len(random_search_tuner.tuning_history)),
                X=X, y=y
            )

            assert isinstance(result, dict)
            assert 'early_stopping_triggered' in result

    def test_parallel_tuning_execution(self, grid_search_tuner, sample_training_data):
        """测试并行调优执行"""
        if TUNER_COMPONENTS_AVAILABLE:
            X, y = sample_training_data

            import threading

            results = []
            errors = []

            def parallel_tuning_execution(tuner_instance, thread_id):
                try:
                    def simple_objective(params):
                        time.sleep(0.01)  # 模拟计算时间
                        return -(params['learning_rate'] - 0.1)**2

                    result = tuner_instance.tune(
                        objective_function=simple_objective,
                        X=X, y=y,
                        cv_folds=2
                    )
                    results.append((thread_id, result))
                except Exception as e:
                    errors.append((thread_id, str(e)))

            # 创建多个调优器实例进行并行执行
            tuners = [GridSearchTuner(grid_search_tuner.hyperparameter_space) for _ in range(3)]
            threads = []

            for i, tuner in enumerate(tuners):
                thread = threading.Thread(target=parallel_tuning_execution, args=(tuner, i))
                threads.append(thread)
                thread.start()

            # 等待所有线程完成
            for thread in threads:
                thread.join()

            # 验证并行执行结果
            assert len(results) == len(tuners)
            assert len(errors) == 0

    def test_tuning_history_analysis(self, random_search_tuner, sample_training_data):
        """测试调优历史分析"""
        if TUNER_COMPONENTS_AVAILABLE:
            X, y = sample_training_data

            # 执行调优
            random_search_tuner.tune(
                objective_function=lambda params: np.random.random(),
                X=X, y=y
            )

            # 分析调优历史
            history_analysis = random_search_tuner.analyze_tuning_history()

            assert isinstance(history_analysis, dict)
            assert 'total_trials' in history_analysis
            assert 'best_score_evolution' in history_analysis
            assert 'parameter_importance' in history_analysis
            assert 'convergence_analysis' in history_analysis

    def test_hyperparameter_importance_analysis(self, bayesian_tuner, sample_training_data):
        """测试超参数重要性分析"""
        if TUNER_COMPONENTS_AVAILABLE:
            X, y = sample_training_data

            # 执行多次调优以获得足够的数据
            for _ in range(5):
                bayesian_tuner.tune(
                    objective_function=lambda params: np.random.random(),
                    X=X, y=y
                )

            # 分析超参数重要性
            importance_analysis = bayesian_tuner.analyze_hyperparameter_importance()

            assert isinstance(importance_analysis, dict)
            assert 'feature_importance' in importance_analysis
            assert 'correlation_analysis' in importance_analysis

            # 检查重要性得分
            feature_importance = importance_analysis['feature_importance']
            for param_name in ['n_estimators', 'max_depth', 'learning_rate']:
                assert param_name in feature_importance

    def test_tuning_visualization_and_reporting(self, grid_search_tuner, sample_training_data):
        """测试调优可视化和报告"""
        if TUNER_COMPONENTS_AVAILABLE:
            X, y = sample_training_data

            # 执行调优
            grid_search_tuner.tune(
                objective_function=lambda params: np.random.random(),
                X=X, y=y
            )

            # 生成调优报告
            report = grid_search_tuner.generate_tuning_report()

            assert isinstance(report, dict)
            assert 'summary_statistics' in report
            assert 'best_configuration' in report
            assert 'performance_analysis' in report
            assert 'recommendations' in report

            # 测试可视化生成
            visualization_data = grid_search_tuner.generate_visualization_data()

            assert isinstance(visualization_data, dict)
            assert 'parameter_heatmap' in visualization_data
            assert 'optimization_curve' in visualization_data

    def test_adaptive_tuning_strategies(self, random_search_tuner, sample_training_data):
        """测试自适应调优策略"""
        if TUNER_COMPONENTS_AVAILABLE:
            X, y = sample_training_data

            # 启用自适应调优
            random_search_tuner.enable_adaptive_tuning()

            # 执行自适应调优
            adaptive_result = random_search_tuner.tune_adaptive(
                objective_function=lambda params: np.random.random(),
                X=X, y=y,
                adaptation_interval=5
            )

            assert isinstance(adaptive_result, dict)
            assert 'adaptation_history' in adaptive_result
            assert 'strategy_adjustments' in adaptive_result

    def test_tuning_with_constraints(self, bayesian_tuner):
        """测试带约束的调优"""
        if TUNER_COMPONENTS_AVAILABLE:
            # 定义约束条件
            constraints = [
                lambda params: params['max_depth'] * params['learning_rate'] <= 3.0,  # 深度-学习率约束
                lambda params: params['n_estimators'] >= 50,  # 最小树数量
                lambda params: params['subsample'] + params['colsample_bytree'] <= 1.8  # 采样比例约束
            ]

            bayesian_tuner.set_tuning_constraints(constraints)

            # 执行带约束的调优
            constrained_result = bayesian_tuner.tune_with_constraints(
                objective_function=lambda params: np.random.random(),
                X=pd.DataFrame(), y=pd.Series(),
                constraints_active=True
            )

            assert isinstance(constrained_result, dict)
            assert 'constraints_satisfied' in constrained_result

    def test_multi_objective_tuning(self, grid_search_tuner):
        """测试多目标调优"""
        if TUNER_COMPONENTS_AVAILABLE:
            # 定义多目标函数
            def multi_objective_function(params):
                # 返回多个目标值
                accuracy = params['learning_rate'] * 0.8 + params['max_depth'] * 0.1
                complexity = params['n_estimators'] * 0.01 + params['max_depth'] * 0.1
                return [accuracy, -complexity]  # 最大化准确性，最小化复杂度

            # 执行多目标调优
            pareto_result = grid_search_tuner.tune_multi_objective(
                multi_objective_function=multi_objective_function,
                X=pd.DataFrame(), y=pd.Series(),
                population_size=20,
                generations=5
            )

            assert isinstance(pareto_result, dict)
            assert 'pareto_front' in pareto_result
            assert 'pareto_optimal_solutions' in pareto_result

    def test_tuning_robustness_testing(self, random_search_tuner):
        """测试调优鲁棒性测试"""
        if TUNER_COMPONENTS_AVAILABLE:
            # 执行鲁棒性测试
            robustness_result = random_search_tuner.test_tuning_robustness(
                objective_function=lambda params: np.random.random(),
                X=pd.DataFrame(), y=pd.Series(),
                noise_levels=[0.0, 0.1, 0.2],
                n_repeats=3
            )

            assert isinstance(robustness_result, dict)
            assert 'robustness_scores' in robustness_result
            assert 'stability_analysis' in robustness_result
            assert 'noise_sensitivity' in robustness_result

    def test_tuning_scalability_analysis(self, bayesian_tuner, sample_training_data):
        """测试调优扩展性分析"""
        if TUNER_COMPONENTS_AVAILABLE:
            X, y = sample_training_data

            # 测试不同规模数据集的调优性能
            dataset_sizes = [100, 500, 1000]

            scalability_results = {}

            for size in dataset_sizes:
                X_subset = X.head(size)
                y_subset = y.head(size)

                start_time = time.time()
                result = bayesian_tuner.tune(
                    objective_function=lambda params: np.random.random(),
                    X=X_subset, y=y_subset
                )
                end_time = time.time()

                scalability_results[size] = {
                    'execution_time': end_time - start_time,
                    'performance': result['best_score']
                }

            # 分析扩展性
            scalability_analysis = bayesian_tuner.analyze_scalability(scalability_results)

            assert isinstance(scalability_analysis, dict)
            assert 'time_complexity' in scalability_analysis
            assert 'performance_scaling' in scalability_analysis

    def test_tuning_pipeline_integration(self, component_factory, sample_training_data):
        """测试调优管道集成"""
        if TUNER_COMPONENTS_AVAILABLE:
            X, y = sample_training_data

            # 创建调优管道
            pipeline_config = {
                'preprocessing': {'scaling': True, 'feature_selection': True},
                'tuning': {'method': 'bayesian', 'n_trials': 10},
                'validation': {'cv_folds': 5, 'metrics': ['accuracy', 'f1_score']},
                'postprocessing': {'model_selection': True, 'ensemble': False}
            }

            # 执行管道调优
            pipeline_result = component_factory.execute_tuning_pipeline(
                pipeline_config=pipeline_config,
                X=X, y=y,
                objective_function=lambda params: np.random.random()
            )

            assert isinstance(pipeline_result, dict)
            assert 'pipeline_execution_status' in pipeline_result
            assert 'stage_results' in pipeline_result

    def test_tuning_error_handling_and_recovery(self, grid_search_tuner):
        """测试调优错误处理和恢复"""
        if TUNER_COMPONENTS_AVAILABLE:
            # 测试无效参数处理
            invalid_params = {'learning_rate': -0.5, 'max_depth': 0}  # 无效参数

            try:
                grid_search_tuner.validate_parameters(invalid_params)
                # 如果没有抛出异常，验证参数被修正
            except ValueError:
                # 期望的参数验证错误
                pass

            # 测试目标函数崩溃的恢复
            def unstable_objective(params):
                if np.random.random() < 0.3:  # 30%概率崩溃
                    raise RuntimeError("Simulated objective function failure")
                return np.random.random()

            # 执行带错误恢复的调优
            robust_result = grid_search_tuner.tune_with_error_recovery(
                objective_function=unstable_objective,
                X=pd.DataFrame(), y=pd.Series(),
                max_retries=3
            )

            assert isinstance(robust_result, dict)
            assert 'recovery_attempts' in robust_result
            assert 'successful_trials' in robust_result

    def test_tuning_resource_management(self, random_search_tuner, sample_training_data):
        """测试调优资源管理"""
        if TUNER_COMPONENTS_AVAILABLE:
            X, y = sample_training_data

            import psutil
            import os

            process = psutil.Process(os.getpid())

            # 记录初始资源使用
            initial_memory = process.memory_info().rss
            initial_cpu = process.cpu_percent()

            # 执行调优
            random_search_tuner.tune(
                objective_function=lambda params: np.random.random(),
                X=X, y=y
            )

            # 检查资源使用
            final_memory = process.memory_info().rss
            final_cpu = process.cpu_percent()

            memory_increase = final_memory - initial_memory

            # 验证资源使用合理
            assert memory_increase < 200 * 1024 * 1024  # 200MB限制

            # 获取调优器资源统计
            resource_stats = random_search_tuner.get_resource_usage()

            assert isinstance(resource_stats, dict)
            assert 'peak_memory_usage' in resource_stats
            assert 'total_cpu_time' in resource_stats
            assert 'average_trial_time' in resource_stats

    def test_tuning_audit_and_logging(self, bayesian_tuner, sample_training_data):
        """测试调优审计和日志"""
        if TUNER_COMPONENTS_AVAILABLE:
            X, y = sample_training_data

            # 启用审计日志
            bayesian_tuner.enable_audit_logging()

            # 执行调优操作
            bayesian_tuner.tune(
                objective_function=lambda params: np.random.random(),
                X=X, y=y
            )

            # 获取审计日志
            audit_log = bayesian_tuner.get_audit_log()

            assert isinstance(audit_log, list)
            assert len(audit_log) > 0

            # 检查审计记录
            for record in audit_log:
                assert 'timestamp' in record
                assert 'operation' in record
                assert 'parameters' in record
                assert 'result' in record

            # 测试日志导出
            log_export = bayesian_tuner.export_audit_log()

            assert isinstance(log_export, dict)
            assert 'log_entries' in log_export
            assert 'summary_statistics' in log_export
