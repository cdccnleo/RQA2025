# RQA2025 模型层功能增强分析报告（续8）

## 2. 功能分析（续）

### 2.5 模型监控（续）

#### 2.5.1 模型漂移检测（续）

**实现建议**（续）：

```python
    def set_baseline(
        self,
        X: np.ndarray,
        predictions: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'ModelDriftDetector':
        """
        设置基准数据
        
        Args:
            X: 特征数据
            predictions: 预测结果
            feature_names: 特征名称
            
        Returns:
            ModelDriftDetector: 自身
        """
        self.baseline_data = X
        self.baseline_predictions = predictions
        
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            self.feature_names = feature_names
        
        # 计算基准特征分布
        self.feature_distributions = {}
        for i, feature_name in enumerate(self.feature_names):
            self.feature_distributions[feature_name] = {
                'mean': np.mean(X[:, i]),
                'std': np.std(X[:, i]),
                'quantiles': np.percentile(X[:, i], [25, 50, 75]),
                'histogram': np.histogram(X[:, i], bins=50)
            }
        
        # 计算基准预测分布
        self.prediction_distribution = {
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'quantiles': np.percentile(predictions, [25, 50, 75]),
            'histogram': np.histogram(predictions, bins=50)
        }
        
        return self
    
    def detect_drift(
        self,
        X: np.ndarray,
        predictions: np.ndarray,
        method: str = 'ks_test',
        plot: bool = True
    ) -> Dict:
        """
        检测数据漂移
        
        Args:
            X: 新的特征数据
            predictions: 新的预测结果
            method: 检测方法，'ks_test'或'wasserstein'
            plot: 是否绘制图表
            
        Returns:
            Dict: 漂移检测结果
        """
        if self.baseline_data is None:
            raise ValueError("Baseline data not set. Call set_baseline() first.")
        
        # 初始化结果
        results = {
            'feature_drift': {},
            'prediction_drift': None,
            'overall_drift': False,
            'plots': {}
        }
        
        # 检测特征漂移
        for i, feature_name in enumerate(self.feature_names):
            # 获取基准和新数据
            baseline_feature = self.baseline_data[:, i]
            new_feature = X[:, i]
            
            # 计算漂移
            if method == 'ks_test':
                statistic, p_value = stats.ks_2samp(baseline_feature, new_feature)
                drift_score = statistic
            elif method == 'wasserstein':
                drift_score = wasserstein_distance(baseline_feature, new_feature)
            else:
                raise ValueError(f"Unknown drift detection method: {method}")
            
            # 自适应阈值计算
            if self.adaptive_threshold and len(self.drift_history) >= self.min_samples:
                # 基于历史漂移分数的动态阈值
                mean_score = np.mean(self.drift_history)
                std_score = np.std(self.drift_history)
                adaptive_threshold = mean_score + 3 * std_score
                has_drift = drift_score > adaptive_threshold
            else:
                # 使用固定阈值
                has_drift = drift_score > self.drift_threshold
            
            # 记录漂移分数用于自适应计算
            if update_threshold:
                self.drift_history.append(drift_score)
                # 保持窗口大小
                if len(self.drift_history) > self.window_size:
                    self.drift_history.pop(0)
            
            # 保存结果
            results['feature_drift'][feature_name] = {
                'drift_score': drift_score,
                'has_drift': has_drift,
                'baseline_stats': self.feature_distributions[feature_name],
                'current_stats': {
                    'mean': np.mean(new_feature),
                    'std': np.std(new_feature),
                    'quantiles': np.percentile(new_feature, [25, 50, 75]),
                    'histogram': np.histogram(new_feature, bins=50)
                }
            }
            
            # 如果有任何特征漂移，标记整体漂移
            if has_drift:
                results['overall_drift'] = True
            
            # 绘制漂移图表
            if plot:
                fig = self._plot_drift_comparison(
                    baseline_feature,
                    new_feature,
                    feature_name,
                    drift_score,
                    has_drift
                )
                results['plots'][feature_name] = fig
                
                if self.output_dir:
                    fig.savefig(
                        os.path.join(self.output_dir, f'drift_{feature_name}.png'),
                        dpi=300,
                        bbox_inches='tight'
                    )
        
        # 检测预测漂移
        if method == 'ks_test':
            statistic, p_value = stats.ks_2samp(
                self.baseline_predictions,
                predictions
            )
            prediction_drift_score = statistic
        elif method == 'wasserstein':
            prediction_drift_score = wasserstein_distance(
                self.baseline_predictions,
                predictions
            )
        
        prediction_has_drift = prediction_drift_score > self.drift_threshold
        
        results['prediction_drift'] = {
            'drift_score': prediction_drift_score,
            'has_drift': prediction_has_drift,
            'baseline_stats': self.prediction_distribution,
            'current_stats': {
                'mean': np.mean(predictions),
                'std': np.std(predictions),
                'quantiles': np.percentile(predictions, [25, 50, 75]),
                'histogram': np.histogram(predictions, bins=50)
            }
        }
        
        # 如果预测漂移，标记整体漂移
        if prediction_has_drift:
            results['overall_drift'] = True
        
        # 绘制预测漂移图表
        if plot:
            fig = self._plot_drift_comparison(
                self.baseline_predictions,
                predictions,
                'Predictions',
                prediction_drift_score,
                prediction_has_drift
            )
            results['plots']['predictions'] = fig
            
            if self.output_dir:
                fig.savefig(
                    os.path.join(self.output_dir, 'drift_predictions.png'),
                    dpi=300,
                    bbox_inches='tight'
                )
        
        return results
    
    def _plot_drift_comparison(
        self,
        baseline_data: np.ndarray,
        new_data: np.ndarray,
        name: str,
        drift_score: float,
        has_drift: bool
    ) -> plt.Figure:
        """
        绘制漂移对比图
        
        Args:
            baseline_data: 基准数据
            new_data: 新数据
            name: 数据名称
            drift_score: 漂移分数
            has_drift: 是否存在漂移
            
        Returns:
            plt.Figure: 图表对象
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制分布对比
        sns.kdeplot(baseline_data, ax=ax1, label='Baseline', color='blue')
        sns.kdeplot(new_data, ax=ax1, label='Current', color='red')
        ax1.set_title(f'{name} Distribution Comparison')
        ax1.legend()
        
        # 绘制Q-Q图
        stats.probplot(new_data, dist="norm", plot=ax2)
        ax2.set_title(f'{name} Q-Q Plot')
        
        # 添加漂移信息
        drift_status = "DRIFT DETECTED" if has_drift else "NO DRIFT"
        fig.suptitle(
            f"{name} Drift Analysis\n"
            f"Drift Score: {drift_score:.4f} ({drift_status})",
            fontsize=12
        )
        
        fig.tight_layout()
        return fig
    
    def generate_drift_report(
        self,
        drift_results: Dict,
        save_path: Optional[str] = None
    ) -> str:
        """
        生成漂移报告
        
        Args:
            drift_results: 漂移检测结果
            save_path: 保存路径
            
        Returns:
            str: 报告内容
        """
        # 创建报告
        report = "# Model Drift Analysis Report\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 添加总体漂移状态
        report += "## Overall Drift Status\n\n"
        drift_status = "DRIFT DETECTED" if drift_results['overall_drift'] else "NO DRIFT"
        report += f"Status: **{drift_status}**\n\n"
        
        # 添加特征漂移分析
        report += "## Feature Drift Analysis\n\n"
        
        for feature_name, drift_info in drift_results['feature_drift'].items():
            report += f"### {feature_name}\n\n"
            report += f"- Drift Score: {drift_info['drift_score']:.4f}\n"
            report += f"- Status: {'DRIFT DETECTED' if drift_info['has_drift'] else 'NO DRIFT'}\n\n"
            
            # 添加统计信息对比
            report += "#### Statistics Comparison\n\n"
            report += "| Metric | Baseline | Current |\n"
            report += "| ------ | -------- | ------- |\n"
            report += f"| Mean | {drift_info['baseline_stats']['mean']:.4f} | {drift_info['current_stats']['mean']:.4f} |\n"
            report += f"| Std | {drift_info['baseline_stats']['std']:.4f} | {drift_info['current_stats']['std']:.4f} |\n"
            report += f"| 25% | {drift_info['baseline_stats']['quantiles'][0]:.4f} | {drift_info['current_stats']['quantiles'][0]:.4f} |\n"
            report += f"| 50% | {drift_info['baseline_stats']['quantiles'][1]:.4f} | {drift_info['current_stats']['quantiles'][1]:.4f} |\n"
            report += f"| 75% | {drift_info['baseline_stats']['quantiles'][2]:.4f} | {drift_info['current_stats']['quantiles'][2]:.4f} |\n\n"
            
            # 添加图表
            if feature_name in drift_results['plots']:
                report += f"![{feature_name} Drift Analysis](drift_{feature_name}.png)\n\n"
        
        # 添加预测漂移分析
        report += "## Prediction Drift Analysis\n\n"
        pred_drift = drift_results['prediction_drift']
        report += f"- Drift Score: {pred_drift['drift_score']:.4f}\n"
        report += f"- Status: {'DRIFT DETECTED' if pred_drift['has_drift'] else 'NO DRIFT'}\n\n"
        
        # 添加预测统计信息对比
        report += "### Statistics Comparison\n\n"
        report += "| Metric | Baseline | Current |\n"
        report += "| ------ | -------- | ------- |\n"
        report += f"| Mean | {pred_drift['baseline_stats']['mean']:.4f} | {pred_drift['current_stats']['mean']:.4f} |\n"
        report += f"| Std | {pred_drift['baseline_stats']['std']:.4f} | {pred_drift['current_stats']['std']:.4f} |\n"
        report += f"| 25% | {pred_drift['baseline_stats']['quantiles'][0]:.4f} | {pred_drift['current_stats']['quantiles'][0]:.4f} |\n"
        report += f"| 50% | {pred_drift['baseline_stats']['quantiles'][1]:.4f} | {pred_drift['current_stats']['quantiles'][1]:.4f} |\n"
        report += f"| 75% | {pred_drift['baseline_stats']['quantiles'][2]:.4f} | {pred_drift['current_stats']['quantiles'][2]:.4f} |\n\n"
        
        # 添加预测图表
        if 'predictions' in drift_results['plots']:
            report += "![Predictions Drift Analysis](drift_predictions.png)\n\n"
        
        # 保存报告
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
```

## 3. 实施计划

根据功能分析，我们制定了以下实施计划，按照优先级分为三个阶段：

### 3.1 阶段一：核心功能优化（预计3周）

#### 3.1.1 模型性能优化（1周）

**目标**：提高模型训练和预测效率

**步骤**：
1. 创建 `src/model/optimization/model_training_optimizer.py` 文件，实现 `ModelTrainingOptimizer` 类
2. 创建 `src/model/optimization/model_prediction_optimizer.py` 文件，实现 `ModelPredictionOptimizer` 类
3. 编写单元测试和集成测试
4. 更新现有代码以使用新的优化器

**交付物**：
- `ModelTrainingOptimizer` 类实现
- `ModelPredictionOptimizer` 类实现
- 测试用例和测试报告
- 性能基准测试报告

#### 3.1.2 模型评估增强（1周）

**目标**：建立全面的模型评估体系

**步骤**：
1. 创建 `src/model/evaluation/model_evaluator.py` 文件，实现 `ModelEvaluator` 类
2. 创建 `src/model/evaluation/cross_validator.py` 文件，实现 `CrossValidator` 类
3. 编写单元测试和集成测试
4. 更新现有代码以使用新的评估工具

**交付物**：
- `ModelEvaluator` 类实现
- `CrossValidator` 类实现
- 测试用例和测试报告
- 评估报告模板

#### 3.1.3 模型集成增强（1周）

**目标**：增强模型集成能力

**步骤**：
1. 创建 `src/model/ensemble/advanced_stacking.py` 文件，实现 `AdvancedStackingEnsemble` 类
2. 创建 `src/model/ensemble/dynamic_selector.py` 文件，实现 `DynamicEnsembleSelector` 类
3. 编写单元测试和集成测试