#!/usr/bin/env python3
"""
大规模训练演示脚本

演示数据管道和分布式训练在金融数据上的应用
    创建时间: 2025年1月
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from deep_learning.data_pipeline import (
        DataPipeline, StreamDataSource
    )
    from deep_learning.distributed_trainer import (
        DistributedTrainer, create_financial_hyperparameter_search,
        TrialResult
    )
    from deep_learning.deep_learning_manager import DeepLearningManager
    from deep_learning.data_preprocessor import DataPreprocessor
    print("✅ 深度学习模块导入成功")
except ImportError as e:
    print(f"❌ 深度学习模块导入失败: {e}")
    sys.exit(1)


class LargeScaleTrainingDemo:
    """大规模训练演示"""

    def __init__(self):
        self.manager = DeepLearningManager("models/large_scale")
        self.preprocessor = DataPreprocessor()
        self.pipeline = None
        self.trainer = None
        self.results = {}

    def setup_data_pipeline(self, data_size: str = "medium") -> DataPipeline:
        """
        设置数据管道

        Args:
            data_size: 数据规模 ("small", "medium", "large")

        Returns:
            配置好的数据管道
        """
        print(f"🔧 设置{data_size}规模数据管道...")

        # 根据数据规模调整配置
        if data_size == "small":
            batch_size = 500
            n_samples = 1000
        elif data_size == "medium":
            batch_size = 1000
            n_samples = 5000
        else:  # large
            batch_size = 2000
            n_samples = 10000

        config = {
            'batch_size': batch_size,
            'validation': {
                'rules': {
                    'data_shape': (-1, 10),
                    'dtype': np.float32,
                    'max_missing_rate': 0.1,
                    'max_outlier_rate': 0.05
                }
            },
            'feature_engineering': {
                'features': [
                    'moving_average',
                    'rsi',
                    'macd',
                    'momentum',
                    'volatility',
                    'time_features',
                    'price_features'
                ],
                'ma_windows': [5, 10, 20],
                'ma_periods': [1, 5, 10]
            }
        }

        pipeline = DataPipeline(config)

        # 创建流数据源用于演示
        source_config = {
            'type': 'stream',
            'batch_size': batch_size,
            'queue_size': 100
        }

        data_source = StreamDataSource(source_config)
        pipeline.set_data_source(data_source)

        self.pipeline = pipeline
        print(f"✅ {data_size}规模数据管道设置完成")
        return pipeline

    def setup_distributed_trainer(self, max_trials: int = 20) -> DistributedTrainer:
        """
        设置分布式训练器

        Args:
            max_trials: 最大试验次数

        Returns:
            配置好的分布式训练器
        """
        print("🔧 设置分布式训练器...")

        config = {
            'search_algorithm': 'random',
            'max_trials': max_trials,
            'max_concurrent': min(4, max_trials),  # 限制并发数
            'metric_name': 'loss',
            'metric_direction': 'minimize'
        }

        trainer = DistributedTrainer(self.manager, config)

        # 配置超参数搜索空间
        search = create_financial_hyperparameter_search()
        trainer.search = search

        self.trainer = trainer
        print(f"✅ 分布式训练器设置完成，最大试验数: {max_trials}")
        return trainer

    def generate_large_dataset(self, n_samples: int = 5000, save_path: str = "data/large_financial.csv") -> str:
        """
        生成大规模数据集

        Args:
            n_samples: 样本数量
            save_path: 保存路径

        Returns:
            数据文件路径
        """
        print(f"📊 生成大规模数据集 ({n_samples} 样本)...")

        # 创建时间序列
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')

        np.random.seed(42)

        # 生成基础价格序列
        base_price = np.cumsum(np.random.randn(n_samples) * 0.01 + 0.001)
        base_price = (base_price - base_price.min()) / (base_price.max() - base_price.min())

        # 创建数据框
        data = {'timestamp': dates, 'close': base_price}

        # 添加OHLC数据
        data['open'] = base_price + np.random.randn(n_samples) * 0.01
        data['high'] = np.maximum(data['open'], base_price) + np.random.rand(n_samples) * 0.02
        data['low'] = np.minimum(data['open'], base_price) - np.random.rand(n_samples) * 0.02

        # 添加成交量
        data['volume'] = np.abs(np.random.randn(n_samples)) * 1000 + 100

        # 添加技术指标
        data['rsi'] = 50 + np.random.randn(n_samples) * 10
        data['macd'] = np.random.randn(n_samples) * 0.001
        data['bb_upper'] = base_price + np.random.rand(n_samples) * 0.05
        data['bb_lower'] = base_price - np.random.rand(n_samples) * 0.05

        # 添加更多特征
        for i in range(6):
            data[f'feature_{i}'] = np.random.randn(n_samples) * 0.1

        # 创建数据框
        df = pd.DataFrame(data)

        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存数据
        df.to_csv(save_path, index=False)
        print(f"✅ 大规模数据集已生成: {save_path}")
        print(f"   数据形状: {df.shape}")
        print(f"   列名: {df.columns.tolist()[:10]}...")

        return save_path

    def run_pipeline_demo(self, max_batches: int = 5) -> Dict[str, Any]:
        """
        运行数据管道演示

        Args:
            max_batches: 最大批次数

        Returns:
            管道运行结果
        """
        print("🪈 运行数据管道演示...")

        if not self.pipeline:
            raise ValueError("数据管道未设置")

        import queue
        output_queue = queue.Queue(maxsize=50)

        # 启动管道
        if not self.pipeline.start_pipeline():
            raise RuntimeError("数据管道启动失败")

        try:
            processed_batches = []

            # 处理数据流
            for i, batch in enumerate(self.pipeline.process_data_stream(output_queue, max_batches)):
                processed_batches.append({
                    'batch_id': batch.batch_id,
                    'original_shape': batch.data.shape if hasattr(batch, 'data') else None,
                    'timestamp': batch.timestamp,
                    'metadata': batch.metadata
                })

                print(f"   处理批次 {i+1}: {batch.batch_id}, 形状: {batch.data.shape}")

            # 获取统计信息
            stats = self.pipeline.get_statistics()

            result = {
                'processed_batches': processed_batches,
                'statistics': stats,
                'execution_time': time.time()
            }

            print(f"✅ 数据管道演示完成，处理了 {len(processed_batches)} 个批次")
            return result

        finally:
            # 停止管道
            self.pipeline.stop_pipeline()

    def run_training_demo(self, processed_data: Dict[str, Any], max_trials: int = 5) -> Optional[TrialResult]:
        """
        运行训练演示

        Args:
            processed_data: 预处理后的数据
            max_trials: 最大试验次数

        Returns:
            最佳试验结果
        """
        print("🧠 运行分布式训练演示...")

        if not self.trainer:
            raise ValueError("分布式训练器未设置")

        # 设置较少的试验次数用于演示
        self.trainer.search.max_trials = max_trials

        # 运行超参数搜索
        start_time = time.time()

        best_trial = self.trainer.train_with_search(
            processed_data=processed_data,
            model_type="LSTM",
            base_config={
                'epochs': 15,  # 减少训练轮数
                'batch_size': 32
            }
        )

        execution_time = time.time() - start_time

        if best_trial:
            print("✅ 分布式训练演示完成")
            print(f"   最佳试验: {best_trial.trial_id}")
            print(f"   最佳指标: {best_trial.metrics}")
            print(f"   执行时间: {execution_time:.2f}秒")

            # 保存训练结果
            output_dir = "models/large_scale/training_results"
            self.trainer.save_search_results(output_dir)

            return best_trial

        return None

    def run_complete_demo(self, data_size: str = "medium", max_trials: int = 8) -> Dict[str, Any]:
        """
        运行完整演示

        Args:
            data_size: 数据规模
            max_trials: 最大试验次数

        Returns:
            演示结果
        """
        print("🎯 开始完整大规模训练演示")
        print("="*60)

        demo_start_time = time.time()

        # 1. 设置数据管道
        pipeline = self.setup_data_pipeline(data_size)

        # 2. 设置分布式训练器
        trainer = self.setup_distributed_trainer(max_trials)

        # 3. 生成大规模数据
        data_file = self.generate_large_dataset(
            n_samples=2000 if data_size == "small" else 5000,
            save_path=f"data/{data_size}_financial.csv"
        )

        # 4. 运行数据管道演示
        pipeline_results = self.run_pipeline_demo(max_batches=3)

        # 5. 创建预处理数据（用于训练演示）
        synthetic_data = self.preprocessor.create_synthetic_data(
            n_samples=1000,
            n_features=8,
            sequence_length=30,
            trend='random'
        )

        processed_data = self.preprocessor.preprocess_price_data(
            synthetic_data,
            target_column='close',
            feature_columns=['open', 'high', 'low', 'close', 'volume', 'feature_6', 'feature_7']
        )

        # 6. 运行训练演示
        training_result = self.run_training_demo(processed_data, max_trials)

        # 7. 汇总结果
        demo_results = {
            'demo_config': {
                'data_size': data_size,
                'max_trials': max_trials,
                'data_file': data_file
            },
            'pipeline_results': pipeline_results,
            'training_result': training_result,
            'execution_time': time.time() - demo_start_time,
            'timestamp': datetime.now().isoformat()
        }

        # 保存演示结果
        results_file = "models/large_scale/demo_results.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(demo_results, f, ensure_ascii=False, indent=2, default=str)

        print("\n🎉 完整演示完成！")
        print(f"   总执行时间: {demo_results['execution_time']:.2f}秒")
        print(f"   结果文件: {results_file}")

        return demo_results

    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """
        生成性能报告

        Args:
            results: 演示结果

        Returns:
            报告文件路径
        """
        print("📊 生成性能报告...")

        report = {
            'title': 'RQA2025大规模训练演示性能报告',
            'generated_at': datetime.now().isoformat(),
            'configuration': results.get('demo_config', {}),
            'execution_summary': {
                'total_time': results.get('execution_time', 0),
                'pipeline_batches': len(results.get('pipeline_results', {}).get('processed_batches', [])),
                'training_trials': results.get('training_result', {}).trial_id if results.get('training_result') else None
            },
            'performance_metrics': {},
            'recommendations': []
        }

        # 分析管道性能
        pipeline_stats = results.get('pipeline_results', {}).get('statistics', {})
        if pipeline_stats:
            total_batches = pipeline_stats.get('total_batches', 0)
            valid_batches = pipeline_stats.get('valid_batches', 0)
            processing_rate = valid_batches / results.get('execution_time', 1)

            report['performance_metrics']['pipeline'] = {
                'total_batches': total_batches,
                'valid_batches': valid_batches,
                'valid_rate': valid_batches / max(total_batches, 1),
                'processing_rate': processing_rate
            }

        # 分析训练性能
        training_result = results.get('training_result')
        if training_result:
            report['performance_metrics']['training'] = {
                'best_loss': training_result.metrics.get('loss', 'N/A'),
                'best_mae': training_result.metrics.get('mae', 'N/A'),
                'training_time': training_result.training_time,
                'hyperparameters': training_result.hyperparameters
            }

        # 生成建议
        if pipeline_stats.get('invalid_batches', 0) > 0:
            report['recommendations'].append("数据质量需要改进，减少无效批次")

        if training_result and training_result.training_time > 300:
            report['recommendations'].append("考虑使用GPU加速或优化模型结构")

        # 保存报告
        report_file = "models/large_scale/performance_report.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        print(f"✅ 性能报告已生成: {report_file}")
        return report_file


def main():
    """主函数"""
    print("🚀 RQA2025大规模训练演示")
    print("="*60)

    # 创建演示实例
    demo = LargeScaleTrainingDemo()

    try:
        # 运行完整演示
        results = demo.run_complete_demo(
            data_size="small",  # 使用小规模数据以便快速演示
            max_trials=3        # 减少试验次数
        )

        # 生成性能报告
        report_file = demo.generate_performance_report(results)

        # 显示总结
        print("\n" + "="*60)
        print("📋 演示总结")
        print("="*60)

        config = results.get('demo_config', {})
        print(f"数据规模: {config.get('data_size', 'N/A')}")
        print(f"最大试验数: {config.get('max_trials', 'N/A')}")
        print(f"数据文件: {config.get('data_file', 'N/A')}")

        pipeline_results = results.get('pipeline_results', {})
        if pipeline_results:
            stats = pipeline_results.get('statistics', {})
            print(f"管道处理批次: {stats.get('total_batches', 0)}")
            print(f"有效批次: {stats.get('valid_batches', 0)}")

        training_result = results.get('training_result')
        if training_result:
            print(f"最佳模型: {training_result.trial_id}")
            print(f"最佳指标: {training_result.metrics}")
            print(f"训练时间: {training_result.training_time:.2f}秒")

        print(f"\n总执行时间: {results.get('execution_time', 0):.2f}秒")
        print(f"性能报告: {report_file}")

        print("\n🎯 演示成功完成！")
        print("   数据管道和分布式训练系统已验证可用")
        print("   可以处理大规模金融数据和模型训练任务")

        return demo, results

    except Exception as e:
        print(f"\n❌ 演示执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    demo, results = main()
