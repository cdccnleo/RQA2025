#!/usr/bin/env python3
"""
深度学习模型训练脚本

训练和评估LSTM、Transformer等深度学习模型
    创建时间: 2025年1月
"""

import sys
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import json
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib后端
plt.switch_backend('Agg')

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from deep_learning.deep_learning_manager import DeepLearningManager
    from deep_learning.data_preprocessor import DataPreprocessor
    print("✅ 深度学习模块导入成功")
except ImportError as e:
    print(f"❌ 深度学习模块导入失败: {e}")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """深度学习模型训练器"""

    def __init__(self, output_dir: str = "models/deep_learning"):
        """
        初始化训练器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.manager = DeepLearningManager(output_dir)
        self.preprocessor = DataPreprocessor()
        self.training_results = {}

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)

        logger.info(f"深度学习模型训练器初始化完成，输出目录: {output_dir}")

    def create_and_train_lstm_model(self,
                                    processed_data: Dict[str, Any],
                                    model_config: Optional[Dict[str, Any]] = None) -> str:
        """
        创建和训练LSTM模型

        Args:
            processed_data: 预处理后的数据
            model_config: 模型配置

        Returns:
            模型名称
        """
        logger.info("创建和训练LSTM模型...")

        # 默认配置
        default_config = {
            'input_shape': (processed_data['X_train'].shape[1], processed_data['X_train'].shape[2]),
            'output_units': 1,
            'lstm_units': [64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        }

        if model_config:
            default_config.update(model_config)

        model_name = f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 创建模型
        model = self.manager.create_lstm_model(
            model_name=model_name,
            **default_config
        )

        # 训练模型
        training_history = self.manager.train_model(
            model_name=model_name,
            X_train=processed_data['X_train'],
            y_train=processed_data['y_train'],
            X_val=processed_data['X_val'],
            y_val=processed_data['y_val'],
            epochs=50,
            batch_size=32,
            patience=10
        )

        # 评估模型
        test_metrics = self.manager.evaluate_model(
            model_name=model_name,
            X_test=processed_data['X_test'],
            y_test=processed_data['y_test']
        )

        # 保存训练结果
        self.training_results[model_name] = {
            'model_type': 'LSTM',
            'config': default_config,
            'training_history': training_history,
            'test_metrics': test_metrics,
            'training_time': training_history['training_time'],
            'epochs_trained': training_history['epochs_trained']
        }

        # 保存模型
        model_path = self.manager.save_model(model_name)

        logger.info(f"LSTM模型训练完成: {model_name}")
        logger.info(f"测试指标: {test_metrics}")

        return model_name

    def create_and_train_transformer_model(self,
                                           processed_data: Dict[str, Any],
                                           model_config: Optional[Dict[str, Any]] = None) -> str:
        """
        创建和训练Transformer模型

        Args:
            processed_data: 预处理后的数据
            model_config: 模型配置

        Returns:
            模型名称
        """
        logger.info("创建和训练Transformer模型...")

        # 默认配置
        default_config = {
            'input_shape': (processed_data['X_train'].shape[1], processed_data['X_train'].shape[2]),
            'output_units': 1,
            'num_heads': 8,
            'ff_dim': 64,
            'num_transformer_blocks': 3,
            'dropout_rate': 0.1,
            'learning_rate': 0.001
        }

        if model_config:
            default_config.update(model_config)

        model_name = f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 创建模型
        model = self.manager.create_transformer_model(
            model_name=model_name,
            **default_config
        )

        # 训练模型
        training_history = self.manager.train_model(
            model_name=model_name,
            X_train=processed_data['X_train'],
            y_train=processed_data['y_train'],
            X_val=processed_data['X_val'],
            y_val=processed_data['y_val'],
            epochs=30,
            batch_size=16,
            patience=8
        )

        # 评估模型
        test_metrics = self.manager.evaluate_model(
            model_name=model_name,
            X_test=processed_data['X_test'],
            y_test=processed_data['y_test']
        )

        # 保存训练结果
        self.training_results[model_name] = {
            'model_type': 'Transformer',
            'config': default_config,
            'training_history': training_history,
            'test_metrics': test_metrics,
            'training_time': training_history['training_time'],
            'epochs_trained': training_history['epochs_trained']
        }

        # 保存模型
        model_path = self.manager.save_model(model_name)

        logger.info(f"Transformer模型训练完成: {model_name}")
        logger.info(f"测试指标: {test_metrics}")

        return model_name

    def create_and_train_bidirectional_lstm_model(self,
                                                  processed_data: Dict[str, Any],
                                                  model_config: Optional[Dict[str, Any]] = None) -> str:
        """
        创建和训练双向LSTM模型

        Args:
            processed_data: 预处理后的数据
            model_config: 模型配置

        Returns:
            模型名称
        """
        logger.info("创建和训练双向LSTM模型...")

        # 默认配置
        default_config = {
            'input_shape': (processed_data['X_train'].shape[1], processed_data['X_train'].shape[2]),
            'output_units': 1,
            'lstm_units': [64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        }

        if model_config:
            default_config.update(model_config)

        model_name = f"bi_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 创建模型
        model = self.manager.create_bidirectional_lstm_model(
            model_name=model_name,
            **default_config
        )

        # 训练模型
        training_history = self.manager.train_model(
            model_name=model_name,
            X_train=processed_data['X_train'],
            y_train=processed_data['y_train'],
            X_val=processed_data['X_val'],
            y_val=processed_data['y_val'],
            epochs=40,
            batch_size=32,
            patience=10
        )

        # 评估模型
        test_metrics = self.manager.evaluate_model(
            model_name=model_name,
            X_test=processed_data['X_test'],
            y_test=processed_data['y_test']
        )

        # 保存训练结果
        self.training_results[model_name] = {
            'model_type': 'BidirectionalLSTM',
            'config': default_config,
            'training_history': training_history,
            'test_metrics': test_metrics,
            'training_time': training_history['training_time'],
            'epochs_trained': training_history['epochs_trained']
        }

        # 保存模型
        model_path = self.manager.save_model(model_name)

        logger.info(f"双向LSTM模型训练完成: {model_name}")
        logger.info(f"测试指标: {test_metrics}")

        return model_name

    def create_and_train_autoencoder_model(self,
                                           processed_data: Dict[str, Any],
                                           model_config: Optional[Dict[str, Any]] = None) -> str:
        """
        创建和训练自编码器模型

        Args:
            processed_data: 预处理后的数据
            model_config: 模型配置

        Returns:
            模型名称
        """
        logger.info("创建和训练自编码器模型...")

        # 准备自编码器数据（使用训练集）
        X_train_flat = processed_data['X_train'].reshape(processed_data['X_train'].shape[0], -1)

        # 默认配置
        default_config = {
            'input_shape': (X_train_flat.shape[1],),
            'encoding_dim': 32,
            'learning_rate': 0.001
        }

        if model_config:
            default_config.update(model_config)

        model_name = f"autoencoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 创建模型
        model, encoder, decoder = self.manager.create_autoencoder_model(
            model_name=model_name,
            **default_config
        )

        # 训练模型
        training_history = self.manager.train_model(
            model_name=model_name,
            X_train=X_train_flat,
            y_train=X_train_flat,
            epochs=100,
            batch_size=64,
            patience=15
        )

        # 评估模型（使用测试集）
        X_test_flat = processed_data['X_test'].reshape(processed_data['X_test'].shape[0], -1)
        test_metrics = self.manager.evaluate_model(
            model_name=model_name,
            X_test=X_test_flat,
            y_test=X_test_flat
        )

        # 保存训练结果
        self.training_results[model_name] = {
            'model_type': 'Autoencoder',
            'config': default_config,
            'training_history': training_history,
            'test_metrics': test_metrics,
            'training_time': training_history['training_time'],
            'epochs_trained': training_history['epochs_trained']
        }

        # 保存模型
        model_path = self.manager.save_model(model_name)

        logger.info(f"自编码器模型训练完成: {model_name}")
        logger.info(f"测试指标: {test_metrics}")

        return model_name

    def plot_training_history(self, model_name: str, save_path: Optional[str] = None):
        """
        绘制训练历史

        Args:
            model_name: 模型名称
            save_path: 保存路径
        """
        if model_name not in self.training_results:
            logger.error(f"模型 {model_name} 的训练结果不存在")
            return

        history = self.training_results[model_name]['training_history']['history']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 损失曲线
        axes[0, 0].plot(history['loss'], label='训练损失')
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='验证损失')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('轮次')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # MAE曲线
        if 'mae' in history:
            axes[0, 1].plot(history['mae'], label='训练MAE')
            if 'val_mae' in history:
                axes[0, 1].plot(history['val_mae'], label='验证MAE')
            axes[0, 1].set_title('MAE曲线')
            axes[0, 1].set_xlabel('轮次')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # MSE曲线
        if 'mse' in history:
            axes[1, 0].plot(history['mse'], label='训练MSE')
            if 'val_mse' in history:
                axes[1, 0].plot(history['val_mse'], label='验证MSE')
            axes[1, 0].set_title('MSE曲线')
            axes[1, 0].set_xlabel('轮次')
            axes[1, 0].set_ylabel('MSE')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # RMSE曲线
        if 'root_mean_squared_error' in history:
            axes[1, 1].plot(history['root_mean_squared_error'], label='训练RMSE')
            if 'val_root_mean_squared_error' in history:
                axes[1, 1].plot(history['val_root_mean_squared_error'], label='验证RMSE')
            axes[1, 1].set_title('RMSE曲线')
            axes[1, 1].set_xlabel('轮次')
            axes[1, 1].set_ylabel('RMSE')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "plots", f"{model_name}_training_history.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"训练历史图表已保存: {save_path}")

    def plot_model_comparison(self, model_names: List[str], save_path: Optional[str] = None):
        """
        绘制模型对比图

        Args:
            model_names: 模型名称列表
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        metrics = ['loss', 'mae', 'mse', 'root_mean_squared_error']

        for i, metric in enumerate(metrics):
            ax = axes[i//2, i % 2]

            for model_name in model_names:
                if model_name in self.training_results:
                    history = self.training_results[model_name]['training_history']['history']
                    if metric in history:
                        ax.plot(history[metric], label=f"{model_name} - 训练")
                    if f"val_{metric}" in history:
                        ax.plot(history[f"val_{metric}"],
                                label=f"{model_name} - 验证",
                                linestyle='--')

            ax.set_title(f'{metric.upper()}对比')
            ax.set_xlabel('轮次')
            ax.set_ylabel(metric.upper())
            ax.legend()
            ax.grid(True)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "plots", "model_comparison.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"模型对比图表已保存: {save_path}")

    def generate_training_report(self, save_path: Optional[str] = None) -> str:
        """
        生成训练报告

        Args:
            save_path: 保存路径

        Returns:
            报告文件路径
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "reports", "training_report.json")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        report = {
            'report_title': 'RQA2025深度学习模型训练报告',
            'generated_at': datetime.now().isoformat(),
            'total_models': len(self.training_results),
            'models': {}
        }

        for model_name, result in self.training_results.items():
            model_info = self.manager.get_model_summary(model_name)

            report['models'][model_name] = {
                'model_type': result['model_type'],
                'config': result['config'],
                'training_metrics': {
                    'final_loss': result['training_history']['final_loss'],
                    'final_val_loss': result['training_history']['final_val_loss'],
                    'training_time': result['training_time'],
                    'epochs_trained': result['epochs_trained']
                },
                'test_metrics': result['test_metrics'],
                'model_info': model_info
            }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"训练报告已生成: {save_path}")
        return save_path

    def run_complete_training_pipeline(self,
                                       n_samples: int = 1000,
                                       n_features: int = 10,
                                       sequence_length: int = 30) -> Dict[str, Any]:
        """
        运行完整的训练流程

        Args:
            n_samples: 样本数量
            n_features: 特征数量
            sequence_length: 序列长度

        Returns:
            训练结果摘要
        """
        logger.info("开始完整的训练流程...")

        # 创建合成数据
        print("🎭 创建合成数据...")
        synthetic_data = self.preprocessor.create_synthetic_data(
            n_samples=n_samples,
            n_features=n_features,
            sequence_length=sequence_length,
            trend='random'
        )

        # 预处理数据
        print("🔄 预处理数据...")
        processed_data = self.preprocessor.preprocess_price_data(
            synthetic_data,
            target_column='close',
            feature_columns=['open', 'high', 'low', 'close', 'volume', 'feature_6', 'feature_7']
        )

        # 训练多个模型
        trained_models = []

        # 1. 训练LSTM模型
        print("🧠 训练LSTM模型...")
        lstm_model = self.create_and_train_lstm_model(processed_data)
        trained_models.append(lstm_model)

        # 2. 训练Transformer模型
        print("🔄 训练Transformer模型...")
        transformer_model = self.create_and_train_transformer_model(processed_data)
        trained_models.append(transformer_model)

        # 3. 训练双向LSTM模型
        print("🔄 训练双向LSTM模型...")
        bi_lstm_model = self.create_and_train_bidirectional_lstm_model(processed_data)
        trained_models.append(bi_lstm_model)

        # 4. 训练自编码器模型
        print("🔄 训练自编码器模型...")
        autoencoder_model = self.create_and_train_autoencoder_model(processed_data)
        trained_models.append(autoencoder_model)

        # 生成图表
        print("📊 生成训练图表...")
        for model_name in trained_models:
            self.plot_training_history(model_name)

        if len(trained_models) > 1:
            self.plot_model_comparison(trained_models)

        # 生成报告
        print("📋 生成训练报告...")
        report_path = self.generate_training_report()

        # 汇总结果
        summary = {
            'total_models_trained': len(trained_models),
            'models': trained_models,
            'data_shape': {
                'train': processed_data['X_train'].shape,
                'val': processed_data['X_val'].shape,
                'test': processed_data['X_test'].shape
            },
            'report_path': report_path,
            'best_model': self._get_best_model(trained_models)
        }

        logger.info("完整的训练流程完成")
        return summary

    def _get_best_model(self, model_names: List[str]) -> str:
        """获取最佳模型"""
        best_model = None
        best_score = float('inf')

        for model_name in model_names:
            if model_name in self.training_results:
                test_metrics = self.training_results[model_name]['test_metrics']
                loss = test_metrics.get('loss', float('inf'))

                if loss < best_score:
                    best_score = loss
                    best_model = model_name

        return best_model


def main():
    """主函数 - 深度学习模型训练演示"""
    print("🧠 RQA2025深度学习模型训练器")
    print("="*60)

    # 初始化训练器
    trainer = ModelTrainer()

    # 运行完整训练流程
    print("\n🚀 开始深度学习模型训练流程...")

    try:
        summary = trainer.run_complete_training_pipeline(
            n_samples=500,      # 减少样本数量以便快速演示
            n_features=8,       # 特征数量
            sequence_length=30  # 序列长度
        )

        print("\n✅ 训练流程完成！")
        print(f"   训练模型数量: {summary['total_models_trained']}")
        print(f"   训练集形状: {summary['data_shape']['train']}")
        print(f"   验证集形状: {summary['data_shape']['val']}")
        print(f"   测试集形状: {summary['data_shape']['test']}")
        print(f"   最佳模型: {summary['best_model']}")
        print(f"   报告路径: {summary['report_path']}")

        # 显示模型性能对比
        print("\n📊 模型性能对比:")
        print("-" * 80)
        print("<12")
        print("-" * 80)

        for model_name in summary['models']:
            if model_name in trainer.training_results:
                result = trainer.training_results[model_name]
                test_metrics = result['test_metrics']
                train_time = result['training_history']['training_time']

                loss = test_metrics.get('loss', 'N/A')
                mae = test_metrics.get('mae', 'N/A')
                mse = test_metrics.get('mse', 'N/A')

                print(f"{model_name:<12} | {loss:<8} | {mae:<8} | {mse:<8} | {train_time:.1f}s")

        print("-" * 80)

        print("\n🎯 训练结果已保存到 models/deep_learning/ 目录")
        print("   - 模型文件: *.h5")
        print("   - 配置文件: *_config.json")
        print("   - 训练历史: *_history.pkl")
        print("   - 图表文件: plots/*.png")
        print("   - 训练报告: reports/*.json")
        return trainer, summary

    except Exception as e:
        logger.error(f"训练流程失败: {e}")
        print(f"❌ 训练流程失败: {e}")
        return None, None


if __name__ == "__main__":
    trainer, summary = main()
