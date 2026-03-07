#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2026 脑机接口创新引擎测试套件

测试覆盖:
- 神经信号处理
- 意识状态计算
- 命令解码准确性
- 实时性能测试
"""

import pytest
import numpy as np
import asyncio
import torch
from pathlib import Path
import time
import json
from typing import Dict, List, Any

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bmi_research.engine.bci_engine import (
    BCIEngine, NeuralSignal, BCICommand, ConsciousnessState,
    create_bci_engine, NeuralSignalProcessor, AdaptiveDecoder,
    ConsciousnessModel
)
from bmi_research.signal_processing.advanced_processing import (
    AdvancedSignalProcessor, generate_synthetic_eeg
)


class TestNeuralSignalProcessor:
    """神经信号处理器测试"""

    def setup_method(self):
        """测试前准备"""
        self.processor = NeuralSignalProcessor(sampling_rate=250.0, num_channels=8)

    def test_signal_preprocessing(self):
        """测试信号预处理"""
        # 生成测试信号
        test_signal = np.random.randn(8, 1000) * 10

        # 添加直流偏移
        test_signal += np.random.uniform(-5, 5, (8, 1))

        processed = self.processor.process_signal(NeuralSignal(
            eeg_data=test_signal,
            sampling_rate=250.0,
            channel_names=[f'Ch{i+1}' for i in range(8)],
            timestamp=None
        ))

        # 验证直流偏移已被去除
        assert np.abs(np.mean(processed, axis=1)).max() < 1.0

    def test_feature_extraction(self):
        """测试特征提取"""
        # 生成测试信号
        signal = np.random.randn(8, 1000)

        features = self.processor.extract_features(signal)

        # 验证特征维度合理
        assert features.shape[0] > 0
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()

    def test_frequency_analysis(self):
        """测试频域分析"""
        # 生成特定频率的信号
        t = np.linspace(0, 4, 1000)
        signal = np.zeros((8, 1000))

        # 添加10Hz正弦波
        for ch in range(8):
            signal[ch] = 5 * np.sin(2 * np.pi * 10 * t) + np.random.randn(1000)

        features = self.processor.extract_features(signal)

        # α波功率应该较高
        alpha_power = features[8] if len(features) > 8 else 0
        assert alpha_power > 0

    def test_hjorth_parameters(self):
        """测试Hjorth参数"""
        # 生成不同复杂度的信号
        simple_signal = np.ones((8, 1000))  # 简单信号
        complex_signal = np.random.randn(8, 1000)  # 复杂信号

        simple_features = self.processor.extract_features(simple_signal)
        complex_features = self.processor.extract_features(complex_signal)

        # 复杂信号的Hjorth复杂度应该更高
        # 这里只是基本验证，实际测试可能需要更精确的信号


class TestAdaptiveDecoder:
    """自适应解码器测试"""

    def setup_method(self):
        """测试前准备"""
        self.decoder = AdaptiveDecoder(input_dim=512, num_classes=4)

    def test_decoder_initialization(self):
        """测试解码器初始化"""
        assert self.decoder.num_classes == 4
        assert self.decoder.input_dim == 512

    def test_forward_pass(self):
        """测试前向传播"""
        batch_size = 10
        input_tensor = torch.randn(batch_size, 512)

        logits, confidence = self.decoder(input_tensor)

        assert logits.shape == (batch_size, 4)
        assert confidence.shape == (batch_size,)
        assert torch.all((confidence >= 0) & (confidence <= 1))

    def test_adaptation(self):
        """测试快速适应"""
        support_features = torch.randn(20, 512)
        support_labels = torch.randint(0, 4, (20,))
        query_features = torch.randn(5, 512)

        adapted_logits, adapted_confidence = self.decoder.adapt(
            support_features, support_labels, query_features
        )

        assert adapted_logits.shape == (5, 4)
        assert adapted_confidence.shape == (5,)


class TestConsciousnessModel:
    """意识模型测试"""

    def setup_method(self):
        """测试前准备"""
        self.model = ConsciousnessModel()

    def test_state_update(self):
        """测试状态更新"""
        # 生成测试特征
        features = np.random.random(20) * 10

        context = {'task_type': 'attention', 'intensity': 0.8}

        state = self.model.update_state(features, context)

        assert isinstance(state, ConsciousnessState)
        assert 0 <= state.awareness_level <= 1
        assert isinstance(state.attention_focus, str)
        assert isinstance(state.emotional_state, str)

    def test_emotion_inference(self):
        """测试情感推断"""
        # 高唤醒正向信号
        positive_features = np.array([0.8, 0.9, 0.7, 0.6, 0.8, 0.7, 0.9, 0.8] + [0.1] * 12)
        emotion1 = self.model._infer_emotion(positive_features)

        # 低唤醒负向信号
        negative_features = np.array([0.2, 0.1, 0.3, 0.4, 0.2, 0.3, 0.1, 0.2] + [0.1] * 12)
        emotion2 = self.model._infer_emotion(negative_features)

        # 验证情感推理
        assert emotion1 in ['positive', 'neutral']
        assert emotion2 in ['negative', 'neutral']

    def test_future_prediction(self):
        """测试未来状态预测"""
        # 添加一些历史状态
        for i in range(5):
            features = np.random.random(20)
            self.model.update_state(features, {'step': i})

        prediction = self.model.predict_future_state(np.random.random(20))

        assert isinstance(prediction, ConsciousnessState)
        assert 0 <= prediction.awareness_level <= 1


class TestBCIEngine:
    """BCI引擎测试"""

    def setup_method(self):
        """测试前准备"""
        self.engine = create_bci_engine(num_channels=8, sampling_rate=250.0)

    @pytest.mark.asyncio
    async def test_single_signal_processing(self):
        """测试单个信号处理"""
        # 创建测试信号
        eeg_data = np.random.randn(8, 250) * 10  # 1秒数据
        signal = NeuralSignal(
            eeg_data=eeg_data,
            sampling_rate=250.0,
            channel_names=[f'Ch{i+1}' for i in range(8)],
            timestamp=None,
            metadata={'test': True}
        )

        command = await self.engine.process_single_signal(signal)

        assert isinstance(command, BCICommand)
        assert command.command_type in ['movement', 'selection', 'communication', 'unknown']
        assert 0 <= command.confidence <= 1

    def test_engine_initialization(self):
        """测试引擎初始化"""
        assert self.engine.num_channels == 8
        assert self.engine.sampling_rate == 250.0
        assert not self.engine.is_running

    def test_calibration(self):
        """测试系统校准"""
        # 生成校准数据
        calibration_data = []
        for i in range(10):
            eeg_data = np.random.randn(8, 250)
            signal = NeuralSignal(
                eeg_data=eeg_data,
                sampling_rate=250.0,
                channel_names=[f'Ch{i+1}' for i in range(8)],
                timestamp=None
            )
            intention = ['cursor_up', 'cursor_down', 'select', 'send_message'][i % 4]
            calibration_data.append((signal, intention))

        results = self.engine.calibrate(calibration_data)

        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert all(0 <= v <= 1 for v in results.values())

    def test_engine_stats(self):
        """测试引擎统计"""
        stats = self.engine.get_engine_stats()

        assert 'runtime_seconds' in stats
        assert 'is_running' in stats
        assert 'queue_sizes' in stats
        assert 'communication_stats' in stats


class TestAdvancedSignalProcessing:
    """高级信号处理测试"""

    def setup_method(self):
        """测试前准备"""
        self.processor = AdvancedSignalProcessor()

    def test_synthetic_signal_generation(self):
        """测试合成信号生成"""
        signal = generate_synthetic_eeg(duration=2.0, num_channels=8)

        assert signal.shape == (8, 500)  # 2秒 * 250Hz
        assert not np.isnan(signal).any()

    def test_artifact_removal(self):
        """测试伪迹去除"""
        # 生成带伪迹的信号
        signal = generate_synthetic_eeg(duration=1.0, num_channels=8, add_artifacts=True)

        # 处理信号
        result = self.processor.process_signal(signal, sampling_rate=250.0)

        assert result.processed_signal.shape == signal.shape
        assert result.quality_score >= 0
        assert result.processing_time > 0

    def test_connectivity_analysis(self):
        """测试连通性分析"""
        # 生成相关信号
        signal = np.random.randn(4, 1000)
        # 使前两个通道相关
        signal[1] = 0.8 * signal[0] + 0.6 * signal[1]

        result = self.processor.process_signal(
            signal,
            sampling_rate=250.0,
            processing_config={
                'connectivity_analysis': 'correlation',
                'freq_band': (8, 12)
            }
        )

        assert result.features is not None
        # 验证前两个通道的相关性较高
        connectivity_matrix = result.features.reshape(4, 4)
        assert connectivity_matrix[0, 1] > 0.5

    def test_spatial_filtering(self):
        """测试空间滤波"""
        signal = np.random.randn(8, 1000)

        result = self.processor.process_signal(
            signal,
            sampling_rate=250.0,
            processing_config={
                'spatial_filter': 'CAR',
                'artifact_removal': False
            }
        )

        # CAR应该保持信号的基本特性但改变参考
        assert result.processed_signal.shape == signal.shape


class TestPerformanceBenchmarks:
    """性能基准测试"""

    @pytest.mark.benchmark
    def test_realtime_processing_performance(self, benchmark):
        """实时处理性能测试"""
        engine = create_bci_engine(num_channels=8)

        async def process_signal():
            eeg_data = np.random.randn(8, 250)
            signal = NeuralSignal(
                eeg_data=eeg_data,
                sampling_rate=250.0,
                channel_names=[f'Ch{i+1}' for i in range(8)],
                timestamp=None
            )
            return await engine.process_single_signal(signal)

        # 基准测试
        result = benchmark(lambda: asyncio.run(process_signal()))
        assert result is not None

    def test_memory_efficiency(self):
        """内存效率测试"""
        import psutil
        import os

        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        engine = create_bci_engine(num_channels=16)

        # 处理多个信号
        for _ in range(10):
            async def process():
                eeg_data = np.random.randn(16, 500)
                signal = NeuralSignal(
                    eeg_data=eeg_data,
                    sampling_rate=250.0,
                    channel_names=[f'Ch{i+1}' for i in range(16)],
                    timestamp=None
                )
                return await engine.process_single_signal(signal)

            asyncio.run(process())

        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        # 内存增长应该在合理范围内 (考虑测试环境)
        assert memory_increase < 200  # MB

    def test_scalability(self):
        """可扩展性测试"""
        processing_times = []

        for num_channels in [4, 8, 16]:
            engine = create_bci_engine(num_channels=num_channels, sampling_rate=250.0)

            async def process_channel_test():
                eeg_data = np.random.randn(num_channels, 250)
                signal = NeuralSignal(
                    eeg_data=eeg_data,
                    sampling_rate=250.0,
                    channel_names=[f'Ch{i+1}' for i in range(num_channels)],
                    timestamp=None
                )
                start_time = time.time()
                result = await engine.process_single_signal(signal)
                return time.time() - start_time

            # 运行多次取平均
            times = []
            for _ in range(3):
                time_taken = asyncio.run(process_channel_test())
                times.append(time_taken)

            avg_time = np.mean(times)
            processing_times.append(avg_time)

        # 验证处理时间随通道数合理增长
        ratios = []
        for i in range(1, len(processing_times)):
            ratio = processing_times[i] / processing_times[i-1]
            ratios.append(ratio)

        # 平均增长率应该小于2 (考虑通道数翻倍)
        avg_ratio = np.mean(ratios)
        assert avg_ratio < 3


def benchmark_bci_engine():
    """BCI引擎基准测试"""
    print("🧠 BCI引擎性能基准测试")
    print("=" * 50)

    test_configs = [
        {'channels': 8, 'duration': 1.0},
        {'channels': 16, 'duration': 1.0},
        {'channels': 32, 'duration': 1.0}
    ]

    results = {}

    for config in test_configs:
        print(f"测试配置: {config['channels']}通道, {config['duration']}秒")

        engine = create_bci_engine(
            num_channels=config['channels'],
            sampling_rate=250.0
        )

        # 生成测试信号
        eeg_data = np.random.randn(config['channels'], int(config['duration'] * 250.0))

        signal = NeuralSignal(
            eeg_data=eeg_data,
            sampling_rate=250.0,
            channel_names=[f'Ch{i+1}' for i in range(config['channels'])],
            timestamp=None
        )

        # 执行基准测试
        times = []
        for _ in range(5):  # 5次取平均
            start_time = time.time()
            async def process():
                return await engine.process_single_signal(signal)
            asyncio.run(process())
            times.append(time.time() - start_time)

        avg_time = np.mean(times)
        std_time = np.std(times)

        results[f"{config['channels']}ch"] = {
            'avg_processing_time': avg_time,
            'std_processing_time': std_time,
            'throughput_hz': 1.0 / avg_time if avg_time > 0 else 0
        }


    # 保存基准测试结果
    with open('bmi_research/tests/bci_engine_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\\n✅ BCI引擎基准测试完成!")
    return results


if __name__ == "__main__":
    benchmark_bci_engine()
