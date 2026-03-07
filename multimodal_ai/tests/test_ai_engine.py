#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2026 AI深度集成创新引擎测试套件

测试覆盖:
- 多模态数据处理
- 认知推理能力
- 自适应学习
- 性能基准测试
"""

import pytest
import numpy as np
import torch
import asyncio
from pathlib import Path
import time
import json
from typing import Dict, List, Any

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from multimodal_ai.engine.ai_engine import (
    AIEngine, MultimodalInput, MultimodalOutput,
    create_ai_engine, VisionProcessor, AudioProcessor,
    TextProcessor, SensorProcessor
)
from multimodal_ai.models.cognitive_models import create_cognitive_controller


class TestModalityProcessors:
    """模态处理器测试"""

    def test_vision_processor(self):
        """测试视觉处理器"""
        processor = VisionProcessor()

        # 测试图像预处理
        test_image = np.random.random((64, 64, 3))
        processed = processor.preprocess(test_image)

        assert processed.shape[0] == 3  # CHW格式
        assert processed.max() <= 1.0
        assert processed.min() >= 0.0

        # 测试特征提取
        features = processor.extract_features(processed)
        assert features.shape[0] == processor.feature_dim
        assert not np.isnan(features).any()

    def test_audio_processor(self):
        """测试音频处理器"""
        processor = AudioProcessor()

        # 测试音频预处理
        test_audio = np.random.random((44100,))  # 1秒音频
        processed = processor.preprocess(test_audio)

        assert processed.shape[0] == 1  # 单声道
        assert processed.max() <= 1.0
        assert processed.min() >= -1.0

        # 测试特征提取
        features = processor.extract_features(processed)
        assert features.shape[0] == processor.feature_dim

    def test_text_processor(self):
        """测试文本处理器"""
        processor = TextProcessor()

        # 测试文本预处理
        test_text = "This is a test sentence for multimodal AI processing."
        processed = processor.preprocess(test_text)

        assert isinstance(processed, np.ndarray)
        assert processed.dtype == np.int64

        # 测试特征提取
        features = processor.extract_features(processed)
        assert features.shape[0] == processor.feature_dim

    def test_sensor_processor(self):
        """测试传感器处理器"""
        processor = SensorProcessor(sensor_dims=10)

        # 测试传感器数据预处理
        test_sensor = np.random.random((10,))
        processed = processor.preprocess(test_sensor)

        assert processed.shape == (10,)
        assert processed.max() <= 1.0
        assert processed.min() >= 0.0

        # 测试特征提取
        features = processor.extract_features(processed)
        assert features.shape[0] == processor.feature_dim


class TestAIEngine:
    """AI引擎测试"""

    def setup_method(self):
        """测试前准备"""
        self.engine = create_ai_engine(modalities=["vision", "text"])

    def test_engine_initialization(self):
        """测试引擎初始化"""
        assert len(self.engine.modalities) == 2
        assert "vision" in self.engine.processors
        assert "text" in self.engine.processors
        assert self.engine.fusion_network is not None

    @pytest.mark.asyncio
    async def test_multimodal_processing(self):
        """测试多模态处理"""
        # 创建测试输入
        input_data = MultimodalInput(
            visual=np.random.random((32, 32, 3)),
            text="Test input for multimodal processing",
            metadata={"test": True}
        )

        # 处理输入
        result = await self.engine.process_multimodal_input(input_data)

        # 验证结果
        assert isinstance(result, MultimodalOutput)
        assert isinstance(result.prediction, str)
        assert 0 <= result.confidence <= 1
        assert result.processing_time > 0
        assert len(result.modalities_used) > 0
        assert "vision" in result.modalities_used
        assert "text" in result.modalities_used

    @pytest.mark.asyncio
    async def test_single_modality_processing(self):
        """测试单模态处理"""
        # 只使用文本模态
        input_data = MultimodalInput(
            text="Single modality test input",
            metadata={"modality_test": True}
        )

        result = await self.engine.process_multimodal_input(input_data)

        assert result.modalities_used == ["text"]
        assert result.prediction is not None

    def test_adaptive_learning(self):
        """测试自适应学习"""
        initial_confidence = self.engine.cognitive_state.confidence_level

        # 提供正面反馈
        feedback = {
            'type': 'correction',
            'focus': 'text_processing',
            'accuracy': 0.9
        }

        self.engine.adapt_to_feedback(feedback)

        # 验证适应效果
        updated_state = self.engine.get_cognitive_state()
        assert updated_state.attention_focus == 'text_processing'
        assert len(updated_state.adaptation_history) > 0

    def test_cognitive_state_management(self):
        """测试认知状态管理"""
        state = self.engine.get_cognitive_state()

        assert hasattr(state, 'working_memory')
        assert hasattr(state, 'attention_focus')
        assert hasattr(state, 'emotional_state')
        assert hasattr(state, 'confidence_level')

    def test_model_persistence(self):
        """测试模型持久化"""
        import tempfile
        import os

        # 保存模型
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            self.engine.save_model(temp_path)
            assert os.path.exists(temp_path)

            # 加载模型
            new_engine = create_ai_engine(modalities=["vision", "text"])
            new_engine.load_model(temp_path)

            # 验证加载成功
            assert new_engine.modalities == self.engine.modalities

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestCognitiveModels:
    """认知模型测试"""

    def test_cognitive_controller(self):
        """测试认知控制器"""
        controller = create_cognitive_controller(feature_dim=128)

        # 测试输入处理
        input_features = torch.randn(128)
        context = {
            'situation': 'test',
            'intensity': 0.7,
            'feedback_available': True,
            'target': 0.8
        }

        result = controller.process_input(input_features, context)

        assert 'output' in result
        assert 'attention_weights' in result
        assert 'emotion' in result
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 1

    def test_adaptive_learning(self):
        """测试自适应学习"""
        controller = create_cognitive_controller()

        # 模拟学习过程
        for i in range(5):
            input_features = torch.randn(256)
            context = {
                'step': i,
                'feedback_available': True,
                'target': np.random.random()
            }

            result = controller.process_input(input_features, context)

            # 验证学习统计
            stats = controller.get_cognitive_stats()
            assert 'learning_stats' in stats
            assert 'memory_stats' in stats

    def test_memory_system(self):
        """测试记忆系统"""
        controller = create_cognitive_controller()

        # 存储多个经验
        for i in range(10):
            input_features = torch.randn(256)
            context = {'experience_id': i}
            controller.process_input(input_features, context)

        # 验证记忆统计
        stats = controller.get_cognitive_stats()
        memory_stats = stats['memory_stats']

        assert memory_stats['working_memory_size'] > 0
        assert memory_stats['episodic_memory_size'] > 0

    def test_emotional_model(self):
        """测试情感模型"""
        controller = create_cognitive_controller()

        # 测试情感更新
        input_features = torch.randn(256)
        context = {'intensity': 0.9}
        result = controller.process_input(input_features, context)

        assert result['emotion'] in ['joy', 'sadness', 'surprise', 'neutral']

        # 测试情感响应
        emotional_response = controller.emotion.get_emotional_response('test_situation')
        assert 'emotion' in emotional_response
        assert 'intensity' in emotional_response


class TestPerformanceBenchmarks:
    """性能基准测试"""

    @pytest.mark.benchmark
    def test_multimodal_processing_performance(self, benchmark):
        """多模态处理性能测试"""
        engine = create_ai_engine(modalities=["vision", "text"])

        async def process_input():
            input_data = MultimodalInput(
                visual=np.random.random((64, 64, 3)),
                text="Performance benchmark test input",
                metadata={"benchmark": True}
            )
            return await engine.process_multimodal_input(input_data)

        # 基准测试
        result = benchmark(lambda: asyncio.run(process_input()))
        assert result is not None

    def test_memory_efficiency(self):
        """内存效率测试"""
        import psutil
        import os

        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        engine = create_ai_engine(modalities=["vision", "audio", "text", "sensor"])

        # 处理大数据
        large_input = MultimodalInput(
            visual=np.random.random((256, 256, 3)),
            audio=np.random.random((44100,)),
            text="Large input test for memory efficiency analysis. " * 100,
            sensor=np.random.random((50,))
        )

        async def process_large_input():
            return await engine.process_multimodal_input(large_input)

        asyncio.run(process_large_input())

        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        # 内存增长应该在合理范围内
        assert memory_increase < 500  # MB

    def test_concurrent_processing(self):
        """并发处理测试"""
        engine = create_ai_engine(modalities=["text"])

        async def concurrent_task(task_id: int):
            input_data = MultimodalInput(
                text=f"Concurrent processing test input {task_id}",
                metadata={"task_id": task_id}
            )
            result = await engine.process_multimodal_input(input_data)
            return result

        async def run_concurrent_tasks():
            tasks = [concurrent_task(i) for i in range(10)]
            results = await asyncio.gather(*tasks)
            return results

        # 执行并发测试
        start_time = time.time()
        results = asyncio.run(run_concurrent_tasks())
        total_time = time.time() - start_time

        assert len(results) == 10
        assert total_time < 30  # 并发处理应该很快完成

    def test_scalability(self):
        """可扩展性测试"""
        processing_times = []

        for num_modalities in [1, 2, 3, 4]:
            modalities = ["vision", "audio", "text", "sensor"][:num_modalities]
            engine = create_ai_engine(modalities=modalities)

            start_time = time.time()

            async def process_with_modalities():
                input_data = MultimodalInput(
                    visual=np.random.random((64, 64, 3)) if "vision" in modalities else None,
                    audio=np.random.random((22050,)) if "audio" in modalities else None,
                    text="Scalability test input" if "text" in modalities else None,
                    sensor=np.random.random((10,)) if "sensor" in modalities else None
                )
                return await engine.process_multimodal_input(input_data)

            asyncio.run(process_with_modalities())
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

        # 验证处理时间随模态数量的合理增长
        for i in range(1, len(processing_times)):
            ratio = processing_times[i] / processing_times[i-1]
            assert ratio < 3  # 处理时间增长应该相对线性


def benchmark_ai_engine():
    """AI引擎基准测试"""
    results = {}

    print("🏃 AI引擎性能基准测试...")

    # 单模态基准测试
    for modality in ["vision", "audio", "text", "sensor"]:
        engine = create_ai_engine(modalities=[modality])

        async def single_modality_test():
            if modality == "vision":
                input_data = MultimodalInput(visual=np.random.random((128, 128, 3)))
            elif modality == "audio":
                input_data = MultimodalInput(audio=np.random.random((44100,)))
            elif modality == "text":
                input_data = MultimodalInput(text="Benchmark test input for AI engine performance evaluation.")
            else:  # sensor
                input_data = MultimodalInput(sensor=np.random.random((20,)))

            start_time = time.time()
            result = await engine.process_multimodal_input(input_data)
            processing_time = time.time() - start_time

            return processing_time, result.confidence

        # 执行多次测试
        times = []
        confidences = []
        for _ in range(5):
            time_taken, confidence = asyncio.run(single_modality_test())
            times.append(time_taken)
            confidences.append(confidence)

        results[modality] = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'avg_confidence': np.mean(confidences),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }

        print(f"{modality}: {results[modality]['avg_time']:.3f}±{results[modality]['std_time']:.3f}s")

    # 保存基准测试结果
    benchmark_file = Path(__file__).parent / 'ai_engine_benchmark_results.json'
    with open(benchmark_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"基准测试结果已保存到: {benchmark_file}")
    return results


if __name__ == "__main__":
    # 运行基准测试
    benchmark_results = benchmark_ai_engine()

    print("\\n📊 基准测试汇总:")
    for modality, result in benchmark_results.items():
        print(f"{modality}: {result['avg_time']:.3f}s (置信度: {result['avg_confidence']:.2%})")

    print("\\n✅ AI引擎基准测试完成!")
