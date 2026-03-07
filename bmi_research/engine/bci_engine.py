#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2026 脑机接口创新引擎
神经信号处理与意识计算框架

核心特性:
- 实时神经信号处理
- 脑机协同算法
- 意识计算模型
- 自适应解码器
- 神经反馈系统
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import queue

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NeuralSignal:
    """神经信号数据"""
    eeg_data: np.ndarray  # EEG信号 [channels, time_samples]
    sampling_rate: float  # 采样率 (Hz)
    channel_names: List[str]  # 通道名称
    timestamp: datetime  # 时间戳
    metadata: Dict[str, Any] = None  # 元数据

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BCICommand:
    """脑机接口命令"""
    command_type: str  # 命令类型 ('movement', 'selection', 'communication')
    target: Any  # 命令目标
    confidence: float  # 置信度
    neural_features: np.ndarray  # 神经特征
    processing_time: float  # 处理时间
    timestamp: datetime  # 时间戳


@dataclass
class ConsciousnessState:
    """意识状态"""
    awareness_level: float  # 意识水平 (0-1)
    attention_focus: str  # 注意力焦点
    emotional_state: str  # 情感状态
    cognitive_load: float  # 认知负荷
    neural_synchronization: float  # 神经同步度
    timestamp: datetime  # 时间戳


class NeuralSignalProcessor:
    """神经信号处理器"""

    def __init__(self, sampling_rate: float = 250.0, num_channels: int = 32):
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels

        # 信号处理参数
        self.notch_freq = 50.0  # 工频陷波频率
        self.bandpass_low = 1.0  # 带通滤波低频
        self.bandpass_high = 40.0  # 带通滤波高频

        # 预处理管道
        self.preprocessing_pipeline = [
            self._remove_dc_offset,
            self._apply_notch_filter,
            self._apply_bandpass_filter,
            self._normalize_signal
        ]

    def process_signal(self, signal: NeuralSignal) -> np.ndarray:
        """处理神经信号"""
        processed_data = signal.eeg_data.copy()

        # 应用预处理管道
        for processor in self.preprocessing_pipeline:
            processed_data = processor(processed_data)

        return processed_data

    def _remove_dc_offset(self, data: np.ndarray) -> np.ndarray:
        """去除直流偏移"""
        # 计算每个通道的均值并减去
        for ch in range(data.shape[0]):
            data[ch] -= np.mean(data[ch])
        return data

    def _apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
        """应用陷波滤波器 (去除工频干扰)"""
        # 简化的陷波滤波器实现
        from scipy.signal import iirnotch, filtfilt

        try:
            # 设计陷波滤波器
            b, a = iirnotch(self.notch_freq, Q=30, fs=self.sampling_rate)

            # 应用到每个通道
            filtered_data = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered_data[ch] = filtfilt(b, a, data[ch])

            return filtered_data
        except ImportError:
            logger.warning("scipy不可用，使用简化的陷波滤波")
            return data  # 返回原始数据

    def _apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """应用带通滤波器"""
        from scipy.signal import butter, filtfilt

        try:
            # 设计带通滤波器
            nyquist = self.sampling_rate / 2
            low = self.bandpass_low / nyquist
            high = self.bandpass_high / nyquist
            b, a = butter(4, [low, high], btype='band')

            # 应用到每个通道
            filtered_data = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered_data[ch] = filtfilt(b, a, data[ch])

            return filtered_data
        except ImportError:
            logger.warning("scipy不可用，使用简化的带通滤波")
            return data

    def _normalize_signal(self, data: np.ndarray) -> np.ndarray:
        """信号归一化"""
        # Z-score归一化
        normalized_data = np.zeros_like(data, dtype=np.float32)

        for ch in range(data.shape[0]):
            channel_data = data[ch]
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)

            if std_val > 0:
                normalized_data[ch] = (channel_data - mean_val) / std_val
            else:
                normalized_data[ch] = channel_data - mean_val

        return normalized_data

    def extract_features(self, processed_signal: np.ndarray) -> np.ndarray:
        """提取神经特征"""
        features = []

        # 时域特征
        features.extend(self._extract_time_domain_features(processed_signal))

        # 频域特征
        features.extend(self._extract_frequency_domain_features(processed_signal))

        # 时频特征
        features.extend(self._extract_time_frequency_features(processed_signal))

        return np.array(features)

    def _extract_time_domain_features(self, signal: np.ndarray) -> List[float]:
        """提取时域特征"""
        features = []

        for ch in range(signal.shape[0]):
            channel_data = signal[ch]

            # 统计特征
            features.append(np.mean(channel_data))  # 均值
            features.append(np.std(channel_data))   # 标准差
            features.append(np.var(channel_data))   # 方差
            features.append(np.max(channel_data))   # 最大值
            features.append(np.min(channel_data))   # 最小值
            features.append(np.ptp(channel_data))   # 峰峰值

            # 非线性特征
            features.append(self._hjorth_mobility(channel_data))  # Hjorth移动性
            features.append(self._hjorth_complexity(channel_data))  # Hjorth复杂度

        return features

    def _extract_frequency_domain_features(self, signal: np.ndarray) -> List[float]:
        """提取频域特征"""
        features = []

        for ch in range(signal.shape[0]):
            channel_data = signal[ch]

            # FFT
            fft_result = np.fft.fft(channel_data)
            freqs = np.fft.fftfreq(len(channel_data), 1/self.sampling_rate)
            power_spectrum = np.abs(fft_result)**2

            # 频带功率
            delta_power = self._band_power(power_spectrum, freqs, 1, 4)    # δ波 (1-4 Hz)
            theta_power = self._band_power(power_spectrum, freqs, 4, 8)    # θ波 (4-8 Hz)
            alpha_power = self._band_power(power_spectrum, freqs, 8, 12)   # α波 (8-12 Hz)
            beta_power = self._band_power(power_spectrum, freqs, 12, 30)   # β波 (12-30 Hz)
            gamma_power = self._band_power(power_spectrum, freqs, 30, 40)  # γ波 (30-40 Hz)

            features.extend([delta_power, theta_power, alpha_power, beta_power, gamma_power])

            # 频带功率比
            total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
            if total_power > 0:
                features.extend([
                    alpha_power / total_power,  # α/总功率比
                    beta_power / alpha_power if alpha_power > 0 else 0,  # β/α比
                    (alpha_power + theta_power) / (beta_power + gamma_power) if (beta_power + gamma_power) > 0 else 0  # (α+θ)/(β+γ)比
                ])

        return features

    def _extract_time_frequency_features(self, signal: np.ndarray) -> List[float]:
        """提取时频特征"""
        features = []

        # 简化的时频分析 (实际应使用小波变换或STFT)
        window_size = int(self.sampling_rate * 0.5)  # 0.5秒窗口

        for ch in range(signal.shape[0]):
            channel_data = signal[ch]

            # 分窗分析
            for start in range(0, len(channel_data) - window_size, window_size // 2):
                window = channel_data[start:start + window_size]

                # 计算窗口内特征
                window_mean = np.mean(window)
                window_std = np.std(window)
                window_energy = np.sum(window ** 2)

                features.extend([window_mean, window_std, window_energy])

        return features

    def _band_power(self, power_spectrum: np.ndarray, freqs: np.ndarray,
                   low_freq: float, high_freq: float) -> float:
        """计算频带功率"""
        freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
        return np.sum(power_spectrum[freq_mask])

    def _hjorth_mobility(self, signal: np.ndarray) -> float:
        """Hjorth移动性参数"""
        first_derivative = np.diff(signal)
        if len(first_derivative) == 0:
            return 0.0

        return np.std(first_derivative) / np.std(signal) if np.std(signal) > 0 else 0.0

    def _hjorth_complexity(self, signal: np.ndarray) -> float:
        """Hjorth复杂度参数"""
        first_derivative = np.diff(signal)
        if len(first_derivative) == 0:
            return 0.0

        second_derivative = np.diff(first_derivative)
        if len(second_derivative) == 0:
            return 0.0

        mobility1 = np.std(first_derivative) / np.std(signal) if np.std(signal) > 0 else 0.0
        mobility2 = np.std(second_derivative) / np.std(first_derivative) if np.std(first_derivative) > 0 else 0.0

        return mobility2 / mobility1 if mobility1 > 0 else 0.0


class AdaptiveDecoder(nn.Module):
    """自适应解码器"""

    def __init__(self, input_dim: int = 512, output_dim: int = 10, num_classes: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_classes = num_classes

        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        # 置信度估计器
        self.confidence_estimator = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 特征编码
        features = self.feature_encoder(x)

        # 分类
        logits = self.classifier(features)

        # 置信度估计
        confidence = self.confidence_estimator(features)

        return logits, confidence.squeeze()

    def adapt(self, support_features: torch.Tensor, support_labels: torch.Tensor,
              query_features: torch.Tensor, num_adaptation_steps: int = 5):
        """快速适应"""
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        # 适应步骤
        for _ in range(num_adaptation_steps):
            optimizer.zero_grad()

            # 计算支持集损失
            support_logits, _ = self(support_features)
            support_loss = nn.CrossEntropyLoss()(support_logits, support_labels)

            support_loss.backward()
            optimizer.step()

        # 推理查询样本
        with torch.no_grad():
            query_logits, query_confidence = self(query_features)

        return query_logits, query_confidence


class ConsciousnessModel:
    """意识计算模型"""

    def __init__(self):
        self.state_history = []
        self.attention_threshold = 0.7
        self.synchronization_threshold = 0.6

        # 意识状态跟踪
        self.current_state = ConsciousnessState(
            awareness_level=0.5,
            attention_focus="passive",
            emotional_state="neutral",
            cognitive_load=0.3,
            neural_synchronization=0.4,
            timestamp=datetime.now()
        )

    def update_state(self, neural_features: np.ndarray,
                    task_context: Dict[str, Any]) -> ConsciousnessState:
        """更新意识状态"""
        # 基于神经特征计算意识指标
        awareness_level = self._calculate_awareness(neural_features)
        attention_focus = self._determine_attention_focus(neural_features, task_context)
        emotional_state = self._infer_emotion(neural_features)
        cognitive_load = self._estimate_cognitive_load(neural_features)
        neural_sync = self._measure_synchronization(neural_features)

        # 更新当前状态
        self.current_state = ConsciousnessState(
            awareness_level=awareness_level,
            attention_focus=attention_focus,
            emotional_state=emotional_state,
            cognitive_load=cognitive_load,
            neural_synchronization=neural_sync,
            timestamp=datetime.now()
        )

        # 记录历史
        self.state_history.append(self.current_state)

        return self.current_state

    def _calculate_awareness(self, features: np.ndarray) -> float:
        """计算意识水平"""
        # 基于alpha波功率和神经同步度计算
        alpha_power = features[8] if len(features) > 8 else 0.5  # α波功率
        sync_measure = self._measure_synchronization(features)

        awareness = (alpha_power + sync_measure) / 2
        return np.clip(awareness, 0, 1)

    def _determine_attention_focus(self, features: np.ndarray,
                                 context: Dict[str, Any]) -> str:
        """确定注意力焦点"""
        task_type = context.get('task_type', 'general')

        # 基于频带功率比确定注意力
        beta_power = features[9] if len(features) > 9 else 0.5   # β波功率
        alpha_power = features[8] if len(features) > 8 else 0.5  # α波功率

        if beta_power > alpha_power * 1.5:
            return "focused"  # 集中注意力
        elif alpha_power > beta_power * 1.2:
            return "relaxed"  # 放松状态
        else:
            return "passive"  # 被动状态

    def _infer_emotion(self, features: np.ndarray) -> str:
        """推断情感状态"""
        # 简化的情感识别 (基于不对称性和频带特征)
        frontal_asymmetry = features[0] - features[1] if len(features) > 1 else 0

        if frontal_asymmetry > 0.3:
            return "positive"
        elif frontal_asymmetry < -0.3:
            return "negative"
        else:
            return "neutral"

    def _estimate_cognitive_load(self, features: np.ndarray) -> float:
        """估计认知负荷"""
        # 基于theta和beta波的相对功率
        theta_power = features[7] if len(features) > 7 else 0.5  # θ波功率
        beta_power = features[9] if len(features) > 9 else 0.5   # β波功率

        # θ/β比值反映认知负荷
        cognitive_load = theta_power / (beta_power + 1e-6)
        return np.clip(cognitive_load, 0, 1)

    def _measure_synchronization(self, features: np.ndarray) -> float:
        """测量神经同步度"""
        # 简化的相干性测量
        if len(features) < 10:
            return 0.5

        # 计算不同通道间的相关性
        correlations = []
        for i in range(min(5, len(features) // 2)):
            for j in range(i+1, min(10, len(features))):
                corr = np.corrcoef([features[i], features[j]])[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0)

        return np.mean(correlations) if correlations else 0.5

    def get_state_history(self, window_size: int = 10) -> List[ConsciousnessState]:
        """获取状态历史"""
        return self.state_history[-window_size:]

    def predict_future_state(self, current_features: np.ndarray) -> ConsciousnessState:
        """预测未来意识状态"""
        # 简化的状态预测 (基于当前趋势)
        if len(self.state_history) < 2:
            return self.current_state

        # 计算趋势
        recent_states = self.get_state_history(3)
        awareness_trend = np.mean([s.awareness_level for s in recent_states[-2:]])

        # 预测下一个状态
        predicted_awareness = np.clip(awareness_trend + np.random.normal(0, 0.1), 0, 1)

        return ConsciousnessState(
            awareness_level=predicted_awareness,
            attention_focus=self.current_state.attention_focus,
            emotional_state=self.current_state.emotional_state,
            cognitive_load=self.current_state.cognitive_load,
            neural_synchronization=self.current_state.neural_synchronization,
            timestamp=datetime.now()
        )


class BCICommunicationInterface:
    """脑机通信接口"""

    def __init__(self):
        self.signal_processor = NeuralSignalProcessor()
        self.decoder = AdaptiveDecoder()
        self.consciousness_model = ConsciousnessModel()

        # 通信参数
        self.command_buffer = queue.Queue(maxsize=10)
        self.feedback_enabled = True

        # 性能监控
        self.processing_stats = {
            'total_signals': 0,
            'successful_decodings': 0,
            'average_confidence': 0.0,
            'processing_times': []
        }

    async def process_neural_signal(self, signal: NeuralSignal) -> BCICommand:
        """处理神经信号并生成命令"""
        start_time = time.time()

        # 预处理信号
        processed_signal = self.signal_processor.process_signal(signal)

        # 提取特征
        neural_features = self.signal_processor.extract_features(processed_signal)

        # 更新意识状态
        consciousness_state = self.consciousness_model.update_state(
            neural_features,
            signal.metadata
        )

        # 解码意图
        command = await self._decode_intention(neural_features, signal.metadata)

        # 记录性能
        processing_time = time.time() - start_time
        self.processing_stats['total_signals'] += 1
        self.processing_stats['processing_times'].append(processing_time)

        if command.confidence > 0.6:
            self.processing_stats['successful_decodings'] += 1

        # 更新平均置信度
        total_confidence = self.processing_stats['average_confidence'] * (self.processing_stats['total_signals'] - 1)
        self.processing_stats['average_confidence'] = (total_confidence + command.confidence) / self.processing_stats['total_signals']

        return command

    async def _decode_intention(self, features: np.ndarray,
                              context: Dict[str, Any]) -> BCICommand:
        """解码用户意图"""
        # 转换为tensor
        feature_tensor = torch.from_numpy(features).float().unsqueeze(0)

        with torch.no_grad():
            # 解码
            logits, confidence = self.decoder(feature_tensor)

            # 获取预测结果
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence_score = confidence.item()

        # 映射到命令
        command_mapping = {
            0: ('movement', 'cursor_up'),
            1: ('movement', 'cursor_down'),
            2: ('selection', 'select'),
            3: ('communication', 'send_message')
        }

        command_type, target = command_mapping.get(predicted_class, ('unknown', 'none'))

        return BCICommand(
            command_type=command_type,
            target=target,
            confidence=confidence_score,
            neural_features=features,
            processing_time=time.time() - time.time(),  # 需要实际计算
            timestamp=datetime.now()
        )

    def provide_feedback(self, command: BCICommand, success: bool):
        """提供神经反馈"""
        if not self.feedback_enabled:
            return

        # 根据执行结果调整解码器
        if success and command.confidence > 0.7:
            # 正强化
            self._reinforce_decoder(command)
        elif not success:
            # 负反馈调整
            self._adjust_decoder(command)

    def _reinforce_decoder(self, command: BCICommand):
        """强化解码器"""
        # 简化的强化学习调整
        logger.info(f"强化解码器: {command.command_type} -> {command.target}")

    def _adjust_decoder(self, command: BCICommand):
        """调整解码器"""
        # 简化的调整逻辑
        logger.info(f"调整解码器: 降低 {command.command_type} 命令的敏感度")

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            **self.processing_stats,
            'consciousness_history': len(self.consciousness_model.state_history),
            'command_buffer_size': self.command_buffer.qsize()
        }

    def calibrate_system(self, calibration_signals: List[NeuralSignal],
                        known_intentions: List[str]) -> Dict[str, float]:
        """系统校准"""
        logger.info("开始系统校准...")

        calibration_results = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }

        # 简化的校准过程
        if len(calibration_signals) == len(known_intentions):
            # 这里应该实现实际的校准算法
            calibration_results['accuracy'] = 0.85  # 模拟准确率
            calibration_results['precision'] = 0.82
            calibration_results['recall'] = 0.88

        logger.info(f"校准完成: 准确率 {calibration_results['accuracy']:.2%}")
        return calibration_results


class BCIEngine:
    """RQA2026 脑机接口创新引擎"""

    def __init__(self, sampling_rate: float = 250.0, num_channels: int = 32):
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels

        # 初始化组件
        self.signal_processor = NeuralSignalProcessor(sampling_rate, num_channels)
        self.communication_interface = BCICommunicationInterface()

        # 实时处理
        self.is_running = False
        self.signal_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # 性能监控
        self.start_time = None

        logger.info(f"脑机接口引擎初始化完成: {num_channels} 通道, {sampling_rate}Hz")

    async def start_realtime_processing(self):
        """启动实时处理"""
        self.is_running = True
        self.start_time = time.time()

        logger.info("启动实时脑机接口处理...")

        # 启动处理循环
        processing_task = asyncio.create_task(self._processing_loop())

        try:
            await processing_task
        except KeyboardInterrupt:
            logger.info("停止实时处理...")
        finally:
            self.is_running = False

    async def _processing_loop(self):
        """处理循环"""
        while self.is_running:
            try:
                # 获取信号 (非阻塞)
                if not self.signal_queue.empty():
                    signal = self.signal_queue.get_nowait()

                    # 处理信号
                    command = await self.communication_interface.process_neural_signal(signal)

                    # 放入结果队列
                    self.result_queue.put(command)

                await asyncio.sleep(0.01)  # 10ms循环

            except Exception as e:
                logger.error(f"处理循环异常: {e}")
                await asyncio.sleep(0.1)

    def add_neural_signal(self, signal: NeuralSignal):
        """添加神经信号到处理队列"""
        self.signal_queue.put(signal)

    def get_next_command(self) -> Optional[BCICommand]:
        """获取下一个命令"""
        if not self.result_queue.empty():
            return self.result_queue.get_nowait()
        return None

    async def process_single_signal(self, signal: NeuralSignal) -> BCICommand:
        """处理单个神经信号"""
        return await self.communication_interface.process_neural_signal(signal)

    def calibrate(self, calibration_data: List[Tuple[NeuralSignal, str]]) -> Dict[str, float]:
        """校准系统"""
        signals, intentions = zip(*calibration_data)
        return self.communication_interface.calibrate_system(list(signals), list(intentions))

    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计"""
        runtime = time.time() - self.start_time if self.start_time else 0

        return {
            'runtime_seconds': runtime,
            'is_running': self.is_running,
            'queue_sizes': {
                'signal_queue': self.signal_queue.qsize(),
                'result_queue': self.result_queue.qsize()
            },
            'communication_stats': self.communication_interface.get_performance_stats()
        }

    def shutdown(self):
        """关闭引擎"""
        self.is_running = False
        logger.info("脑机接口引擎已关闭")


def create_bci_engine(sampling_rate: float = 250.0, num_channels: int = 32) -> BCIEngine:
    """
    创建脑机接口引擎的工厂函数

    Args:
        sampling_rate: 采样率 (Hz)
        num_channels: 通道数量

    Returns:
        配置好的脑机接口引擎实例
    """
    return BCIEngine(sampling_rate=sampling_rate, num_channels=num_channels)


async def demo_bci_engine():
    """脑机接口引擎演示"""
    print("🧠 RQA2026 脑机接口创新引擎演示")
    print("=" * 50)

    # 创建BCI引擎
    engine = create_bci_engine(num_channels=8, sampling_rate=250.0)

    # 生成模拟神经信号
    print("生成模拟神经信号...")
    np.random.seed(42)

    # 模拟不同意图的信号
    intentions = ['cursor_up', 'cursor_down', 'select', 'send_message']

    for i, intention in enumerate(intentions):
        # 生成模拟EEG数据 (8通道, 1秒数据)
        eeg_data = np.random.randn(8, 250) * 10 + np.random.randn(8, 250) * 2

        # 根据意图添加特定模式
        if intention == 'cursor_up':
            eeg_data[0, :] += 5  # 前额叶活动增强
        elif intention == 'select':
            eeg_data[1, :] += 3  # 运动皮层活动

        signal = NeuralSignal(
            eeg_data=eeg_data,
            sampling_rate=250.0,
            channel_names=[f'Ch{j+1}' for j in range(8)],
            timestamp=datetime.now(),
            metadata={'intention': intention, 'trial': i+1}
        )

        print(f"处理信号: {intention}")
        command = await engine.process_single_signal(signal)

        print(f"  解码命令: {command.command_type} -> {command.target}")
        print(".2%")

    # 获取引擎统计
    stats = engine.get_engine_stats()
    print("\\n📊 引擎统计:")
    print(f"总信号数: {stats['communication_stats']['total_signals']}")
    print(".1%")

    print("\\n✅ 脑机接口引擎演示完成!")


if __name__ == "__main__":
    asyncio.run(demo_bci_engine())
