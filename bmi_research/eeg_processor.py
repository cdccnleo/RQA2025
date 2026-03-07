#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2026脑机接口EEG信号处理模块

实现EEG信号的实时处理、注意力监测和交易辅助决策。

作者: RQA2026脑机接口引擎团队
时间: 2025年12月3日
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, welch
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import time
from datetime import datetime


class EEGSignalProcessor:
    """EEG信号处理器"""

    def __init__(self,
                 sampling_rate: int = 250,
                 buffer_size: int = 1000,
                 num_channels: int = 4):
        """
        初始化EEG信号处理器

        Args:
            sampling_rate: 采样率 (Hz)
            buffer_size: 缓冲区大小
            num_channels: EEG通道数量
        """
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.num_channels = num_channels

        # EEG通道标签 (标准10-20系统简化版)
        self.channel_names = ['Fz', 'Cz', 'Pz', 'Oz'] if num_channels == 4 else \
                           [f'Ch{i+1}' for i in range(num_channels)]

        # 信号缓冲区
        self.signal_buffers = [deque(maxlen=buffer_size) for _ in range(num_channels)]

        # 频段定义 (Hz)
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 45)
        }

        # 注意力状态阈值
        self.attention_thresholds = {
            'high_attention': {'alpha_beta_ratio': 0.7, 'beta_power': 0.6},
            'medium_attention': {'alpha_beta_ratio': 0.4, 'beta_power': 0.4},
            'low_attention': {'alpha_beta_ratio': 0.2, 'beta_power': 0.2}
        }

        # 历史数据存储
        self.attention_history = []
        self.power_history = {band: [] for band in self.frequency_bands.keys()}

        # 实时状态
        self.current_attention_level = 'unknown'
        self.last_update_time = time.time()

    def add_eeg_data(self, eeg_data: np.ndarray):
        """
        添加新的EEG数据

        Args:
            eeg_data: EEG数据数组 (channels, samples) 或 (samples,) for single channel
        """
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)

        channels, samples = eeg_data.shape

        for ch in range(min(channels, self.num_channels)):
            for sample in eeg_data[ch]:
                self.signal_buffers[ch].append(sample)

        self.last_update_time = time.time()

    def preprocess_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """
        EEG信号预处理

        Args:
            signal_data: 原始EEG信号

        Returns:
            np.ndarray: 预处理后的信号
        """
        # 去趋势
        signal_data = signal.detrend(signal_data)

        # 带通滤波 (0.5-45 Hz, 适合EEG)
        nyquist = self.sampling_rate / 2
        low_freq = 0.5 / nyquist
        high_freq = 45 / nyquist
        b, a = butter(4, [low_freq, high_freq], btype='band')
        signal_data = filtfilt(b, a, signal_data)

        # 去除工频干扰 (50/60 Hz)
        notch_freq = 50.0 / nyquist  # 假设50Hz工频
        b_notch, a_notch = butter(4, [notch_freq - 0.01, notch_freq + 0.01], btype='bandstop')
        signal_data = filtfilt(b_notch, a_notch, signal_data)

        return signal_data

    def extract_frequency_bands(self, signal_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取不同频段的功率

        Args:
            signal_data: 预处理后的EEG信号

        Returns:
            Dict[str, np.ndarray]: 各频段功率
        """
        freq_bands_power = {}

        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # 设计带通滤波器
            nyquist = self.sampling_rate / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            b, a = butter(4, [low, high], btype='band')

            # 滤波
            filtered_signal = filtfilt(b, a, signal_data)

            # 计算功率
            power = np.mean(filtered_signal ** 2)

            freq_bands_power[band_name] = power

        return freq_bands_power

    def calculate_attention_metrics(self, freq_bands_power: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        计算注意力相关指标

        Args:
            freq_bands_power: 各频段功率

        Returns:
            Dict[str, float]: 注意力指标
        """
        # Alpha/Beta比率 (注意力指标)
        alpha_power = freq_bands_power.get('alpha', 0)
        beta_power = freq_bands_power.get('beta', 0)
        alpha_beta_ratio = alpha_power / beta_power if beta_power > 0 else 0

        # Theta/Beta比率 (认知负荷指标)
        theta_power = freq_bands_power.get('theta', 0)
        theta_beta_ratio = theta_power / beta_power if beta_power > 0 else 0

        # Beta功率相对水平
        total_power = sum(freq_bands_power.values())
        beta_relative = beta_power / total_power if total_power > 0 else 0

        # Alpha抑制指数 (专注度指标)
        alpha_relative = alpha_power / total_power if total_power > 0 else 0

        return {
            'alpha_beta_ratio': alpha_beta_ratio,
            'theta_beta_ratio': theta_beta_ratio,
            'beta_relative_power': beta_relative,
            'alpha_relative_power': alpha_relative,
            'total_power': total_power
        }

    def classify_attention_level(self, attention_metrics: Dict[str, float]) -> str:
        """
        分类注意力水平

        Args:
            attention_metrics: 注意力指标

        Returns:
            str: 注意力水平 ('high', 'medium', 'low')
        """
        alpha_beta_ratio = attention_metrics['alpha_beta_ratio']
        beta_power = attention_metrics['beta_relative_power']

        if alpha_beta_ratio <= self.attention_thresholds['high_attention']['alpha_beta_ratio'] and \
           beta_power >= self.attention_thresholds['high_attention']['beta_power']:
            return 'high'
        elif alpha_beta_ratio <= self.attention_thresholds['medium_attention']['alpha_beta_ratio'] and \
             beta_power >= self.attention_thresholds['medium_attention']['beta_power']:
            return 'medium'
        else:
            return 'low'

    def analyze_attention_state(self) -> Dict[str, Any]:
        """
        分析当前注意力状态

        Returns:
            Dict[str, Any]: 注意力分析结果
        """
        if not all(len(buffer) >= 100 for buffer in self.signal_buffers):
            return {
                'status': 'insufficient_data',
                'message': '需要更多EEG数据进行分析'
            }

        # 获取当前信号数据
        current_signals = []
        for buffer in self.signal_buffers:
            signal_data = np.array(list(buffer))
            processed_signal = self.preprocess_signal(signal_data)
            current_signals.append(processed_signal)

        current_signals = np.array(current_signals)

        # 计算各通道的频段功率
        channel_powers = {}
        for ch in range(self.num_channels):
            freq_bands_power = self.extract_frequency_bands(current_signals[ch])
            channel_powers[self.channel_names[ch]] = freq_bands_power

            # 更新历史记录
            for band, power in freq_bands_power.items():
                if len(self.power_history[band]) >= 100:  # 保持历史记录长度
                    self.power_history[band].pop(0)
                self.power_history[band].append(power)

        # 计算平均注意力指标
        avg_powers = {}
        for band in self.frequency_bands.keys():
            channel_band_powers = [channel_powers[ch][band] for ch in self.channel_names]
            avg_powers[band] = np.mean(channel_band_powers)

        attention_metrics = self.calculate_attention_metrics(avg_powers)
        attention_level = self.classify_attention_level(attention_metrics)

        # 更新注意力历史
        attention_record = {
            'timestamp': datetime.now(),
            'level': attention_level,
            'metrics': attention_metrics,
            'channel_powers': channel_powers
        }
        self.attention_history.append(attention_record)

        # 保持历史记录长度
        if len(self.attention_history) > 50:
            self.attention_history.pop(0)

        self.current_attention_level = attention_level

        return {
            'status': 'success',
            'attention_level': attention_level,
            'attention_metrics': attention_metrics,
            'channel_powers': channel_powers,
            'confidence': self._calculate_confidence(attention_metrics)
        }

    def _calculate_confidence(self, attention_metrics: Dict[str, float]) -> float:
        """计算分析置信度"""
        # 基于信号质量和一致性计算置信度
        alpha_beta_ratio = attention_metrics['alpha_beta_ratio']
        beta_power = attention_metrics['beta_relative_power']

        # 理想范围内的置信度更高
        ratio_confidence = 1.0 if 0.1 <= alpha_beta_ratio <= 1.0 else 0.5
        power_confidence = 1.0 if beta_power >= 0.3 else 0.7

        return (ratio_confidence + power_confidence) / 2

    def generate_trading_signals(self, attention_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于注意力状态生成交易信号

        Args:
            attention_analysis: 注意力分析结果

        Returns:
            Dict[str, Any]: 交易信号
        """
        if attention_analysis['status'] != 'success':
            return {'signals': [], 'recommendations': []}

        attention_level = attention_analysis['attention_level']
        confidence = attention_analysis['confidence']

        signals = []
        recommendations = []

        # 基于注意力水平生成信号
        if attention_level == 'high' and confidence > 0.7:
            signals.append({
                'type': 'HIGH_ATTENTION_TRADING',
                'strength': confidence,
                'description': '检测到高度专注状态，适合进行复杂交易决策',
                'action': 'proceed_with_analysis'
            })
            recommendations.append('当前脑电波显示高度专注，建议进行技术分析和交易执行')

        elif attention_level == 'medium' and confidence > 0.6:
            signals.append({
                'type': 'MODERATE_ATTENTION_MONITORING',
                'strength': confidence,
                'description': '中等注意力水平，适合市场监控',
                'action': 'monitor_only'
            })
            recommendations.append('中等专注度，建议继续监控市场但避免重大交易决策')

        elif attention_level == 'low':
            signals.append({
                'type': 'LOW_ATTENTION_WARNING',
                'strength': confidence,
                'description': '注意力不集中，建议暂停交易活动',
                'action': 'pause_trading'
            })
            recommendations.append('检测到注意力不足，建议休息或暂停交易决策')

        return {
            'signals': signals,
            'recommendations': recommendations,
            'attention_level': attention_level,
            'confidence': confidence
        }

    def create_sample_eeg_data(self, duration_seconds: int = 10) -> np.ndarray:
        """
        创建示例EEG数据用于演示

        Args:
            duration_seconds: 数据时长(秒)

        Returns:
            np.ndarray: 示例EEG数据
        """
        num_samples = duration_seconds * self.sampling_rate
        time_points = np.linspace(0, duration_seconds, num_samples)

        # 为每个通道生成不同的信号模式
        eeg_data = np.zeros((self.num_channels, num_samples))

        for ch in range(self.num_channels):
            # 基础alpha波 (10 Hz)
            alpha_wave = 50 * np.sin(2 * np.pi * 10 * time_points)

            # 添加beta波 (20 Hz)
            beta_wave = 30 * np.sin(2 * np.pi * 20 * time_points + ch * np.pi/4)

            # 添加theta波 (6 Hz)
            theta_wave = 20 * np.sin(2 * np.pi * 6 * time_points + ch * np.pi/2)

            # 添加随机噪声
            noise = 10 * np.random.randn(num_samples)

            # 组合信号
            eeg_data[ch] = alpha_wave + beta_wave + theta_wave + noise

        return eeg_data

    def visualize_eeg_analysis(self, attention_analysis: Dict[str, Any]):
        """可视化EEG分析结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # 1. 注意力水平时间序列
            if self.attention_history:
                timestamps = [record['timestamp'] for record in self.attention_history[-20:]]
                levels = [record['level'] for record in self.attention_history[-20:]]

                level_mapping = {'high': 3, 'medium': 2, 'low': 1, 'unknown': 0}
                level_values = [level_mapping.get(level, 0) for level in levels]

                axes[0, 0].plot(range(len(level_values)), level_values, 'b-o')
                axes[0, 0].set_title('Attention Level Over Time')
                axes[0, 0].set_yticks([1, 2, 3])
                axes[0, 0].set_yticklabels(['Low', 'Medium', 'High'])
                axes[0, 0].set_xlabel('Time Points')
                axes[0, 0].set_ylabel('Attention Level')

            # 2. 当前频段功率分布
            if attention_analysis['status'] == 'success':
                channel_powers = attention_analysis['channel_powers']
                bands = list(self.frequency_bands.keys())
                powers = [channel_powers[self.channel_names[0]].get(band, 0) for band in bands]

                axes[0, 1].bar(bands, powers, color='green', alpha=0.7)
                axes[0, 1].set_title('Frequency Band Powers')
                axes[0, 1].set_ylabel('Power')
                axes[0, 1].tick_params(axis='x', rotation=45)

            # 3. 注意力指标雷达图
            if attention_analysis['status'] == 'success':
                metrics = attention_analysis['attention_metrics']
                categories = ['Alpha/Beta Ratio', 'Theta/Beta Ratio', 'Beta Power', 'Alpha Power']
                values = [
                    metrics['alpha_beta_ratio'] * 10,  # 放大显示
                    metrics['theta_beta_ratio'] * 10,
                    metrics['beta_relative_power'] * 100,
                    metrics['alpha_relative_power'] * 100
                ]

                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                values += values[:1]  # 闭合雷达图
                angles += angles[:1]

                axes[1, 0].polar(angles, values)
                axes[1, 0].set_title('Attention Metrics Radar', pad=20)
                axes[1, 0].set_thetagrids(np.degrees(angles[:-1]), categories)

            # 4. 信号质量指示器
            confidence = attention_analysis.get('confidence', 0)
            axes[1, 1].bar(['Confidence'], [confidence], color='blue', alpha=0.7)
            axes[1, 1].set_title('Analysis Confidence')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_ylabel('Confidence Level')

            plt.tight_layout()
            plt.savefig('eeg_attention_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

            print("📊 EEG分析可视化图已保存为: eeg_attention_analysis.png")

        except Exception as e:
            print(f"⚠️ 可视化失败: {e}")


def demonstrate_eeg_attention_monitoring():
    """演示EEG注意力监测"""
    print("🧠 RQA2026脑机接口EEG注意力监测演示")
    print("=" * 60)

    # 创建EEG处理器
    processor = EEGSignalProcessor(sampling_rate=250, num_channels=4)

    # 生成示例EEG数据 (10秒)
    print("📡 生成示例EEG数据...")
    eeg_data = processor.create_sample_eeg_data(duration_seconds=10)

    # 分批添加数据并分析
    batch_size = 250  # 1秒数据
    analysis_results = []

    print("🔍 实时EEG分析...")
    for i in range(0, eeg_data.shape[1], batch_size):
        batch_data = eeg_data[:, i:i+batch_size]
        if batch_data.shape[1] > 0:
            processor.add_eeg_data(batch_data)

            # 每2秒进行一次注意力分析
            if (i // batch_size) % 2 == 0 and i > 0:
                attention_result = processor.analyze_attention_state()
                if attention_result['status'] == 'success':
                    analysis_results.append(attention_result)
                    print(f"🧠 时间点 {i//250}s: 注意力水平 = {attention_result['attention_level']} "
                          f"(置信度: {attention_result['confidence']:.2f})")

    # 综合分析结果
    if analysis_results:
        print("\n🎯 综合分析结果:")
        latest_result = analysis_results[-1]
        attention_level = latest_result['attention_level']
        metrics = latest_result['attention_metrics']

        print(f"📊 最终注意力水平: {attention_level}")
        print(f"  Alpha/Beta比率: {metrics['alpha_beta_ratio']:.3f}")
        print(f"  Theta/Beta比率: {metrics['theta_beta_ratio']:.3f}")
        print(f"  Beta相对功率: {metrics['beta_relative_power']:.3f}")
        print(f"  Alpha相对功率: {metrics['alpha_relative_power']:.3f}")
        # 生成交易信号
        trading_signals = processor.generate_trading_signals(latest_result)

        print("\n📈 交易信号:")
        if trading_signals['signals']:
            for signal in trading_signals['signals']:
                print(f"  • {signal['type']}: {signal['description']}")
        else:
            print("  📊 无明确交易信号")

        print("\n💡 建议:")
        for rec in trading_signals['recommendations']:
            print(f"  • {rec}")

        # 可视化 (可选)
        try:
            processor.visualize_eeg_analysis(latest_result)
        except Exception as e:
            print(f"⚠️ 可视化出错: {e}")

    print("
🧠 脑机接口优势:"    print("  • 实时注意力监测")
    print("  • 交易决策辅助")
    print("  • 风险控制预警")
    print("  • 个性化交易策略")
    print("  • 情绪状态量化")

    print("\n🎊 RQA2026脑机接口EEG注意力监测演示完成!")
    print("为量化交易带来了革命性的生物反馈技术！")

    return {
        'processor': processor,
        'analysis_results': analysis_results,
        'final_result': latest_result if analysis_results else None
    }


if __name__ == "__main__":
    results = demonstrate_eeg_attention_monitoring()
