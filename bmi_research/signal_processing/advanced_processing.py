#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2026 高级神经信号处理
脑机接口信号处理算法

包含:
- 自适应滤波
- 独立成分分析 (ICA)
- 共空间模式 (CSP)
- 相干性分析
- 脑网络分析
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from scipy import signal, stats
from sklearn.decomposition import FastICA
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


@dataclass
class ProcessingResult:
    """处理结果"""
    processed_signal: np.ndarray
    features: np.ndarray
    artifacts_removed: List[str]
    quality_score: float
    processing_time: float


class AdaptiveFilter:
    """自适应滤波器"""

    def __init__(self, filter_type: str = "LMS", num_taps: int = 32,
                 learning_rate: float = 0.01):
        self.filter_type = filter_type
        self.num_taps = num_taps
        self.learning_rate = learning_rate

        # 滤波器系数
        self.weights = np.zeros(num_taps)
        self.buffer = np.zeros(num_taps)

    def adapt(self, input_signal: np.ndarray,
              reference_signal: np.ndarray) -> np.ndarray:
        """自适应滤波"""
        if self.filter_type == "LMS":
            return self._lms_adapt(input_signal, reference_signal)
        elif self.filter_type == "RLS":
            return self._rls_adapt(input_signal, reference_signal)
        else:
            raise ValueError(f"不支持的滤波器类型: {self.filter_type}")

    def _lms_adapt(self, input_signal: np.ndarray,
                  reference_signal: np.ndarray) -> np.ndarray:
        """LMS自适应滤波"""
        output = np.zeros_like(input_signal)
        error_history = []

        for i in range(len(input_signal)):
            # 更新输入缓冲区
            self.buffer[1:] = self.buffer[:-1]
            self.buffer[0] = input_signal[i]

            # 计算输出
            y = np.dot(self.weights, self.buffer)

            # 计算误差 (假设参考信号是期望的噪声)
            error = reference_signal[i] - y if i < len(reference_signal) else 0

            # 更新权重
            self.weights += self.learning_rate * error * self.buffer

            output[i] = y
            error_history.append(error)

        return output

    def _rls_adapt(self, input_signal: np.ndarray,
                  reference_signal: np.ndarray) -> np.ndarray:
        """RLS自适应滤波"""
        # RLS算法实现 (简化的版本)
        n = len(input_signal)
        output = np.zeros(n)

        # 初始化
        P = np.eye(self.num_taps) / 0.1  # 逆相关矩阵
        lambda_factor = 0.99  # 遗忘因子

        for i in range(n):
            # 更新输入向量
            x = np.roll(self.buffer, 1)
            x[0] = input_signal[i]

            # 计算增益向量
            k = P @ x / (lambda_factor + x.T @ P @ x)

            # 计算输出
            y = self.weights.T @ x

            # 计算误差
            error = reference_signal[i] - y if i < len(reference_signal) else 0

            # 更新权重
            self.weights += k * error

            # 更新逆相关矩阵
            P = (P - np.outer(k, x.T @ P)) / lambda_factor

            output[i] = y
            self.buffer = x

        return output


class ArtifactRemover:
    """伪迹去除器"""

    def __init__(self):
        self.artifact_detectors = {
            'eye_blink': self._detect_eye_blinks,
            'muscle': self._detect_muscle_artifacts,
            'line_noise': self._detect_line_noise,
            'movement': self._detect_movement_artifacts
        }

    def remove_artifacts(self, signal: np.ndarray,
                        sampling_rate: float) -> Tuple[np.ndarray, List[str]]:
        """去除伪迹"""
        cleaned_signal = signal.copy()
        removed_artifacts = []

        # 检测和去除各种伪迹
        for artifact_type, detector in self.artifact_detectors.items():
            if detector(cleaned_signal, sampling_rate):
                cleaned_signal = self._remove_artifact(cleaned_signal, artifact_type)
                removed_artifacts.append(artifact_type)

        return cleaned_signal, removed_artifacts

    def _detect_eye_blinks(self, signal: np.ndarray, sampling_rate: float) -> bool:
        """检测眼跳伪迹"""
        # 基于前端通道的尖峰检测
        frontal_channels = signal[:4] if signal.shape[0] >= 4 else signal

        # 计算瞬时能量
        energy = np.sum(frontal_channels ** 2, axis=0)

        # 检测尖峰
        threshold = np.mean(energy) + 3 * np.std(energy)
        peaks = energy > threshold

        return np.sum(peaks) > len(energy) * 0.01  # 1%的样本点

    def _detect_muscle_artifacts(self, signal: np.ndarray, sampling_rate: float) -> bool:
        """检测肌肉伪迹"""
        # 基于高频功率检测
        from scipy.signal import welch

        for ch in range(signal.shape[0]):
            freqs, psd = welch(signal[ch], fs=sampling_rate, nperseg=256)

            # 高频段功率 (30-100 Hz)
            high_freq_power = np.sum(psd[(freqs >= 30) & (freqs <= 100)])
            total_power = np.sum(psd[(freqs >= 1) & (freqs <= 40)])

            if total_power > 0 and high_freq_power / total_power > 0.3:
                return True

        return False

    def _detect_line_noise(self, signal: np.ndarray, sampling_rate: float) -> bool:
        """检测工频干扰"""
        from scipy.signal import welch

        # 检测50Hz或60Hz峰值
        line_freqs = [50.0, 60.0]

        for ch in range(signal.shape[0]):
            freqs, psd = welch(signal[ch], fs=sampling_rate, nperseg=1024)

            for line_freq in line_freqs:
                # 查找最接近工频的频率
                freq_idx = np.argmin(np.abs(freqs - line_freq))

                # 检查是否为显著峰值
                if freq_idx > 0 and freq_idx < len(psd) - 1:
                    peak_ratio = psd[freq_idx] / np.mean(psd)
                    if peak_ratio > 5.0:  # 5倍于平均功率
                        return True

        return False

    def _detect_movement_artifacts(self, signal: np.ndarray, sampling_rate: float) -> bool:
        """检测运动伪迹"""
        # 基于信号变化率的检测
        for ch in range(signal.shape[0]):
            # 计算一阶导数
            derivative = np.diff(signal[ch])

            # 计算瞬时变化率
            change_rate = np.abs(derivative) / (np.max(np.abs(signal[ch])) + 1e-6)

            # 检测剧烈变化
            threshold = np.mean(change_rate) + 5 * np.std(change_rate)
            if np.sum(change_rate > threshold) > len(change_rate) * 0.05:  # 5%的样本
                return True

        return False

    def _remove_artifact(self, signal: np.ndarray, artifact_type: str) -> np.ndarray:
        """去除特定类型的伪迹"""
        if artifact_type == 'eye_blink':
            return self._remove_eye_blinks(signal)
        elif artifact_type == 'muscle':
            return self._remove_muscle_artifacts(signal)
        elif artifact_type == 'line_noise':
            return self._remove_line_noise(signal)
        elif artifact_type == 'movement':
            return self._remove_movement_artifacts(signal)
        else:
            return signal

    def _remove_eye_blinks(self, signal: np.ndarray) -> np.ndarray:
        """去除眼跳伪迹"""
        # 使用ICA分离眼跳成分
        ica = FastICA(n_components=min(signal.shape[0], 8), random_state=42)
        ica_components = ica.fit_transform(signal.T).T

        # 识别眼跳成分 (通常在前端通道最强)
        blink_component = np.argmax(np.abs(ica_components[:4]).max(axis=0))

        # 重构信号，去除眼跳成分
        ica_components[blink_component] = 0
        cleaned_signal = ica.inverse_transform(ica_components.T).T

        return cleaned_signal

    def _remove_muscle_artifacts(self, signal: np.ndarray) -> np.ndarray:
        """去除肌肉伪迹"""
        # 高通滤波去除低频成分，保留肌肉噪声
        from scipy.signal import butter, filtfilt

        b, a = butter(4, 20.0 / (250.0 / 2), btype='high')  # 20Hz高通
        filtered_signal = np.zeros_like(signal)

        for ch in range(signal.shape[0]):
            filtered_signal[ch] = filtfilt(b, a, signal[ch])

        # 从原始信号中减去高频成分
        return signal - filtered_signal * 0.5

    def _remove_line_noise(self, signal: np.ndarray) -> np.ndarray:
        """去除工频干扰"""
        # 使用自适应滤波器
        filter_instance = AdaptiveFilter("LMS", num_taps=64, learning_rate=0.01)

        cleaned_signal = np.zeros_like(signal)
        for ch in range(signal.shape[0]):
            # 使用相邻通道作为参考
            reference_ch = (ch + 1) % signal.shape[0]
            cleaned_signal[ch] = signal[ch] - filter_instance.adapt(
                signal[ch], signal[reference_ch]
            )

        return cleaned_signal

    def _remove_movement_artifacts(self, signal: np.ndarray) -> np.ndarray:
        """去除运动伪迹"""
        # 使用中值滤波去除尖峰
        from scipy.signal import medfilt

        cleaned_signal = np.zeros_like(signal)
        for ch in range(signal.shape[0]):
            cleaned_signal[ch] = medfilt(signal[ch], kernel_size=5)

        return cleaned_signal


class SpatialFilter:
    """空间滤波器"""

    def __init__(self, filter_type: str = "CAR"):
        self.filter_type = filter_type

    def apply_filter(self, signal: np.ndarray) -> np.ndarray:
        """应用空间滤波"""
        if self.filter_type == "CAR":
            return self._common_average_reference(signal)
        elif self.filter_type == "LAP":
            return self._laplacian_filter(signal)
        elif self.filter_type == "CSP":
            return self._common_spatial_patterns(signal)
        else:
            raise ValueError(f"不支持的空间滤波器: {self.filter_type}")

    def _common_average_reference(self, signal: np.ndarray) -> np.ndarray:
        """共平均参考 (CAR)"""
        # 计算所有通道的平均值
        average_reference = np.mean(signal, axis=0)

        # 从每个通道减去平均值
        car_signal = signal - average_reference

        return car_signal

    def _laplacian_filter(self, signal: np.ndarray) -> np.ndarray:
        """拉普拉斯滤波器"""
        laplacian_signal = np.zeros_like(signal)

        # 简化的2D拉普拉斯滤波 (假设标准10-20系统布局)
        # 这里使用简化的邻域平均
        for ch in range(signal.shape[0]):
            if ch == 0:  # Fz
                neighbors = [1, 2, 3] if signal.shape[0] > 3 else [1]
            elif ch == signal.shape[0] - 1:  # 最后通道
                neighbors = [ch-1, ch-2] if ch > 1 else [ch-1]
            else:
                neighbors = [ch-1, ch+1]

            # 计算拉普拉斯值
            neighbor_avg = np.mean(signal[neighbors], axis=0)
            laplacian_signal[ch] = signal[ch] - neighbor_avg

        return laplacian_signal

    def _common_spatial_patterns(self, signal: np.ndarray) -> np.ndarray:
        """共空间模式 (CSP) - 简化的二分类版本"""
        # CSP需要标签数据，这里提供简化的实现
        n_channels = signal.shape[0]

        # 生成随机的空间滤波器 (实际应基于训练数据)
        np.random.seed(42)
        csp_filters = np.random.randn(n_channels, n_channels)

        # 应用滤波器
        csp_signal = csp_filters @ signal

        return csp_signal


class ConnectivityAnalyzer:
    """连通性分析器"""

    def __init__(self):
        self.connectivity_measures = {
            'coherence': self._coherence,
            'phase_locking_value': self._phase_locking_value,
            'correlation': self._correlation,
            'granger_causality': self._granger_causality
        }

    def analyze_connectivity(self, signal: np.ndarray, method: str = 'coherence',
                           freq_band: Tuple[float, float] = (8, 12)) -> np.ndarray:
        """分析神经连通性"""
        if method not in self.connectivity_measures:
            raise ValueError(f"不支持的连通性方法: {method}")

        return self.connectivity_measures[method](signal, freq_band)

    def _coherence(self, signal: np.ndarray, freq_band: Tuple[float, float]) -> np.ndarray:
        """相干性分析"""
        from scipy.signal import coherence

        n_channels = signal.shape[0]
        coherence_matrix = np.zeros((n_channels, n_channels))

        # 计算每对通道的相干性
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                freqs, coh = coherence(signal[i], signal[j], fs=250.0, nperseg=256)

                # 在指定频带内平均
                band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
                if np.sum(band_mask) > 0:
                    avg_coherence = np.mean(coh[band_mask])
                else:
                    avg_coherence = 0

                coherence_matrix[i, j] = coherence_matrix[j, i] = avg_coherence

        return coherence_matrix

    def _phase_locking_value(self, signal: np.ndarray, freq_band: Tuple[float, float]) -> np.ndarray:
        """相位锁定值 (PLV)"""
        from scipy.signal import hilbert

        n_channels = signal.shape[0]
        plv_matrix = np.zeros((n_channels, n_channels))

        # 带通滤波到指定频带
        from scipy.signal import butter, filtfilt
        b, a = butter(4, [freq_band[0]/(250.0/2), freq_band[1]/(250.0/2)], btype='band')

        filtered_signals = np.zeros_like(signal)
        for ch in range(n_channels):
            filtered_signals[ch] = filtfilt(b, a, signal[ch])

        # 计算解析信号
        analytic_signals = hilbert(filtered_signals)

        # 计算每对通道的PLV
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                phase_i = np.angle(analytic_signals[i])
                phase_j = np.angle(analytic_signals[j])

                # 计算相位差
                phase_diff = phase_i - phase_j

                # 计算PLV
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_matrix[i, j] = plv_matrix[j, i] = plv

        return plv_matrix

    def _correlation(self, signal: np.ndarray, freq_band: Tuple[float, float] = None) -> np.ndarray:
        """相关性分析"""
        # 计算Pearson相关系数
        correlation_matrix = np.corrcoef(signal)
        return correlation_matrix

    def _granger_causality(self, signal: np.ndarray, freq_band: Tuple[float, float] = None) -> np.ndarray:
        """格兰杰因果性 (简化的时域版本)"""
        from statsmodels.tsa.stattools import grangercausalitytests

        n_channels = signal.shape[0]
        gc_matrix = np.zeros((n_channels, n_channels))

        max_lag = 10  # 最大滞后

        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    try:
                        # 使用格兰杰因果性检验
                        data = np.column_stack([signal[j], signal[i]])  # X causes Y
                        result = grangercausalitytests(data, max_lag, verbose=False)

                        # 使用F检验的p值
                        f_test_p = result[max_lag][0]['ssr_ftest'][1]
                        gc_matrix[j, i] = 1 - f_test_p  # 转换为因果强度

                    except:
                        gc_matrix[j, i] = 0

        return gc_matrix


class AdvancedSignalProcessor:
    """高级信号处理器"""

    def __init__(self):
        self.adaptive_filter = AdaptiveFilter()
        self.artifact_remover = ArtifactRemover()
        self.spatial_filter = SpatialFilter()
        self.connectivity_analyzer = ConnectivityAnalyzer()

    def process_signal(self, signal: np.ndarray, sampling_rate: float = 250.0,
                      processing_config: Dict[str, Any] = None) -> ProcessingResult:
        """高级信号处理流程"""
        import time
        start_time = time.time()

        if processing_config is None:
            processing_config = {
                'artifact_removal': True,
                'spatial_filter': 'CAR',
                'connectivity_analysis': 'coherence',
                'freq_band': (8, 12)
            }

        processed_signal = signal.copy()
        artifacts_removed = []
        features = None

        # 1. 伪迹去除
        if processing_config.get('artifact_removal', True):
            processed_signal, artifacts_removed = self.artifact_remover.remove_artifacts(
                processed_signal, sampling_rate
            )

        # 2. 空间滤波
        spatial_filter_type = processing_config.get('spatial_filter', 'CAR')
        if spatial_filter_type:
            self.spatial_filter = SpatialFilter(spatial_filter_type)
            processed_signal = self.spatial_filter.apply_filter(processed_signal)

        # 3. 连通性分析
        connectivity_method = processing_config.get('connectivity_analysis')
        if connectivity_method:
            freq_band = processing_config.get('freq_band', (8, 12))
            connectivity_matrix = self.connectivity_analyzer.analyze_connectivity(
                processed_signal, connectivity_method, freq_band
            )
            features = connectivity_matrix.flatten()

        # 4. 质量评估
        quality_score = self._assess_signal_quality(processed_signal)

        processing_time = time.time() - start_time

        return ProcessingResult(
            processed_signal=processed_signal,
            features=features if features is not None else processed_signal.flatten(),
            artifacts_removed=artifacts_removed,
            quality_score=quality_score,
            processing_time=processing_time
        )

    def _assess_signal_quality(self, signal: np.ndarray) -> float:
        """评估信号质量"""
        quality_scores = []

        # 1. 信噪比
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean((signal - np.mean(signal, axis=1, keepdims=True)) ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        quality_scores.append(np.clip(snr / 20, 0, 1))  # 归一化到[0,1]

        # 2. 通道一致性 (相关性)
        correlations = []
        for i in range(signal.shape[0]):
            for j in range(i+1, signal.shape[0]):
                corr = np.corrcoef(signal[i], signal[j])[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0)

        avg_correlation = np.mean(correlations) if correlations else 0
        quality_scores.append(avg_correlation)

        # 3. 动态范围
        dynamic_range = np.ptp(signal, axis=1).mean() / (np.std(signal, axis=1).mean() + 1e-6)
        quality_scores.append(np.clip(dynamic_range / 5, 0, 1))

        return np.mean(quality_scores)


# 工具函数
def generate_synthetic_eeg(duration: float = 1.0, sampling_rate: float = 250.0,
                          num_channels: int = 8, add_artifacts: bool = True) -> np.ndarray:
    """生成合成EEG信号用于测试"""
    n_samples = int(duration * sampling_rate)

    # 生成基础脑电信号
    signal = np.zeros((num_channels, n_samples))

    # 添加不同频带的振荡
    freqs = [10, 20, 40]  # α, β, γ波
    for ch in range(num_channels):
        for freq in freqs:
            t = np.linspace(0, duration, n_samples)
            amplitude = np.random.uniform(1, 5)
            phase = np.random.uniform(0, 2*np.pi)
            signal[ch] += amplitude * np.sin(2*np.pi*freq*t + phase)

    # 添加噪声
    signal += np.random.normal(0, 2, (num_channels, n_samples))

    # 添加伪迹 (可选)
    if add_artifacts:
        # 眼跳伪迹
        blink_times = np.random.choice(n_samples, size=3, replace=False)
        for blink_time in blink_times:
            start = max(0, blink_time - 10)
            end = min(n_samples, blink_time + 10)
            signal[0, start:end] += np.random.normal(0, 10, end - start)

        # 肌肉伪迹
        muscle_times = np.random.choice(n_samples, size=5, replace=False)
        for muscle_time in muscle_times:
            start = max(0, muscle_time - 25)
            end = min(n_samples, muscle_time + 25)
            for ch in range(num_channels):
                signal[ch, start:end] += np.random.normal(0, 3, end - start)

    return signal


def benchmark_signal_processing():
    """信号处理基准测试"""
    print("🔬 高级信号处理基准测试")
    print("=" * 50)

    processor = AdvancedSignalProcessor()

    # 测试不同规模的信号
    test_configs = [
        {'duration': 1.0, 'channels': 8},
        {'duration': 5.0, 'channels': 16},
        {'duration': 10.0, 'channels': 32}
    ]

    results = {}

    for config in test_configs:
        print(f"测试配置: {config['duration']}秒, {config['channels']}通道")

        # 生成测试信号
        signal = generate_synthetic_eeg(**config)

        # 处理信号
        import time
        start_time = time.time()
        result = processor.process_signal(signal, sampling_rate=250.0)
        processing_time = time.time() - start_time

        results[f"{config['channels']}ch_{config['duration']}s"] = {
            'processing_time': processing_time,
            'quality_score': result.quality_score,
            'artifacts_removed': len(result.artifacts_removed),
            'feature_dim': len(result.features)
        }

        print(f"  去除伪迹: {len(result.artifacts_removed)}")
        print(f"  特征维度: {len(result.features)}")

    # 保存结果
    import json
    with open('bmi_research/tests/signal_processing_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\\n✅ 信号处理基准测试完成!")
    return results


if __name__ == "__main__":
    benchmark_signal_processing()
