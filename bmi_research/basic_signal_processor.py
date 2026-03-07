# -*- coding: utf-8 -*-
"""
Basic Neural Signal Processor for BMI Research
"""

import numpy as np
from typing import Dict, Any

class BasicSignalProcessor:
    """Basic neural signal processor for demonstration"""

    def __init__(self):
        self.sampling_rate = 500  # Hz

    def simulate_eeg(self, duration=10, n_channels=4):
        """Simulate EEG data for demonstration"""
        n_samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, n_samples)

        data = []
        for ch in range(n_channels):
            # Basic alpha rhythm (8-12 Hz)
            alpha = 2 * np.sin(2 * np.pi * 10 * t)
            # Add noise
            noise = 0.5 * np.random.randn(n_samples)
            signal = alpha + noise
            data.append(signal)

        return np.array(data)

    def basic_analysis(self, data):
        """Basic signal analysis"""
        results = {}
        for ch in range(data.shape[0]):
            ch_data = data[ch]
            results[f"ch_{ch}"] = {
                "mean": float(np.mean(ch_data)),
                "std": float(np.std(ch_data)),
                "max": float(np.max(ch_data)),
                "min": float(np.min(ch_data))
            }
        return results

if __name__ == "__main__":
    processor = BasicSignalProcessor()
    data = processor.simulate_eeg(duration=5, n_channels=2)
    results = processor.basic_analysis(data)
    print("Basic EEG Analysis Results:")
    print(results)
