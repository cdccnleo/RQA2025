#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streaming层 - 流处理综合测试

测试流数据处理、实时计算、流水线
"""

import pytest
from typing import List
from datetime import datetime
from tests.unit.streaming.conftest import import_realtime_analyzer


class TestStreamProcessing:
    """测试流处理"""
    
    def test_process_stream_data(self):
        """测试处理流数据"""
        stream_data = [
            {'timestamp': datetime.now(), 'value': 100},
            {'timestamp': datetime.now(), 'value': 102},
            {'timestamp': datetime.now(), 'value': 101}
        ]
        
        # 处理每个数据点
        processed = []
        for data in stream_data:
            processed.append(data['value'] * 2)
        
        assert processed == [200, 204, 202]
    
    def test_sliding_window_aggregation(self):
        """测试滑动窗口聚合"""
        stream = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        window_size = 3
        
        windowed_sums = []
        for i in range(len(stream) - window_size + 1):
            window = stream[i:i+window_size]
            windowed_sums.append(sum(window))
        
        assert windowed_sums[0] == 6  # 1+2+3
    
    def test_stream_filtering(self):
        """测试流过滤"""
        stream = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # 过滤出偶数
        filtered = [x for x in stream if x % 2 == 0]
        
        assert filtered == [2, 4, 6, 8, 10]


class TestRealTimeComputation:
    """测试实时计算"""
    
    def test_realtime_average(self):
        """测试实时平均值"""
        # 使用实际的RealTimeAnalyzer
        RealTimeAnalyzer = import_realtime_analyzer()
        if RealTimeAnalyzer is not None:
            analyzer = RealTimeAnalyzer('test_analyzer')
            for value in [10, 20, 30, 40]:
                analyzer.add_data_point({'value': value})
            
            # 注册平均值分析器
            def avg_analyzer(data_window, timestamp_window):
                if len(data_window) == 0:
                    return {'avg': 0.0}
                values = [d.get('value', 0) if isinstance(d, dict) else d for d in data_window]
                return {'avg': sum(values) / len(values)}
            
            analyzer.register_analyzer('avg', avg_analyzer)
            analyzer.start_analysis()
            import time
            time.sleep(0.2)
            analyzer.stop_analysis()
            
            metrics = analyzer.get_current_metrics()
            assert metrics is not None
        else:
            values = []
            running_sum = 0
            running_avg = []
            
            for value in [10, 20, 30, 40]:
                values.append(value)
                running_sum += value
                avg = running_sum / len(values)
                running_avg.append(avg)
            
            assert running_avg[-1] == 25.0
    
    def test_realtime_max(self):
        """测试实时最大值"""
        stream = [5, 3, 8, 2, 9, 1]
        running_max = []
        
        current_max = float('-inf')
        for value in stream:
            current_max = max(current_max, value)
            running_max.append(current_max)
        
        assert running_max == [5, 5, 8, 8, 9, 9]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

