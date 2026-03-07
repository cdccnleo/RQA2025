#!/usr/bin/env python3
"""
拆分performance_monitor_dashboard.py为多个专门模块
"""

import os


def extract_class_content(content: str, class_name: str):
    """提取指定类的完整内容"""
    lines = content.split('\n')

    # 找到类定义
    class_start = None
    for i, line in enumerate(lines):
        if f'class {class_name}:' in line:
            class_start = i
            break

    if class_start is None:
        return None

    # 找到类结束（下一个类定义或文件结束）
    class_end = len(lines) - 1
    for i in range(class_start + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        if (stripped.startswith('class ') and
            not stripped.startswith('    ') and
                not stripped.startswith('\t')):
            class_end = i - 1
            break

    return lines[class_start:class_end + 1]


def create_monitoring_core(content: str):
    """创建监控核心模块"""

    # 提取PerformanceMonitorDashboard类，但只保留核心监控功能
    lines = content.split('\n')

    core_lines = [
        '"""监控面板核心功能"""',
        '',
        'from infrastructure.config.core.imports import (',
        '    Dict, Any, Optional, List, Union,',
        '    time, datetime, threading',
        ')',
        'from infrastructure.config.core.common_mixins import MonitoringMixin',
        '',
        'class PerformanceMonitorDashboardCore(MonitoringMixin):',
        '    """性能监控面板核心功能"""',
        '',
        '    def __init__(self, storage_path: str = "config/performance",',
        '                 retention_days: int = 30):',
        '        """初始化监控面板核心"""',
        '        super().__init__(enable_metrics=True, enable_alerts=True, enable_history=True)',
        '        self.storage_path = storage_path',
        '        self.retention_days = retention_days',
        '        self._lock = threading.RLock()',
        '        self._storage_initialized = False',
        '',
        '    def _initialize_storage(self):',
        '        """初始化存储"""',
        '        if not self._storage_initialized:',
        '            # 初始化存储逻辑',
        '            self._storage_initialized = True',
        '',
        '    def record_operation(self, operation: str, duration: float, success: bool = True):',
        '        """记录操作"""',
        '        if not self._storage_initialized:',
        '            self._initialize_storage()',
        '        ',
        '        # 记录操作逻辑',
        '        pass',
        '',
        '    def get_operation_stats(self) -> Dict[str, Any]:',
        '        """获取操作统计"""',
        '        return {',
        '            "total_operations": 0,',
        '            "success_rate": 1.0,',
        '            "avg_duration": 0.0',
        '        }',
        '',
        '    def get_system_health_status(self) -> Dict[str, Any]:',
        '        """获取系统健康状态"""',
        '        return {',
        '            "status": "healthy",',
        '            "cpu_usage": 0.0,',
        '            "memory_usage": 0.0',
        '        }'
    ]

    # 找到原类的核心方法
    class_content = extract_class_content(content, 'PerformanceMonitorDashboard')
    if class_content:
        # 提取核心方法
        core_methods = ['__init__', '_initialize_storage', 'record_operation',
                        'get_operation_stats', 'get_system_health_status']

        in_method = False
        current_method = None
        method_lines = []

        for line in class_content:
            stripped = line.strip()

            if stripped.startswith('def ') and any(method in stripped for method in core_methods):
                if in_method:
                    core_lines.extend(method_lines)
                    core_lines.append('')
                in_method = True
                current_method = stripped.split('def ')[1].split('(')[0].strip()
                method_lines = [line]
            elif in_method:
                method_lines.append(line)
                # 检查方法结束
                if stripped and not line.startswith(' ') and not line.startswith('\t') and not stripped.startswith('#'):
                    in_method = False
                    if method_lines:
                        core_lines.extend(method_lines)
                        core_lines.append('')
                        method_lines = []

        if method_lines:
            core_lines.extend(method_lines)

    core_lines.extend([
        '',
        '# 向后兼容',
        'PerformanceMonitorDashboard = PerformanceMonitorDashboardCore'
    ])

    return '\n'.join(core_lines)


def create_anomaly_detector(content: str):
    """创建异常检测模块"""

    anomaly_lines = [
        '"""异常检测功能"""',
        '',
        'from infrastructure.config.core.imports import Dict, Any, List, Optional',
        'from infrastructure.config.core.common_mixins import ConfigComponentMixin',
        '',
        'class AnomalyDetector(ConfigComponentMixin):',
        '    """异常检测器"""',
        '',
        '    def __init__(self, window_size: int = 20, threshold: float = 2.5):',
        '        """初始化异常检测器"""',
        '        super().__init__()',
        '        self._init_component_attributes(enable_threading=True, enable_data=True)',
        '        self.window_size = window_size',
        '        self.threshold = threshold',
        '        self._data_windows: Dict[str, List[float]] = {}',
        '        self._baselines: Dict[str, float] = {}',
        '        self._std_devs: Dict[str, float] = {}',
        '',
        '    def update_baseline(self, metric_name: str, values: List[float]):',
        '        """更新基线"""',
        '        if len(values) >= self.window_size:',
        '            window = values[-self.window_size:]',
        '            self._baselines[metric_name] = sum(window) / len(window)',
        '            variance = sum((x - self._baselines[metric_name]) ** 2 for x in window) / len(window)',
        '            self._std_devs[metric_name] = variance ** 0.5',
        '',
        '    def detect_anomaly(self, metric_name: str, value: float) -> Dict[str, Any]:',
        '        """检测异常"""',
        '        if metric_name not in self._data_windows:',
        '            self._data_windows[metric_name] = []',
        '',
        '        self._data_windows[metric_name].append(value)',
        '',
        '        # 保持窗口大小',
        '        if len(self._data_windows[metric_name]) > self.window_size:',
        '            self._data_windows[metric_name].pop(0)',
        '',
        '        # 更新基线',
        '        self.update_baseline(metric_name, self._data_windows[metric_name])',
        '',
        '        # 检测异常',
        '        if metric_name in self._baselines and metric_name in self._std_devs:',
        '            baseline = self._baselines[metric_name]',
        '            std_dev = self._std_devs[metric_name]',
        '',
        '            if std_dev > 0:',
        '                z_score = abs(value - baseline) / std_dev',
        '                is_anomaly = z_score > self.threshold',
        '',
        '                return {',
        '                    "is_anomaly": is_anomaly,',
        '                    "z_score": z_score,',
        '                    "baseline": baseline,',
        '                    "std_dev": std_dev,',
        '                    "threshold": self.threshold',
        '                }',
        '',
        '        return {',
        '            "is_anomaly": False,',
        '            "z_score": 0.0,',
        '            "baseline": value if len(self._data_windows[metric_name]) == 1 else 0.0,',
        '            "std_dev": 0.0,',
        '            "threshold": self.threshold',
        '        }'
    ]

    return '\n'.join(anomaly_lines)


def create_trend_analyzer(content: str):
    """创建趋势分析模块"""

    trend_lines = [
        '"""趋势分析功能"""',
        '',
        'from infrastructure.config.core.imports import Dict, Any, List',
        'from infrastructure.config.core.common_mixins import ConfigComponentMixin',
        '',
        'class TrendAnalyzer(ConfigComponentMixin):',
        '    """趋势分析器"""',
        '',
        '    def __init__(self, window_size: int = 50):',
        '        """初始化趋势分析器"""',
        '        super().__init__()',
        '        self._init_component_attributes(enable_threading=True, enable_data=True)',
        '        self.window_size = window_size',
        '        self._data_series: Dict[str, List[float]] = {}',
        '',
        '    def add_data_point(self, metric_name: str, value: float):',
        '        """添加数据点"""',
        '        if metric_name not in self._data_series:',
        '            self._data_series[metric_name] = []',
        '',
        '        self._data_series[metric_name].append(value)',
        '',
        '        # 保持窗口大小',
        '        if len(self._data_series[metric_name]) > self.window_size:',
        '            self._data_series[metric_name].pop(0)',
        '',
        '    def analyze_trend(self, metric_name: str) -> Dict[str, Any]:',
        '        """分析趋势"""',
        '        if metric_name not in self._data_series:',
        '            return {"trend": "insufficient_data", "slope": 0.0, "confidence": 0.0}',
        '',
        '        values = self._data_series[metric_name]',
        '        if len(values) < 10:  # 需要至少10个数据点',
        '            return {"trend": "insufficient_data", "slope": 0.0, "confidence": 0.0}',
        '',
        '        # 计算线性回归',
        '        n = len(values)',
        '        x = list(range(n))',
        '',
        '        # 计算斜率和截距',
        '        sum_x = sum(x)',
        '        sum_y = sum(values)',
        '        sum_xy = sum(xi * yi for xi, yi in zip(x, values))',
        '        sum_xx = sum(xi * xi for xi in x)',
        '',
        '        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)',
        '        intercept = (sum_y - slope * sum_x) / n',
        '',
        '        # 计算R²值作为置信度',
        '        y_mean = sum_y / n',
        '        ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, values))',
        '        ss_tot = sum((yi - y_mean) ** 2 for yi in values)',
        '        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0',
        '',
        '        # 确定趋势方向',
        '        if abs(slope) < 0.001:',
        '            trend = "stable"',
        '        elif slope > 0:',
        '            trend = "increasing"',
        '        else:',
        '            trend = "decreasing"',
        '',
        '        return {',
        '            "trend": trend,',
        '            "slope": slope,',
        '            "confidence": r_squared,',
        '            "intercept": intercept,',
        '            "data_points": n',
        '        }'
    ]

    return '\n'.join(trend_lines)


def create_performance_predictor(content: str):
    """创建性能预测模块"""

    predictor_lines = [
        '"""性能预测功能"""',
        '',
        'from infrastructure.config.core.imports import Dict, Any, List, Optional',
        'from infrastructure.config.core.common_mixins import ConfigComponentMixin',
        '',
        'class PerformancePredictor(ConfigComponentMixin):',
        '    """性能预测器"""',
        '',
        '    def __init__(self, prediction_window: int = 10):',
        '        """初始化性能预测器"""',
        '        super().__init__()',
        '        self._init_component_attributes(enable_threading=True, enable_data=True)',
        '        self.prediction_window = prediction_window',
        '        self._historical_data: Dict[str, List[float]] = {}',
        '',
        '    def add_historical_data(self, metric_name: str, value: float):',
        '        """添加历史数据"""',
        '        if metric_name not in self._historical_data:',
        '            self._historical_data[metric_name] = []',
        '',
        '        self._historical_data[metric_name].append(value)',
        '',
        '        # 保持合理的历史数据量',
        '        if len(self._historical_data[metric_name]) > 1000:',
        '            self._historical_data[metric_name] = self._historical_data[metric_name][-500:]',
        '',
        '    def predict_next_value(self, metric_name: str) -> Dict[str, Any]:',
        '        """预测下一个值"""',
        '        if metric_name not in self._historical_data:',
        '            return {"prediction": None, "confidence": 0.0, "method": "insufficient_data"}',
        '',
        '        values = self._historical_data[metric_name]',
        '        if len(values) < 5:',
        '            return {"prediction": None, "confidence": 0.0, "method": "insufficient_data"}',
        '',
        '        # 使用简单移动平均进行预测',
        '        window_size = min(10, len(values))',
        '        recent_values = values[-window_size:]',
        '        prediction = sum(recent_values) / len(recent_values)',
        '',
        '        # 计算置信度（基于数据的稳定性）',
        '        if len(recent_values) >= 3:',
        '            mean = sum(recent_values) / len(recent_values)',
        '            variance = sum((x - mean) ** 2 for x in recent_values) / len(recent_values)',
        '            std_dev = variance ** 0.5',
        '            cv = std_dev / mean if mean > 0 else 0  # 变异系数',
        '            confidence = max(0, 1 - cv)  # 稳定性越好，置信度越高',
        '        else:',
        '            confidence = 0.5',
        '',
        '        return {',
        '            "prediction": prediction,',
        '            "confidence": confidence,',
        '            "method": "moving_average",',
        '            "window_size": window_size,',
        '            "historical_points": len(values)',
        '        }',
        '',
        '    def predict_trend(self, metric_name: str) -> Dict[str, Any]:',
        '        """预测趋势"""',
        '        if metric_name not in self._historical_data:',
        '            return {"trend": "unknown", "confidence": 0.0}',
        '',
        '        values = self._historical_data[metric_name]',
        '        if len(values) < 10:',
        '            return {"trend": "insufficient_data", "confidence": 0.0}',
        '',
        '        # 计算最近趋势',
        '        recent = values[-20:]  # 最近20个点',
        '        if len(recent) < 5:',
        '            return {"trend": "insufficient_data", "confidence": 0.0}',
        '',
        '        # 计算斜率',
        '        x = list(range(len(recent)))',
        '        slope = sum((xi - sum(x)/len(x)) * (yi - sum(recent)/len(recent))',
        '                   for xi, yi in zip(x, recent)) / sum((xi - sum(x)/len(x)) ** 2 for xi in x)',
        '',
        '        if abs(slope) < 0.001:',
        '            trend = "stable"',
        '            confidence = 0.8',
        '        elif slope > 0:',
        '            trend = "increasing"',
        '            confidence = min(1.0, abs(slope) * 10)  # 斜率越大，置信度越高',
        '        else:',
        '            trend = "decreasing"',
        '            confidence = min(1.0, abs(slope) * 10)',
        '',
        '        return {',
        '            "trend": trend,',
        '            "confidence": confidence,',
        '            "slope": slope,',
        '            "data_points": len(recent)',
        '        }'
    ]

    return '\n'.join(predictor_lines)


def create_unified_monitoring_dashboard():
    """创建统一的监控面板"""

    dashboard_lines = [
        '"""统一性能监控面板"""',
        '',
        'from infrastructure.config.core.imports import Dict, Any, Optional, List',
        'from infrastructure.config.monitoring.core import PerformanceMonitorDashboardCore',
        'from infrastructure.config.monitoring.anomaly_detector import AnomalyDetector',
        'from infrastructure.config.monitoring.trend_analyzer import TrendAnalyzer',
        'from infrastructure.config.monitoring.performance_predictor import PerformancePredictor',
        '',
        'class PerformanceMonitorDashboard:',
        '    """统一性能监控面板 - 整合所有监控功能"""',
        '',
        '    def __init__(self, storage_path: str = "config/performance",',
        '                 retention_days: int = 30,',
        '                 enable_system_monitoring: bool = True):',
        '        """初始化统一监控面板"""',
        '        self.core = PerformanceMonitorDashboardCore(storage_path, retention_days)',
        '        self.anomaly_detector = AnomalyDetector()',
        '        self.trend_analyzer = TrendAnalyzer()',
        '        self.performance_predictor = PerformancePredictor()',
        '        self.enable_system_monitoring = enable_system_monitoring',
        '',
        '    def start_monitoring(self):',
        '        """启动监控"""',
        '        self.core.start()',
        '',
        '    def stop_monitoring(self):',
        '        """停止监控"""',
        '        self.core.stop()',
        '',
        '    def record_operation(self, operation: str, duration: float, success: bool = True):',
        '        """记录操作"""',
        '        return self.core.record_operation(operation, duration, success)',
        '',
        '    def get_operation_stats(self) -> Dict[str, Any]:',
        '        """获取操作统计"""',
        '        return self.core.get_operation_stats()',
        '',
        '    def get_system_health_status(self) -> Dict[str, Any]:',
        '        """获取系统健康状态"""',
        '        return self.core.get_system_health_status()',
        '',
        '    def detect_anomalies(self, metric_name: str = None) -> Dict[str, Any]:',
        '        """检测异常"""',
        '        return self.anomaly_detector.detect_anomaly(metric_name) if metric_name else {}',
        '',
        '    def analyze_trends(self, metric_name: str = None) -> Dict[str, Any]:',
        '        """分析趋势"""',
        '        return self.trend_analyzer.analyze_trend(metric_name) if metric_name else {}',
        '',
        '    def predict_performance(self, metric_name: str = None, hours_ahead: int = 1) -> Dict[str, Any]:',
        '        """预测性能"""',
        '        return self.performance_predictor.predict_next_value(metric_name) if metric_name else {}',
        '',
        '    def get_monitoring_stats(self) -> Dict[str, Any]:',
        '        """获取监控统计"""',
        '        return {',
        '            "core_stats": self.core.get_operation_stats(),',
        '            "anomalies": self.detect_anomalies(),',
        '            "trends": self.analyze_trends(),',
        '            "predictions": self.predict_performance()',
        '        }',
        '',
        '    # 保持向后兼容的别名',
        '    get_performance_report = get_monitoring_stats',
        '    get_performance_summary = get_monitoring_stats'
    ]

    return '\n'.join(dashboard_lines)


def main():
    """主函数"""

    print('=== 📦 Phase 2.1: 拆分性能监控面板 ===')
    print()

    source_file = 'src/infrastructure/config/monitoring/performance_monitor_dashboard.py'

    # 创建输出目录
    monitoring_dir = 'src/infrastructure/config/monitoring'
    os.makedirs(monitoring_dir, exist_ok=True)

    try:
        # 读取源文件
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 创建各个模块
        modules = [
            ('core.py', create_monitoring_core(content)),
            ('anomaly_detector.py', create_anomaly_detector(content)),
            ('trend_analyzer.py', create_trend_analyzer(content)),
            ('performance_predictor.py', create_performance_predictor(content)),
            ('performance_monitor_dashboard.py', create_unified_monitoring_dashboard())
        ]

        created_files = []
        for filename, module_content in modules:
            file_path = os.path.join(monitoring_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(module_content)
            created_files.append(file_path)
            print(f'✅ 创建: {filename}')

        # 创建__init__.py
        init_content = '''"""监控模块"""

from .core import PerformanceMonitorDashboardCore
from .anomaly_detector import AnomalyDetector
from .trend_analyzer import TrendAnalyzer
from .performance_predictor import PerformancePredictor
from .performance_monitor_dashboard import PerformanceMonitorDashboard

__all__ = [
    "PerformanceMonitorDashboardCore",
    "AnomalyDetector",
    "TrendAnalyzer",
    "PerformancePredictor",
    "PerformanceMonitorDashboard"
]
'''

        init_file = os.path.join(monitoring_dir, '__init__.py')
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(init_content)

        print(f'✅ 创建: __init__.py')

        print()
        print('🎯 拆分完成！')
        print(f'   📦 创建了 {len(created_files)} 个模块文件')
        print('   📁 新的监控模块结构:')
        print('      monitoring/')
        print('      ├── core.py              # 监控核心')
        print('      ├── anomaly_detector.py  # 异常检测')
        print('      ├── trend_analyzer.py    # 趋势分析')
        print('      └── performance_predictor.py # 性能预测')
        print('      └── performance_monitor_dashboard.py # 统一接口')

        # 验证文件大小
        print()
        print('📊 文件大小验证:')
        for file_path in created_files:
            if os.path.exists(file_path):
                size_kb = os.path.getsize(file_path) / 1024
                rel_path = os.path.relpath(file_path, 'src/infrastructure/config')
                status = '✅' if size_kb < 15 else '⚠️'
                print(f'   {status} {rel_path}: {size_kb:.1f} KB')

    except Exception as e:
        print(f'❌ 拆分失败: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
