#!/usr/bin/env python3
"""
容量规划分析器
分析系统资源使用情况，制定性能优化和容量规划策略
"""

import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import psutil
import threading
from dataclasses import dataclass, asdict
# import matplotlib.pyplot as plt
# import seaborn as sns

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ResourceUsage:
    """资源使用数据类"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, float]
    gpu_usage: Optional[Dict[str, float]] = None
    process_count: int = 0
    load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class CapacityRecommendation:
    """容量建议数据类"""
    resource_type: str
    current_usage: float
    recommended_capacity: float
    growth_rate: float
    time_to_capacity: int  # 天
    priority: str  # HIGH, MEDIUM, LOW
    cost_estimate: float
    optimization_suggestions: List[str]


class CapacityPlanningAnalyzer:
    """容量规划分析器"""

    def __init__(self, output_dir: str = "reports/capacity_planning"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

        # 容量规划配置
        self.capacity_config = {
            'cpu_warning_threshold': 80.0,  # CPU使用率告警阈值
            'memory_warning_threshold': 85.0,  # 内存使用率告警阈值
            'disk_warning_threshold': 90.0,  # 磁盘使用率告警阈值
            'network_warning_threshold': 80.0,  # 网络使用率告警阈值
            'gpu_warning_threshold': 85.0,  # GPU使用率告警阈值
            'growth_analysis_days': 30,  # 增长分析天数
            'capacity_buffer': 0.2,  # 容量缓冲系数
            'cost_per_cpu_core': 100,  # 每CPU核心成本（元/月）
            'cost_per_gb_memory': 50,  # 每GB内存成本（元/月）
            'cost_per_gb_storage': 10,  # 每GB存储成本（元/月）
            'cost_per_gpu': 2000,  # 每GPU成本（元/月）
        }

        # 资源使用历史
        self.resource_history: List[ResourceUsage] = []
        self.monitoring = False
        self.monitor_thread = None

        # 分析结果
        self.analysis_results = {
            'current_usage': {},
            'trend_analysis': {},
            'capacity_recommendations': [],
            'optimization_suggestions': [],
            'cost_analysis': {}
        }

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def start_monitoring(self, interval: float = 60.0):
        """开始资源监控"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"开始资源监控，间隔: {interval}秒")

    def stop_monitoring(self):
        """停止资源监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("停止资源监控")

    def _monitor_resources(self, interval: float):
        """监控系统资源"""
        while self.monitoring:
            try:
                usage = self._collect_resource_usage()
                self.resource_history.append(usage)

                # 保持历史数据在合理范围内
                if len(self.resource_history) > 1000:
                    self.resource_history = self.resource_history[-500:]

                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"资源监控错误: {e}")
                time.sleep(interval)

    def _collect_resource_usage(self) -> ResourceUsage:
        """收集资源使用情况"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)

        # 内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100

        # 网络IO
        network_io = psutil.net_io_counters()
        network_usage = {
            'bytes_sent': network_io.bytes_sent,
            'bytes_recv': network_io.bytes_recv,
            'packets_sent': network_io.packets_sent,
            'packets_recv': network_io.packets_recv
        }

        # GPU使用率（如果可用）
        gpu_usage = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_usage = {}
                for i in range(torch.cuda.device_count()):
                    gpu_usage[f'gpu_{i}'] = {
                        'memory_allocated': torch.cuda.memory_allocated(i) / 1024**3,  # GB
                        'memory_reserved': torch.cuda.memory_reserved(i) / 1024**3,  # GB
                        'utilization': 0.0  # 需要额外工具获取
                    }
        except ImportError:
            pass

        # 进程数量
        process_count = len(psutil.pids())

        # 负载平均值
        load_average = psutil.getloadavg()

        return ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            network_io=network_usage,
            gpu_usage=gpu_usage,
            process_count=process_count,
            load_average=load_average
        )

    def analyze_current_usage(self) -> Dict[str, Any]:
        """分析当前资源使用情况"""
        if not self.resource_history:
            self.logger.warning("没有资源使用历史数据")
            return {}

        latest_usage = self.resource_history[-1]

        # 计算平均值
        recent_data = self.resource_history[-10:]  # 最近10个数据点
        avg_cpu = np.mean([u.cpu_percent for u in recent_data])
        avg_memory = np.mean([u.memory_percent for u in recent_data])
        avg_disk = np.mean([u.disk_percent for u in recent_data])

        # 检查告警
        alerts = []
        if avg_cpu > self.capacity_config['cpu_warning_threshold']:
            alerts.append(f"CPU使用率过高: {avg_cpu:.1f}%")
        if avg_memory > self.capacity_config['memory_warning_threshold']:
            alerts.append(f"内存使用率过高: {avg_memory:.1f}%")
        if avg_disk > self.capacity_config['disk_warning_threshold']:
            alerts.append(f"磁盘使用率过高: {avg_disk:.1f}%")

        current_usage = {
            'timestamp': latest_usage.timestamp,
            'cpu_percent': avg_cpu,
            'memory_percent': avg_memory,
            'disk_percent': avg_disk,
            'process_count': latest_usage.process_count,
            'load_average': latest_usage.load_average,
            'alerts': alerts,
            'gpu_usage': latest_usage.gpu_usage
        }

        self.analysis_results['current_usage'] = current_usage
        return current_usage

    def analyze_trends(self) -> Dict[str, Any]:
        """分析资源使用趋势"""
        if len(self.resource_history) < 10:
            self.logger.warning("历史数据不足，无法进行趋势分析")
            return {}

        # 转换为DataFrame进行分析
        df = pd.DataFrame([
            {
                'timestamp': u.timestamp,
                'cpu_percent': u.cpu_percent,
                'memory_percent': u.memory_percent,
                'disk_percent': u.disk_percent,
                'process_count': u.process_count
            }
            for u in self.resource_history
        ])

        # 计算趋势
        trends = {}
        for column in ['cpu_percent', 'memory_percent', 'disk_percent', 'process_count']:
            if len(df) > 1:
                # 线性回归计算增长率
                x = np.arange(len(df))
                y = df[column].values
                slope = np.polyfit(x, y, 1)[0]
                trends[column] = {
                    'current_value': df[column].iloc[-1],
                    'average_value': df[column].mean(),
                    'growth_rate': slope,
                    'trend': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                }

        self.analysis_results['trend_analysis'] = trends
        return trends

    def generate_capacity_recommendations(self) -> List[CapacityRecommendation]:
        """生成容量建议"""
        if not self.resource_history:
            return []

        recommendations = []
        trends = self.analyze_trends()

        # CPU容量建议
        if 'cpu_percent' in trends:
            cpu_trend = trends['cpu_percent']
            current_cpu = cpu_trend['current_value']
            growth_rate = cpu_trend['growth_rate']

            # 计算达到阈值的时间
            threshold = self.capacity_config['cpu_warning_threshold']
            if growth_rate > 0:
                time_to_capacity = int((threshold - current_cpu) / growth_rate)
            else:
                time_to_capacity = 365  # 无增长，设为1年

            # 计算建议容量
            recommended_capacity = current_cpu * (1 + self.capacity_config['capacity_buffer'])

            # 成本估算
            current_cores = psutil.cpu_count()
            additional_cores = max(0, int(current_cores * 0.2))  # 建议增加20%核心
            cost_estimate = additional_cores * self.capacity_config['cost_per_cpu_core']

            recommendations.append(CapacityRecommendation(
                resource_type='CPU',
                current_usage=current_cpu,
                recommended_capacity=recommended_capacity,
                growth_rate=growth_rate,
                time_to_capacity=time_to_capacity,
                priority='HIGH' if time_to_capacity < 30 else 'MEDIUM' if time_to_capacity < 90 else 'LOW',
                cost_estimate=cost_estimate,
                optimization_suggestions=[
                    '优化CPU密集型任务',
                    '启用任务调度优化',
                    '考虑使用更高效的算法'
                ]
            ))

        # 内存容量建议
        if 'memory_percent' in trends:
            memory_trend = trends['memory_percent']
            current_memory = memory_trend['current_value']
            growth_rate = memory_trend['growth_rate']

            threshold = self.capacity_config['memory_warning_threshold']
            if growth_rate > 0:
                time_to_capacity = int((threshold - current_memory) / growth_rate)
            else:
                time_to_capacity = 365

            recommended_capacity = current_memory * (1 + self.capacity_config['capacity_buffer'])

            # 成本估算
            current_memory_gb = psutil.virtual_memory().total / (1024**3)
            additional_memory_gb = max(0, current_memory_gb * 0.3)  # 建议增加30%内存
            cost_estimate = additional_memory_gb * self.capacity_config['cost_per_gb_memory']

            recommendations.append(CapacityRecommendation(
                resource_type='Memory',
                current_usage=current_memory,
                recommended_capacity=recommended_capacity,
                growth_rate=growth_rate,
                time_to_capacity=time_to_capacity,
                priority='HIGH' if time_to_capacity < 30 else 'MEDIUM' if time_to_capacity < 90 else 'LOW',
                cost_estimate=cost_estimate,
                optimization_suggestions=[
                    '优化内存使用',
                    '启用内存缓存',
                    '定期清理无用对象'
                ]
            ))

        # 磁盘容量建议
        if 'disk_percent' in trends:
            disk_trend = trends['disk_percent']
            current_disk = disk_trend['current_value']
            growth_rate = disk_trend['growth_rate']

            threshold = self.capacity_config['disk_warning_threshold']
            if growth_rate > 0:
                time_to_capacity = int((threshold - current_disk) / growth_rate)
            else:
                time_to_capacity = 365

            recommended_capacity = current_disk * (1 + self.capacity_config['capacity_buffer'])

            # 成本估算
            current_disk_gb = psutil.disk_usage('/').total / (1024**3)
            additional_disk_gb = max(0, current_disk_gb * 0.5)  # 建议增加50%存储
            cost_estimate = additional_disk_gb * self.capacity_config['cost_per_gb_storage']

            recommendations.append(CapacityRecommendation(
                resource_type='Storage',
                current_usage=current_disk,
                recommended_capacity=recommended_capacity,
                growth_rate=growth_rate,
                time_to_capacity=time_to_capacity,
                priority='HIGH' if time_to_capacity < 30 else 'MEDIUM' if time_to_capacity < 90 else 'LOW',
                cost_estimate=cost_estimate,
                optimization_suggestions=[
                    '清理临时文件',
                    '启用数据压缩',
                    '实施数据归档策略'
                ]
            ))

        self.analysis_results['capacity_recommendations'] = recommendations
        return recommendations

    def generate_optimization_suggestions(self) -> List[str]:
        """生成优化建议"""
        suggestions = []

        # 基于当前使用情况生成建议
        current_usage = self.analyze_current_usage()

        if current_usage.get('cpu_percent', 0) > 70:
            suggestions.extend([
                '考虑增加CPU核心数',
                '优化计算密集型任务',
                '启用任务并行处理'
            ])

        if current_usage.get('memory_percent', 0) > 80:
            suggestions.extend([
                '增加系统内存',
                '优化内存使用模式',
                '启用内存缓存机制'
            ])

        if current_usage.get('disk_percent', 0) > 85:
            suggestions.extend([
                '扩展存储容量',
                '清理无用文件',
                '启用数据压缩'
            ])

        # 基于趋势分析生成建议
        trends = self.analyze_trends()
        for resource, trend in trends.items():
            if trend['trend'] == 'increasing' and trend['growth_rate'] > 0.1:
                suggestions.append(f'{resource}使用率快速增长，建议提前扩容')

        # 通用优化建议
        suggestions.extend([
            '实施资源监控和告警',
            '建立容量规划流程',
            '定期进行性能优化',
            '考虑使用云原生架构'
        ])

        self.analysis_results['optimization_suggestions'] = suggestions
        return suggestions

    def generate_cost_analysis(self) -> Dict[str, Any]:
        """生成成本分析"""
        recommendations = self.generate_capacity_recommendations()

        total_cost = sum(rec.cost_estimate for rec in recommendations)
        cost_breakdown = {
            rec.resource_type: rec.cost_estimate for rec in recommendations
        }

        # 按优先级分组
        high_priority_cost = sum(
            rec.cost_estimate for rec in recommendations
            if rec.priority == 'HIGH'
        )
        medium_priority_cost = sum(
            rec.cost_estimate for rec in recommendations
            if rec.priority == 'MEDIUM'
        )
        low_priority_cost = sum(
            rec.cost_estimate for rec in recommendations
            if rec.priority == 'LOW'
        )

        cost_analysis = {
            'total_monthly_cost': total_cost,
            'cost_breakdown': cost_breakdown,
            'priority_cost_breakdown': {
                'high_priority': high_priority_cost,
                'medium_priority': medium_priority_cost,
                'low_priority': low_priority_cost
            },
            'roi_analysis': {
                'estimated_performance_improvement': '20-30%',
                'estimated_cost_savings': '15-25%',
                'payback_period_months': 6
            }
        }

        self.analysis_results['cost_analysis'] = cost_analysis
        return cost_analysis

    def generate_report(self) -> str:
        """生成容量规划报告"""
        # 执行所有分析
        current_usage = self.analyze_current_usage()
        trends = self.analyze_trends()
        recommendations = self.generate_capacity_recommendations()
        suggestions = self.generate_optimization_suggestions()
        cost_analysis = self.generate_cost_analysis()

        # 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_recommendations': len(recommendations),
                'high_priority_count': len([r for r in recommendations if r.priority == 'HIGH']),
                'total_monthly_cost': cost_analysis['total_monthly_cost'],
                'critical_alerts': len(current_usage.get('alerts', []))
            },
            'current_usage': current_usage,
            'trend_analysis': trends,
            'capacity_recommendations': [asdict(r) for r in recommendations],
            'optimization_suggestions': suggestions,
            'cost_analysis': cost_analysis
        }

        # 保存报告
        report_file = self.output_dir / \
            f"capacity_planning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        # 生成Markdown报告
        md_report = self._generate_markdown_report(report)
        md_file = self.output_dir / \
            f"capacity_planning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_report)

        self.logger.info(f"容量规划报告已生成: {report_file}")
        return str(report_file)

    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """生成Markdown格式报告"""
        md_content = f"""# 容量规划分析报告

**生成时间**: {report['timestamp']}  
**分析概要**: {report['summary']['total_recommendations']} 个建议，{report['summary']['high_priority_count']} 个高优先级

## 📊 当前资源使用情况

### 系统状态
- **CPU使用率**: {report['current_usage'].get('cpu_percent', 0):.1f}%
- **内存使用率**: {report['current_usage'].get('memory_percent', 0):.1f}%
- **磁盘使用率**: {report['current_usage'].get('disk_percent', 0):.1f}%
- **进程数量**: {report['current_usage'].get('process_count', 0)}

### 告警信息
"""

        alerts = report['current_usage'].get('alerts', [])
        if alerts:
            for alert in alerts:
                md_content += f"- ⚠️ {alert}\n"
        else:
            md_content += "- ✅ 无告警\n"

        md_content += f"""
## 📈 趋势分析

### 资源使用趋势
"""

        for resource, trend in report['trend_analysis'].items():
            md_content += f"""
#### {resource}
- **当前值**: {trend['current_value']:.1f}
- **平均值**: {trend['average_value']:.1f}
- **增长率**: {trend['growth_rate']:.3f}
- **趋势**: {trend['trend']}
"""

        md_content += f"""
## 💡 容量建议

### 高优先级建议
"""

        high_priority = [r for r in report['capacity_recommendations'] if r['priority'] == 'HIGH']
        for rec in high_priority:
            md_content += f"""
#### {rec['resource_type']}
- **当前使用**: {rec['current_usage']:.1f}%
- **建议容量**: {rec['recommended_capacity']:.1f}%
- **达到容量时间**: {rec['time_to_capacity']} 天
- **月成本**: ¥{rec['cost_estimate']:.0f}
- **优化建议**:
"""
            for suggestion in rec['optimization_suggestions']:
                md_content += f"  - {suggestion}\n"

        md_content += f"""
## 💰 成本分析

### 总成本
- **月度总成本**: ¥{report['cost_analysis']['total_monthly_cost']:.0f}
- **高优先级成本**: ¥{report['cost_analysis']['priority_cost_breakdown']['high_priority']:.0f}
- **中优先级成本**: ¥{report['cost_analysis']['priority_cost_breakdown']['medium_priority']:.0f}
- **低优先级成本**: ¥{report['cost_analysis']['priority_cost_breakdown']['low_priority']:.0f}

### 投资回报分析
- **预计性能提升**: {report['cost_analysis']['roi_analysis']['estimated_performance_improvement']}
- **预计成本节省**: {report['cost_analysis']['roi_analysis']['estimated_cost_savings']}
- **投资回收期**: {report['cost_analysis']['roi_analysis']['payback_period_months']} 个月

## 🔧 优化建议

"""

        for suggestion in report['optimization_suggestions']:
            md_content += f"- {suggestion}\n"

        md_content += f"""
## 📋 行动计划

### 立即行动（1-2周）
"""

        for rec in high_priority:
            md_content += f"- 扩容 {rec['resource_type']} 资源\n"

        md_content += f"""
### 短期计划（1-2月）
- 实施监控和告警系统
- 优化资源使用模式
- 建立容量规划流程

### 长期规划（3-6月）
- 考虑云原生架构
- 实施自动化扩容
- 建立性能基准

---
**报告生成器**: 容量规划分析器  
**版本**: 1.0.0
"""

        return md_content


def main():
    """主函数"""
    analyzer = CapacityPlanningAnalyzer()

    # 开始监控
    analyzer.start_monitoring(interval=30.0)  # 30秒间隔

    try:
        # 监控一段时间
        print("开始容量规划分析...")
        print("监控中，按 Ctrl+C 停止...")

        # 监控5分钟
        time.sleep(300)

    except KeyboardInterrupt:
        print("\n停止监控...")
    finally:
        analyzer.stop_monitoring()

        # 生成报告
        report_file = analyzer.generate_report()
        print(f"容量规划报告已生成: {report_file}")


if __name__ == "__main__":
    main()
