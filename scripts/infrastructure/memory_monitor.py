#!/usr/bin/env python3
"""
内存监控脚本
持续监控内存使用情况并检测内存暴涨问题
"""

import sys
import gc
import psutil
import time
import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class MemoryMonitor:
    """内存监控器"""

    def __init__(self, monitoring_duration: int = 300, check_interval: int = 5):
        self.project_root = Path(project_root)
        self.report_dir = self.project_root / 'reports' / 'infrastructure'
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.monitoring_duration = monitoring_duration  # 监控时长（秒）
        self.check_interval = check_interval  # 检查间隔（秒）
        self.stop_monitoring = False

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.report_dir / 'memory_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.monitoring_data = {
            'start_time': datetime.now().isoformat(),
            'duration': monitoring_duration,
            'check_interval': check_interval,
            'memory_samples': [],
            'alerts': [],
            'summary': {}
        }

        # 内存暴涨阈值
        self.memory_surge_threshold = 50  # MB
        self.memory_growth_threshold = 10  # MB/分钟
        self.gc_threshold = 1000  # 垃圾回收对象数量

    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        memory_info = psutil.virtual_memory()
        process = psutil.Process()

        return {
            'timestamp': datetime.now().isoformat(),
            'system_total': memory_info.total,
            'system_available': memory_info.available,
            'system_percent': memory_info.percent,
            'process_rss': process.memory_info().rss,
            'process_vms': process.memory_info().vms,
            'process_percent': process.memory_percent(),
            'gc_stats': gc.get_stats(),
            'gc_count': gc.get_count()
        }

    def detect_memory_surge(self, current_sample: Dict[str, Any], previous_sample: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """检测内存暴涨"""
        alerts = []

        if previous_sample:
            # 计算内存增长
            memory_growth_mb = (current_sample['process_rss'] -
                                previous_sample['process_rss']) / 1024 / 1024
            time_diff_minutes = (datetime.fromisoformat(current_sample['timestamp']) -
                                 datetime.fromisoformat(previous_sample['timestamp'])).total_seconds() / 60

            if time_diff_minutes > 0:
                growth_rate = memory_growth_mb / time_diff_minutes

                # 检查内存暴涨
                if memory_growth_mb > self.memory_surge_threshold:
                    alerts.append({
                        'type': 'memory_surge',
                        'severity': 'critical',
                        'message': f'内存暴涨: {memory_growth_mb:.2f} MB',
                        'timestamp': current_sample['timestamp'],
                        'value': memory_growth_mb
                    })

                # 检查内存增长率
                if growth_rate > self.memory_growth_threshold:
                    alerts.append({
                        'type': 'memory_growth_rate',
                        'severity': 'warning',
                        'message': f'内存增长率过高: {growth_rate:.2f} MB/分钟',
                        'timestamp': current_sample['timestamp'],
                        'value': growth_rate
                    })

        # 检查系统内存使用
        if current_sample['system_percent'] > 80:
            alerts.append({
                'type': 'system_memory_high',
                'severity': 'warning',
                'message': f'系统内存使用率过高: {current_sample["system_percent"]:.1f}%',
                'timestamp': current_sample['timestamp'],
                'value': current_sample['system_percent']
            })

        # 检查垃圾回收
        gc_stats = current_sample['gc_stats']
        if gc_stats and len(gc_stats) > 0:
            total_collected = sum(stat['collected'] for stat in gc_stats)
            if total_collected > self.gc_threshold:
                alerts.append({
                    'type': 'gc_high',
                    'severity': 'info',
                    'message': f'垃圾回收对象过多: {total_collected}',
                    'timestamp': current_sample['timestamp'],
                    'value': total_collected
                })

        return alerts

    def analyze_memory_trend(self) -> Dict[str, Any]:
        """分析内存趋势"""
        if len(self.monitoring_data['memory_samples']) < 2:
            return {}

        samples = self.monitoring_data['memory_samples']

        # 计算趋势
        memory_values = [s['process_rss'] for s in samples]
        timestamps = [datetime.fromisoformat(s['timestamp']) for s in samples]

        # 计算内存增长率
        if len(memory_values) >= 2:
            total_growth = (memory_values[-1] - memory_values[0]) / 1024 / 1024  # MB
            total_time = (timestamps[-1] - timestamps[0]).total_seconds() / 60  # 分钟

            if total_time > 0:
                average_growth_rate = total_growth / total_time
            else:
                average_growth_rate = 0
        else:
            average_growth_rate = 0

        # 计算内存使用峰值
        max_memory = max(memory_values) / 1024 / 1024  # MB
        min_memory = min(memory_values) / 1024 / 1024  # MB
        avg_memory = sum(memory_values) / len(memory_values) / 1024 / 1024  # MB

        return {
            'total_samples': len(samples),
            'monitoring_duration_minutes': total_time if len(timestamps) >= 2 else 0,
            'average_growth_rate_mb_per_minute': average_growth_rate,
            'max_memory_mb': max_memory,
            'min_memory_mb': min_memory,
            'avg_memory_mb': avg_memory,
            'memory_volatility': max_memory - min_memory
        }

    def generate_recommendations(self) -> List[str]:
        """生成内存优化建议"""
        recommendations = []

        trend = self.analyze_memory_trend()
        if trend:
            if trend.get('average_growth_rate_mb_per_minute', 0) > 5:
                recommendations.append("内存持续增长，建议检查内存泄漏")

            if trend.get('memory_volatility', 0) > 100:
                recommendations.append("内存波动较大，建议优化内存分配策略")

            if trend.get('avg_memory_mb', 0) > 500:
                recommendations.append("平均内存使用较高，建议优化数据结构")

        # 基于告警生成建议
        alert_types = [alert['type'] for alert in self.monitoring_data['alerts']]

        if 'memory_surge' in alert_types:
            recommendations.append("检测到内存暴涨，建议立即检查内存泄漏")

        if 'memory_growth_rate' in alert_types:
            recommendations.append("内存增长率过高，建议优化内存使用")

        if 'system_memory_high' in alert_types:
            recommendations.append("系统内存使用率过高，建议增加系统内存或优化应用")

        if 'gc_high' in alert_types:
            recommendations.append("垃圾回收频繁，建议优化对象生命周期管理")

        if not recommendations:
            recommendations.append("内存使用正常，建议继续监控")

        return recommendations

    def save_monitoring_report(self):
        """保存监控报告"""
        # 分析趋势
        trend = self.analyze_memory_trend()
        self.monitoring_data['summary'] = trend

        # 生成建议
        recommendations = self.generate_recommendations()
        self.monitoring_data['recommendations'] = recommendations

        # 保存完整报告
        report_file = self.report_dir / 'memory_monitoring_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.monitoring_data, f, ensure_ascii=False, indent=2)

        # 生成Markdown报告
        md_report = self._generate_markdown_report()
        md_file = self.report_dir / 'memory_monitoring_report.md'
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_report)

        self.logger.info(f"监控报告已保存到: {self.report_dir}")

    def _generate_markdown_report(self) -> str:
        """生成Markdown格式报告"""
        md_content = f"""# 内存监控报告

## 监控概述
- 开始时间: {self.monitoring_data['start_time']}
- 监控时长: {self.monitoring_data['duration']} 秒
- 检查间隔: {self.monitoring_data['check_interval']} 秒
- 样本数量: {len(self.monitoring_data['memory_samples'])}

## 内存趋势分析
"""

        summary = self.monitoring_data['summary']
        if summary:
            md_content += f"- 监控时长: {summary.get('monitoring_duration_minutes', 0):.1f} 分钟\n"
            md_content += f"- 平均增长率: {summary.get('average_growth_rate_mb_per_minute', 0):.2f} MB/分钟\n"
            md_content += f"- 最大内存: {summary.get('max_memory_mb', 0):.1f} MB\n"
            md_content += f"- 最小内存: {summary.get('min_memory_mb', 0):.1f} MB\n"
            md_content += f"- 平均内存: {summary.get('avg_memory_mb', 0):.1f} MB\n"
            md_content += f"- 内存波动: {summary.get('memory_volatility', 0):.1f} MB\n"

        md_content += "\n## 告警统计\n"
        alert_counts = {}
        for alert in self.monitoring_data['alerts']:
            alert_type = alert['type']
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1

        for alert_type, count in alert_counts.items():
            md_content += f"- {alert_type}: {count} 次\n"

        md_content += "\n## 优化建议\n"
        for rec in self.monitoring_data.get('recommendations', []):
            md_content += f"- {rec}\n"

        return md_content

    def monitor_memory(self):
        """监控内存使用"""
        self.logger.info(f"开始内存监控，时长: {self.monitoring_duration}秒，间隔: {self.check_interval}秒")

        start_time = time.time()
        previous_sample = None

        while not self.stop_monitoring and (time.time() - start_time) < self.monitoring_duration:
            try:
                # 获取内存信息
                current_sample = self.get_memory_info()
                self.monitoring_data['memory_samples'].append(current_sample)

                # 检测内存暴涨
                alerts = self.detect_memory_surge(current_sample, previous_sample)
                self.monitoring_data['alerts'].extend(alerts)

                # 记录告警
                for alert in alerts:
                    self.logger.warning(f"{alert['severity'].upper()}: {alert['message']}")

                # 输出当前状态
                memory_mb = current_sample['process_rss'] / 1024 / 1024
                system_percent = current_sample['system_percent']
                self.logger.info(f"内存使用: {memory_mb:.1f} MB, 系统内存: {system_percent:.1f}%")

                previous_sample = current_sample

                # 等待下次检查
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                self.logger.info("监控被用户中断")
                break
            except Exception as e:
                self.logger.error(f"监控过程中发生错误: {e}")
                break

        self.logger.info("内存监控完成")

    def run(self):
        """运行内存监控"""
        try:
            # 启动监控线程
            monitor_thread = threading.Thread(target=self.monitor_memory)
            monitor_thread.daemon = True
            monitor_thread.start()

            # 等待监控完成
            monitor_thread.join()

            # 保存报告
            self.save_monitoring_report()

            # 输出摘要
            summary = self.analyze_memory_trend()
            print(f"\n=== 内存监控摘要 ===")
            print(f"监控样本: {len(self.monitoring_data['memory_samples'])}")
            print(f"告警数量: {len(self.monitoring_data['alerts'])}")
            if summary:
                print(f"平均增长率: {summary.get('average_growth_rate_mb_per_minute', 0):.2f} MB/分钟")
                print(f"内存波动: {summary.get('memory_volatility', 0):.1f} MB")

        except Exception as e:
            self.logger.error(f"内存监控失败: {e}")
            raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='内存监控脚本')
    parser.add_argument('--duration', type=int, default=300, help='监控时长（秒）')
    parser.add_argument('--interval', type=int, default=5, help='检查间隔（秒）')

    args = parser.parse_args()

    monitor = MemoryMonitor(
        monitoring_duration=args.duration,
        check_interval=args.interval
    )
    monitor.run()
