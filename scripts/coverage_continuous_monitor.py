#!/usr/bin/env python3
"""
持续测试覆盖率监控系统

建立可持续的质量监控机制，定期检查和报告测试覆盖率状态
"""

import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import schedule
import threading
import psutil
import sqlite3

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/coverage_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ContinuousCoverageMonitor:
    """持续测试覆盖率监控器"""

    def __init__(self, project_root: str, db_path: str = "data/coverage_monitor.db"):
        self.project_root = Path(project_root)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 监控配置
        self.config = {
            'monitoring_interval': 3600,  # 1小时
            'alert_thresholds': {
                'overall_coverage': 70.0,
                'layer_coverage': 60.0,
                'file_coverage': 50.0
            },
            'report_formats': ['html', 'json', 'xml'],
            'notification_channels': ['console', 'file'],
            'layers': {
                'infrastructure': 'src/infrastructure',
                'data': 'src/data',
                'features': 'src/features',
                'strategy': 'src/strategy',
                'trading': 'src/trading',
                'risk': 'src/risk'
            }
        }

        # 监控状态
        self.monitoring_active = False
        self.last_report_time = None
        self.baseline_coverage = {}

        # 初始化数据库
        self.init_database()

    def init_database(self):
        """初始化监控数据库"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # 创建覆盖率历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS coverage_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    coverage REAL NOT NULL,
                    statements INTEGER,
                    missed INTEGER,
                    branches INTEGER,
                    partial INTEGER
                )
            ''')

            # 创建告警历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    layer TEXT,
                    file_path TEXT
                )
            ''')

            # 创建基准线表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    layer TEXT NOT NULL UNIQUE,
                    baseline_coverage REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')

            conn.commit()
            conn.close()
            logger.info("✅ 监控数据库初始化完成")
        except Exception as e:
            logger.error(f"❌ 数据库初始化失败: {e}")

    def start_monitoring(self):
        """启动持续监控"""
        if self.monitoring_active:
            logger.warning("⚠️ 监控已启动")
            return

        logger.info("🚀 启动持续测试覆盖率监控...")
        self.monitoring_active = True

        # 设置定期任务
        schedule.every(self.config['monitoring_interval']).seconds.do(self.run_monitoring_cycle)

        # 在后台运行调度器
        monitor_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        monitor_thread.start()

        logger.info(f"✅ 监控已启动，间隔: {self.config['monitoring_interval']}秒")

    def stop_monitoring(self):
        """停止监控"""
        logger.info("🛑 停止持续测试覆盖率监控...")
        self.monitoring_active = False
        schedule.clear()

    def _run_scheduler(self):
        """运行调度器"""
        while self.monitoring_active:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次

    def run_monitoring_cycle(self):
        """执行监控周期"""
        try:
            logger.info("🔍 执行监控周期...")

            # 1. 生成覆盖率报告
            coverage_data = self.generate_coverage_report()

            # 2. 分析覆盖率数据
            analysis = self.analyze_coverage_data(coverage_data)

            # 3. 检查告警条件
            alerts = self.check_alerts(analysis)

            # 4. 保存到数据库
            self.save_to_database(coverage_data, alerts)

            # 5. 生成报告
            self.generate_reports(analysis, alerts)

            # 6. 发送通知
            self.send_notifications(alerts)

            self.last_report_time = datetime.now()
            logger.info("✅ 监控周期完成")

        except Exception as e:
            logger.error(f"❌ 监控周期执行失败: {e}")

    def generate_coverage_report(self) -> Dict[str, Any]:
        """生成覆盖率报告"""
        try:
            # 运行pytest并生成覆盖率报告
            cmd = [
                sys.executable, "-m", "pytest",
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=json:coverage.json",
                "tests/"
            ]

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )

            # 读取覆盖率数据
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r', encoding='utf-8') as f:
                    coverage_data = json.load(f)
            else:
                coverage_data = {}

            return coverage_data

        except Exception as e:
            logger.error(f"生成覆盖率报告失败: {e}")
            return {}

    def analyze_coverage_data(self, coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析覆盖率数据"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'overall': {},
            'layers': {},
            'files': {},
            'trends': {},
            'recommendations': []
        }

        if not coverage_data:
            return analysis

        try:
            # 计算整体覆盖率
            totals = coverage_data.get('totals', {})
            analysis['overall'] = {
                'coverage': totals.get('percent_covered', 0),
                'statements': totals.get('num_statements', 0),
                'missed': totals.get('missing_lines', 0),
                'branches': totals.get('num_branches', 0),
                'partial': totals.get('num_partial_branches', 0)
            }

            # 分析各层覆盖率
            for layer_name, layer_path in self.config['layers'].items():
                layer_files = {
                    file_path: data
                    for file_path, data in coverage_data.get('files', {}).items()
                    if layer_path in file_path
                }

                if layer_files:
                    layer_analysis = self._analyze_layer_coverage(layer_files)
                    analysis['layers'][layer_name] = layer_analysis

            # 生成建议
            analysis['recommendations'] = self._generate_recommendations(analysis)

        except Exception as e:
            logger.error(f"分析覆盖率数据失败: {e}")

        return analysis

    def _analyze_layer_coverage(self, layer_files: Dict[str, Any]) -> Dict[str, Any]:
        """分析层级覆盖率"""
        if not layer_files:
            return {}

        total_statements = 0
        total_missed = 0
        total_branches = 0
        total_partial = 0

        for file_data in layer_files.values():
            summary = file_data.get('summary', {})
            total_statements += summary.get('num_statements', 0)
            total_missed += summary.get('missing_lines', 0)
            total_branches += summary.get('num_branches', 0)
            total_partial += summary.get('num_partial_branches', 0)

        coverage = 0.0
        if total_statements > 0:
            coverage = ((total_statements - total_missed) / total_statements) * 100

        return {
            'coverage': round(coverage, 2),
            'statements': total_statements,
            'missed': total_missed,
            'branches': total_branches,
            'partial': total_partial,
            'files_count': len(layer_files)
        }

    def check_alerts(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查告警条件"""
        alerts = []

        try:
            # 检查整体覆盖率
            overall_coverage = analysis.get('overall', {}).get('coverage', 0)
            threshold = self.config['alert_thresholds']['overall_coverage']

            if overall_coverage < threshold:
                alerts.append({
                    'type': 'overall_coverage_low',
                    'message': '.2f',
                    'severity': 'high' if overall_coverage < threshold - 10 else 'medium',
                    'layer': 'overall'
                })

            # 检查各层覆盖率
            for layer_name, layer_data in analysis.get('layers', {}).items():
                layer_coverage = layer_data.get('coverage', 0)
                layer_threshold = self.config['alert_thresholds']['layer_coverage']

                if layer_coverage < layer_threshold:
                    alerts.append({
                        'type': 'layer_coverage_low',
                        'message': '.2f',
                        'severity': 'medium',
                        'layer': layer_name
                    })

        except Exception as e:
            logger.error(f"检查告警失败: {e}")

        return alerts

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        overall_coverage = analysis.get('overall', {}).get('coverage', 0)

        if overall_coverage < 60:
            recommendations.append("🔴 紧急: 整体测试覆盖率过低，优先编写核心业务逻辑的单元测试")
        elif overall_coverage < 70:
            recommendations.append("🟡 警告: 测试覆盖率需提升，重点关注边界条件和异常处理")
        else:
            recommendations.append("🟢 良好: 测试覆盖率达标，继续保持和扩展测试范围")

        # 检查各层覆盖率
        for layer_name, layer_data in analysis.get('layers', {}).items():
            layer_coverage = layer_data.get('coverage', 0)
            if layer_coverage < 50:
                recommendations.append(f"🔴 {layer_name}层覆盖率严重不足，需要紧急补充测试")

        return recommendations

    def save_to_database(self, coverage_data: Dict[str, Any], alerts: List[Dict[str, Any]]):
        """保存数据到数据库"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # 保存覆盖率数据
            timestamp = datetime.now().isoformat()
            for layer_name, layer_data in coverage_data.get('files', {}).items():
                summary = layer_data.get('summary', {})
                # 处理missing_lines，可能是列表或整数
                missing_lines = summary.get('missing_lines', [])
                if isinstance(missing_lines, list):
                    missed_count = len(missing_lines)
                else:
                    missed_count = missing_lines if isinstance(missing_lines, int) else 0

                cursor.execute('''
                    INSERT INTO coverage_history
                    (timestamp, layer, coverage, statements, missed, branches, partial)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp,
                    layer_name.split('/')[1] if '/' in layer_name else 'unknown',
                    summary.get('percent_covered', 0),
                    summary.get('num_statements', 0),
                    missed_count,
                    summary.get('num_branches', 0),
                    summary.get('num_partial_branches', 0)
                ))

            # 保存告警数据
            for alert in alerts:
                cursor.execute('''
                    INSERT INTO alerts (timestamp, alert_type, message, severity, layer)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    timestamp,
                    alert['type'],
                    alert['message'],
                    alert['severity'],
                    alert.get('layer')
                ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"保存到数据库失败: {e}")

    def generate_reports(self, analysis: Dict[str, Any], alerts: List[Dict[str, Any]]):
        """生成报告"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 生成JSON报告
            report_data = {
                'timestamp': analysis['timestamp'],
                'analysis': analysis,
                'alerts': alerts,
                'system_info': self._get_system_info()
            }

            reports_dir = self.project_root / "reports" / "coverage_monitoring"
            reports_dir.mkdir(parents=True, exist_ok=True)

            json_file = reports_dir / f"coverage_report_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            logger.info(f"📊 报告已生成: {json_file}")

        except Exception as e:
            logger.error(f"生成报告失败: {e}")

    def send_notifications(self, alerts: List[Dict[str, Any]]):
        """发送通知"""
        if not alerts:
            return

        try:
            # 控制台通知
            print("\n" + "="*60)
            print("🚨 测试覆盖率告警")
            print("="*60)

            for alert in alerts:
                severity_icon = {
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(alert['severity'], '⚪')

                print(f"{severity_icon} {alert['message']}")

            print("="*60)

            # 文件日志
            logger.warning(f"检测到 {len(alerts)} 个覆盖率告警")

        except Exception as e:
            logger.error(f"发送通知失败: {e}")

    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'python_version': sys.version,
                'platform': sys.platform
            }
        except:
            return {}

    def get_coverage_trends(self, days: int = 7) -> Dict[str, Any]:
        """获取覆盖率趋势"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # 查询最近N天的覆盖率数据
            start_date = (datetime.now() - timedelta(days=days)).isoformat()

            cursor.execute('''
                SELECT timestamp, layer, coverage
                FROM coverage_history
                WHERE timestamp >= ?
                ORDER BY timestamp
            ''', (start_date,))

            rows = cursor.fetchall()
            conn.close()

            # 整理趋势数据
            trends = {}
            for row in rows:
                timestamp, layer, coverage = row
                if layer not in trends:
                    trends[layer] = []
                trends[layer].append({
                    'timestamp': timestamp,
                    'coverage': coverage
                })

            return trends

        except Exception as e:
            logger.error(f"获取覆盖率趋势失败: {e}")
            return {}

    def set_baseline(self, layer: str, coverage: float):
        """设置基准覆盖率"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            now = datetime.now().isoformat()
            cursor.execute('''
                INSERT OR REPLACE INTO baselines
                (layer, baseline_coverage, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            ''', (layer, coverage, now, now))

            conn.commit()
            conn.close()

            logger.info(f"✅ 已设置 {layer} 基准覆盖率: {coverage}%")

        except Exception as e:
            logger.error(f"设置基准失败: {e}")

    def get_baseline(self, layer: str) -> Optional[float]:
        """获取基准覆盖率"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute('SELECT baseline_coverage FROM baselines WHERE layer = ?', (layer,))
            row = cursor.fetchone()
            conn.close()

            return row[0] if row else None

        except Exception as e:
            logger.error(f"获取基准失败: {e}")
            return None


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='持续测试覆盖率监控系统')
    parser.add_argument('--project-root', default='.', help='项目根目录')
    parser.add_argument('--db-path', default='data/coverage_monitor.db', help='数据库路径')
    parser.add_argument('--command', choices=['start', 'stop', 'status', 'report', 'baseline'],
                        default='status', help='命令')
    parser.add_argument('--layer', help='层名称（用于baseline命令）')
    parser.add_argument('--coverage', type=float, help='覆盖率值（用于baseline命令）')
    parser.add_argument('--once', action='store_true', help='只执行一次监控周期')

    args = parser.parse_args()

    # 创建监控器
    monitor = ContinuousCoverageMonitor(args.project_root, args.db_path)

    if args.command == 'start':
        monitor.start_monitoring()
        print("✅ 持续监控已启动，按Ctrl+C停止...")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("\n🛑 监控已停止")

    elif args.command == 'stop':
        monitor.stop_monitoring()
        print("🛑 监控已停止")

    elif args.command == 'status':
        print("📊 监控状态:")
        print(f"  - 活动状态: {'是' if monitor.monitoring_active else '否'}")
        print(f"  - 最后报告: {monitor.last_report_time}")
        print(f"  - 数据库: {monitor.db_path.exists()}")

    elif args.command == 'report':
        if args.once:
            print("🔍 执行一次性监控...")
            monitor.run_monitoring_cycle()
        else:
            trends = monitor.get_coverage_trends()
            print("📈 覆盖率趋势:")
            for layer, data in trends.items():
                if data:
                    latest = data[-1]['coverage']
                    print(".2f")
    elif args.command == 'baseline':
        if args.layer and args.coverage is not None:
            monitor.set_baseline(args.layer, args.coverage)
        elif args.layer:
            baseline = monitor.get_baseline(args.layer)
            if baseline:
                print(".2f")
            else:
                print(f"❌ 未找到 {args.layer} 的基准数据")
        else:
            print("❌ 请指定层名称和覆盖率值")


if __name__ == "__main__":
    main()
