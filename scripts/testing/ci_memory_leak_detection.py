#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CI/CD内存泄漏检测脚本

专门用于持续集成流程的内存泄漏检测工具：
1. 自动化内存泄漏检测
2. 生成详细报告
3. 设置阈值和告警
4. 集成到CI/CD流程
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class MemoryLeakReport:
    """内存泄漏报告"""
    timestamp: str
    total_leaks: int
    total_memory_mb: float
    memory_growth_mb: float
    initial_memory_mb: float
    final_memory_mb: float
    leak_details: List[Dict[str, Any]]
    test_results: Dict[str, Any]
    status: str  # 'pass', 'warning', 'fail'


class CIMemoryLeakDetector:
    """CI/CD内存泄漏检测器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.reports_dir = project_root / "reports" / "ci_memory_leak"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'memory_threshold_mb': 50,  # 内存增长阈值
            'leak_count_threshold': 5,   # 泄漏数量阈值
            'test_timeout_seconds': 300,  # 测试超时时间
            'enable_auto_fix': True,     # 启用自动修复
            'generate_report': True,      # 生成报告
            'fail_on_leak': False,       # 发现泄漏时是否失败
        }

    def run_memory_leak_detection(self) -> MemoryLeakReport:
        """运行内存泄漏检测"""
        print("🔍 开始CI/CD内存泄漏检测...")

        # 记录初始内存
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # 运行内存清理
        self._run_memory_cleanup()

        # 运行内存泄漏检测
        leaks = self._run_leak_detection()

        # 运行测试
        test_results = self._run_tests()

        # 记录最终内存
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory

        # 生成报告
        report = self._generate_report(
            initial_memory, final_memory, memory_growth, leaks, test_results
        )

        # 保存报告
        self._save_report(report)

        # 输出结果
        self._print_summary(report)

        return report

    def _run_memory_cleanup(self):
        """运行内存清理"""
        print("🧹 运行内存清理...")
        try:
            from scripts.testing.aggressive_memory_fix import AggressiveMemoryFixer
            fixer = AggressiveMemoryFixer()
            fixer.run_aggressive_fix()
            print("✅ 内存清理完成")
        except Exception as e:
            print(f"⚠️  内存清理失败: {e}")

    def _run_leak_detection(self) -> List[Dict[str, Any]]:
        """运行泄漏检测"""
        print("🔍 运行泄漏检测...")
        try:
            from scripts.testing.comprehensive_memory_leak_detector import ComprehensiveMemoryLeakDetector
            detector = ComprehensiveMemoryLeakDetector()
            detector.run_comprehensive_detection()
            return detector.detected_leaks
        except Exception as e:
            print(f"❌ 泄漏检测失败: {e}")
            return []

    def _run_tests(self) -> Dict[str, Any]:
        """运行测试"""
        print("🧪 运行基础设施测试...")

        test_cmd = [
            sys.executable, "scripts/testing/run_infrastructure_tests.py", "core"
        ]

        env = os.environ.copy()
        env.update({
            'CI_MODE': 'true',
            'DISABLE_HEAVY_IMPORTS': 'true',
            'ENABLE_MEMORY_OPTIMIZATION': 'true',
            'PROMETHEUS_ISOLATED': 'true'
        })

        try:
            result = subprocess.run(
                test_cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config['test_timeout_seconds']
            )

            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'duration': time.time()
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'timeout',
                'duration': self.config['test_timeout_seconds']
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'duration': 0
            }

    def _generate_report(self, initial_memory: float, final_memory: float,
                         memory_growth: float, leaks: List[Dict[str, Any]],
                         test_results: Dict[str, Any]) -> MemoryLeakReport:
        """生成报告"""

        # 计算总泄漏内存
        total_memory = sum(leak.get('memory_mb', 0) for leak in leaks) if leaks else 0

        # 确定状态
        status = 'pass'
        if memory_growth > self.config.get('memory_threshold_mb', 50):
            status = 'fail'
        elif len(leaks) > self.config.get('leak_count_threshold', 5):
            status = 'warning'
        elif total_memory > self.config.get('memory_threshold_mb', 50):
            status = 'warning'

        # 转换泄漏详情
        leak_details = []
        for leak in leaks:
            leak_details.append({
                'type': leak.leak_type,
                'module': leak.module_path,
                'class': leak.class_name,
                'count': leak.instance_count,
                'memory_mb': leak.memory_size,
                'description': leak.description
            })

        return MemoryLeakReport(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            total_leaks=len(leaks),
            total_memory_mb=total_memory,
            memory_growth_mb=memory_growth,
            initial_memory_mb=initial_memory,
            final_memory_mb=final_memory,
            leak_details=leak_details,
            test_results=test_results,
            status=status
        )

    def _save_report(self, report: MemoryLeakReport):
        """保存报告"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_file = self.reports_dir / f"memory_leak_report_{timestamp}.json"

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)

        print(f"📄 报告已保存: {report_file}")

    def _print_summary(self, report: MemoryLeakReport):
        """打印总结"""
        print("\n📊 CI/CD内存泄漏检测总结")
        print("=" * 60)

        print(f"状态: {report.status.upper()}")
        print(f"时间: {report.timestamp}")
        print(f"内存增长: {report.memory_growth_mb:.2f} MB")
        print(f"泄漏数量: {report.total_leaks}")
        print(f"泄漏内存: {report.total_memory_mb:.2f} MB")
        print(f"测试结果: {'✅ 成功' if report.test_results.get('success') else '❌ 失败'}")

        if report.leak_details:
            print("\n🔍 泄漏详情:")
            for leak in report.leak_details:
                print(f"  - {leak['type']}: {leak['memory_mb']:.2f} MB ({leak['description']})")

        # 根据状态设置退出码
        if report.status == 'fail' and self.config['fail_on_leak']:
            sys.exit(1)
        elif report.status == 'warning':
            print("⚠️  检测到警告级别的内存泄漏")
        else:
            print("✅ 内存泄漏检测通过")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="CI/CD内存泄漏检测")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--threshold", type=float, default=50, help="内存阈值(MB)")
    parser.add_argument("--fail-on-leak", action="store_true", help="发现泄漏时失败")
    parser.add_argument("--no-auto-fix", action="store_true", help="禁用自动修复")

    args = parser.parse_args()

    # 加载配置
    config = {
        'memory_threshold_mb': args.threshold,
        'fail_on_leak': args.fail_on_leak,
        'enable_auto_fix': not args.no_auto_fix,
    }

    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config.update(json.load(f))
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")

    # 运行检测
    detector = CIMemoryLeakDetector(config)
    report = detector.run_memory_leak_detection()

    # 输出结果
    print(f"\n🎯 检测完成，状态: {report.status.upper()}")


if __name__ == "__main__":
    main()
