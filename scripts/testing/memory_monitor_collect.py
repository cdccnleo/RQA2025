#!/usr/bin/env python3
"""
内存监控收集脚本 - 定位pytest收集阶段内存暴涨的文件
"""

import sys
import time
import subprocess
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryMonitor:
    """内存监控器"""

    def __init__(self, threshold_mb: int = 500, timeout: int = 30):
        self.threshold_mb = threshold_mb
        self.timeout = timeout
        self.results = []

    def monitor_pytest_collect(self, test_file: Path) -> Tuple[str, float, int, str]:
        """
        监控单个测试文件的pytest收集过程

        Returns:
            Tuple[状态, 内存峰值MB, 返回码, 错误信息]
        """
        cmd = [
            sys.executable, '-m', 'pytest',
            '--collect-only', '-v',
            str(test_file),
            '--tb=no'  # 减少输出
        ]

        logger.info(f"开始监控: {test_file}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )

            p = psutil.Process(proc.pid)
            max_mem = 0
            start_time = time.time()

            while proc.poll() is None:
                try:
                    mem = p.memory_info().rss / 1024 / 1024
                    max_mem = max(max_mem, mem)

                    # 检查超时
                    if time.time() - start_time > self.timeout:
                        proc.kill()
                        return 'TIMEOUT', max_mem, -1, f"收集超时({self.timeout}s)"

                except psutil.NoSuchProcess:
                    break
                except Exception as e:
                    logger.warning(f"监控进程异常: {e}")

                time.sleep(0.5)

            # 等待进程结束
            try:
                stdout, stderr = proc.communicate(timeout=5)
                return_code = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                return 'TIMEOUT', max_mem, -1, "进程通信超时"

            error_msg = stderr.strip() if stderr else ""

            if return_code == 0:
                return 'SUCCESS', max_mem, return_code, error_msg
            else:
                return 'ERROR', max_mem, return_code, error_msg

        except Exception as e:
            return 'EXCEPTION', 0, -1, str(e)

    def scan_test_files(self, test_dir: Path) -> List[Path]:
        """扫描测试文件"""
        test_files = []
        for pattern in ['test_*.py', '*_test.py']:
            test_files.extend(test_dir.glob(pattern))
        return sorted(test_files)

    def run_memory_scan(self, test_dir: Path) -> Dict:
        """运行内存扫描"""
        logger.info(f"开始扫描目录: {test_dir}")

        test_files = self.scan_test_files(test_dir)
        logger.info(f"找到 {len(test_files)} 个测试文件")

        results = {
            'scan_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_dir': str(test_dir),
            'threshold_mb': self.threshold_mb,
            'timeout': self.timeout,
            'files': []
        }

        for test_file in test_files:
            logger.info(f"监控文件: {test_file.name}")

            # 清理内存
            gc.collect()

            status, mem_peak, return_code, error_msg = self.monitor_pytest_collect(test_file)

            result = {
                'file': test_file.name,
                'path': str(test_file),
                'status': status,
                'memory_peak_mb': round(mem_peak, 2),
                'return_code': return_code,
                'error_msg': error_msg,
                'is_problematic': (
                    status in ['TIMEOUT', 'ERROR', 'EXCEPTION'] or
                    mem_peak > self.threshold_mb
                )
            }

            results['files'].append(result)

            # 输出结果
            if result['is_problematic']:
                logger.warning(f"⚠️ {test_file.name} - 内存峰值: {mem_peak:.1f}MB, 状态: {status}")
                if error_msg:
                    logger.error(f"   错误: {error_msg}")
            else:
                logger.info(f"✅ {test_file.name} - 内存峰值: {mem_peak:.1f}MB, 状态: {status}")

        return results

    def generate_report(self, results: Dict) -> str:
        """生成报告"""
        problematic_files = [f for f in results['files'] if f['is_problematic']]
        total_files = len(results['files'])
        problem_count = len(problematic_files)

        report = f"""
=== 内存监控收集报告 ===
扫描时间: {results['scan_time']}
测试目录: {results['test_dir']}
总文件数: {total_files}
问题文件数: {problem_count}
内存阈值: {results['threshold_mb']}MB
超时设置: {results['timeout']}s

问题文件详情:
"""

        if problematic_files:
            for file_info in problematic_files:
                report += f"""
文件: {file_info['file']}
状态: {file_info['status']}
内存峰值: {file_info['memory_peak_mb']}MB
返回码: {file_info['return_code']}
错误信息: {file_info['error_msg']}
"""
        else:
            report += "✅ 未发现内存问题文件\n"

        return report


def main():
    """主函数"""
    # 设置测试目录
    project_root = Path(__file__).parent.parent.parent
    test_dirs = [
        project_root / 'tests' / 'unit' / 'infrastructure' / 'config',
        project_root / 'tests' / 'unit' / 'infrastructure' / 'logging',
        project_root / 'tests' / 'unit' / 'infrastructure' / 'utils',
        project_root / 'tests' / 'unit' / 'infrastructure'
    ]

    monitor = MemoryMonitor(threshold_mb=500, timeout=30)

    all_results = {}

    for test_dir in test_dirs:
        if test_dir.exists():
            logger.info(f"\n{'='*50}")
            logger.info(f"扫描目录: {test_dir}")
            logger.info(f"{'='*50}")

            results = monitor.run_memory_scan(test_dir)
            all_results[str(test_dir)] = results

            # 生成并保存报告
            report = monitor.generate_report(results)
            logger.info(report)

            # 保存详细结果到JSON文件
            report_file = project_root / 'reports' / 'testing' / \
                f'memory_scan_{test_dir.name}_{int(time.time())}.json'
            report_file.parent.mkdir(parents=True, exist_ok=True)

            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info(f"详细结果已保存到: {report_file}")

    # 生成总体报告
    total_problematic = sum(
        len([f for f in results['files'] if f['is_problematic']])
        for results in all_results.values()
    )

    logger.info(f"\n{'='*50}")
    logger.info(f"总体扫描完成，发现 {total_problematic} 个问题文件")
    logger.info(f"{'='*50}")


if __name__ == '__main__':
    main()
