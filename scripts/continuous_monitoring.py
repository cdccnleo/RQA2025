#!/usr/bin/env python3
"""
RQA2025持续语法监控系统
定期检查和修复语法错误，确保代码质量
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContinuousSyntaxMonitor:
    """持续语法监控器"""

    def __init__(self, root_dir: str, report_dir: str = "reports/syntax_monitoring"):
        self.root_dir = Path(root_dir)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # 监控统计
        self.stats = {
            'total_scans': 0,
            'total_files': 0,
            'error_files': 0,
            'fixed_files': 0,
            'last_scan': None,
            'scan_history': []
        }

        # 监控配置
        self.monitoring_config = {
            'scan_interval': 300,  # 5分钟
            'auto_fix': True,
            'fix_attempts_limit': 3,
            'excluded_dirs': ['__pycache__', '.git', 'node_modules', '.vscode'],
            'included_extensions': ['.py']
        }

        self.load_stats()

    def load_stats(self):
        """加载统计数据"""
        stats_file = self.report_dir / "monitoring_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
            except Exception as e:
                logger.warning(f"加载统计数据失败: {e}")

    def save_stats(self):
        """保存统计数据"""
        stats_file = self.report_dir / "monitoring_stats.json"
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存统计数据失败: {e}")

    def scan_and_monitor(self) -> Dict[str, Any]:
        """执行扫描和监控"""
        logger.info("🔍 开始持续语法监控扫描...")

        start_time = datetime.now()

        # 执行扫描
        result = self._scan_python_files()

        # 生成报告
        report = self._generate_report(result, start_time)

        # 保存统计
        self._update_stats(result)
        self.save_stats()

        return report

    def _scan_python_files(self) -> Dict[str, Any]:
        """扫描Python文件"""
        result = {
            'total_files': 0,
            'error_files': [],
            'syntax_errors': [],
            'fixed_files': [],
            'scan_time': datetime.now().isoformat()
        }

        # 查找所有Python文件
        python_files = []
        for ext in self.monitoring_config['included_extensions']:
            for file_path in self.root_dir.rglob(f"*{ext}"):
                if not any(excluded in str(file_path) for excluded in self.monitoring_config['excluded_dirs']):
                    python_files.append(file_path)

        result['total_files'] = len(python_files)
        logger.info(f"发现 {len(python_files)} 个Python文件")

        # 检查每个文件
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 尝试编译检查语法
                compile(content, str(file_path), 'exec')

            except SyntaxError as e:
                error_info = {
                    'file': str(file_path),
                    'error_type': 'SyntaxError',
                    'message': str(e),
                    'line': e.lineno,
                    'offset': e.offset
                }
                result['error_files'].append(str(file_path))
                result['syntax_errors'].append(error_info)

                logger.warning(f"语法错误: {file_path} - {e}")

                # 自动修复
                if self.monitoring_config['auto_fix']:
                    fixed = self._auto_fix_file(file_path, e)
                    if fixed:
                        result['fixed_files'].append(str(file_path))

            except UnicodeDecodeError:
                logger.warning(f"编码错误，跳过文件: {file_path}")
            except Exception as e:
                logger.error(f"检查文件时出错: {file_path} - {e}")

        return result

    def _auto_fix_file(self, file_path: Path, error: SyntaxError) -> bool:
        """自动修复文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 应用修复策略
            content = self._apply_fixes(content, error)

            if content != original_content:
                # 验证修复结果
                try:
                    compile(content, str(file_path), 'exec')
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.info(f"✅ 自动修复成功: {file_path}")
                    return True
                except SyntaxError:
                    logger.warning(f"自动修复后仍有语法错误: {file_path}")
                    return False

        except Exception as e:
            logger.error(f"修复文件时出错: {file_path} - {e}")

        return False

    def _apply_fixes(self, content: str, error: SyntaxError) -> str:
        """应用修复策略"""
        error_msg = str(error)

        # 根据错误类型应用不同的修复策略
        if 'EOL while scanning string literal' in error_msg:
            content = self._fix_string_literals(content)
        elif 'unexpected indent' in error_msg or 'unindent' in error_msg:
            content = self._fix_indentation(content)
        elif 'invalid syntax' in error_msg and '{' in content:
            content = self._fix_dict_syntax(content)
        elif 'invalid character' in error_msg and '：' in error_msg:
            content = self._fix_chinese_colon(content)

        return content

    def _fix_string_literals(self, content: str) -> str:
        """修复字符串字面量错误"""
        import re
        content = re.sub(r'"""([^"]*?)$', r'"""\1"""', content, flags=re.MULTILINE)
        content = re.sub(r"'''([^']*?)$", r"'''\1'''", content, flags=re.MULTILINE)
        return content

    def _fix_indentation(self, content: str) -> str:
        """修复缩进错误"""
        lines = content.split('\n')
        fixed_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped and not line.startswith(' ') and not line.startswith('\t'):
                # 检查是否应该缩进
                if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except ')):
                    # 这些语句通常需要缩进
                    pass  # 保持原样
            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_dict_syntax(self, content: str) -> str:
        """修复字典语法错误"""
        import re
        content = re.sub(r'(\w+)\s*=\s*\{\s*\n\s*([^}]*?)\n\s*\}',
                         r'\1 = {\n        \2\n    }', content, flags=re.MULTILINE)
        return content

    def _fix_chinese_colon(self, content: str) -> str:
        """修复中文冒号错误"""
        return content.replace('：', ':')

    def _generate_report(self, result: Dict[str, Any], start_time: datetime) -> Dict[str, Any]:
        """生成报告"""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        report = {
            'scan_time': result['scan_time'],
            'duration_seconds': duration,
            'total_files': result['total_files'],
            'error_files_count': len(result['error_files']),
            'fixed_files_count': len(result['fixed_files']),
            'error_rate': len(result['error_files']) / max(result['total_files'], 1),
            'error_files': result['error_files'],
            'fixed_files': result['fixed_files'],
            'syntax_errors': result['syntax_errors'][:10],  # 只显示前10个错误
            'recommendations': self._generate_recommendations(result)
        }

        # 保存报告
        self._save_report(report)

        return report

    def _generate_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []

        error_rate = len(result['error_files']) / max(result['total_files'], 1)

        if error_rate > 0.1:
            recommendations.append("⚠️ 语法错误率较高，建议进行全面代码检查")
        elif error_rate > 0.05:
            recommendations.append("⚠️ 有语法错误，建议及时修复")
        else:
            recommendations.append("✅ 语法错误率正常")

        if len(result['fixed_files']) > 0:
            recommendations.append(f"✅ 自动修复了 {len(result['fixed_files'])} 个文件")

        if len(result['error_files']) > len(result['fixed_files']):
            recommendations.append("💡 建议手动检查未能自动修复的文件")

        return recommendations

    def _save_report(self, report: Dict[str, Any]):
        """保存报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.report_dir / f"syntax_report_{timestamp}.json"

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存报告失败: {e}")

    def _update_stats(self, result: Dict[str, Any]):
        """更新统计数据"""
        self.stats['total_scans'] += 1
        self.stats['total_files'] = result['total_files']
        self.stats['error_files'] = len(result['error_files'])
        self.stats['fixed_files'] += len(result['fixed_files'])
        self.stats['last_scan'] = result['scan_time']

        # 记录扫描历史
        self.stats['scan_history'].append({
            'timestamp': result['scan_time'],
            'total_files': result['total_files'],
            'error_files': len(result['error_files']),
            'fixed_files': len(result['fixed_files'])
        })

        # 只保留最近的100条记录
        self.stats['scan_history'] = self.stats['scan_history'][-100:]

    def print_summary(self, report: Dict[str, Any]):
        """打印摘要"""
        print("\n" + "="*60)
        print("📊 RQA2025持续语法监控报告")
        print("="*60)

        print(f"扫描时间: {report['scan_time']}")
        print(f"扫描耗时: {report['duration_seconds']:.2f}秒")
        print(f"总文件数: {report['total_files']}")
        print(f"错误文件数: {report['error_files_count']}")
        print(f"修复文件数: {report['fixed_files_count']}")
        print(f"错误率: {report['error_rate']:.2%}")

        print("\n🔧 建议:")
        for rec in report['recommendations']:
            print(f"  {rec}")

        if report['error_files']:
            print("\n❌ 发现语法错误的文件的文件:")
            for file in report['error_files'][:5]:
                print(f"  {file}")
            if len(report['error_files']) > 5:
                print(f"  ... 还有 {len(report['error_files']) - 5} 个文件")

        if report['fixed_files']:
            print("\n✅ 已自动修复的文件:")
            for file in report['fixed_files'][:5]:
                print(f"  {file}")
            if len(report['fixed_files']) > 5:
                print(f"  ... 还有 {len(report['fixed_files']) - 5} 个文件")

    def start_monitoring_loop(self):
        """开始监控循环"""
        logger.info("🚀 启动持续语法监控循环...")

        try:
            while True:
                logger.info("开始新一轮扫描...")
                report = self.scan_and_monitor()
                self.print_summary(report)

                logger.info(f"等待 {self.monitoring_config['scan_interval']} 秒后进行下次扫描...")
                time.sleep(self.monitoring_config['scan_interval'])

        except KeyboardInterrupt:
            logger.info("收到停止信号，正在退出...")
        except Exception as e:
            logger.error(f"监控循环异常: {e}")


def test_monitoring_system():
    """测试监控系统"""
    print("🧪 测试持续语法监控系统...")

    try:
        monitor = ContinuousSyntaxMonitor("src")

        # 执行一次扫描
        report = monitor.scan_and_monitor()
        monitor.print_summary(report)

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='RQA2025持续语法监控系统')
    parser.add_argument('--test', action='store_true', help='运行测试模式')
    parser.add_argument('--once', action='store_true', help='只运行一次扫描')
    parser.add_argument('--dir', default='src', help='监控目录')
    parser.add_argument('--interval', type=int, default=300, help='扫描间隔(秒)')

    args = parser.parse_args()

    if args.test:
        test_monitoring_system()
        return

    # 创建监控器
    monitor = ContinuousSyntaxMonitor(args.dir)

    if args.interval != 300:
        monitor.monitoring_config['scan_interval'] = args.interval

    if args.once:
        # 只运行一次
        report = monitor.scan_and_monitor()
        monitor.print_summary(report)
    else:
        # 开始持续监控
        monitor.start_monitoring_loop()


if __name__ == "__main__":
    main()
