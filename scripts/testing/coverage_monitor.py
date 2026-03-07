#!/usr/bin/env python3
"""
测试覆盖率监控脚本

自动化收集和报告测试覆盖率数据，支持：
1. 分层覆盖率收集
2. 覆盖率趋势分析
3. 覆盖率报告生成
4. 覆盖率目标监控
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import argparse

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 简化导入，避免依赖问题


class CoverageMonitor:
    """测试覆盖率监控器"""

    def __init__(self, config_path: str = "config/testing/coverage_config.json"):
        # 覆盖率目标配置
        self.coverage_targets = {
            'base_model.py': 100.0,
            'model_manager.py': 100.0,
            'model_evaluator.py': 85.0,
            'trainer.py': 85.0,
            'concrete_models.py': 85.0,
            'deployer.py': 85.0,
            'serving.py': 80.0,
            'version_manager.py': 80.0,
            'inference/*': 70.0,
            'overall': 80.0
        }

        self.config = self._load_config(config_path)
        self.reports_dir = Path("reports/coverage")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'test_directories': [
                'tests/unit/models',
                'tests/unit/features',
                'tests/unit/data',
                'tests/unit/trading',
                'tests/unit/risk'
            ],
            'source_directories': [
                'src/models',
                'src/features',
                'src/data',
                'src/trading',
                'src/risk'
            ],
            'coverage_targets': self.coverage_targets,
            'report_formats': ['term', 'html', 'json'],
            'output_directory': 'reports/coverage'
        }

    def collect_coverage(self, source_dir: str, test_dir: str,
                         output_format: str = 'term') -> Dict[str, Any]:
        """收集指定目录的测试覆盖率"""
        try:
            print(f"收集覆盖率: {source_dir} (测试: {test_dir})")

            # 构建pytest命令
            cmd = [
                'python', '-m', 'pytest',
                test_dir,
                '--cov=' + source_dir,
                '--cov-report=' + output_format
            ]

            if output_format == 'html':
                html_dir = f"htmlcov/{Path(source_dir).name}"
                cmd.extend(['--cov-report', f'html:{html_dir}'])

            # 执行测试
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=project_root
            )

            if result.returncode == 0:
                print(f"✅ 覆盖率收集成功: {source_dir}")
                return self._parse_coverage_output(result.stdout, source_dir)
            else:
                print(f"❌ 覆盖率收集失败: {source_dir}")
                print(f"错误: {result.stderr}")
                return {'error': result.stderr}

        except Exception as e:
            print(f"❌ 覆盖率收集异常: {source_dir} - {e}")
            return {'error': str(e)}

    def _parse_coverage_output(self, output: str, source_dir: str) -> Dict[str, Any]:
        """解析覆盖率输出"""
        try:
            lines = output.split('\n')
            coverage_data = {
                'source_dir': source_dir,
                'timestamp': datetime.now().isoformat(),
                'modules': {},
                'total_coverage': 0.0
            }

            # 查找覆盖率行
            for line in lines:
                if 'TOTAL' in line and '%' in line:
                    # 解析总覆盖率
                    parts = line.split()
                    for part in parts:
                        if '%' in part:
                            coverage_data['total_coverage'] = float(part.replace('%', ''))
                            break
                elif source_dir in line and '%' in line:
                    # 解析模块覆盖率
                    parts = line.split()
                    if len(parts) >= 4:
                        module_name = parts[0].split('/')[-1]
                        coverage_str = parts[3]
                        if '%' in coverage_str:
                            coverage = float(coverage_str.replace('%', ''))
                            coverage_data['modules'][module_name] = coverage

            return coverage_data

        except Exception as e:
            print(f"解析覆盖率输出失败: {e}")
            return {'error': f'解析失败: {e}'}

    def collect_all_coverage(self) -> Dict[str, Any]:
        """收集所有目录的测试覆盖率"""
        all_coverage = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'details': {},
            'targets_achieved': [],
            'targets_missing': []
        }

        total_coverage = 0.0
        module_count = 0

        for source_dir in self.config['source_directories']:
            # 找到对应的测试目录
            test_dir = self._find_test_directory(source_dir)
            if test_dir:
                coverage_data = self.collect_coverage(source_dir, test_dir)
                if 'error' not in coverage_data:
                    all_coverage['details'][source_dir] = coverage_data

                    # 累计总覆盖率
                    if coverage_data['total_coverage'] > 0:
                        total_coverage += coverage_data['total_coverage']
                        module_count += 1

                    # 检查目标达成情况
                    self._check_coverage_targets(coverage_data, all_coverage)

        # 计算总体覆盖率
        if module_count > 0:
            all_coverage['summary']['overall_coverage'] = total_coverage / module_count
        else:
            all_coverage['summary']['overall_coverage'] = 0.0

        all_coverage['summary']['modules_tested'] = module_count
        all_coverage['summary']['total_modules'] = len(self.config['source_directories'])

        return all_coverage

    def _find_test_directory(self, source_dir: str) -> Optional[str]:
        """查找对应的测试目录"""
        source_name = Path(source_dir).name
        for test_dir in self.config['test_directories']:
            if source_name in test_dir or source_name.replace('src/', '') in test_dir:
                return test_dir
        return None

    def _check_coverage_targets(self, coverage_data: Dict[str, Any],
                                all_coverage: Dict[str, Any]):
        """检查覆盖率目标达成情况"""
        source_dir = coverage_data['source_dir']
        source_name = Path(source_dir).name

        for module, target in self.coverage_targets.items():
            if module in source_name or module == 'overall':
                if source_name in coverage_data['modules']:
                    actual_coverage = coverage_data['modules'][source_name]
                    if actual_coverage >= target:
                        all_coverage['targets_achieved'].append({
                            'module': source_name,
                            'target': target,
                            'actual': actual_coverage
                        })
                    else:
                        all_coverage['targets_missing'].append({
                            'module': source_name,
                            'target': target,
                            'actual': actual_coverage,
                            'gap': target - actual_coverage
                        })

    def generate_report(self, coverage_data: Dict[str, Any]) -> str:
        """生成覆盖率报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_file = self.reports_dir / f"coverage_report_{timestamp}.json"

        # 保存详细数据
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(coverage_data, f, indent=2, ensure_ascii=False)

        # 生成摘要报告
        summary_file = self.reports_dir / f"coverage_summary_{timestamp}.md"
        self._generate_markdown_summary(coverage_data, summary_file)

        return str(report_file)

    def _generate_markdown_summary(self, coverage_data: Dict[str, Any],
                                   output_file: Path):
        """生成Markdown格式的摘要报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 测试覆盖率报告\n\n")
            f.write(f"**生成时间**: {coverage_data['timestamp']}\n\n")

            # 总体摘要
            f.write("## 📊 总体摘要\n\n")
            f.write(f"- **总体覆盖率**: {coverage_data['summary']['overall_coverage']:.2f}%\n")
            f.write(
                f"- **已测试模块**: {coverage_data['summary']['modules_tested']}/{coverage_data['summary']['total_modules']}\n")
            f.write(f"- **目标达成**: {len(coverage_data['targets_achieved'])} 个\n")
            f.write(f"- **目标未达成**: {len(coverage_data['targets_missing'])} 个\n\n")

            # 目标达成情况
            if coverage_data['targets_achieved']:
                f.write("## ✅ 目标达成模块\n\n")
                f.write("| 模块 | 目标 | 实际 | 状态 |\n")
                f.write("|------|------|------|------|\n")
                for target in coverage_data['targets_achieved']:
                    f.write(
                        f"| {target['module']} | {target['target']}% | {target['actual']:.2f}% | ✅ 达成 |\n")
                f.write("\n")

            # 目标未达成情况
            if coverage_data['targets_missing']:
                f.write("## ⚠️ 目标未达成模块\n\n")
                f.write("| 模块 | 目标 | 实际 | 差距 | 状态 |\n")
                f.write("|------|------|------|------|------|\n")
                for target in coverage_data['targets_missing']:
                    f.write(
                        f"| {target['module']} | {target['target']}% | {target['actual']:.2f}% | {target['gap']:.2f}% | ⚠️ 未达成 |\n")
                f.write("\n")

            # 详细覆盖率
            f.write("## 📈 详细覆盖率\n\n")
            for source_dir, details in coverage_data['details'].items():
                if 'error' not in details:
                    f.write(f"### {source_dir}\n\n")
                    f.write(f"- **总体覆盖率**: {details['total_coverage']:.2f}%\n")
                    f.write(f"- **模块数量**: {len(details['modules'])}\n\n")

                    if details['modules']:
                        f.write("| 模块 | 覆盖率 |\n")
                        f.write("|------|--------|\n")
                        for module, coverage in details['modules'].items():
                            f.write(f"| {module} | {coverage:.2f}% |\n")
                        f.write("\n")

    def run_monitoring(self) -> Dict[str, Any]:
        """运行覆盖率监控"""
        print("🚀 开始测试覆盖率监控...")
        print(f"📁 项目根目录: {project_root}")
        print(f"📊 覆盖率目标: {self.coverage_targets}")

        # 收集覆盖率数据
        coverage_data = self.collect_all_coverage()

        # 生成报告
        report_file = self.generate_report(coverage_data)

        # 输出摘要
        print("\n📊 覆盖率监控完成!")
        print(f"📈 总体覆盖率: {coverage_data['summary']['overall_coverage']:.2f}%")
        print(f"✅ 目标达成: {len(coverage_data['targets_achieved'])} 个")
        print(f"⚠️ 目标未达成: {len(coverage_data['targets_missing'])} 个")
        print(f"📄 详细报告: {report_file}")

        return coverage_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试覆盖率监控工具')
    parser.add_argument('--config', default='config/testing/coverage_config.json',
                        help='配置文件路径')
    parser.add_argument('--source-dir', help='指定源目录')
    parser.add_argument('--test-dir', help='指定测试目录')
    parser.add_argument('--output-format', default='term',
                        choices=['term', 'html', 'json'],
                        help='输出格式')

    args = parser.parse_args()

    monitor = CoverageMonitor(args.config)

    if args.source_dir and args.test_dir:
        # 收集指定目录的覆盖率
        coverage_data = monitor.collect_coverage(
            args.source_dir,
            args.test_dir,
            args.output_format
        )
        print(json.dumps(coverage_data, indent=2, ensure_ascii=False))
    else:
        # 运行完整监控
        monitor.run_monitoring()


if __name__ == '__main__':
    main()
