#!/usr/bin/env python3
"""
修复命名规范问题的脚本

修复不符合Python命名规范的类名、函数名等
"""

import re
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class NamingConventionFixer:
    """命名规范修复器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.fixed_files = []

    def fix_file(self, file_path: Path) -> bool:
        """修复单个文件的命名规范问题"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 修复类名：将中文类名转换为英文类名
            content = self._fix_class_names(content, file_path)

            # 修复函数名：将中文函数名转换为英文函数名
            content = self._fix_function_names(content, file_path)

            # 修复变量名：将中文变量名转换为英文变量名
            content = self._fix_variable_names(content, file_path)

            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixed_files.append(str(file_path))
                return True

            return False

        except Exception as e:
            print(f"❌ 修复文件失败 {file_path}: {e}")
            return False

    def _fix_class_names(self, content: str, file_path: Path) -> str:
        """修复类名"""
        # 匹配中文类名模式
        patterns = [
            (r'class\s+Web服务Component(\d+):', 'WebServiceComponent'),
            (r'class\s+API网关Component(\d+):', 'ApiGatewayComponent'),
            (r'class\s+接口文件Component(\d+):', 'InterfaceComponent'),
            (r'class\s+策略决策Component(\d+):', 'StrategyDecisionComponent'),
            (r'class\s+风险检查Component(\d+):', 'RiskCheckComponent'),
            (r'class\s+交易执行Component(\d+):', 'TradingExecutionComponent'),
            (r'class\s+回测分析Component(\d+):', 'BacktestAnalysisComponent'),
            (r'class\s+引擎监控Component(\d+):', 'EngineMonitorComponent'),
        ]

        for pattern, replacement_prefix in patterns:
            def replace_class(match):
                number = match.group(1)
                return f"class {replacement_prefix}{number}:"

            content = re.sub(pattern, replace_class, content)

        return content

    def _fix_function_names(self, content: str, file_path: Path) -> str:
        """修复函数名"""
        # 匹配中文函数名
        patterns = [
            (r'def\s+获取组件信息\s*\(', 'def get_component_info('),
            (r'def\s+处理数据\s*\(', 'def process_data('),
            (r'def\s+初始化组件\s*\(', 'def initialize_component('),
            (r'def\s+执行操作\s*\(', 'def execute_operation('),
            (r'def\s+检查状态\s*\(', 'def check_status('),
            (r'def\s+获取配置\s*\(', 'def get_configuration('),
            (r'def\s+设置配置\s*\(', 'def set_configuration('),
            (r'def\s+启动服务\s*\(', 'def start_service('),
            (r'def\s+停止服务\s*\(', 'def stop_service('),
        ]

        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)

        return content

    def _fix_variable_names(self, content: str, file_path: Path) -> str:
        """修复变量名"""
        # 匹配中文变量名
        patterns = [
            (r'(\s+)组件名称(\s*)=', r'\1component_name\2='),
            (r'(\s+)创建时间(\s*)=', r'\1creation_time\2='),
            (r'(\s+)描述信息(\s*)=', r'\1description\2='),
            (r'(\s+)版本号(\s*)=', r'\1version\2='),
            (r'(\s+)配置信息(\s*)=', r'\1config_info\2='),
            (r'(\s+)状态信息(\s*)=', r'\1status_info\2='),
        ]

        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)

        return content

    def scan_and_fix_directory(self, directory: Path) -> Dict[str, Any]:
        """扫描并修复目录中的命名规范问题"""
        python_files = list(directory.rglob("*.py"))

        total_files = len(python_files)
        fixed_count = 0

        for file_path in python_files:
            if self.fix_file(file_path):
                fixed_count += 1

        return {
            "total_files": total_files,
            "fixed_files": fixed_count,
            "fixed_file_list": self.fixed_files
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成修复报告"""
        report = {
            "timestamp": str(datetime.now()),
            "summary": {
                "total_python_files": results["total_files"],
                "fixed_files": results["fixed_files"]
            },
            "fixed_files": results["fixed_file_list"],
            "details": "修复了类名、函数名和变量名的命名规范问题"
        }

        return json.dumps(report, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description='修复命名规范问题')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--directory', help='指定修复目录')
    parser.add_argument('--report', action='store_true', help='生成报告')

    args = parser.parse_args()

    fixer = NamingConventionFixer(args.project)

    if args.directory:
        target_dir = Path(args.directory)
    else:
        target_dir = fixer.src_dir

    print("🔧 开始修复命名规范问题...")
    results = fixer.scan_and_fix_directory(target_dir)

    print(f"✅ 修复完成: {results['fixed_files']}/{results['total_files']} 个文件已修复")

    if results['fixed_files'] > 0:
        print("\n修复的文件列表:")
        for file in results['fixed_file_list'][:10]:  # 显示前10个
            print(f"  - {file}")

        if len(results['fixed_file_list']) > 10:
            print(f"  ... 还有 {len(results['fixed_file_list']) - 10} 个文件")

    if args.report:
        report_content = fixer.generate_report(results)
        report_file = fixer.project_root / "reports" / \
            f"naming_convention_fix_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"📊 报告已保存: {report_file}")


if __name__ == "__main__":
    main()
