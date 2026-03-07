#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志记录器集成脚本

将引擎层的统一日志记录器集成到现有组件中，替换现有的日志记录代码。
"""

from src.engine.logging.unified_logger import configure_engine_logging
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class UnifiedLoggerIntegrator:
    """统一日志记录器集成器"""

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]
        self.src_dir = self.project_root / "src"
        self.backup_dir = self.project_root / "backup" / "logging_integration"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # 需要集成的组件列表
        self.target_components = [
            "src/trading",
            "src/data",
            "src/features",
            "src/infrastructure",
            "src/engine"
        ]

        # 日志配置
        self.logging_config = {
            'level': 'INFO',
            'console_output': True,
            'log_file': 'logs/unified_engine.log',
            'propagate': False
        }

    def backup_file(self, file_path: Path) -> Path:
        """备份文件"""
        relative_path = file_path.relative_to(self.project_root)
        backup_path = self.backup_dir / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists():
            import shutil
            shutil.copy2(file_path, backup_path)
            print(f"✅ 已备份: {file_path} -> {backup_path}")

        return backup_path

    def find_python_files(self, directory: Path) -> List[Path]:
        """查找Python文件"""
        python_files = []
        if directory.exists():
            for file_path in directory.rglob("*.py"):
                if not file_path.name.startswith("__"):
                    python_files.append(file_path)
        return python_files

    def analyze_logging_usage(self, file_path: Path) -> Dict[str, Any]:
        """分析文件中的日志使用情况"""
        if not file_path.exists():
            return {}

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        analysis = {
            'file_path': file_path,
            'has_logging_import': False,
            'has_logger_definition': False,
            'logging_calls': [],
            'logger_name': None,
            'import_lines': [],
            'logger_lines': []
        }

        lines = content.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()

            # 检查logging导入
            if 'import logging' in line:
                analysis['has_logging_import'] = True
                analysis['import_lines'].append((i, line))

            # 检查logger定义
            if 'logger = logging.getLogger' in line:
                analysis['has_logger_definition'] = True
                analysis['logger_lines'].append((i, line))

                # 提取logger名称
                match = re.search(r'logging\.getLogger\(([^)]+)\)', line)
                if match:
                    logger_name = match.group(1).strip('"\'')
                    analysis['logger_name'] = logger_name

            # 检查日志调用
            if any(method in line for method in ['logger.debug', 'logger.info', 'logger.warning', 'logger.error', 'logger.critical']):
                analysis['logging_calls'].append((i, line))

        return analysis

    def generate_unified_logger_code(self, analysis: Dict[str, Any]) -> Tuple[str, List[str]]:
        """生成统一日志记录器代码"""
        if not analysis['has_logging_import'] and not analysis['has_logger_definition']:
            return "", []

        file_path = analysis['file_path']
        module_name = file_path.stem
        logger_name = analysis['logger_name'] or f"src.{file_path.relative_to(self.src_dir).parent}.{module_name}"

        # 生成导入语句
        import_code = "from src.engine.logging.unified_logger import get_unified_logger\n"

        # 生成logger定义
        logger_code = f"logger = get_unified_logger('{logger_name}')\n"

        # 生成替换建议
        replacements = []
        for line_num, line in analysis['logging_calls']:
            # 简单的日志调用替换
            if 'logger.debug(' in line:
                replacements.append(f"第{line_num + 1}行: {line.strip()} -> logger.debug(...)")
            elif 'logger.info(' in line:
                replacements.append(f"第{line_num + 1}行: {line.strip()} -> logger.info(...)")
            elif 'logger.warning(' in line:
                replacements.append(f"第{line_num + 1}行: {line.strip()} -> logger.warning(...)")
            elif 'logger.error(' in line:
                replacements.append(f"第{line_num + 1}行: {line.strip()} -> logger.error(...)")
            elif 'logger.critical(' in line:
                replacements.append(f"第{line_num + 1}行: {line.strip()} -> logger.critical(...)")

        return import_code + logger_code, replacements

    def integrate_file(self, file_path: Path) -> Dict[str, Any]:
        """集成单个文件"""
        print(f"\n🔍 分析文件: {file_path}")

        # 备份文件
        backup_path = self.backup_file(file_path)

        # 分析日志使用情况
        analysis = self.analyze_logging_usage(file_path)

        if not analysis['has_logging_import'] and not analysis['has_logger_definition']:
            print(f"⏭️  跳过: 未发现日志使用")
            return {'status': 'skipped', 'reason': 'no_logging_usage'}

        # 生成统一日志记录器代码
        unified_code, replacements = self.generate_unified_logger_code(analysis)

        if not unified_code:
            print(f"⏭️  跳过: 无法生成统一日志代码")
            return {'status': 'skipped', 'reason': 'cannot_generate_code'}

        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 替换导入语句
        lines = content.split('\n')
        new_lines = []
        import_added = False
        logger_added = False

        for i, line in enumerate(lines):
            # 跳过原有的logging导入
            if 'import logging' in line:
                continue

            # 跳过原有的logger定义
            if 'logger = logging.getLogger' in line:
                continue

            # 在import语句后添加统一日志记录器导入
            if not import_added and (line.startswith('import ') or line.startswith('from ')):
                new_lines.append(line)
                if not any('unified_logger' in l for l in new_lines):
                    new_lines.append(
                        "from src.engine.logging.unified_logger import get_unified_logger")
                    import_added = True
            elif not import_added and line.strip() == '':
                # 在空行后添加导入
                new_lines.append(line)
                new_lines.append("from src.engine.logging.unified_logger import get_unified_logger")
                import_added = True
            else:
                new_lines.append(line)

        # 在类定义前添加logger定义
        final_lines = []
        logger_added = False

        for line in new_lines:
            # 在第一个类定义前添加logger
            if not logger_added and (line.startswith('class ') or line.startswith('def ')):
                if analysis['logger_name']:
                    logger_line = f"logger = get_unified_logger('{analysis['logger_name']}')"
                else:
                    module_name = file_path.stem
                    logger_line = f"logger = get_unified_logger('{module_name}')"
                final_lines.append(logger_line)
                final_lines.append("")
                logger_added = True
            final_lines.append(line)

        # 如果没有找到合适的位置，在文件开头添加
        if not logger_added:
            if analysis['logger_name']:
                logger_line = f"logger = get_unified_logger('{analysis['logger_name']}')"
            else:
                module_name = file_path.stem
                logger_line = f"logger = get_unified_logger('{module_name}')"
            final_lines.insert(0, logger_line)
            final_lines.insert(1, "")

        # 写入新内容
        new_content = '\n'.join(final_lines)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ 已集成: {file_path}")
        print(f"   替换建议: {len(replacements)} 个日志调用")

        return {
            'status': 'integrated',
            'backup_path': backup_path,
            'replacements': replacements,
            'logger_name': analysis['logger_name']
        }

    def integrate_component(self, component_path: str) -> Dict[str, Any]:
        """集成整个组件"""
        component_dir = self.project_root / component_path
        print(f"\n🚀 开始集成组件: {component_path}")

        if not component_dir.exists():
            print(f"❌ 组件目录不存在: {component_dir}")
            return {'status': 'error', 'reason': 'directory_not_found'}

        python_files = self.find_python_files(component_dir)
        print(f"📁 找到 {len(python_files)} 个Python文件")

        results = {
            'total_files': len(python_files),
            'integrated_files': 0,
            'skipped_files': 0,
            'error_files': 0,
            'details': []
        }

        for file_path in python_files:
            try:
                result = self.integrate_file(file_path)
                results['details'].append({
                    'file': str(file_path),
                    'result': result
                })

                if result['status'] == 'integrated':
                    results['integrated_files'] += 1
                elif result['status'] == 'skipped':
                    results['skipped_files'] += 1
                else:
                    results['error_files'] += 1

            except Exception as e:
                print(f"❌ 集成失败: {file_path} - {e}")
                results['error_files'] += 1
                results['details'].append({
                    'file': str(file_path),
                    'result': {'status': 'error', 'reason': str(e)}
                })

        print(f"\n📊 集成结果:")
        print(f"   总文件数: {results['total_files']}")
        print(f"   已集成: {results['integrated_files']}")
        print(f"   已跳过: {results['skipped_files']}")
        print(f"   错误: {results['error_files']}")

        return results

    def run_integration(self) -> Dict[str, Any]:
        """运行集成流程"""
        print("🎯 开始统一日志记录器集成")
        print("=" * 50)

        # 配置引擎日志
        configure_engine_logging(self.logging_config)

        overall_results = {
            'components': {},
            'total_files': 0,
            'total_integrated': 0,
            'total_skipped': 0,
            'total_errors': 0
        }

        for component_path in self.target_components:
            results = self.integrate_component(component_path)
            overall_results['components'][component_path] = results

            overall_results['total_files'] += results['total_files']
            overall_results['total_integrated'] += results['integrated_files']
            overall_results['total_skipped'] += results['skipped_files']
            overall_results['total_errors'] += results['error_files']

        print("\n" + "=" * 50)
        print("📈 总体集成结果:")
        print(f"   总文件数: {overall_results['total_files']}")
        print(f"   已集成: {overall_results['total_integrated']}")
        print(f"   已跳过: {overall_results['total_skipped']}")
        print(f"   错误: {overall_results['total_errors']}")

        return overall_results

    def create_integration_report(self, results: Dict[str, Any]) -> None:
        """创建集成报告"""
        import time

        report_path = self.project_root / "reports" / "project" / "unified_logger_integration_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 统一日志记录器集成报告\n\n")
            f.write(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 总体结果\n\n")
            f.write(f"- 总文件数: {results['total_files']}\n")
            f.write(f"- 已集成: {results['total_integrated']}\n")
            f.write(f"- 已跳过: {results['total_skipped']}\n")
            f.write(f"- 错误: {results['total_errors']}\n\n")

            f.write("## 组件详细结果\n\n")
            for component, result in results['components'].items():
                f.write(f"### {component}\n\n")
                f.write(f"- 总文件数: {result['total_files']}\n")
                f.write(f"- 已集成: {result['integrated_files']}\n")
                f.write(f"- 已跳过: {result['skipped_files']}\n")
                f.write(f"- 错误: {result['error_files']}\n\n")

                if result['details']:
                    f.write("#### 详细结果\n\n")
                    for detail in result['details']:
                        f.write(f"- {detail['file']}: {detail['result']['status']}\n")
                    f.write("\n")

        print(f"📄 集成报告已生成: {report_path}")


def main():
    """主函数"""

    integrator = UnifiedLoggerIntegrator()

    try:
        results = integrator.run_integration()
        integrator.create_integration_report(results)

        print("\n✅ 统一日志记录器集成完成!")

    except Exception as e:
        print(f"\n❌ 集成过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
