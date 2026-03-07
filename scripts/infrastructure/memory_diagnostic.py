#!/usr/bin/env python3
"""
内存问题诊断和修复脚本
识别和解决基础设施层的内存暴涨问题
"""

import os
import sys
import gc
import psutil
import tracemalloc
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class MemoryDiagnostic:
    """内存问题诊断器"""

    def __init__(self):
        self.project_root = Path(project_root)
        self.report_dir = self.project_root / 'reports' / 'infrastructure'
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # 禁用复杂日志以避免递归
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.diagnostic_results = {
            'timestamp': datetime.now().isoformat(),
            'memory_usage': {},
            'memory_leaks': [],
            'performance_issues': [],
            'fixes_applied': [],
            'recommendations': []
        }

    def analyze_memory_usage(self) -> Dict[str, Any]:
        """分析内存使用情况"""
        print("分析内存使用情况...")

        # 获取系统内存信息
        memory_info = psutil.virtual_memory()
        process = psutil.Process()

        memory_data = {
            'system_total': memory_info.total,
            'system_available': memory_info.available,
            'system_percent': memory_info.percent,
            'process_rss': process.memory_info().rss,
            'process_vms': process.memory_info().vms,
            'process_percent': process.memory_percent(),
            'gc_stats': gc.get_stats(),
            'gc_count': gc.get_count()
        }

        print(f"系统内存使用: {memory_data['system_percent']:.1f}%")
        print(f"进程内存使用: {memory_data['process_rss'] / 1024 / 1024:.1f} MB")

        return memory_data

    def detect_memory_leaks(self) -> List[Dict[str, Any]]:
        """检测内存泄漏"""
        print("检测内存泄漏...")

        leaks = []

        # 启动内存跟踪
        tracemalloc.start()

        try:
            # 分析基础设施模块
            infrastructure_modules = [
                'src.infrastructure.logging',
                'src.infrastructure.config',
                'src.infrastructure.monitoring',
                'src.infrastructure.database',
                'src.infrastructure.cache',
                'src.infrastructure.utils'
            ]

            for module_name in infrastructure_modules:
                try:
                    # 记录导入前的内存
                    snapshot1 = tracemalloc.take_snapshot()

                    # 尝试导入模块
                    __import__(module_name)

                    # 记录导入后的内存
                    snapshot2 = tracemalloc.take_snapshot()

                    # 比较快照
                    top_stats = snapshot2.compare_to(snapshot1, 'lineno')

                    if top_stats:
                        for stat in top_stats[:5]:  # 只检查前5个最大的变化
                            if stat.size_diff > 1024 * 1024:  # 大于1MB的变化
                                leaks.append({
                                    'module': module_name,
                                    'file': stat.traceback.format()[-1],
                                    'size_diff': stat.size_diff,
                                    'size_diff_mb': stat.size_diff / 1024 / 1024
                                })

                except Exception as e:
                    leaks.append({
                        'module': module_name,
                        'error': str(e),
                        'type': 'import_error'
                    })

        finally:
            tracemalloc.stop()

        return leaks

    def analyze_circular_references(self) -> List[Dict[str, Any]]:
        """分析循环引用"""
        print("分析循环引用...")

        circular_refs = []

        # 检查常见的循环引用模式
        problematic_patterns = [
            'infrastructure_logger',
            'unified_logging_interface',
            'enhanced_log_manager',
            'config_manager',
            'monitoring_service'
        ]

        for pattern in problematic_patterns:
            try:
                # 查找包含该模式的模块
                for root, dirs, files in os.walk(self.project_root / 'src' / 'infrastructure'):
                    for file in files:
                        if file.endswith('.py') and pattern in file:
                            file_path = Path(root) / file
                            circular_refs.extend(self._check_file_circular_refs(file_path))
            except Exception as e:
                circular_refs.append({
                    'pattern': pattern,
                    'error': str(e)
                })

        return circular_refs

    def _check_file_circular_refs(self, file_path: Path) -> List[Dict[str, Any]]:
        """检查单个文件的循环引用"""
        refs = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查常见的循环引用模式
            import_patterns = [
                r'from\s+\.\.logging\.infrastructure_logger\s+import',
                r'from\s+\.\.config\.config_manager\s+import',
                r'from\s+\.\.monitoring\.monitoring_service\s+import',
                r'import\s+infrastructure_logger',
                r'import\s+config_manager',
                r'import\s+monitoring_service'
            ]

            for pattern in import_patterns:
                if re.search(pattern, content):
                    refs.append({
                        'file': str(file_path),
                        'pattern': pattern,
                        'line': content.count(pattern)
                    })

        except Exception as e:
            refs.append({
                'file': str(file_path),
                'error': str(e)
            })

        return refs

    def optimize_memory_usage(self) -> List[Dict[str, Any]]:
        """优化内存使用"""
        print("优化内存使用...")

        optimizations = []

        # 1. 强制垃圾回收
        before_gc = gc.get_count()
        collected = gc.collect()
        after_gc = gc.get_count()

        optimizations.append({
            'type': 'garbage_collection',
            'before': before_gc,
            'after': after_gc,
            'collected': collected
        })

        # 2. 优化导入
        import_optimizations = self._optimize_imports()
        optimizations.extend(import_optimizations)

        # 3. 优化数据结构
        data_structure_optimizations = self._optimize_data_structures()
        optimizations.extend(data_structure_optimizations)

        return optimizations

    def _optimize_imports(self) -> List[Dict[str, Any]]:
        """优化导入语句"""
        optimizations = []

        # 检查延迟导入
        lazy_import_files = [
            'src/infrastructure/logging/infrastructure_logger.py',
            'src/infrastructure/config/core/config_manager.py',
            'src/infrastructure/monitoring/monitoring_service.py'
        ]

        for file_path in lazy_import_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查是否有延迟导入
                    if 'import' in content and 'def' in content:
                        # 简单的延迟导入检查
                        optimizations.append({
                            'type': 'lazy_import_check',
                            'file': file_path,
                            'status': 'checked'
                        })

                except Exception as e:
                    optimizations.append({
                        'type': 'lazy_import_check',
                        'file': file_path,
                        'error': str(e)
                    })

        return optimizations

    def _optimize_data_structures(self) -> List[Dict[str, Any]]:
        """优化数据结构"""
        optimizations = []

        # 检查缓存大小
        cache_dirs = [
            'cache',
            'enhanced_cache',
            'feature_cache',
            'test_cache'
        ]

        for cache_dir in cache_dirs:
            cache_path = self.project_root / cache_dir
            if cache_path.exists():
                try:
                    total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
                    file_count = len(list(cache_path.rglob('*')))

                    if total_size > 100 * 1024 * 1024:  # 大于100MB
                        optimizations.append({
                            'type': 'cache_cleanup',
                            'cache_dir': cache_dir,
                            'size_mb': total_size / 1024 / 1024,
                            'file_count': file_count,
                            'recommendation': '清理缓存文件'
                        })

                except Exception as e:
                    optimizations.append({
                        'type': 'cache_cleanup',
                        'cache_dir': cache_dir,
                        'error': str(e)
                    })

        return optimizations

    def fix_memory_issues(self) -> List[Dict[str, Any]]:
        """修复内存问题"""
        print("修复内存问题...")

        fixes = []

        # 1. 修复日志系统的递归问题
        logger_fixes = self._fix_logger_recursion()
        fixes.extend(logger_fixes)

        # 2. 修复配置管理器的内存泄漏
        config_fixes = self._fix_config_memory_leaks()
        fixes.extend(config_fixes)

        # 3. 修复监控服务的内存问题
        monitoring_fixes = self._fix_monitoring_memory_issues()
        fixes.extend(monitoring_fixes)

        return fixes

    def _fix_logger_recursion(self) -> List[Dict[str, Any]]:
        """修复日志系统递归问题"""
        fixes = []

        logger_file = self.project_root / 'src' / 'infrastructure' / 'logging' / 'infrastructure_logger.py'

        if logger_file.exists():
            try:
                with open(logger_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查是否存在递归调用
                if 'super().log' in content and 'self._log' in content:
                    # 创建备份
                    backup_file = logger_file.with_suffix('.py.backup')
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        f.write(content)

                    # 修复递归问题
                    fixed_content = self._fix_logger_recursion_content(content)

                    with open(logger_file, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)

                    fixes.append({
                        'type': 'logger_recursion_fix',
                        'file': str(logger_file),
                        'backup': str(backup_file),
                        'status': 'fixed'
                    })

            except Exception as e:
                fixes.append({
                    'type': 'logger_recursion_fix',
                    'file': str(logger_file),
                    'error': str(e)
                })

        return fixes

    def _fix_logger_recursion_content(self, content: str) -> str:
        """修复日志递归内容"""
        # 简单的递归修复
        lines = content.split('\n')
        fixed_lines = []

        for line in lines:
            # 避免在_log方法中调用super().log
            if 'def _log' in line:
                fixed_lines.append(line)
                fixed_lines.append('        # 防止递归调用')
                fixed_lines.append('        if hasattr(self, "_logging") and self._logging:')
                fixed_lines.append('            return')
                fixed_lines.append('        self._logging = True')
            elif 'super().log' in line and 'def _log' in '\n'.join(fixed_lines[-10:]):
                # 替换为直接调用父类方法
                fixed_lines.append('        # 直接调用父类方法避免递归')
                fixed_lines.append('        logging.Logger.log(self, level, msg, *args, **kwargs)')
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_config_memory_leaks(self) -> List[Dict[str, Any]]:
        """修复配置管理器内存泄漏"""
        fixes = []

        config_files = [
            'src/infrastructure/config/core/config_manager.py',
            'src/infrastructure/config/core/config_storage.py'
        ]

        for config_file in config_files:
            file_path = self.project_root / config_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查是否有内存泄漏模式
                    if '_cache' in content or '_config_data' in content:
                        # 添加清理方法
                        if 'def cleanup' not in content:
                            backup_file = file_path.with_suffix('.py.backup')
                            with open(backup_file, 'w', encoding='utf-8') as f:
                                f.write(content)

                            # 添加清理方法
                            cleanup_method = '''
    def cleanup(self):
        """清理内存缓存"""
        if hasattr(self, '_cache'):
            self._cache.clear()
        if hasattr(self, '_config_data'):
            self._config_data.clear()
        gc.collect()
'''
                            content += cleanup_method

                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(content)

                            fixes.append({
                                'type': 'config_cleanup_method',
                                'file': str(file_path),
                                'backup': str(backup_file),
                                'status': 'added'
                            })

                except Exception as e:
                    fixes.append({
                        'type': 'config_cleanup_method',
                        'file': str(file_path),
                        'error': str(e)
                    })

        return fixes

    def _fix_monitoring_memory_issues(self) -> List[Dict[str, Any]]:
        """修复监控服务内存问题"""
        fixes = []

        monitoring_files = [
            'src/infrastructure/monitoring/monitoring_service.py',
            'src/infrastructure/monitoring/automation_monitor.py'
        ]

        for monitoring_file in monitoring_files:
            file_path = self.project_root / monitoring_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查是否有无限增长的指标存储
                    if 'metrics' in content and 'append' in content:
                        # 添加指标限制
                        if 'MAX_METRICS' not in content:
                            backup_file = file_path.with_suffix('.py.backup')
                            with open(backup_file, 'w', encoding='utf-8') as f:
                                f.write(content)

                            # 添加指标限制
                            limit_code = '''
    MAX_METRICS = 10000  # 最大指标数量
    
    def _limit_metrics(self):
        """限制指标数量防止内存泄漏"""
        if hasattr(self, 'metrics') and len(self.metrics) > self.MAX_METRICS:
            # 保留最新的指标
            self.metrics = self.metrics[-self.MAX_METRICS:]
'''
                            content += limit_code

                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(content)

                            fixes.append({
                                'type': 'monitoring_metrics_limit',
                                'file': str(file_path),
                                'backup': str(backup_file),
                                'status': 'added'
                            })

                except Exception as e:
                    fixes.append({
                        'type': 'monitoring_metrics_limit',
                        'file': str(file_path),
                        'error': str(e)
                    })

        return fixes

    def generate_recommendations(self) -> List[str]:
        """生成内存优化建议"""
        print("生成内存优化建议...")

        recommendations = [
            "定期运行垃圾回收: gc.collect()",
            "使用延迟导入减少启动内存",
            "限制缓存大小和TTL",
            "避免循环引用",
            "使用弱引用处理回调",
            "定期清理临时文件",
            "监控内存使用趋势",
            "使用内存分析工具",
            "优化数据结构",
            "实现内存泄漏检测"
        ]

        return recommendations

    def save_diagnostic_report(self):
        """保存诊断报告"""
        print("保存诊断报告...")

        # 保存完整报告
        report_file = self.report_dir / 'memory_diagnostic_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.diagnostic_results, f, ensure_ascii=False, indent=2)

        # 生成Markdown报告
        md_report = self._generate_markdown_report()
        md_file = self.report_dir / 'memory_diagnostic_report.md'
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_report)

        print(f"诊断报告已保存到: {self.report_dir}")

    def _generate_markdown_report(self) -> str:
        """生成Markdown格式报告"""
        md_content = f"""# 内存问题诊断报告

## 概述
- 诊断时间: {self.diagnostic_results['timestamp']}
- 系统内存使用: {self.diagnostic_results['memory_usage'].get('system_percent', 0):.1f}%
- 进程内存使用: {self.diagnostic_results['memory_usage'].get('process_rss', 0) / 1024 / 1024:.1f} MB

## 内存泄漏检测
"""

        if self.diagnostic_results['memory_leaks']:
            md_content += "### 发现的内存泄漏\n"
            for leak in self.diagnostic_results['memory_leaks']:
                md_content += f"- **模块**: {leak.get('module', 'N/A')}\n"
                md_content += f"- **文件**: {leak.get('file', 'N/A')}\n"
                md_content += f"- **大小变化**: {leak.get('size_diff_mb', 0):.2f} MB\n\n"
        else:
            md_content += "### 未发现明显内存泄漏\n\n"

        md_content += "## 性能问题\n"
        if self.diagnostic_results['performance_issues']:
            for issue in self.diagnostic_results['performance_issues']:
                md_content += f"- {issue}\n"
        else:
            md_content += "- 未发现明显性能问题\n"

        md_content += "\n## 修复措施\n"
        if self.diagnostic_results['fixes_applied']:
            for fix in self.diagnostic_results['fixes_applied']:
                md_content += f"- **类型**: {fix.get('type', 'N/A')}\n"
                md_content += f"- **文件**: {fix.get('file', 'N/A')}\n"
                md_content += f"- **状态**: {fix.get('status', 'N/A')}\n\n"
        else:
            md_content += "- 未应用修复措施\n"

        md_content += "\n## 优化建议\n"
        for rec in self.diagnostic_results['recommendations']:
            md_content += f"- {rec}\n"

        return md_content

    def run(self):
        """运行内存诊断流程"""
        print("开始内存诊断流程...")

        try:
            # 分析内存使用
            self.diagnostic_results['memory_usage'] = self.analyze_memory_usage()

            # 检测内存泄漏
            self.diagnostic_results['memory_leaks'] = self.detect_memory_leaks()

            # 分析循环引用
            circular_refs = self.analyze_circular_references()
            self.diagnostic_results['performance_issues'].extend([
                f"循环引用: {ref.get('file', 'N/A')}" for ref in circular_refs
            ])

            # 优化内存使用
            optimizations = self.optimize_memory_usage()
            self.diagnostic_results['performance_issues'].extend([
                f"优化: {opt.get('type', 'N/A')}" for opt in optimizations
            ])

            # 修复内存问题
            self.diagnostic_results['fixes_applied'] = self.fix_memory_issues()

            # 生成建议
            self.diagnostic_results['recommendations'] = self.generate_recommendations()

            # 保存报告
            self.save_diagnostic_report()

            print("内存诊断流程完成")

            # 输出摘要
            print(f"\n=== 内存诊断报告 ===")
            print(f"内存使用: {self.diagnostic_results['memory_usage'].get('system_percent', 0):.1f}%")
            print(f"发现泄漏: {len(self.diagnostic_results['memory_leaks'])}")
            print(f"性能问题: {len(self.diagnostic_results['performance_issues'])}")
            print(f"修复措施: {len(self.diagnostic_results['fixes_applied'])}")
            print(f"优化建议: {len(self.diagnostic_results['recommendations'])}")

        except Exception as e:
            print(f"内存诊断流程失败: {e}")
            raise


if __name__ == '__main__':
    import re  # 添加re模块导入

    diagnostic = MemoryDiagnostic()
    diagnostic.run()
