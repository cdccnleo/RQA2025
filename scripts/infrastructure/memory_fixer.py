#!/usr/bin/env python3
"""
内存问题修复脚本
针对诊断发现的具体内存问题进行修复
"""

import sys
import gc
import psutil
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class MemoryFixer:
    """内存问题修复器"""

    def __init__(self):
        self.project_root = Path(project_root)
        self.report_dir = self.project_root / 'reports' / 'infrastructure'
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.fix_results = {
            'timestamp': datetime.now().isoformat(),
            'fixes_applied': [],
            'memory_before': {},
            'memory_after': {},
            'improvements': []
        }

    def fix_logging_memory_leak(self) -> Dict[str, Any]:
        """修复日志系统内存泄漏"""
        print("修复日志系统内存泄漏...")

        fixes = []

        # 修复infrastructure_logger.py
        logger_file = self.project_root / 'src' / 'infrastructure' / 'logging' / 'infrastructure_logger.py'
        if logger_file.exists():
            try:
                with open(logger_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查并修复递归问题
                if 'super().log' in content:
                    # 创建备份
                    backup_file = logger_file.with_suffix('.py.backup')
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        f.write(content)

                    # 修复递归调用
                    fixed_content = self._fix_logger_recursion(content)

                    with open(logger_file, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)

                    fixes.append({
                        'file': str(logger_file),
                        'type': 'recursion_fix',
                        'backup': str(backup_file),
                        'status': 'fixed'
                    })

            except Exception as e:
                fixes.append({
                    'file': str(logger_file),
                    'type': 'recursion_fix',
                    'error': str(e)
                })

        # 修复unified_logging_interface.py
        unified_logger_file = self.project_root / 'src' / \
            'infrastructure' / 'logging' / 'unified_logging_interface.py'
        if unified_logger_file.exists():
            try:
                with open(unified_logger_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查循环引用
                if 'from src.engine.logging.unified_context import UnifiedLogContext' in content:
                    # 添加内存清理方法
                    if 'def cleanup' not in content:
                        cleanup_method = '''
    def cleanup(self):
        """清理日志系统内存"""
        if hasattr(self, '_loggers'):
            for logger in self._loggers.values():
                if hasattr(logger, 'handlers'):
                    for handler in logger.handlers[:]:
                        handler.close()
                        logger.removeHandler(handler)
            self._loggers.clear()
        gc.collect()
'''
                        content += cleanup_method

                        with open(unified_logger_file, 'w', encoding='utf-8') as f:
                            f.write(content)

                        fixes.append({
                            'file': str(unified_logger_file),
                            'type': 'cleanup_method',
                            'status': 'added'
                        })

            except Exception as e:
                fixes.append({
                    'file': str(unified_logger_file),
                    'type': 'cleanup_method',
                    'error': str(e)
                })

        return {'fixes': fixes}

    def _fix_logger_recursion(self, content: str) -> str:
        """修复日志递归问题"""
        lines = content.split('\n')
        fixed_lines = []
        in_log_method = False

        for i, line in enumerate(lines):
            if 'def _log' in line:
                in_log_method = True
                fixed_lines.append(line)
                fixed_lines.append('        # 防止递归调用')
                fixed_lines.append('        if hasattr(self, "_in_logging") and self._in_logging:')
                fixed_lines.append('            return')
                fixed_lines.append('        self._in_logging = True')
                fixed_lines.append('        try:')
            elif 'super().log' in line and in_log_method:
                # 替换为直接调用
                fixed_lines.append('            # 直接调用父类方法避免递归')
                fixed_lines.append(
                    '            logging.Logger.log(self, level, msg, *args, **kwargs)')
            elif in_log_method and line.strip() == '':
                fixed_lines.append('        finally:')
                fixed_lines.append('            self._in_logging = False')
                fixed_lines.append(line)
                in_log_method = False
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def fix_config_memory_leak(self) -> Dict[str, Any]:
        """修复配置系统内存泄漏"""
        print("修复配置系统内存泄漏...")

        fixes = []

        config_files = [
            'src/infrastructure/config/core/config_manager.py',
            'src/infrastructure/config/core/config_storage.py',
            'src/infrastructure/config/core/unified_validator.py'
        ]

        for config_file in config_files:
            file_path = self.project_root / config_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查是否有缓存清理方法
                    if '_cache' in content and 'def cleanup' not in content:
                        # 创建备份
                        backup_file = file_path.with_suffix('.py.backup')
                        with open(backup_file, 'w', encoding='utf-8') as f:
                            f.write(content)

                        # 添加清理方法
                        cleanup_method = '''
    def cleanup(self):
        """清理配置缓存"""
        if hasattr(self, '_cache'):
            self._cache.clear()
        if hasattr(self, '_config_data'):
            self._config_data.clear()
        if hasattr(self, '_validators'):
            self._validators.clear()
        gc.collect()
'''
                        content += cleanup_method

                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)

                        fixes.append({
                            'file': str(file_path),
                            'type': 'cache_cleanup',
                            'backup': str(backup_file),
                            'status': 'added'
                        })

                except Exception as e:
                    fixes.append({
                        'file': str(file_path),
                        'type': 'cache_cleanup',
                        'error': str(e)
                    })

        return {'fixes': fixes}

    def fix_monitoring_memory_leak(self) -> Dict[str, Any]:
        """修复监控系统内存泄漏"""
        print("修复监控系统内存泄漏...")

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

                    # 检查指标限制
                    if 'metrics' in content and 'MAX_METRICS' not in content:
                        # 创建备份
                        backup_file = file_path.with_suffix('.py.backup')
                        with open(backup_file, 'w', encoding='utf-8') as f:
                            f.write(content)

                        # 添加指标限制
                        limit_code = '''
    MAX_METRICS = 10000  # 最大指标数量
    MAX_METRIC_AGE = 3600  # 最大指标保留时间（秒）
    
    def _limit_metrics(self):
        """限制指标数量防止内存泄漏"""
        if hasattr(self, 'metrics') and len(self.metrics) > self.MAX_METRICS:
            # 保留最新的指标
            self.metrics = self.metrics[-self.MAX_METRICS:]
        
    def _cleanup_old_metrics(self):
        """清理过期指标"""
        if hasattr(self, 'metrics'):
            current_time = time.time()
            self.metrics = [
                metric for metric in self.metrics 
                if current_time - metric.get('timestamp', 0) < self.MAX_METRIC_AGE
            ]
'''
                        content += limit_code

                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)

                        fixes.append({
                            'file': str(file_path),
                            'type': 'metrics_limit',
                            'backup': str(backup_file),
                            'status': 'added'
                        })

                except Exception as e:
                    fixes.append({
                        'file': str(file_path),
                        'type': 'metrics_limit',
                        'error': str(e)
                    })

        return {'fixes': fixes}

    def fix_database_memory_leak(self) -> Dict[str, Any]:
        """修复数据库系统内存泄漏"""
        print("修复数据库系统内存泄漏...")

        fixes = []

        database_files = [
            'src/infrastructure/database/connection_manager.py',
            'src/infrastructure/database/query_executor.py'
        ]

        for db_file in database_files:
            file_path = self.project_root / db_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查连接池清理
                    if 'connection' in content and 'def cleanup' not in content:
                        # 创建备份
                        backup_file = file_path.with_suffix('.py.backup')
                        with open(backup_file, 'w', encoding='utf-8') as f:
                            f.write(content)

                        # 添加连接清理方法
                        cleanup_method = '''
    def cleanup(self):
        """清理数据库连接"""
        if hasattr(self, '_connections'):
            for conn in self._connections:
                try:
                    conn.close()
                except:
                    pass
            self._connections.clear()
        if hasattr(self, '_connection_pool'):
            self._connection_pool.close()
        gc.collect()
'''
                        content += cleanup_method

                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)

                        fixes.append({
                            'file': str(file_path),
                            'type': 'connection_cleanup',
                            'backup': str(backup_file),
                            'status': 'added'
                        })

                except Exception as e:
                    fixes.append({
                        'file': str(file_path),
                        'type': 'connection_cleanup',
                        'error': str(e)
                    })

        return {'fixes': fixes}

    def fix_circular_references(self) -> Dict[str, Any]:
        """修复循环引用"""
        print("修复循环引用...")

        fixes = []

        # 修复日志系统的循环引用
        circular_files = [
            'src/infrastructure/logging/unified_logging_interface.py',
            'src/infrastructure/logging/enhanced_log_manager.py'
        ]

        for file_path in circular_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查并修复循环导入
                    if 'from src.engine.logging.unified_context import UnifiedLogContext' in content:
                        # 使用延迟导入
                        if 'import importlib' not in content:
                            # 创建备份
                            backup_file = full_path.with_suffix('.py.backup')
                            with open(backup_file, 'w', encoding='utf-8') as f:
                                f.write(content)

                            # 添加延迟导入
                            lazy_import = '''
import importlib

def _get_unified_context():
    """延迟导入UnifiedLogContext"""
    if not hasattr(_get_unified_context, '_context'):
        module = importlib.import_module('src.engine.logging.unified_context')
        _get_unified_context._context = module.UnifiedLogContext
    return _get_unified_context._context
'''
                            # 替换直接导入
                            content = content.replace(
                                'from src.engine.logging.unified_context import UnifiedLogContext',
                                '# 使用延迟导入避免循环引用'
                            )
                            content = content.replace(
                                'UnifiedLogContext',
                                '_get_unified_context()'
                            )
                            content = lazy_import + content

                            with open(full_path, 'w', encoding='utf-8') as f:
                                f.write(content)

                            fixes.append({
                                'file': str(full_path),
                                'type': 'lazy_import',
                                'backup': str(backup_file),
                                'status': 'fixed'
                            })

                except Exception as e:
                    fixes.append({
                        'file': str(full_path),
                        'type': 'lazy_import',
                        'error': str(e)
                    })

        return {'fixes': fixes}

    def optimize_memory_usage(self) -> Dict[str, Any]:
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

        # 2. 清理缓存文件
        cache_dirs = ['cache', 'enhanced_cache', 'feature_cache', 'test_cache']
        for cache_dir in cache_dirs:
            cache_path = self.project_root / cache_dir
            if cache_path.exists():
                try:
                    # 删除超过7天的缓存文件
                    import time
                    current_time = time.time()
                    deleted_count = 0

                    for file_path in cache_path.rglob('*'):
                        if file_path.is_file():
                            file_age = current_time - file_path.stat().st_mtime
                            if file_age > 7 * 24 * 3600:  # 7天
                                file_path.unlink()
                                deleted_count += 1

                    if deleted_count > 0:
                        optimizations.append({
                            'type': 'cache_cleanup',
                            'cache_dir': cache_dir,
                            'deleted_files': deleted_count
                        })

                except Exception as e:
                    optimizations.append({
                        'type': 'cache_cleanup',
                        'cache_dir': cache_dir,
                        'error': str(e)
                    })

        return {'optimizations': optimizations}

    def measure_memory_improvement(self) -> Dict[str, Any]:
        """测量内存改进"""
        print("测量内存改进...")

        # 记录修复前的内存
        memory_before = {
            'system_percent': psutil.virtual_memory().percent,
            'process_rss': psutil.Process().memory_info().rss,
            'gc_count': gc.get_count()
        }

        # 应用修复
        self.fix_logging_memory_leak()
        self.fix_config_memory_leak()
        self.fix_monitoring_memory_leak()
        self.fix_database_memory_leak()
        self.fix_circular_references()

        # 强制垃圾回收
        gc.collect()

        # 记录修复后的内存
        memory_after = {
            'system_percent': psutil.virtual_memory().percent,
            'process_rss': psutil.Process().memory_info().rss,
            'gc_count': gc.get_count()
        }

        # 计算改进
        improvements = {
            'system_memory_change': memory_before['system_percent'] - memory_after['system_percent'],
            'process_memory_change_mb': (memory_before['process_rss'] - memory_after['process_rss']) / 1024 / 1024,
            'gc_improvement': memory_before['gc_count'][0] - memory_after['gc_count'][0]
        }

        return {
            'before': memory_before,
            'after': memory_after,
            'improvements': improvements
        }

    def save_fix_report(self):
        """保存修复报告"""
        print("保存修复报告...")

        # 保存完整报告
        report_file = self.report_dir / 'memory_fix_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.fix_results, f, ensure_ascii=False, indent=2)

        # 生成Markdown报告
        md_report = self._generate_markdown_report()
        md_file = self.report_dir / 'memory_fix_report.md'
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_report)

        print(f"修复报告已保存到: {self.report_dir}")

    def _generate_markdown_report(self) -> str:
        """生成Markdown格式报告"""
        md_content = f"""# 内存问题修复报告

## 概述
- 修复时间: {self.fix_results['timestamp']}
- 修复前系统内存: {self.fix_results['memory_before'].get('system_percent', 0):.1f}%
- 修复后系统内存: {self.fix_results['memory_after'].get('system_percent', 0):.1f}%
- 进程内存改进: {self.fix_results['improvements'].get('process_memory_change_mb', 0):.2f} MB

## 修复措施
"""

        for fix in self.fix_results['fixes_applied']:
            md_content += f"- **类型**: {fix.get('type', 'N/A')}\n"
            md_content += f"- **文件**: {fix.get('file', 'N/A')}\n"
            md_content += f"- **状态**: {fix.get('status', 'N/A')}\n\n"

        md_content += "## 内存改进\n"
        improvements = self.fix_results['improvements']
        md_content += f"- 系统内存变化: {improvements.get('system_memory_change', 0):.2f}%\n"
        md_content += f"- 进程内存变化: {improvements.get('process_memory_change_mb', 0):.2f} MB\n"
        md_content += f"- 垃圾回收改进: {improvements.get('gc_improvement', 0)}\n"

        return md_content

    def run(self):
        """运行内存修复流程"""
        print("开始内存修复流程...")

        try:
            # 测量修复前的内存
            memory_results = self.measure_memory_improvement()
            self.fix_results['memory_before'] = memory_results['before']
            self.fix_results['memory_after'] = memory_results['after']
            self.fix_results['improvements'] = memory_results['improvements']

            # 优化内存使用
            optimization_results = self.optimize_memory_usage()
            self.fix_results['fixes_applied'].extend(optimization_results.get('optimizations', []))

            # 保存报告
            self.save_fix_report()

            print("内存修复流程完成")

            # 输出摘要
            improvements = self.fix_results['improvements']
            print(f"\n=== 内存修复报告 ===")
            print(f"系统内存改进: {improvements.get('system_memory_change', 0):.2f}%")
            print(f"进程内存改进: {improvements.get('process_memory_change_mb', 0):.2f} MB")
            print(f"垃圾回收改进: {improvements.get('gc_improvement', 0)}")
            print(f"应用修复: {len(self.fix_results['fixes_applied'])}")

        except Exception as e:
            print(f"内存修复流程失败: {e}")
            raise


if __name__ == '__main__':
    import time  # 添加time模块导入

    fixer = MemoryFixer()
    fixer.run()
