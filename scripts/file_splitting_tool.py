#!/usr/bin/env python3
"""
基础设施层文件拆分工具

自动分析和拆分超大文件，提高代码可维护性。

作者: RQA2025 Team
版本: 1.0.0
更新: 2025年9月21日
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class ClassInfo:
    """类信息"""
    name: str
    start_line: int
    end_line: int
    content: str
    dependencies: List[str]
    is_abstract: bool = False


@dataclass
class FunctionInfo:
    """函数信息"""
    name: str
    start_line: int
    end_line: int
    content: str
    is_method: bool = False


@dataclass
class FileSplitPlan:
    """文件拆分计划"""
    original_file: str
    target_directory: str
    splits: Dict[str, List[str]]  # 文件名 -> 类/函数列表


class FileSplitter:
    """文件拆分器"""

    def __init__(self):
        self.large_files = [
            "src/infrastructure/cache/multi_level_cache.py",
            "src/infrastructure/cache/redis_adapter_unified.py",
            "src/infrastructure/cache/unified_cache_manager_refactored.py",
            "src/infrastructure/config/core/unified_manager.py",
            "src/infrastructure/config/monitoring/performance_monitor_dashboard.py",
            "src/infrastructure/health/enhanced_health_checker.py",
            "src/infrastructure/logging/microservice_manager.py",
            "src/infrastructure/logging/micro_service.py"
        ]

    def analyze_file_structure(self, file_path: str) -> Tuple[List[ClassInfo], List[FunctionInfo]]:
        """分析文件结构"""

        classes = []
        functions = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')

            # 使用正则表达式解析类和函数定义
            class_pattern = r'^class\s+(\w+)'
            function_pattern = r'^(?:def|async def)\s+(\w+)'

            current_class = None
            brace_count = 0

            for i, line in enumerate(lines, 1):
                # 检查类定义
                class_match = re.match(class_pattern, line.strip())
                if class_match:
                    if current_class:
                        # 结束之前的类
                        current_class.end_line = i - 1
                        current_class.content = '\n'.join(
                            lines[current_class.start_line-1:current_class.end_line])
                        classes.append(current_class)

                    # 开始新类
                    class_name = class_match.group(1)
                    current_class = ClassInfo(
                        name=class_name,
                        start_line=i,
                        end_line=0,
                        content="",
                        dependencies=[],
                        is_abstract='ABC' in line or '(ABC)' in line
                    )

                # 检查函数定义
                func_match = re.match(function_pattern, line.strip())
                if func_match and not current_class:
                    # 模块级函数
                    func_name = func_match.group(1)
                    functions.append(FunctionInfo(
                        name=func_name,
                        start_line=i,
                        end_line=i,  # 简化处理
                        content=line,
                        is_method=False
                    ))

                # 统计大括号以确定类边界
                brace_count += line.count('{') - line.count('}')

            # 处理最后一个类
            if current_class:
                current_class.end_line = len(lines)
                current_class.content = '\n'.join(
                    lines[current_class.start_line-1:current_class.end_line])
                classes.append(current_class)

        except Exception as e:
            print(f"  ❌ 分析文件失败 {file_path}: {e}")

        return classes, functions

    def create_split_plan(self, file_path: str) -> FileSplitPlan:
        """创建拆分计划"""

        classes, functions = self.analyze_file_structure(file_path)

        # 根据文件类型创建不同的拆分策略
        if 'multi_level_cache.py' in file_path:
            return self._create_multi_level_cache_split_plan(file_path, classes)
        elif 'redis_adapter_unified.py' in file_path:
            return self._create_redis_adapter_split_plan(file_path, classes)
        elif 'unified_cache_manager_refactored.py' in file_path:
            return self._create_cache_manager_split_plan(file_path, classes)
        elif 'unified_manager.py' in file_path:
            return self._create_config_manager_split_plan(file_path, classes)
        elif 'performance_monitor_dashboard.py' in file_path:
            return self._create_monitor_dashboard_split_plan(file_path, classes)
        elif 'enhanced_health_checker.py' in file_path:
            return self._create_health_checker_split_plan(file_path, classes)
        elif 'microservice_manager.py' in file_path:
            return self._create_microservice_manager_split_plan(file_path, classes)
        elif 'micro_service.py' in file_path:
            return self._create_microservice_split_plan(file_path, classes)
        else:
            # 默认拆分策略
            return self._create_default_split_plan(file_path, classes)

    def _create_multi_level_cache_split_plan(self, file_path: str, classes: List[ClassInfo]) -> FileSplitPlan:
        """创建多级缓存拆分计划"""

        base_dir = Path(file_path).parent / "multi_level_cache"

        splits = {
            "interfaces.py": ["CacheTierInterface", "CacheTier"],
            "config.py": ["TierConfig", "MultiLevelConfig"],
            "tiers/memory_tier.py": ["MemoryTier"],
            "tiers/redis_tier.py": ["RedisTier"],
            "tiers/disk_tier.py": ["DiskTier"],
            "core.py": ["MultiLevelCache"],
            "consistency.py": ["CacheConsistencyChecker"],
            "manager.py": ["MultiLevelCacheManager"],
            "factory.py": ["MultiLevelCacheFactory"],
            "optimizer.py": ["CachePerformanceOptimizer"]
        }

        return FileSplitPlan(
            original_file=file_path,
            target_directory=str(base_dir),
            splits=splits
        )

    def _create_redis_adapter_split_plan(self, file_path: str, classes: List[ClassInfo]) -> FileSplitPlan:
        """创建Redis适配器拆分计划"""

        base_dir = Path(file_path).parent / "redis_adapter"

        splits = {
            "interfaces.py": ["IRedisAdapter", "RedisConfig"],
            "connection.py": ["RedisConnection", "ConnectionPool"],
            "operations.py": ["RedisOperations", "PipelineManager"],
            "serialization.py": ["RedisSerializer", "CompressionHandler"],
            "monitoring.py": ["RedisMonitor", "PerformanceTracker"],
            "core.py": ["RedisAdapterUnified"],
            "factory.py": ["RedisAdapterFactory"]
        }

        return FileSplitPlan(
            original_file=file_path,
            target_directory=str(base_dir),
            splits=splits
        )

    def _create_cache_manager_split_plan(self, file_path: str, classes: List[ClassInfo]) -> FileSplitPlan:
        """创建缓存管理器拆分计划"""

        base_dir = Path(file_path).parent / "unified_cache_manager"

        splits = {
            "interfaces.py": ["ICacheManager", "ICacheStrategy"],
            "config.py": ["CacheManagerConfig", "CacheStrategyConfig"],
            "strategies.py": ["CacheStrategy", "EvictionStrategy"],
            "storage.py": ["CacheStorage", "PersistentStorage"],
            "monitoring.py": ["CacheMonitor", "PerformanceMetrics"],
            "core.py": ["UnifiedCacheManagerRefactored"],
            "factory.py": ["CacheManagerFactory"]
        }

        return FileSplitPlan(
            original_file=file_path,
            target_directory=str(base_dir),
            splits=splits
        )

    def _create_config_manager_split_plan(self, file_path: str, classes: List[ClassInfo]) -> FileSplitPlan:
        """创建配置管理器拆分计划"""

        base_dir = Path(file_path).parent / "unified_manager"

        splits = {
            "interfaces.py": ["IConfigManager", "IConfigProvider"],
            "config.py": ["ConfigSettings", "ConfigValidation"],
            "providers.py": ["ConfigProvider", "FileProvider"],
            "validation.py": ["ConfigValidator", "SchemaValidator"],
            "monitoring.py": ["ConfigMonitor", "ConfigMetrics"],
            "core.py": ["UnifiedConfigManager"],
            "factory.py": ["ConfigManagerFactory"]
        }

        return FileSplitPlan(
            original_file=file_path,
            target_directory=str(base_dir),
            splits=splits
        )

    def _create_monitor_dashboard_split_plan(self, file_path: str, classes: List[ClassInfo]) -> FileSplitPlan:
        """创建监控仪表板拆分计划"""

        base_dir = Path(file_path).parent / "performance_monitor"

        splits = {
            "interfaces.py": ["IMonitorDashboard", "IMetricsCollector"],
            "metrics.py": ["MetricsCollector", "PerformanceMetrics"],
            "charts.py": ["ChartGenerator", "DashboardRenderer"],
            "alerts.py": ["AlertManager", "ThresholdManager"],
            "storage.py": ["MetricsStorage", "HistoricalData"],
            "core.py": ["PerformanceMonitorDashboard"],
            "factory.py": ["MonitorDashboardFactory"]
        }

        return FileSplitPlan(
            original_file=file_path,
            target_directory=str(base_dir),
            splits=splits
        )

    def _create_health_checker_split_plan(self, file_path: str, classes: List[ClassInfo]) -> FileSplitPlan:
        """创建健康检查器拆分计划"""

        base_dir = Path(file_path).parent / "enhanced_health_checker"

        splits = {
            "interfaces.py": ["IHealthChecker", "IHealthMonitor"],
            "checkers.py": ["HealthChecker", "ServiceChecker"],
            "monitors.py": ["HealthMonitor", "SystemMonitor"],
            "metrics.py": ["HealthMetrics", "PerformanceMetrics"],
            "alerts.py": ["HealthAlert", "AlertManager"],
            "core.py": ["EnhancedHealthChecker"],
            "factory.py": ["HealthCheckerFactory"]
        }

        return FileSplitPlan(
            original_file=file_path,
            target_directory=str(base_dir),
            splits=splits
        )

    def _create_microservice_manager_split_plan(self, file_path: str, classes: List[ClassInfo]) -> FileSplitPlan:
        """创建微服务管理器拆分计划"""

        base_dir = Path(file_path).parent / "microservice_manager"

        splits = {
            "interfaces.py": ["IMicroserviceManager", "IServiceRegistry"],
            "registry.py": ["ServiceRegistry", "ServiceDiscovery"],
            "communication.py": ["ServiceCommunicator", "MessageHandler"],
            "monitoring.py": ["ServiceMonitor", "HealthChecker"],
            "config.py": ["ServiceConfig", "ClusterConfig"],
            "core.py": ["MicroserviceManager"],
            "factory.py": ["MicroserviceManagerFactory"]
        }

        return FileSplitPlan(
            original_file=file_path,
            target_directory=str(base_dir),
            splits=splits
        )

    def _create_microservice_split_plan(self, file_path: str, classes: List[ClassInfo]) -> FileSplitPlan:
        """创建微服务拆分计划"""

        base_dir = Path(file_path).parent / "microservice"

        splits = {
            "interfaces.py": ["IMicroservice", "IServiceLifecycle"],
            "lifecycle.py": ["ServiceLifecycle", "StartupManager"],
            "communication.py": ["ServiceCommunicator", "EventHandler"],
            "monitoring.py": ["ServiceMonitor", "MetricsCollector"],
            "config.py": ["ServiceConfig", "EnvironmentConfig"],
            "core.py": ["MicroService"],
            "factory.py": ["MicroserviceFactory"]
        }

        return FileSplitPlan(
            original_file=file_path,
            target_directory=str(base_dir),
            splits=splits
        )

    def _create_default_split_plan(self, file_path: str, classes: List[ClassInfo]) -> FileSplitPlan:
        """创建默认拆分计划"""

        base_name = Path(file_path).stem
        base_dir = Path(file_path).parent / base_name

        # 简单地将类平均分配到几个文件中
        splits = {}
        classes_per_file = max(1, len(classes) // 3)

        for i in range(0, len(classes), classes_per_file):
            file_name = f"part_{i//classes_per_file + 1}.py"
            class_names = [cls.name for cls in classes[i:i+classes_per_file]]
            splits[file_name] = class_names

        return FileSplitPlan(
            original_file=file_path,
            target_directory=str(base_dir),
            splits=splits
        )

    def execute_split_plan(self, plan: FileSplitPlan, dry_run: bool = True) -> bool:
        """执行拆分计划"""

        print(f"🔧 {'预览' if dry_run else '执行'}文件拆分: {Path(plan.original_file).name}")
        print(f"  目标目录: {plan.target_directory}")

        if dry_run:
            print("  📋 拆分计划:")
            for target_file, classes in plan.splits.items():
                print(f"    • {target_file}: {', '.join(classes)}")
            return True

        try:
            # 创建目标目录
            target_dir = Path(plan.target_directory)
            target_dir.mkdir(parents=True, exist_ok=True)

            # 读取原文件
            with open(plan.original_file, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # 分析原文件的导入和依赖
            imports, dependencies = self._extract_imports_and_dependencies(original_content)

            # 为每个目标文件生成内容
            for target_file, class_names in plan.splits.items():
                file_content = self._generate_file_content(
                    target_file, class_names, original_content,
                    imports, dependencies, plan.original_file
                )

                target_path = target_dir / target_file
                target_path.parent.mkdir(parents=True, exist_ok=True)

                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(file_content)

                # 校验文件语法
                syntax_valid, syntax_error = self._validate_file_syntax(target_path)
                if syntax_valid:
                    print(f"  ✅ 创建文件: {target_path} ({len(class_names)} 个类)")
                else:
                    print(
                        f"  ⚠️ 创建文件: {target_path} ({len(class_names)} 个类) - 语法错误: {syntax_error}")

            # 创建__init__.py文件
            self._create_init_file(target_dir, plan.splits)

            print(f"  🎉 拆分完成! 创建了 {len(plan.splits)} 个文件")

            # 可选：备份原文件
            backup_path = Path(plan.original_file).with_suffix('.backup.py')
            if not backup_path.exists():
                import shutil
                shutil.copy2(plan.original_file, backup_path)
                print(f"  💾 原文件已备份: {backup_path}")

            return True

        except Exception as e:
            print(f"  ❌ 拆分失败: {e}")
            return False

    def _extract_imports_and_dependencies(self, content: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """提取导入和依赖关系"""

        lines = content.split('\n')
        imports = []
        dependencies = {}

        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)

        return imports, dependencies

    def _generate_file_content(self, target_file: str, class_names: List[str],
                               original_content: str, imports: List[str],
                               dependencies: Dict[str, List[str]], original_file: str) -> str:
        """生成目标文件内容"""

        lines = original_content.split('\n')
        selected_lines = []

        # 添加文件头
        file_header = f'''"""
{Path(target_file).stem} - 从 {Path(original_file).name} 拆分

自动生成的文件，包含以下类：
{', '.join(class_names)}
"""

'''

        # 筛选相关类的内容
        in_target_class = False
        current_class = None

        for i, line in enumerate(lines):
            # 检查是否是目标类的开始
            for class_name in class_names:
                if re.match(rf'^class\s+{class_name}\b', line.strip()):
                    in_target_class = True
                    current_class = class_name
                    break

            # 检查是否是其他类的开始（结束当前类）
            if in_target_class and re.match(r'^class\s+\w+', line.strip()):
                other_class = re.match(r'^class\s+(\w+)', line.strip()).group(1)
                if other_class not in class_names:
                    in_target_class = False
                    current_class = None

            # 检查缩进减少（可能表示类结束）
            if in_target_class and current_class and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                # 检查这是否是类外的代码
                next_lines = lines[i:i+5] if i+5 < len(lines) else lines[i:]
                if not any(re.match(rf'^class\s+{current_class}\b', l.strip()) for l in next_lines):
                    # 如果接下来的几行都没有当前类的定义，可能已经出了类的范围
                    in_target_class = False
                    current_class = None

            if in_target_class or line.strip() in imports:
                selected_lines.append(line)

        # 如果没有找到任何类内容，至少包含导入
        if not selected_lines:
            selected_lines = imports.copy()

        return file_header + '\n'.join(selected_lines)

    def _validate_file_syntax(self, file_path: Path) -> Tuple[bool, str]:
        """校验文件语法

        Returns:
            (is_valid, error_message)
        """
        try:
            import py_compile
            py_compile.compile(str(file_path), doraise=True)
            return True, ""
        except py_compile.PyCompileError as e:
            return False, str(e)
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"未知错误: {e}"

    def _create_init_file(self, target_dir: Path, splits: Dict[str, List[str]]):
        """创建__init__.py文件"""

        init_content = '''"""
自动生成的包初始化文件
"""

'''

        # 为每个拆分的文件添加导入
        for target_file, class_names in splits.items():
            module_name = Path(target_file).stem
            for class_name in class_names:
                init_content += f"from .{module_name} import {class_name}\n"

        # 添加__all__
        all_classes = []
        for class_names in splits.values():
            all_classes.extend(class_names)

        init_content += f"\n__all__ = {all_classes}\n"

        init_file = target_dir / "__init__.py"
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(init_content)

        # 校验__init__.py文件语法
        syntax_valid, syntax_error = self._validate_file_syntax(init_file)
        if syntax_valid:
            print(f"  📦 创建包文件: {init_file}")
        else:
            print(f"  ⚠️ 创建包文件: {init_file} - 语法错误: {syntax_error}")

    def run_file_splitting_analysis(self, dry_run: bool = True) -> Dict[str, Any]:
        """运行文件拆分分析"""

        print("🔍 开始文件拆分分析...")
        print("=" * 60)

        results = {
            'total_files_analyzed': len(self.large_files),
            'successful_splits': 0,
            'failed_splits': 0,
            'plans': []
        }

        for file_path in self.large_files:
            if not Path(file_path).exists():
                print(f"  ⚠️ 文件不存在: {file_path}")
                continue

            try:
                print(f"\\n📄 分析文件: {Path(file_path).name}")

                # 创建拆分计划
                plan = self.create_split_plan(file_path)
                results['plans'].append({
                    'file': file_path,
                    'target_directory': plan.target_directory,
                    'num_splits': len(plan.splits),
                    'total_classes': sum(len(classes) for classes in plan.splits.values())
                })

                # 执行拆分
                success = self.execute_split_plan(plan, dry_run=dry_run)

                if success:
                    results['successful_splits'] += 1
                    print(f"  ✅ 拆分计划创建成功 ({len(plan.splits)} 个目标文件)")
                else:
                    results['failed_splits'] += 1
                    print("  ❌ 拆分失败")

            except Exception as e:
                print(f"  ❌ 分析失败: {e}")
                results['failed_splits'] += 1

        print("\n📊 分析结果汇总:")
        print(f"  总文件数: {results['total_files_analyzed']}")
        print(f"  成功分析: {results['successful_splits']}")
        print(f"  失败分析: {results['failed_splits']}")
        print(f"  执行模式: {'预览模式' if dry_run else '实际拆分'}")

        if results['plans']:
            print("\\n📋 拆分计划详情:")
            for plan in results['plans']:
                print(
                    f"  • {Path(plan['file']).name} → {plan['num_splits']} 个文件 ({plan['total_classes']} 个类)")

        return results


def main():
    """主函数"""

    print("🔧 基础设施层文件拆分工具")
    print("=" * 50)

    splitter = FileSplitter()

    # 询问用户是否执行实际拆分
    print("此工具将分析和拆分8个超大文件。")
    print("\\n请选择执行模式:")
    print("  (p) 预览模式 - 只显示拆分计划")
    print("  (e) 执行模式 - 实际执行文件拆分")
    print("  (q) 退出")

    choice = input("\\n请选择 (p/e/q): ").lower().strip()

    if choice == 'q':
        print("👋 退出文件拆分工具")
        return
    elif choice == 'p':
        dry_run = True
        print("👁️  进入预览模式...")
    else:
        dry_run = False
        print("🔧 开始实际拆分...")
        print("⚠️  注意: 这将修改源代码文件，建议先备份!")

    # 执行分析
    try:
        results = splitter.run_file_splitting_analysis(dry_run=dry_run)

        if not dry_run and results['successful_splits'] > 0:
            print("\\n✅ 文件拆分完成!")
            print("💡 建议:")
            print("  1. 运行测试确保拆分后功能正常")
            print("  2. 检查导入路径是否正确")
            print("  3. 更新相关文档和依赖")

    except Exception as e:
        print(f"\\n❌ 执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
