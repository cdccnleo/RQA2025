#!/usr/bin/env python3
"""
统一基础设施集成迁移脚本
将业务层直接使用基础设施层的代码迁移到统一集成层
"""

import os
import re


class UnifiedIntegrationMigrator:
    """统一基础设施集成迁移器"""

    def __init__(self):
        self.migration_stats = {
            'files_processed': 0,
            'imports_migrated': 0,
            'errors': []
        }

        # 定义迁移映射 - 导入语句映射 (支持多种格式)
        self.import_migration_map = {
            'src/data': [
                # 直接导入语句 (更宽泛的匹配)
                ('from src.infrastructure.logging import get_infrastructure_logger',
                 'from src.core.integration import get_data_adapter'),
                ('from src.infrastructure.logging.unified_logger import get_unified_logger',
                 'from src.core.integration import get_data_adapter'),
                ('from src.infrastructure.logging.unified_logger import UnifiedLogger',
                 'from src.core.integration import get_data_adapter'),
                ('from src.infrastructure.config.unified_manager import UnifiedConfigManager',
                 'from src.core.integration import get_data_adapter'),
                ('from src.infrastructure.config.unified_config import UnifiedConfigManager',
                 'from src.core.integration import get_data_adapter'),
                ('from src.infrastructure.cache.unified_cache import UnifiedCacheManager',
                 'from src.core.integration import get_data_adapter'),
                ('from src.infrastructure.monitoring.unified_monitoring import UnifiedMonitoring',
                 'from src.core.integration import get_data_adapter'),
                ('from src.infrastructure.health.enhanced_health_checker import EnhancedHealthChecker',
                 'from src.core.integration import get_data_adapter'),
                ('from src.infrastructure.health.health_checker import HealthChecker',
                 'from src.core.integration import get_data_adapter'),
                # 其他基础设施组件
                ('from src.infrastructure.event.event_bus import EventBus',
                 'from src.core.integration import get_data_adapter'),
                ('from src.infrastructure.utils.exceptions import DataLoaderError',
                 'from src.core.integration import get_data_adapter'),
                ('from src.infrastructure.interfaces.standard_interfaces import',
                 'from src.core.integration import get_data_adapter'),
                ('from src.infrastructure.error.exceptions import',
                 'from src.core.integration import get_data_adapter'),
                ('from src.infrastructure.monitoring.metrics import',
                 'from src.core.integration import get_data_adapter'),
            ],
            'src/features': [
                # 直接导入语句
                ('from src.infrastructure.logging import get_infrastructure_logger',
                 'from src.core.integration import get_features_adapter'),
                ('from src.infrastructure.logging.unified_logger import get_unified_logger',
                 'from src.core.integration import get_features_adapter'),
                ('from src.infrastructure.logging.unified_logger import UnifiedLogger',
                 'from src.core.integration import get_features_adapter'),
                ('from src.infrastructure.config.unified_manager import UnifiedConfigManager',
                 'from src.core.integration import get_features_adapter'),
                ('from src.infrastructure.config.unified_config import UnifiedConfigManager',
                 'from src.core.integration import get_features_adapter'),
                ('from src.infrastructure.cache.unified_cache import UnifiedCacheManager',
                 'from src.core.integration import get_features_adapter'),
                ('from src.infrastructure.monitoring.unified_monitoring import UnifiedMonitoring',
                 'from src.core.integration import get_features_adapter'),
                ('from src.infrastructure.health.enhanced_health_checker import EnhancedHealthChecker',
                 'from src.core.integration import get_features_adapter'),
                ('from src.infrastructure.health.health_checker import HealthChecker',
                 'from src.core.integration import get_features_adapter')
            ],
            'src/trading': [
                # 直接导入语句
                ('from src.infrastructure.logging import get_infrastructure_logger',
                 'from src.core.integration import get_trading_adapter'),
                ('from src.infrastructure.logging.unified_logger import get_unified_logger',
                 'from src.core.integration import get_trading_adapter'),
                ('from src.infrastructure.logging.unified_logger import UnifiedLogger',
                 'from src.core.integration import get_trading_adapter'),
                ('from src.infrastructure.config.unified_manager import UnifiedConfigManager',
                 'from src.core.integration import get_trading_adapter'),
                ('from src.infrastructure.config.unified_config import UnifiedConfigManager',
                 'from src.core.integration import get_trading_adapter'),
                ('from src.infrastructure.cache.unified_cache import UnifiedCacheManager',
                 'from src.core.integration import get_trading_adapter'),
                ('from src.infrastructure.monitoring.unified_monitoring import UnifiedMonitoring',
                 'from src.core.integration import get_trading_adapter'),
                ('from src.infrastructure.health.enhanced_health_checker import EnhancedHealthChecker',
                 'from src.core.integration import get_trading_adapter'),
                ('from src.infrastructure.health.health_checker import HealthChecker',
                 'from src.core.integration import get_trading_adapter')
            ],
            'src/risk': [
                # 直接导入语句
                ('from src.infrastructure.logging import get_infrastructure_logger',
                 'from src.core.integration import get_risk_adapter'),
                ('from src.infrastructure.logging.unified_logger import get_unified_logger',
                 'from src.core.integration import get_risk_adapter'),
                ('from src.infrastructure.logging.unified_logger import UnifiedLogger',
                 'from src.core.integration import get_risk_adapter'),
                ('from src.infrastructure.config.unified_manager import UnifiedConfigManager',
                 'from src.core.integration import get_risk_adapter'),
                ('from src.infrastructure.config.unified_config import UnifiedConfigManager',
                 'from src.core.integration import get_risk_adapter'),
                ('from src.infrastructure.cache.unified_cache import UnifiedCacheManager',
                 'from src.core.integration import get_risk_adapter'),
                ('from src.infrastructure.monitoring.unified_monitoring import UnifiedMonitoring',
                 'from src.core.integration import get_risk_adapter'),
                ('from src.infrastructure.health.enhanced_health_checker import EnhancedHealthChecker',
                 'from src.core.integration import get_risk_adapter'),
                ('from src.infrastructure.health.health_checker import HealthChecker',
                 'from src.core.integration import get_risk_adapter')
            ]
        }

        # 定义使用方式映射 - 如何使用适配器
        self.usage_migration_map = {
            'src/data': {
                'get_infrastructure_logger': 'get_data_adapter().get_data_infrastructure_manager().get_logger()',
                'get_unified_logger': 'get_data_adapter().get_data_infrastructure_manager().get_logger()',
                'UnifiedConfigManager': 'get_data_adapter().get_data_config_bridge()',
                'UnifiedCacheManager': 'get_data_adapter().get_data_cache_bridge()',
                'UnifiedMonitoring': 'get_data_adapter().get_data_monitoring_bridge()',
                'EnhancedHealthChecker': 'get_data_adapter().get_data_health_bridge()'
            },
            'src/features': {
                'get_infrastructure_logger': 'get_features_adapter().get_features_infrastructure_bridge().get_logger()',
                'get_unified_logger': 'get_features_adapter().get_features_infrastructure_bridge().get_logger()',
                'UnifiedConfigManager': 'get_features_adapter().get_features_config_manager()',
                'UnifiedCacheManager': 'get_features_adapter().get_features_cache_manager()',
                'UnifiedMonitoring': 'get_features_adapter().get_features_monitoring()',
                'EnhancedHealthChecker': 'get_features_adapter().get_features_health_checker()'
            },
            'src/trading': {
                'get_infrastructure_logger': 'get_trading_adapter().get_trading_engine().get_logger()',
                'get_unified_logger': 'get_trading_adapter().get_trading_engine().get_logger()',
                'UnifiedConfigManager': 'get_trading_adapter().get_trading_config_manager()',
                'UnifiedCacheManager': 'get_trading_adapter().get_trading_cache_manager()',
                'UnifiedMonitoring': 'get_trading_adapter().get_trading_monitoring()',
                'EnhancedHealthChecker': 'get_trading_adapter().get_trading_health_checker()'
            },
            'src/risk': {
                'get_infrastructure_logger': 'get_risk_adapter().get_risk_manager().get_logger()',
                'get_unified_logger': 'get_risk_adapter().get_risk_manager().get_logger()',
                'UnifiedConfigManager': 'get_risk_adapter().get_risk_config_manager()',
                'UnifiedCacheManager': 'get_risk_adapter().get_risk_cache_manager()',
                'UnifiedMonitoring': 'get_risk_adapter().get_risk_monitoring()',
                'EnhancedHealthChecker': 'get_risk_adapter().get_risk_health_checker()'
            }
        }

    def migrate_layer(self, layer_path: str, layer_type: str) -> None:
        """迁移指定层的代码"""
        if not os.path.exists(layer_path):
            print(f"警告: 路径 {layer_path} 不存在")
            return

        print(f"\n开始迁移 {layer_type} 层...")

        # 查找所有Python文件
        python_files = []
        for root, dirs, files in os.walk(layer_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))

        print(f"找到 {len(python_files)} 个Python文件")

        # 迁移每个文件
        for file_path in python_files:
            try:
                self._migrate_file(file_path, layer_type)
            except Exception as e:
                error_msg = f"迁移文件失败 {file_path}: {e}"
                print(f"错误: {error_msg}")
                self.migration_stats['errors'].append(error_msg)

        print(f"{layer_type} 层迁移完成")

    def _migrate_file(self, file_path: str, layer_type: str) -> None:
        """迁移单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            import_map = self.import_migration_map.get(layer_type, {})
            usage_map = self.usage_migration_map.get(layer_type, {})

            # 在试运行模式下显示文件信息
            if self.migration_stats.get('verbose') and self.migration_stats.get('dry_run'):
                print(f"🔍 检查文件: {file_path}")
                # 显示文件中的导入语句 (包括try-except块中的)
                import_lines = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('from src.infrastructure'):
                        import_lines.append(line)
                    elif 'from src.infrastructure' in line:
                        # 处理在字符串中的导入语句
                        import_lines.append(line)

                if import_lines:
                    print(f"  📋 发现导入语句 ({len(import_lines)}个):")
                    for line in import_lines[:3]:  # 只显示前3个
                        print(f"    • {line}")
                    if len(import_lines) > 3:
                        print(f"    ... 还有 {len(import_lines) - 3} 个导入语句")
                else:
                    print("  📋 未发现需要迁移的导入语句")

            # 替换导入语句 (使用正则表达式匹配)
            import_migrated = False
            for old_import, new_import in import_map:
                if re.search(old_import, content):
                    content = re.sub(old_import, new_import, content)
                    self.migration_stats['imports_migrated'] += 1
                    import_migrated = True

                    if self.migration_stats.get('verbose') and self.migration_stats.get('dry_run'):
                        print(f"  🔄 匹配到: {old_import}")
                        print(f"  ✅ 替换为: {new_import}")

            # 替换使用方式 - 更精确的匹配
            if isinstance(usage_map, dict):
                for old_usage, new_usage in usage_map.items():
                    # 匹配类实例化: ClassName(
                    pattern1 = r'\b' + re.escape(old_usage) + r'\s*\('
                    content = re.sub(pattern1, new_usage + '(', content)

                    # 匹配类名使用: ClassName.method()
                    pattern2 = r'\b' + re.escape(old_usage) + r'\s*\.'
                    content = re.sub(pattern2, new_usage + '.', content)
            elif isinstance(usage_map, list):
                for old_usage, new_usage in usage_map:
                    # 匹配类实例化: ClassName(
                    pattern1 = r'\b' + re.escape(old_usage) + r'\s*\('
                    content = re.sub(pattern1, new_usage + '(', content)

                    # 匹配类名使用: ClassName.method()
                    pattern2 = r'\b' + re.escape(old_usage) + r'\s*\.'
                    content = re.sub(pattern2, new_usage + '.', content)

            # 特殊处理：添加适配器获取语句
            if content != original_content:
                # 在文件开头添加适配器导入和获取
                lines = content.split('\n')
                new_lines = []

                # 检查是否已经有了适配器导入
                has_adapter_import = any('from src.core.integration import get_' +
                                         layer_type + '_adapter' in line for line in lines)
                has_adapter_usage = any('get_' + layer_type +
                                        '_adapter()' in line for line in lines)

                if not has_adapter_import and has_adapter_usage:
                    # 找到第一个import语句的位置
                    first_import_idx = -1
                    for i, line in enumerate(lines):
                        if line.strip().startswith('from ') or line.strip().startswith('import '):
                            first_import_idx = i
                            break

                    if first_import_idx >= 0:
                        # 在第一个import后插入适配器获取代码
                        adapter_var = f"{layer_type}_adapter"
                        adapter_getter = f"{adapter_var} = get_{layer_type}_adapter()"
                        new_lines.extend(lines[:first_import_idx + 1])
                        new_lines.append(f"# 获取{adapter_var}适配器")
                        new_lines.append(adapter_getter)
                        new_lines.append("")
                        new_lines.extend(lines[first_import_idx + 1:])
                        content = '\n'.join(new_lines)

            # 如果内容有变化，写入文件
            if content != original_content:
                if self.migration_stats.get('dry_run'):
                    print(f"🔍 [试运行] 发现需要迁移: {file_path}")
                    if self.migration_stats.get('verbose'):
                        # 显示具体的更改
                        lines_old = original_content.split('\n')
                        lines_new = content.split('\n')
                        print(f"  📝 更改详情:")
                        for i, (old_line, new_line) in enumerate(zip(lines_old, lines_new)):
                            if old_line != new_line:
                                print(f"    第{i+1}行:")
                                print(f"      旧: {old_line}")
                                print(f"      新: {new_line}")
                else:
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"✅ 已迁移: {file_path}")
                    except Exception as e:
                        error_msg = f"写入文件失败 {file_path}: {e}"
                        print(f"❌ {error_msg}")
                        self.migration_stats['errors'].append(error_msg)
                        return

                self.migration_stats['files_processed'] += 1

        except Exception as e:
            raise Exception(f"处理文件 {file_path} 时出错: {e}")

    def generate_migration_report(self) -> str:
        """生成迁移报告"""
        report = f"""
统一基础设施集成迁移报告
============================

迁移统计:
- 处理文件数: {self.migration_stats['files_processed']}
- 迁移导入数: {self.migration_stats['imports_migrated']}
- 错误数量: {len(self.migration_stats['errors'])}

迁移详情:
================

数据层迁移:
- 导入语句替换: 基础设施层导入 → 统一集成层导入
- 使用方式更新: 直接使用 → 适配器模式使用
- 主要文件: src/data/ 目录下所有Python文件

特征层迁移:
- 导入语句替换: 基础设施层导入 → 统一集成层导入
- 使用方式更新: 直接使用 → 适配器模式使用
- 主要文件: src/features/ 目录下所有Python文件

交易层迁移:
- 导入语句替换: 基础设施层导入 → 统一集成层导入
- 使用方式更新: 直接使用 → 适配器模式使用
- 主要文件: src/trading/ 目录下所有Python文件

风控层迁移:
- 导入语句替换: 基础设施层导入 → 统一集成层导入
- 使用方式更新: 直接使用 → 适配器模式使用
- 主要文件: src/risk/ 目录下所有Python文件

迁移策略:
==========

1. 导入语句迁移:
   - 将 'from src.infrastructure.*' 替换为 'from src.core.integration import get_*_adapter'
   - 保持原有功能不变，只是改变访问方式

2. 使用方式迁移:
   - 将直接实例化基础设施组件改为通过适配器获取
   - 例如: UnifiedConfigManager() → get_data_adapter().get_data_config_bridge()

3. 错误处理:
   - 所有迁移操作都有异常处理
   - 迁移失败的文件会被记录在错误列表中

4. 验证机制:
   - 迁移完成后需要运行测试验证功能正常
   - 检查导入是否成功，功能是否正常工作

错误详情:
==========
"""
        for error in self.migration_stats['errors']:
            report += f"- {error}\n"

        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='统一基础设施集成迁移工具')
    parser.add_argument('--batch', action='store_true', help='启用批量迁移模式')
    parser.add_argument('--all-layers', action='store_true', help='迁移所有业务层')
    parser.add_argument('--layer', type=str, help='指定要迁移的业务层 (data/features/trading/risk)')
    parser.add_argument('--dry-run', action='store_true', help='仅显示将要进行的更改，不实际修改文件')
    parser.add_argument('--force', action='store_true', help='强制覆盖已存在的备份文件')
    parser.add_argument('--verbose', action='store_true', help='显示详细的迁移过程')

    args = parser.parse_args()

    migrator = UnifiedIntegrationMigrator()

    # 设置迁移模式
    if args.dry_run:
        migrator.migration_stats['dry_run'] = True
        print("🔍 启用试运行模式 - 仅显示更改，不会实际修改文件")
    else:
        migrator.migration_stats['dry_run'] = False

    if args.verbose:
        migrator.migration_stats['verbose'] = True
        print("📝 启用详细输出模式")

    # 确定要迁移的层
    layers_to_migrate = []

    if args.layer:
        layer_map = {
            'data': ('src/data', 'data'),
            'features': ('src/features', 'features'),
            'trading': ('src/trading', 'trading'),
            'risk': ('src/risk', 'risk')
        }
        if args.layer in layer_map:
            layers_to_migrate = [layer_map[args.layer]]
            print(f"🎯 指定迁移层: {args.layer}")
        else:
            print(f"❌ 无效的层名称: {args.layer}")
            print("有效的层名称: data, features, trading, risk")
            return
    elif args.all_layers:
        layers_to_migrate = [
            ('src/data', 'data'),
            ('src/features', 'features'),
            ('src/trading', 'trading'),
            ('src/risk', 'risk')
        ]
        print("🚀 迁移所有业务层")
    else:
        # 默认迁移所有层
        layers_to_migrate = [
            ('src/data', 'data'),
            ('src/features', 'features'),
            ('src/trading', 'trading'),
            ('src/risk', 'risk')
        ]
        print("📦 默认迁移所有业务层")

    print("=" * 60)
    print("🏗️  开始统一基础设施集成迁移")
    print("=" * 60)

    # 显示迁移计划
    print("\n📋 迁移计划:")
    for layer_path, layer_type in layers_to_migrate:
        print(f"  • {layer_type}层: {layer_path}")
    print()

    # 执行迁移
    for layer_path, layer_type in layers_to_migrate:
        migrator.migrate_layer(layer_path, layer_type)

    # 生成报告
    report = migrator.generate_migration_report()
    print(report)

    # 保存报告
    report_filename = 'UNIFIED_INTEGRATION_MIGRATION_REPORT.md'
    if args.dry_run:
        report_filename = 'UNIFIED_INTEGRATION_MIGRATION_DRY_RUN_REPORT.md'

    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ 迁移报告已保存: {report_filename}")
    except Exception as e:
        print(f"❌ 保存报告失败: {e}")

    print("\n" + "=" * 60)
    if args.dry_run:
        print("🔍 试运行完成！请检查上述输出以确认迁移效果")
        print("💡 要实际执行迁移，请移除 --dry-run 参数")
    else:
        print("🎉 迁移完成！")
        print("\n📝 后续步骤建议:")
        print("1. 🔍 运行测试验证功能正常")
        print("2. 📋 检查导入语句是否正确")
        print("3. 🧪 验证业务功能是否正常工作")
        print("4. 📊 查看迁移报告了解详细信息")
        print("5. 🔧 如有问题，请参考错误详情进行修复")
    print("=" * 60)


if __name__ == "__main__":
    main()
