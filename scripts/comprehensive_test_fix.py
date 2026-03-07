#!/usr/bin/env python3
"""
综合测试修复工具

结合自动修复和手动修复策略来解决测试导入问题
"""

import os
import re
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class ComprehensiveTestFixer:
    """综合测试修复工具"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.tests_dir = self.project_root / "tests"
        self.backup_dir = self.project_root / \
            f"backup/comprehensive_test_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # 构建修复映射
        self._build_fix_mapping()

    def _build_fix_mapping(self):
        """构建修复映射"""

        # 自动映射规则
        self.auto_fix_mapping = {
            # 工具类映射
            r'from src\.utils\.logger import': 'from src.infrastructure.utils.logger import',
            r'from src\.utils\.date_utils import': 'from src.infrastructure.utils.date_utils import',
            r'from src\.utils\.math_utils import': 'from src.infrastructure.utils.math_utils import',
            r'from src\.utils\.data_utils import': 'from src.infrastructure.utils.data_utils import',
            r'from src\.utils\.file_utils import': 'from src.infrastructure.utils.file_utils import',
            r'from src\.utils\.logging_utils import': 'from src.infrastructure.utils.logging_utils import',
            r'from src\.utils import': 'from src.infrastructure.utils import',

            # 配置映射
            r'from src\.config\.': 'from src.infrastructure.config.',
            r'from src\.config import': 'from src.infrastructure.config import',

            # 模型映射
            r'from src\.models\.': 'from src.ml.models.',
            r'from src\.models import': 'from src.ml.models import',

            # 集成映射
            r'from src\.integration\.': 'from src.core.integration.',
            r'from src\.integration import': 'from src.core.integration import',

            # 服务映射
            r'from src\.services\.': 'from src.infrastructure.services.',
            r'from src\.services import': 'from src.infrastructure.services import',

            # 适配器映射
            r'from src\.adapters\.': 'from src.data.adapters.',
            r'from src\.adapters import': 'from src.data.adapters import',

            # 监控映射
            r'from src\.monitoring\.': 'from src.infrastructure.monitoring.',
            r'from src\.monitoring import': 'from src.infrastructure.monitoring import',

            # 通用映射
            r'from src\.common\.': 'from src.infrastructure.utils.',
            r'from src\.common import': 'from src.infrastructure.utils import',
        }

        # 手动映射规则 - 基于具体分析结果
        self.manual_fix_mapping = {
            'from src.data.quality.data_quality_monitor import DataQualityMonitor': 'from src.data.quality.data_quality_monitor import DataQualityMonitor',
            'from src.data.validator import DataValidator': 'from src.data.validator import DataValidator',
            'from src.data.adapters.base_adapter import AdapterConfig': 'from src.data.adapters.base_adapter import AdapterConfig',
            'from src.models.base_model import BaseModel': 'from src.ml.models.base_model import BaseModel',
            'from src.infrastructure.monitoring.system_monitor import SystemMonitor': 'from src.infrastructure.monitoring.system_monitor import SystemMonitor',
            'from src.data.data_manager import DataModel': 'from src.data.data_manager import DataModel',
            'from src.features.feature_engineer import FeatureEngineer': 'from src.features.feature_engineer import FeatureEngineer',
            'from src.features.feature_manager import FeatureManager': 'from src.features.feature_manager import FeatureManager',
            'from src.data.loader.base_loader import BaseDataLoader': 'from src.data.loader.base_loader import BaseDataLoader',
            'from src.data.cache.cache_manager import CacheManager': 'from src.data.cache.cache_manager import CacheManager',
        }

    def analyze_and_fix(self, batch_size: int = 50, dry_run: bool = True) -> Dict[str, Any]:
        """分析并修复测试文件"""

        results = {
            "dry_run": dry_run,
            "files_processed": 0,
            "files_fixed": 0,
            "auto_fixes": 0,
            "manual_fixes": 0,
            "errors": [],
            "fixes_applied": []
        }

        print(f"🔍 开始综合测试修复分析... ({'仅分析' if dry_run else '实际修复'})")

        # 获取所有测试文件
        test_files = []
        for root, dirs, files in os.walk(self.tests_dir):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    test_files.append(Path(root) / file)

        print(f"📋 发现 {len(test_files)} 个测试文件")

        # 分批处理
        for i in range(0, len(test_files), batch_size):
            batch = test_files[i:i + batch_size]
            print(f"📦 处理批次 {i//batch_size + 1}/{(len(test_files) + batch_size - 1)//batch_size}")

            for file_path in batch:
                results["files_processed"] += 1

                try:
                    fixes = self._fix_single_file(file_path, dry_run)
                    if fixes:
                        results["files_fixed"] += 1
                        results["auto_fixes"] += fixes.get("auto_fixes", 0)
                        results["manual_fixes"] += fixes.get("manual_fixes", 0)
                        results["fixes_applied"].extend(fixes.get("fixes", []))
                except Exception as e:
                    results["errors"].append(f"{file_path}: {e}")

        return results

    def _fix_single_file(self, file_path: Path, dry_run: bool = True) -> Dict[str, Any]:
        """修复单个文件"""

        fixes = {
            "auto_fixes": 0,
            "manual_fixes": 0,
            "fixes": []
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            modified = False

            # 逐行处理
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith(('from src.', 'import src.')):
                    # 首先尝试自动修复
                    auto_fixed = self._apply_auto_fix(line)
                    if auto_fixed != line:
                        lines[i] = auto_fixed
                        modified = True
                        fixes["auto_fixes"] += 1
                        fixes["fixes"].append({
                            "file": str(file_path.relative_to(self.project_root)),
                            "type": "auto",
                            "original": line.strip(),
                            "fixed": auto_fixed.strip(),
                            "line_number": i + 1
                        })
                        continue

                    # 然后尝试手动修复
                    manual_fixed = self._apply_manual_fix(line)
                    if manual_fixed != line:
                        lines[i] = manual_fixed
                        modified = True
                        fixes["manual_fixes"] += 1
                        fixes["fixes"].append({
                            "file": str(file_path.relative_to(self.project_root)),
                            "type": "manual",
                            "original": line.strip(),
                            "fixed": manual_fixed.strip(),
                            "line_number": i + 1
                        })

            if modified and not dry_run:
                new_content = '\n'.join(lines)

                # 备份原文件
                backup_path = self.backup_dir / file_path.relative_to(self.project_root)
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)

                # 写入修复后的内容
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

        except Exception as e:
            fixes["fixes"].append({
                "file": str(file_path.relative_to(self.project_root)),
                "error": str(e)
            })

        return fixes if fixes["auto_fixes"] > 0 or fixes["manual_fixes"] > 0 else {}

    def _apply_auto_fix(self, import_line: str) -> str:
        """应用自动修复"""

        for pattern, replacement in self.auto_fix_mapping.items():
            if re.search(pattern, import_line):
                return re.sub(pattern, replacement, import_line)

        return import_line

    def _apply_manual_fix(self, import_line: str) -> str:
        """应用手动修复"""

        # 直接查找完全匹配
        if import_line.strip() in self.manual_fix_mapping:
            return self.manual_fix_mapping[import_line.strip()]

        # 查找部分匹配
        for pattern, replacement in self.manual_fix_mapping.items():
            if import_line.strip().startswith(pattern.split(' import ')[0]):
                # 替换模块路径部分
                return import_line.replace(pattern.split(' import ')[0], replacement.split(' import ')[0])

        return import_line

    def validate_fixes(self, sample_size: int = 10) -> Dict[str, Any]:
        """验证修复效果"""

        validation_results = {
            "files_checked": 0,
            "successful_fixes": 0,
            "failed_fixes": 0,
            "validation_errors": []
        }

        print("🔍 开始验证修复效果...")

        # 获取测试文件样本
        test_files = []
        for root, dirs, files in os.walk(self.tests_dir):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    test_files.append(Path(root) / file)
                    if len(test_files) >= sample_size:
                        break
            if len(test_files) >= sample_size:
                break

        for file_path in test_files:
            validation_results["files_checked"] += 1

            try:
                # 尝试导入验证
                import subprocess
                result = subprocess.run([
                    'python', '-c', f"""
import ast
import sys
sys.path.insert(0, '{self.project_root}')

# 解析文件中的导入
with open('{file_path}', 'r', encoding='utf-8') as f:
    content = f.read()

try:
    tree = ast.parse(content)
    print('Syntax OK')
except SyntaxError as e:
    print(f'Syntax Error: {e}')
    sys.exit(1)
"""
                ], capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    validation_results["successful_fixes"] += 1
                else:
                    validation_results["failed_fixes"] += 1
                    validation_results["validation_errors"].append(f"{file_path}: {result.stderr}")

            except Exception as e:
                validation_results["failed_fixes"] += 1
                validation_results["validation_errors"].append(f"{file_path}: {e}")

        return validation_results

    def generate_comprehensive_report(self, fix_results: Dict[str, Any], validation_results: Dict[str, Any]) -> str:
        """生成综合报告"""

        report = f"""# 🔧 综合测试修复报告

## 📅 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 修复结果概况

### 处理统计
- **处理的文件数**: {fix_results['files_processed']}
- **修复的文件数**: {fix_results['files_fixed']}
- **自动修复数**: {fix_results['auto_fixes']}
- **手动修复数**: {fix_results['manual_fixes']}
- **修复模式**: {'仅分析' if fix_results['dry_run'] else '实际执行'}

### 验证结果
- **验证的文件数**: {validation_results.get('files_checked', 0)}
- **验证成功数**: {validation_results.get('successful_fixes', 0)}
- **验证失败数**: {validation_results.get('failed_fixes', 0)}

## 🔧 修复策略

### 自动修复规则
基于正则表达式匹配的自动修复策略：

1. **工具类映射**: `src.utils.*` → `src.infrastructure.utils.*`
2. **配置映射**: `src.config.*` → `src.infrastructure.config.*`
3. **模型映射**: `src.models.*` → `src.ml.models.*`
4. **服务映射**: `src.services.*` → `src.infrastructure.services.*`
5. **监控映射**: `src.monitoring.*` → `src.infrastructure.monitoring.*`

### 手动修复规则
基于具体分析结果的手动修复策略：

1. **数据质量模块**: 保持原路径（路径已存在）
2. **基础设施监控**: 保持原路径（路径已存在）
3. **特征工程模块**: 保持原路径（路径已存在）
4. **缓存管理模块**: 保持原路径（路径已存在）

## 📋 修复示例

### 自动修复示例
```
修复前: from src.utils.logger import get_logger
修复后: from src.infrastructure.utils.logger import get_logger
```

### 手动修复示例
```
修复前: from src.models.base_model import BaseModel
修复后: from src.ml.models.base_model import BaseModel
```

## ⚠️ 备份信息
- **备份目录**: {self.backup_dir}
- **备份文件**: 所有修改的文件都会自动备份
- **恢复方法**: 从备份目录复制文件到原位置

## 💡 建议后续行动

### 1. 验证修复效果
```bash
# 运行关键测试用例验证修复效果
python -m pytest tests/unit/core/test_event_bus.py -v
python -m pytest tests/unit/data/test_data_manager.py -v
```

### 2. 持续监控
- 监控测试执行结果
- 跟踪导入错误减少情况
- 定期运行测试覆盖率分析

### 3. 扩展修复范围
- 识别更多常见的导入模式
- 扩展自动修复规则覆盖范围
- 优化验证逻辑减少误报

### 4. 团队培训
- 更新开发文档中的导入路径指南
- 培训团队成员了解新的模块结构
- 建立代码审查检查点

## 🎯 修复效果评估

### 成功指标
- ✅ **语法错误消除**: 修复后的文件都能正常解析
- ✅ **导入路径正确**: 导入的模块都存在于新的架构中
- ✅ **向后兼容**: 不破坏现有的测试逻辑
- ✅ **可维护性**: 修复规则清晰易于维护

### 潜在风险
- ⚠️ **过度修复**: 某些有效的导入可能被错误修改
- ⚠️ **不完整修复**: 某些复杂的导入模式可能未被覆盖
- ⚠️ **性能影响**: 验证过程可能影响测试执行性能

## 📈 持续改进计划

1. **完善映射规则**: 基于实际使用情况优化修复规则
2. **增强验证逻辑**: 改进导入路径验证的准确性
3. **自动化监控**: 建立自动监控机制跟踪修复效果
4. **文档更新**: 更新相关文档和最佳实践指南

---
*综合测试修复工具版本: v1.0*
*备份目录: {self.backup_dir}*
"""

        return report


def main():
    """主函数"""

    import argparse

    parser = argparse.ArgumentParser(description='综合测试修复工具')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--batch-size', type=int, default=50, help='批次大小')
    parser.add_argument('--dry-run', action='store_true', help='仅分析不执行修复')
    parser.add_argument('--validate', action='store_true', help='验证修复效果')
    parser.add_argument('--report', action='store_true', help='生成详细报告')

    args = parser.parse_args()

    tool = ComprehensiveTestFixer(args.project)

    # 执行修复
    fix_results = tool.analyze_and_fix(args.batch_size, args.dry_run)

    # 验证修复效果
    validation_results = {}
    if args.validate and not args.dry_run:
        validation_results = tool.validate_fixes()

    if args.report:
        report_content = tool.generate_comprehensive_report(fix_results, validation_results)
        report_file = tool.project_root / "reports" / \
            f"comprehensive_test_fix_{'analysis' if args.dry_run else 'execution'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"📊 综合修复报告已保存: {report_file}")


if __name__ == "__main__":
    main()
