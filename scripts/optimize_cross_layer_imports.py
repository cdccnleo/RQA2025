#!/usr/bin/env python3
"""
优化跨层级导入脚本

优化基础设施层中的跨层级导入，减少不必要的依赖关系
"""

import re
from pathlib import Path
from typing import Dict, List, Any


class CrossLayerImportOptimizer:
    """跨层级导入优化器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.infrastructure_dir = self.project_root / "src" / "infrastructure"

        # 定义合理的跨层级导入
        self.allowed_cross_layer_imports = {
            # 基础设施层可以导入的层级
            "src.core": ["base.py", "interfaces.py", "layer_interfaces.py"],
            "src.utils": ["logger.py", "common.py", "helpers.py"],
            "src.data": ["interfaces.py", "base.py"],
            "src.engine": ["interfaces.py", "base.py"],
            "src.features": ["interfaces.py"],
            "src.ml": ["interfaces.py"],
            "src.trading": ["interfaces.py"],
            "src.risk": ["interfaces.py"],
            "src.monitoring": ["interfaces.py"],
        }

        # 不合理的跨层级导入模式
        self.unreasonable_patterns = [
            r"from src\.engine\..*import.*",  # 基础设施层不应该直接导入引擎层具体实现
            r"from src\.data\..*import.*",    # 基础设施层不应该直接导入数据层具体实现
            r"from src\.ml\..*import.*",      # 基础设施层不应该直接导入ML层具体实现
            r"from src\.trading\..*import.*",  # 基础设施层不应该直接导入交易层具体实现
            r"from src\.risk\..*import.*",    # 基础设施层不应该直接导入风控层具体实现
        ]

    def find_cross_layer_imports(self) -> List[Dict[str, Any]]:
        """查找跨层级导入"""
        cross_layer_imports = []

        for py_file in self.infrastructure_dir.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = content.split('\n')
                for i, line in enumerate(lines):
                    line = line.strip()

                    # 检查import语句
                    if line.startswith('import ') or line.startswith('from '):
                        if 'src.' in line and not line.startswith('from src.infrastructure'):
                            # 这是跨层级导入
                            import_info = self._analyze_import(line)

                            # 检查是否是合理的导入
                            is_reasonable = self._is_reasonable_import(line, py_file)

                            cross_layer_imports.append({
                                "file": str(py_file.relative_to(self.project_root)),
                                "line": i + 1,
                                "import": line,
                                "target_layer": import_info.get("layer", "unknown"),
                                "target_module": import_info.get("module", "unknown"),
                                "is_reasonable": is_reasonable,
                                "reason": "合理" if is_reasonable else "不合理"
                            })

            except Exception as e:
                print(f"  检查文件 {py_file} 时出错: {e}")

        return cross_layer_imports

    def _analyze_import(self, import_line: str) -> Dict[str, str]:
        """分析导入语句"""
        import_info = {}

        # 解析import语句
        if import_line.startswith('from '):
            # from src.layer.module import something
            match = re.search(r'from (src\.([^.]+)\.([^.\s]+))', import_line)
            if match:
                import_info["full_path"] = match.group(1)
                import_info["layer"] = match.group(2)
                import_info["module"] = match.group(3)
        elif import_line.startswith('import '):
            # import src.layer.module
            match = re.search(r'import (src\.([^.]+)\.([^.\s]+))', import_line)
            if match:
                import_info["full_path"] = match.group(1)
                import_info["layer"] = match.group(2)
                import_info["module"] = match.group(3)

        return import_info

    def _is_reasonable_import(self, import_line: str, file_path: Path) -> bool:
        """检查导入是否合理"""
        import_info = self._analyze_import(import_line)

        # 检查是否在允许的导入列表中
        target_layer = import_info.get("layer", "")
        target_module = import_info.get("module", "")

        if target_layer in self.allowed_cross_layer_imports:
            allowed_modules = self.allowed_cross_layer_imports[target_layer]
            if target_module in allowed_modules:
                return True

        # 检查是否匹配不合理的模式
        for pattern in self.unreasonable_patterns:
            if re.search(pattern, import_line):
                return False

        # 检查是否是基础设施层特有的合理导入
        # 基础设施层的web模块可以导入其他层的接口
        if "web" in str(file_path) or "api" in str(file_path):
            if "interfaces" in import_line or "base" in import_line:
                return True

        # 检查是否是监控相关的合理导入
        if "monitor" in str(file_path).lower():
            if "interfaces" in import_line:
                return True

        return False

    def optimize_imports(self, cross_layer_imports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """优化跨层级导入"""
        optimization_results = {
            "total_imports": len(cross_layer_imports),
            "reasonable_imports": 0,
            "unreasonable_imports": 0,
            "optimized_imports": 0,
            "optimization_details": []
        }

        for import_info in cross_layer_imports:
            if import_info["is_reasonable"]:
                optimization_results["reasonable_imports"] += 1
            else:
                optimization_results["unreasonable_imports"] += 1

                # 尝试优化不合理的导入
                optimization = self._optimize_single_import(import_info)
                if optimization["optimized"]:
                    optimization_results["optimized_imports"] += 1
                    optimization_results["optimization_details"].append(optimization)

        return optimization_results

    def _optimize_single_import(self, import_info: Dict[str, Any]) -> Dict[str, Any]:
        """优化单个导入"""
        file_path = self.project_root / import_info["file"]
        line_number = import_info["line"]
        original_import = import_info["import"]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.read().split('\n')

            # 根据不同的不合理导入类型进行优化
            optimized_import = self._get_optimized_import(original_import, file_path)

            if optimized_import and optimized_import != original_import:
                lines[line_number - 1] = optimized_import

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))

                return {
                    "file": import_info["file"],
                    "original": original_import,
                    "optimized": optimized_import,
                    "optimized": True,
                    "reason": "替换为更合理的导入"
                }

        except Exception as e:
            return {
                "file": import_info["file"],
                "original": original_import,
                "optimized": original_import,
                "optimized": False,
                "reason": f"优化失败: {e}"
            }

        return {
            "file": import_info["file"],
            "original": original_import,
            "optimized": original_import,
            "optimized": False,
            "reason": "无法优化"
        }

    def _get_optimized_import(self, original_import: str, file_path: Path) -> str:
        """获取优化的导入语句"""
        # 如果是导入具体实现类，改为导入接口
        if "from src.engine" in original_import and "import" in original_import:
            return original_import.replace("from src.engine", "from src.engine.interfaces import")

        if "from src.data" in original_import and "import" in original_import:
            return original_import.replace("from src.data", "from src.data.interfaces import")

        if "from src.ml" in original_import and "import" in original_import:
            return original_import.replace("from src.ml", "from src.ml.interfaces import")

        if "from src.trading" in original_import and "import" in original_import:
            return original_import.replace("from src.trading", "from src.trading.interfaces import")

        if "from src.risk" in original_import and "import" in original_import:
            return original_import.replace("from src.risk", "from src.risk.interfaces import")

        # 如果是基础设施层内部的跨模块导入，改为相对导入
        if "from src.infrastructure" in original_import and "config" in original_import:
            if "utils" in str(file_path):
                return original_import.replace("from src.infrastructure.config", "from ..config")

        return original_import

    def generate_optimization_report(self, cross_layer_imports: List[Dict[str, Any]], optimization_results: Dict[str, Any]) -> str:
        """生成优化报告"""
        import datetime

        report = f"""# 跨层级导入优化报告

## 📊 优化概览

**优化时间**: {datetime.datetime.now().isoformat()}
**总导入数**: {optimization_results['total_imports']} 个
**合理导入**: {optimization_results['reasonable_imports']} 个
**不合理导入**: {optimization_results['unreasonable_imports']} 个
**已优化**: {optimization_results['optimized_imports']} 个

---

## 🔍 跨层级导入分析

"""

        # 按合理性分组显示导入
        reasonable_imports = [imp for imp in cross_layer_imports if imp["is_reasonable"]]
        unreasonable_imports = [imp for imp in cross_layer_imports if not imp["is_reasonable"]]

        report += "### 合理的跨层级导入\n\n"
        if reasonable_imports:
            for imp in reasonable_imports[:10]:  # 只显示前10个
                report += f"- `{imp['file']}` → {imp['import']}\n"
            if len(reasonable_imports) > 10:
                report += f"- ... 还有 {len(reasonable_imports) - 10} 个合理导入\n"
        else:
            report += "无合理跨层级导入\n"

        report += f"""

### 不合理的跨层级导入
"""

        if unreasonable_imports:
            for imp in unreasonable_imports[:10]:  # 只显示前10个
                report += f"- `{imp['file']}` → {imp['import']}\n"
            if len(unreasonable_imports) > 10:
                report += f"- ... 还有 {len(unreasonable_imports) - 10} 个不合理导入\n"
        else:
            report += "无不合理跨层级导入\n"

        # 显示优化详情
        if optimization_results['optimization_details']:
            report += f"""

## ⚡ 优化详情

### 已完成的优化
"""
            for opt in optimization_results['optimization_details']:
                report += f"#### {opt['file']}\n"
                report += f"- **原导入**: {opt['original']}\n"
                report += f"- **优化后**: {opt['optimized']}\n"
                report += f"- **原因**: {opt['reason']}\n\n"

        # 优化建议
        report += f"""

## 💡 优化建议

### 导入合理性评估
- **合理导入比例**: {optimization_results['reasonable_imports'] / max(optimization_results['total_imports'], 1) * 100:.1f}% ({optimization_results['reasonable_imports']}/{optimization_results['total_imports']})
- **优化覆盖率**: {optimization_results['optimized_imports'] / max(optimization_results['unreasonable_imports'], 1) * 100:.1f}% ({optimization_results['optimized_imports']}/{optimization_results['unreasonable_imports']})

### 进一步优化建议

1. **接口导入**: 优先使用接口导入而不是具体实现导入
   ```python
   # 推荐
   from src.engine.interfaces import IEngineComponent

   # 避免
   from src.engine.core import EngineCore
   ```

2. **依赖注入**: 使用依赖注入模式减少直接依赖
   ```python
   # 推荐
   def __init__(self, engine: IEngineComponent):
       self.engine = engine
   ```

3. **相对导入**: 在同一层级内使用相对导入
   ```python
   # 推荐
   from ..config import ConfigManager

   # 避免
   from src.infrastructure.config import ConfigManager
   ```

4. **抽象层**: 通过抽象层隔离跨层级依赖
   ```python
   # 在基础设施层创建适配器
   class EngineAdapter:
       def __init__(self, engine: IEngineComponent):
           self.engine = engine
   ```

---

**优化工具**: scripts/optimize_cross_layer_imports.py
**优化标准**: 基于架构分层原则
**优化状态**: ✅ 完成
"""

        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='跨层级导入优化工具')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--output', help='输出报告文件')
    parser.add_argument('--dry-run', action='store_true', help='仅分析不执行优化')

    args = parser.parse_args()

    optimizer = CrossLayerImportOptimizer(args.project)

    # 查找跨层级导入
    print("🔍 查找跨层级导入...")
    cross_layer_imports = optimizer.find_cross_layer_imports()
    print(f"  发现 {len(cross_layer_imports)} 个跨层级导入")

    if args.dry_run:
        print("🔍 干运行模式 - 仅分析不执行优化")
        optimization_results = optimizer.optimize_imports([])  # 空列表用于计算统计
    else:
        print("⚡ 执行导入优化...")
        optimization_results = optimizer.optimize_imports(cross_layer_imports)

    print(f"  合理导入: {optimization_results['reasonable_imports']} 个")
    print(f"  不合理导入: {optimization_results['unreasonable_imports']} 个")
    print(f"  已优化: {optimization_results['optimized_imports']} 个")

    # 生成报告
    report = optimizer.generate_optimization_report(cross_layer_imports, optimization_results)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
    else:
        print(report)


if __name__ == "__main__":
    main()
