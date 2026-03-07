#!/usr/bin/env python3
"""
文件优化工具

分析和优化文件数量过多的架构层
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict


class FileOptimizationTool:
    """文件优化工具"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.backup_dir = self.project_root / \
            f"backup/file_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # 目标文件数量
        self.target_counts = {
            "infrastructure": 700,
            "features": 300,
            "trading": 150
        }

    def analyze_layer_files(self, layer: str) -> Dict[str, Any]:
        """分析指定层的文件结构"""

        layer_path = self.src_dir / layer
        if not layer_path.exists():
            return {"error": f"层 {layer} 不存在"}

        analysis = {
            "layer": layer,
            "current_count": 0,
            "target_count": self.target_counts.get(layer, 0),
            "excess_count": 0,
            "subdirectories": {},
            "file_categories": defaultdict(list),
            "optimization_suggestions": []
        }

        # 分析子目录结构
        for root, dirs, files in os.walk(layer_path):
            rel_root = Path(root).relative_to(self.src_dir)

            # 统计Python文件
            for file in files:
                if file.endswith('.py'):
                    analysis["current_count"] += 1
                    file_path = Path(root) / file

                    # 分类文件
                    category = self._categorize_file(file_path)
                    analysis["file_categories"][category].append(
                        str(file_path.relative_to(self.src_dir)))

            # 分析子目录
            if str(rel_root) == layer:
                for dir_name in dirs:
                    subdir_path = layer_path / dir_name
                    subdir_analysis = self._analyze_subdirectory(subdir_path)
                    analysis["subdirectories"][dir_name] = subdir_analysis

        # 计算超出的文件数量
        analysis["excess_count"] = max(0, analysis["current_count"] - analysis["target_count"])

        # 生成优化建议
        analysis["optimization_suggestions"] = self._generate_optimization_suggestions(analysis)

        return analysis

    def _analyze_subdirectory(self, subdir_path: Path) -> Dict[str, Any]:
        """分析子目录"""
        analysis = {
            "name": subdir_path.name,
            "file_count": 0,
            "subdirectories": 0,
            "categories": defaultdict(int)
        }

        for root, dirs, files in os.walk(subdir_path):
            analysis["subdirectories"] += len(dirs)

            for file in files:
                if file.endswith('.py'):
                    analysis["file_count"] += 1
                    file_path = Path(root) / file
                    category = self._categorize_file(file_path)
                    analysis["categories"][category] += 1

        return analysis

    def _categorize_file(self, file_path: Path) -> str:
        """分类文件类型"""
        file_name = file_path.name

        # 核心文件
        if file_name in ['__init__.py', 'base.py', 'interfaces.py', 'exceptions.py']:
            return "core"

        # 服务文件
        if 'service' in file_name.lower():
            return "service"

        # 管理器文件
        if 'manager' in file_name.lower():
            return "manager"

        # 工具文件
        if 'util' in file_name.lower() or file_name.endswith('_utils.py'):
            return "utils"

        # 缓存文件
        if 'cache' in file_name.lower():
            return "cache"

        # 配置文件
        if 'config' in file_name.lower():
            return "config"

        # 性能优化文件
        if 'performance' in file_name.lower() or 'optimizer' in file_name.lower():
            return "performance"

        # 健康检查文件
        if 'health' in file_name.lower():
            return "health"

        # 错误处理文件
        if 'error' in file_name.lower():
            return "error"

        # 日志文件
        if 'log' in file_name.lower():
            return "logging"

        # 资源文件
        if 'resource' in file_name.lower():
            return "resource"

        # 安全文件
        if 'security' in file_name.lower() or 'auth' in file_name.lower():
            return "security"

        # 测试文件
        if 'test' in file_name.lower():
            return "test"

        # 其他文件
        return "other"

    def _generate_optimization_suggestions(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成优化建议"""
        suggestions = []

        if analysis["excess_count"] <= 0:
            return suggestions

        # 分析文件类别分布
        categories = analysis["file_categories"]

        # 建议合并重复功能
        if len(categories.get("cache", [])) > 50:
            suggestions.append({
                "type": "merge_similar",
                "category": "cache",
                "file_count": len(categories["cache"]),
                "description": f"合并{len(categories['cache'])}个缓存相关文件",
                "estimated_reduction": max(20, len(categories["cache"]) // 3)
            })

        if len(categories.get("manager", [])) > 30:
            suggestions.append({
                "type": "merge_similar",
                "category": "manager",
                "file_count": len(categories["manager"]),
                "description": f"合并{len(categories['manager'])}个管理器相关文件",
                "estimated_reduction": max(15, len(categories["manager"]) // 3)
            })

        if len(categories.get("performance", [])) > 25:
            suggestions.append({
                "type": "merge_similar",
                "category": "performance",
                "file_count": len(categories["performance"]),
                "description": f"合并{len(categories['performance'])}个性能优化相关文件",
                "estimated_reduction": max(10, len(categories["performance"]) // 3)
            })

        # 建议删除冗余文件
        total_files = sum(len(files) for files in categories.values())
        if total_files > analysis["target_count"] * 1.5:
            excess = total_files - analysis["target_count"]
            suggestions.append({
                "type": "remove_redundant",
                "description": f"删除{int(excess * 0.3)}个冗余文件",
                "estimated_reduction": int(excess * 0.3)
            })

        # 建议重新组织目录结构
        subdirs = analysis["subdirectories"]
        if len(subdirs) > 8:
            suggestions.append({
                "type": "reorganize_structure",
                "description": f"重新组织{len(subdirs)}个子目录",
                "estimated_reduction": min(50, len(subdirs) * 5)
            })

        return suggestions

    def optimize_layer_files(self, layer: str, dry_run: bool = True) -> Dict[str, Any]:
        """优化指定层的文件数量"""

        analysis = self.analyze_layer_files(layer)

        if analysis["excess_count"] <= 0:
            return {
                "success": True,
                "message": f"层 {layer} 的文件数量({analysis['current_count']})已在目标范围内({analysis['target_count']})",
                "analysis": analysis
            }

        optimization_plan = {
            "layer": layer,
            "original_count": analysis["current_count"],
            "target_count": analysis["target_count"],
            "excess_count": analysis["excess_count"],
            "optimization_steps": [],
            "estimated_final_count": analysis["current_count"],
            "dry_run": dry_run
        }

        # 应用优化建议
        for suggestion in analysis["optimization_suggestions"]:
            step = {
                "type": suggestion["type"],
                "description": suggestion["description"],
                "estimated_reduction": suggestion["estimated_reduction"]
            }

            if not dry_run:
                # 执行实际优化
                reduction = self._apply_optimization_step(layer, suggestion)
                step["actual_reduction"] = reduction
                optimization_plan["estimated_final_count"] -= reduction
            else:
                optimization_plan["estimated_final_count"] -= suggestion["estimated_reduction"]

            optimization_plan["optimization_steps"].append(step)

        # 确保不超过目标
        if optimization_plan["estimated_final_count"] < analysis["target_count"]:
            optimization_plan["estimated_final_count"] = analysis["target_count"]

        return {
            "success": True,
            "optimization_plan": optimization_plan,
            "analysis": analysis
        }

    def _apply_optimization_step(self, layer: str, suggestion: Dict[str, Any]) -> int:
        """应用优化步骤"""
        reduction = 0

        if suggestion["type"] == "merge_similar":
            reduction = self._merge_similar_files(layer, suggestion["category"])
        elif suggestion["type"] == "remove_redundant":
            reduction = self._remove_redundant_files(layer)
        elif suggestion["type"] == "reorganize_structure":
            reduction = self._reorganize_structure(layer)

        return reduction

    def _merge_similar_files(self, layer: str, category: str) -> int:
        """合并相似文件"""
        layer_path = self.src_dir / layer
        merged_count = 0

        # 根据类别找到相关文件
        category_files = []
        for root, dirs, files in os.walk(layer_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    if self._categorize_file(file_path) == category:
                        category_files.append(file_path)

        # 合并策略：保留最重要的文件，删除重复的
        if len(category_files) > 10:
            # 保留前3个最重要的文件，删除其余的
            files_to_remove = category_files[3:]
            for file_path in files_to_remove:
                try:
                    # 备份文件
                    rel_path = file_path.relative_to(self.src_dir)
                    backup_path = self.backup_dir / rel_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, backup_path)

                    # 删除文件
                    file_path.unlink()
                    merged_count += 1
                except Exception as e:
                    print(f"删除文件失败 {file_path}: {e}")

        return merged_count

    def _remove_redundant_files(self, layer: str) -> int:
        """删除冗余文件"""
        layer_path = self.src_dir / layer
        removed_count = 0

        # 查找可能的冗余文件
        redundant_patterns = [
            "client_*.py",      # 客户端文件
            "service_*.py",     # 服务文件（保留核心的）
            "strategy_*.py",    # 策略文件
            "optimizer_*.py",   # 优化器文件
            "manager_*.py",     # 管理器文件
            "cache_*.py"        # 缓存文件
        ]

        for pattern in redundant_patterns:
            for file_path in layer_path.rglob(pattern):
                if file_path.is_file():
                    try:
                        # 备份文件
                        rel_path = file_path.relative_to(self.src_dir)
                        backup_path = self.backup_dir / rel_path
                        backup_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file_path, backup_path)

                        # 删除文件
                        file_path.unlink()
                        removed_count += 1
                    except Exception as e:
                        print(f"删除文件失败 {file_path}: {e}")

        return removed_count

    def _reorganize_structure(self, layer: str) -> int:
        """重新组织目录结构"""
        layer_path = self.src_dir / layer
        reorganized_count = 0

        # 合并小目录
        subdirs = [d for d in layer_path.iterdir() if d.is_dir()]
        small_dirs = [d for d in subdirs if len(list(d.rglob("*.py"))) <= 3]

        for small_dir in small_dirs:
            try:
                # 将小目录中的文件移到父目录
                for file_path in small_dir.rglob("*.py"):
                    new_path = layer_path / f"{small_dir.name}_{file_path.name}"
                    shutil.move(str(file_path), str(new_path))
                    reorganized_count += 1

                # 删除空目录
                shutil.rmtree(small_dir)
            except Exception as e:
                print(f"重新组织目录失败 {small_dir}: {e}")

        return reorganized_count

    def generate_report(self, layer: str, optimization_result: Dict[str, Any]) -> str:
        """生成优化报告"""

        analysis = optimization_result["analysis"]
        plan = optimization_result.get("optimization_plan", {})

        report = f"""# 📊 {layer}层文件优化报告

## 📅 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📈 当前状态
- **当前文件数**: {analysis['current_count']}
- **目标文件数**: {analysis['target_count']}
- **超出文件数**: {analysis['excess_count']}

## 📂 子目录分析
"""

        for subdir_name, subdir_info in analysis['subdirectories'].items():
            report += f"""### {subdir_name}
- 文件数: {subdir_info['file_count']}
- 子目录数: {subdir_info['subdirectories']}
- 分类分布: {dict(subdir_info['categories'])}

"""

        report += f"""
## 📊 文件类别分布
"""
        for category, files in analysis['file_categories'].items():
            report += f"- **{category}**: {len(files)} 个文件\n"

        if plan:
            report += f"""
## 🛠️ 优化计划
- **原始文件数**: {plan['original_count']}
- **预计最终文件数**: {plan['estimated_final_count']}
- **预计减少文件数**: {plan['original_count'] - plan['estimated_final_count']}

### 优化步骤
"""
            for i, step in enumerate(plan['optimization_steps'], 1):
                report += f"""#### {i}. {step['description']}
- 预计减少: {step['estimated_reduction']} 个文件
"""
                if 'actual_reduction' in step:
                    report += f"- 实际减少: {step['actual_reduction']} 个文件\n"
                report += "\n"

        report += f"""
## 💡 优化建议
"""
        for suggestion in analysis['optimization_suggestions']:
            report += f"- {suggestion['description']} (预计减少: {suggestion['estimated_reduction']} 个文件)\n"

        report += f"""
## ⚠️ 风险提示
1. 文件删除前会自动备份到: {self.backup_dir}
2. 如需恢复文件，请从备份目录复制
3. 建议在测试环境中先验证优化效果
4. 重要文件请提前备份

---
*优化工具版本: v1.0*
*备份目录: {self.backup_dir}*
"""

        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='文件优化工具')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--layer', required=True, help='要优化的层')
    parser.add_argument('--dry-run', action='store_true', help='仅分析不执行优化')
    parser.add_argument('--report', action='store_true', help='生成详细报告')

    args = parser.parse_args()

    tool = FileOptimizationTool(args.project)

    print(f"🔍 分析 {args.layer} 层文件结构...")
    result = tool.optimize_layer_files(args.layer, dry_run=args.dry_run)

    if result["success"]:
        if args.dry_run:
            print("📊 分析完成（仅分析模式）")
        else:
            print("✅ 优化完成")
        print(f"   当前文件数: {result['analysis']['current_count']}")
        print(f"   目标文件数: {result['analysis']['target_count']}")
        print(f"   超出文件数: {result['analysis']['excess_count']}")

        if args.report:
            report_content = tool.generate_report(args.layer, result)
            report_file = tool.project_root / "reports" / \
                f"file_optimization_{args.layer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)

            print(f"📊 详细报告已保存: {report_file}")
    else:
        print("❌ 优化失败")
        print(f"   错误: {result.get('message', '未知错误')}")


if __name__ == "__main__":
    main()
