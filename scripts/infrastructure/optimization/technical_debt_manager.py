#!/usr/bin/env python3
"""
RQA2025 技术债务管理脚本

功能：
1. 自动识别技术债务
2. 跟踪债务解决进度
3. 生成债务报告
4. 管理债务优先级
"""

import sys
import json
import datetime
from pathlib import Path
from typing import Dict, List
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TechnicalDebtManager:
    """技术债务管理器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.debt_file = self.project_root / "docs" / "technical_debt.json"
        self.debt_file.parent.mkdir(exist_ok=True)

        # 债务类型定义
        self.debt_types = {
            'coverage_gap': '测试覆盖率不足',
            'test_failure': '测试失败',
            'missing_module': '模块缺失',
            'import_error': '导入错误',
            'dependency_conflict': '依赖冲突',
            'performance_issue': '性能问题',
            'code_quality': '代码质量问题',
            'documentation': '文档缺失'
        }

        # 优先级定义
        self.priorities = {
            'critical': '严重',
            'high': '高',
            'medium': '中',
            'low': '低'
        }

        # 加载现有债务
        self.debts = self.load_debts()

    def load_debts(self) -> List[Dict]:
        """加载现有技术债务"""
        if self.debt_file.exists():
            try:
                with open(self.debt_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载技术债务文件失败: {e}")
                return []
        return []

    def save_debts(self):
        """保存技术债务"""
        try:
            with open(self.debt_file, 'w', encoding='utf-8') as f:
                json.dump(self.debts, f, indent=2, ensure_ascii=False)
            logger.info(f"技术债务已保存到: {self.debt_file}")
        except Exception as e:
            logger.error(f"保存技术债务失败: {e}")

    def add_debt(self, debt: Dict):
        """添加技术债务"""
        # 生成唯一ID
        debt['id'] = f"TD-{datetime.datetime.now().strftime('%Y%m%d')}-{len(self.debts)+1:03d}"
        debt['created_at'] = datetime.datetime.now().isoformat()
        debt['status'] = 'open'

        self.debts.append(debt)
        self.save_debts()

        logger.info(f"添加技术债务: {debt['id']} - {debt['description']}")

    def update_debt(self, debt_id: str, updates: Dict):
        """更新技术债务"""
        for debt in self.debts:
            if debt['id'] == debt_id:
                debt.update(updates)
                debt['updated_at'] = datetime.datetime.now().isoformat()
                self.save_debts()
                logger.info(f"更新技术债务: {debt_id}")
                return True
        return False

    def close_debt(self, debt_id: str, resolution: str = ""):
        """关闭技术债务"""
        return self.update_debt(debt_id, {
            'status': 'closed',
            'resolution': resolution,
            'closed_at': datetime.datetime.now().isoformat()
        })

    def get_debts_by_layer(self, layer: str) -> List[Dict]:
        """获取指定层的技术债务"""
        return [debt for debt in self.debts if debt.get('layer') == layer]

    def get_debts_by_priority(self, priority: str) -> List[Dict]:
        """获取指定优先级的技术债务"""
        return [debt for debt in self.debts if debt.get('priority') == priority]

    def get_open_debts(self) -> List[Dict]:
        """获取开放的技术债务"""
        return [debt for debt in self.debts if debt.get('status') == 'open']

    def identify_coverage_debts(self, layer_results: List[Dict]) -> List[Dict]:
        """识别覆盖率相关的技术债务"""
        debts = []

        for result in layer_results:
            layer_name = result['layer']
            coverage_result = result.get('coverage_result', {})
            current_coverage = coverage_result.get('coverage', 0)

            # 根据层设置目标覆盖率
            target_coverage = {
                'infrastructure': 90,
                'data': 80,
                'features': 80,
                'models': 80,
                'trading': 80,
                'backtest': 80
            }.get(layer_name, 80)

            if current_coverage < target_coverage:
                debts.append({
                    'type': 'coverage_gap',
                    'description': f'{layer_name}层覆盖率{current_coverage}%低于目标{target_coverage}%',
                    'priority': 'high' if layer_name in ['infrastructure', 'data'] else 'medium',
                    'layer': layer_name,
                    'current_coverage': current_coverage,
                    'target_coverage': target_coverage
                })

        return debts

    def identify_test_failure_debts(self, layer_results: List[Dict]) -> List[Dict]:
        """识别测试失败相关的技术债务"""
        debts = []

        for result in layer_results:
            layer_name = result['layer']
            test_result = result.get('test_result', {})
            failed = test_result.get('failed', 0)
            error = test_result.get('error', 0)

            if failed > 0 or error > 0:
                debts.append({
                    'type': 'test_failure',
                    'description': f'{layer_name}层有{failed}个失败测试，{error}个错误测试',
                    'priority': 'high',
                    'layer': layer_name,
                    'failed_tests': failed,
                    'error_tests': error
                })

        return debts

    def identify_missing_module_debts(self) -> List[Dict]:
        """识别缺失模块的技术债务"""
        debts = []

        # 检查缺失的模块
        missing_modules = [
            {
                'layer': 'features',
                'module': 'src/features/processors/technical',
                'description': '技术指标处理器模块缺失',
                'priority': 'high'
            },
            {
                'layer': 'data',
                'module': 'src/data/adapters',
                'description': '数据适配器模块测试缺失',
                'priority': 'high'
            },
            {
                'layer': 'data',
                'module': 'src/data/cache',
                'description': '数据缓存模块测试缺失',
                'priority': 'medium'
            }
        ]

        for module in missing_modules:
            module_path = self.project_root / module['module']
            if not module_path.exists():
                debts.append({
                    'type': 'missing_module',
                    'description': module['description'],
                    'priority': module['priority'],
                    'layer': module['layer'],
                    'module_path': module['module']
                })

        return debts

    def generate_debt_report(self) -> str:
        """生成技术债务报告"""
        report = []
        report.append("# RQA2025 技术债务报告")
        report.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 总体统计
        total_debts = len(self.debts)
        open_debts = len(self.get_open_debts())
        closed_debts = total_debts - open_debts

        report.append("## 总体统计")
        report.append(f"- 总债务数: {total_debts}")
        report.append(f"- 开放债务: {open_debts}")
        report.append(f"- 已解决债务: {closed_debts}")
        report.append(
            f"- 解决率: {closed_debts/total_debts*100:.1f}%" if total_debts > 0 else "- 解决率: 0%")
        report.append("")

        # 按优先级统计
        report.append("## 按优先级统计")
        for priority in ['critical', 'high', 'medium', 'low']:
            priority_debts = self.get_debts_by_priority(priority)
            open_priority_debts = [d for d in priority_debts if d.get('status') == 'open']
            report.append(f"- {self.priorities[priority]}优先级: {len(open_priority_debts)}个开放债务")
        report.append("")

        # 按层统计
        report.append("## 按层统计")
        layers = ['infrastructure', 'data', 'features', 'models', 'trading', 'backtest']
        for layer in layers:
            layer_debts = self.get_debts_by_layer(layer)
            open_layer_debts = [d for d in layer_debts if d.get('status') == 'open']
            report.append(f"- {layer.capitalize()}层: {len(open_layer_debts)}个开放债务")
        report.append("")

        # 开放债务详情
        open_debts = self.get_open_debts()
        if open_debts:
            report.append("## 开放债务详情")
            for debt in open_debts:
                priority_emoji = {
                    'critical': '🔴',
                    'high': '🟠',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(debt.get('priority', 'medium'), '🟡')

                report.append(f"### {priority_emoji} {debt['id']}")
                report.append(f"- **描述**: {debt['description']}")
                report.append(f"- **类型**: {self.debt_types.get(debt.get('type', 'unknown'), '未知')}")
                report.append(
                    f"- **优先级**: {self.priorities.get(debt.get('priority', 'medium'), '中')}")
                report.append(f"- **层**: {debt.get('layer', '未知')}")
                report.append(f"- **创建时间**: {debt.get('created_at', '未知')}")
                report.append("")

        return "\n".join(report)

    def save_debt_report(self):
        """保存技术债务报告"""
        report_content = self.generate_debt_report()
        report_file = self.project_root / "docs" / "technical_debt_report.md"

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"技术债务报告已保存: {report_file}")
        except Exception as e:
            logger.error(f"保存技术债务报告失败: {e}")

    def auto_identify_debts(self, layer_results: List[Dict]):
        """自动识别技术债务"""
        logger.info("开始自动识别技术债务...")

        # 识别覆盖率债务
        coverage_debts = self.identify_coverage_debts(layer_results)
        for debt in coverage_debts:
            self.add_debt(debt)

        # 识别测试失败债务
        test_failure_debts = self.identify_test_failure_debts(layer_results)
        for debt in test_failure_debts:
            self.add_debt(debt)

        # 识别缺失模块债务
        missing_module_debts = self.identify_missing_module_debts()
        for debt in missing_module_debts:
            self.add_debt(debt)

        logger.info(
            f"自动识别完成，新增 {len(coverage_debts) + len(test_failure_debts) + len(missing_module_debts)} 个技术债务")


def main():
    """主函数"""
    manager = TechnicalDebtManager()

    # 示例：添加一些技术债务
    if len(manager.debts) == 0:
        logger.info("添加示例技术债务...")

        # 添加覆盖率债务
        manager.add_debt({
            'type': 'coverage_gap',
            'description': '数据层覆盖率不足，当前11.97%，目标80%',
            'priority': 'high',
            'layer': 'data',
            'current_coverage': 11.97,
            'target_coverage': 80
        })

        # 添加测试失败债务
        manager.add_debt({
            'type': 'test_failure',
            'description': '特征层存在huggingface依赖问题',
            'priority': 'high',
            'layer': 'features',
            'failed_tests': 5,
            'error_tests': 2
        })

        # 添加缺失模块债务
        manager.add_debt({
            'type': 'missing_module',
            'description': '技术指标处理器模块缺失',
            'priority': 'high',
            'layer': 'features',
            'module_path': 'src/features/processors/technical'
        })

    # 生成报告
    manager.save_debt_report()

    print("✅ 技术债务管理完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
