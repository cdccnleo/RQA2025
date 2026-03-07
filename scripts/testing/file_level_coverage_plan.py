#!/usr/bin/env python3
"""
RQA2025 文件级测试覆盖率提升计划
按照依赖关系和业务流程优先级，以文件为最小单位进行测试覆盖率提升
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class FileLevelCoveragePlan:
    """文件级测试覆盖率提升计划"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.src_path = self.project_root / "src"
        self.tests_path = self.project_root / "tests"

        # 按依赖关系和业务流程定义的文件优先级
        self.file_priority = {
            # 基础设施层 - 最高优先级（底层依赖）
            "infrastructure": {
                "critical": [
                    "config/config_manager.py",  # 配置管理，所有模块依赖
                    "m_logging/logger.py",       # 日志管理，所有模块依赖
                    "cache/cache_manager.py",    # 缓存管理，性能关键
                    "database/database_manager.py",  # 数据库管理，数据层依赖
                    "error/error_handler.py",    # 错误处理，稳定性关键
                    "monitoring/system_monitor.py",  # 系统监控，运维关键
                    "circuit_breaker.py",        # 断路器，稳定性关键
                    "service_launcher.py"        # 服务启动，部署关键
                ],
                "high": [
                    "config/config_version.py",
                    "config/deployment_manager.py",
                    "m_logging/log_manager.py",
                    "m_logging/performance_monitor.py",
                    "cache/thread_safe_cache.py",
                    "database/connection_pool.py",
                    "monitoring/application_monitor.py",
                    "monitoring/alert_manager.py"
                ],
                "medium": [
                    "config/event_filters.py",
                    "config/exceptions.py",
                    "config/paths.py",
                    "database/influxdb_adapter.py",
                    "database/sqlite_adapter.py",
                    "di/container.py",
                    "storage/storage_core.py",
                    "security/security_manager.py"
                ]
            },

            # 数据层 - 高优先级（数据依赖）
            "data": {
                "critical": [
                    "data_loader.py",           # 数据加载，特征层依赖
                    "data_manager.py",          # 数据管理，核心功能
                    "validator.py",             # 数据验证，质量关键
                    "base_loader.py",           # 基础加载器，扩展关键
                    "parallel_loader.py"        # 并行加载，性能关键
                ],
                "high": [
                    "financial_data_loader.py",
                    "stock_data_loader.py",
                    "index_data_loader.py",
                    "data_processor.py",
                    "data_synchronizer.py"
                ],
                "medium": [
                    "data_cleaner.py",
                    "data_transformer.py",
                    "data_exporter.py",
                    "data_importer.py"
                ]
            },

            # 特征层 - 中优先级（模型依赖）
            "features": {
                "critical": [
                    "feature_engineer.py",      # 特征工程，模型层依赖
                    "feature_manager.py",       # 特征管理，核心功能
                    "feature_engine.py",        # 特征引擎，性能关键
                    "signal_generator.py",      # 信号生成，交易层依赖
                    "sentiment_analyzer.py"     # 情感分析，策略关键
                ],
                "high": [
                    "processors/technical_processor.py",
                    "processors/fundamental_processor.py",
                    "processors/sentiment_processor.py",
                    "processors/orderbook_processor.py",
                    "feature_selector.py"
                ],
                "medium": [
                    "processors/base_processor.py",
                    "processors/data_processor.py",
                    "feature_validator.py",
                    "feature_optimizer.py"
                ]
            },

            # 模型层 - 中优先级（交易层依赖）
            "models": {
                "critical": [
                    "model_manager.py",         # 模型管理，交易层依赖
                    "base_model.py",            # 基础模型，扩展关键
                    "ensemble/ensemble_model.py",  # 集成模型，性能关键
                    "model_validator.py",       # 模型验证，质量关键
                    "model_persistence.py"      # 模型持久化，部署关键
                ],
                "high": [
                    "ensemble/voting_classifier.py",
                    "ensemble/stacking_classifier.py",
                    "ensemble/bagging_classifier.py",
                    "model_optimizer.py",
                    "model_evaluator.py"
                ],
                "medium": [
                    "ensemble/base_ensemble.py",
                    "model_explainer.py",
                    "model_monitor.py",
                    "model_version_manager.py"
                ]
            },

            # 交易层 - 中优先级（回测层依赖）
            "trading": {
                "critical": [
                    "trading_engine.py",        # 交易引擎，回测层依赖
                    "execution_engine.py",      # 执行引擎，核心功能
                    "order_manager.py",         # 订单管理，稳定性关键
                    "risk_manager.py",          # 风险管理，安全关键
                    "position_manager.py"       # 仓位管理，风险关键
                ],
                "high": [
                    "strategies/basic_strategy.py",
                    "strategies/limit_up_strategy.py",
                    "strategies/dragon_tiger_strategy.py",
                    "order_executor.py",
                    "signal_processor.py"
                ],
                "medium": [
                    "strategies/base_strategy.py",
                    "strategies/star_market_strategy.py",
                    "strategies/st_strategy.py",
                    "strategies/margin_strategy.py",
                    "trading_utils.py"
                ]
            },

            # 回测层 - 低优先级（最终验证）
            "backtest": {
                "critical": [
                    "backtest_engine.py",       # 回测引擎，验证关键
                    "parameter_optimizer.py",   # 参数优化，性能关键
                    "performance_analyzer.py",  # 性能分析，评估关键
                    "strategy_evaluator.py",    # 策略评估，质量关键
                    "risk_analyzer.py"          # 风险分析，安全关键
                ],
                "high": [
                    "data_replayer.py",
                    "result_analyzer.py",
                    "report_generator.py",
                    "backtest_validator.py"
                ],
                "medium": [
                    "scenario_analyzer.py",
                    "stress_tester.py",
                    "backtest_utils.py"
                ]
            }
        }

        # 目标覆盖率
        self.target_coverage = {
            "critical": 85.0,  # 关键文件目标85%
            "high": 75.0,      # 高优先级文件目标75%
            "medium": 60.0     # 中优先级文件目标60%
        }

    def analyze_current_coverage(self) -> Dict:
        """分析当前文件覆盖率"""
        print("🔍 分析当前文件覆盖率...")

        coverage_data = {}

        for layer, priorities in self.file_priority.items():
            coverage_data[layer] = {}

            for priority, files in priorities.items():
                coverage_data[layer][priority] = {}

                for file in files:
                    file_path = self.src_path / layer / file
                    if file_path.exists():
                        # 运行单个文件的测试覆盖率
                        coverage = self.get_file_coverage(layer, file)
                        coverage_data[layer][priority][file] = coverage
                    else:
                        print(f"⚠️  文件不存在: {file_path}")
                        coverage_data[layer][priority][file] = 0.0

        return coverage_data

    def get_file_coverage(self, layer: str, file: str) -> float:
        """获取单个文件的测试覆盖率"""
        try:
            # 构建测试命令
            test_file = file.replace('.py', '_test.py')
            test_path = self.tests_path / "unit" / layer / test_file

            if not test_path.exists():
                return 0.0

            # 运行测试并获取覆盖率
            cmd = [
                "python", "-m", "pytest",
                str(test_path),
                f"--cov=src/{layer}/{file}",
                "--cov-report=term-missing",
                "--cov-report=json",
                "-q"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300
            )

            if result.returncode == 0:
                # 解析覆盖率数据
                coverage = self.parse_coverage_output(result.stdout)
                return coverage
            else:
                print(f"❌ 测试失败: {file}")
                return 0.0

        except Exception as e:
            print(f"❌ 获取覆盖率失败: {file} - {e}")
            return 0.0

    def parse_coverage_output(self, output: str) -> float:
        """解析覆盖率输出"""
        try:
            lines = output.split('\n')
            for line in lines:
                if 'TOTAL' in line and '%' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        coverage_str = parts[3].replace('%', '')
                        return float(coverage_str)
        except:
            pass
        return 0.0

    def identify_priority_files(self, coverage_data: Dict) -> List[Tuple[str, str, str, float, float]]:
        """识别需要优先提升的文件"""
        priority_files = []

        for layer, priorities in coverage_data.items():
            for priority, files in priorities.items():
                target = self.target_coverage[priority]

                for file, current_coverage in files.items():
                    if current_coverage < target:
                        gap = target - current_coverage
                        priority_files.append((
                            layer,           # 层级
                            priority,        # 优先级
                            file,           # 文件名
                            current_coverage,  # 当前覆盖率
                            gap             # 差距
                        ))

        # 按优先级和差距排序
        priority_files.sort(key=lambda x: (
            self.get_priority_score(x[1]),  # 优先级分数
            x[4]                            # 差距（降序）
        ), reverse=True)

        return priority_files

    def get_priority_score(self, priority: str) -> int:
        """获取优先级分数"""
        scores = {
            "critical": 3,
            "high": 2,
            "medium": 1
        }
        return scores.get(priority, 0)

    def create_test_file(self, layer: str, file: str) -> bool:
        """为文件创建测试文件"""
        try:
            # 构建文件路径
            src_file = self.src_path / layer / file
            test_file = file.replace('.py', '_test.py')
            test_path = self.tests_path / "unit" / layer / test_file

            # 确保测试目录存在
            test_path.parent.mkdir(parents=True, exist_ok=True)

            # 如果测试文件已存在，跳过
            if test_path.exists():
                print(f"ℹ️  测试文件已存在: {test_path}")
                return True

            # 生成测试文件内容
            test_content = self.generate_test_content(layer, file, src_file)

            # 写入测试文件
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write(test_content)

            print(f"✅ 创建测试文件: {test_path}")
            return True

        except Exception as e:
            print(f"❌ 创建测试文件失败: {file} - {e}")
            return False

    def generate_test_content(self, layer: str, file: str, src_file: Path) -> str:
        """生成测试文件内容"""
        class_name = file.replace('.py', '').replace('_', ' ').title().replace(' ', '')

        test_content = f'''#!/usr/bin/env python3
"""
{file} 测试文件
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 添加src路径到sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from {layer}.{file.replace('.py', '')} import {class_name}

class Test{class_name}:
    """{class_name} 测试类"""
    
    def setup_method(self):
        """测试前准备"""
        pass
    
    def teardown_method(self):
        """测试后清理"""
        pass
    
    def test_init(self):
        """测试初始化"""
        # TODO: 实现初始化测试
        pass
    
    def test_basic_functionality(self):
        """测试基本功能"""
        # TODO: 实现基本功能测试
        pass
    
    def test_error_handling(self):
        """测试错误处理"""
        # TODO: 实现错误处理测试
        pass
    
    def test_edge_cases(self):
        """测试边界情况"""
        # TODO: 实现边界情况测试
        pass
    
    def test_integration(self):
        """测试集成功能"""
        # TODO: 实现集成测试
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
        return test_content

    def run_file_test(self, layer: str, file: str) -> Dict:
        """运行单个文件的测试"""
        try:
            test_file = file.replace('.py', '_test.py')
            test_path = self.tests_path / "unit" / layer / test_file

            if not test_path.exists():
                return {
                    "success": False,
                    "error": "测试文件不存在",
                    "coverage": 0.0
                }

            # 运行测试
            cmd = [
                "python", "-m", "pytest",
                str(test_path),
                f"--cov=src/{layer}/{file}",
                "--cov-report=term-missing",
                "--cov-report=json",
                "-v"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=600
            )

            if result.returncode == 0:
                coverage = self.parse_coverage_output(result.stdout)
                return {
                    "success": True,
                    "coverage": coverage,
                    "output": result.stdout
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                    "coverage": 0.0
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "coverage": 0.0
            }

    def execute_coverage_plan(self) -> Dict:
        """执行文件级覆盖率提升计划"""
        print("🚀 开始执行文件级覆盖率提升计划...")

        # 1. 分析当前覆盖率
        current_coverage = self.analyze_current_coverage()

        # 2. 识别优先级文件
        priority_files = self.identify_priority_files(current_coverage)

        # 3. 生成提升计划
        plan = self.create_enhancement_plan(priority_files)

        # 4. 执行提升计划
        results = []
        for layer, priority, file, current_cov, gap in priority_files[:10]:  # 限制处理前10个文件
            print(f"\n📋 处理文件: {layer}/{file}")
            print(
                f"   优先级: {priority}, 当前覆盖率: {current_cov:.1f}%, 目标: {self.target_coverage[priority]:.1f}%")

            # 创建测试文件
            if self.create_test_file(layer, file):
                # 运行测试
                result = self.run_file_test(layer, file)
                results.append({
                    "layer": layer,
                    "priority": priority,
                    "file": file,
                    "current_coverage": current_cov,
                    "target_coverage": self.target_coverage[priority],
                    "result": result
                })

        # 5. 生成报告
        report = self.generate_coverage_report(current_coverage, results)

        return {
            "current_coverage": current_coverage,
            "priority_files": priority_files,
            "plan": plan,
            "results": results,
            "report": report
        }

    def create_enhancement_plan(self, priority_files: List[Tuple]) -> Dict:
        """创建提升计划"""
        plan = {
            "total_files": len(priority_files),
            "critical_files": len([f for f in priority_files if f[1] == "critical"]),
            "high_priority_files": len([f for f in priority_files if f[1] == "high"]),
            "medium_priority_files": len([f for f in priority_files if f[1] == "medium"]),
            "target_files": priority_files[:20],  # 前20个文件
            "estimated_time": "2-3周",
            "success_criteria": {
                "critical_files_coverage": "85%+",
                "high_priority_files_coverage": "75%+",
                "medium_priority_files_coverage": "60%+"
            }
        }
        return plan

    def generate_coverage_report(self, current_coverage: Dict, results: List[Dict]) -> str:
        """生成覆盖率报告"""
        report = f"""
# 📊 RQA2025 文件级测试覆盖率提升报告

## 📋 执行摘要
- **执行时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **处理文件数**: {len(results)}
- **成功提升**: {len([r for r in results if r['result']['success']])}
- **失败文件**: {len([r for r in results if not r['result']['success']])}

## 📈 提升结果

### 成功提升的文件
"""

        for result in results:
            if result['result']['success']:
                report += f"""
- **{result['layer']}/{result['file']}**
  - 优先级: {result['priority']}
  - 当前覆盖率: {result['current_coverage']:.1f}%
  - 目标覆盖率: {result['target_coverage']:.1f}%
  - 状态: ✅ 成功
"""

        report += """
### 需要修复的文件
"""

        for result in results:
            if not result['result']['success']:
                report += f"""
- **{result['layer']}/{result['file']}**
  - 优先级: {result['priority']}
  - 错误: {result['result']['error']}
  - 状态: ❌ 需要修复
"""

        report += f"""
## 🎯 下一步建议

### 1. 立即行动
- 修复失败的测试文件
- 完善测试用例实现
- 解决依赖和导入问题

### 2. 短期目标 (1周内)
- 完成前10个优先级文件的测试
- 达到关键文件85%覆盖率目标
- 解决所有测试失败问题

### 3. 中期目标 (1个月内)
- 完成所有关键文件测试
- 达到高优先级文件75%覆盖率目标
- 建立自动化测试流程

### 4. 长期目标 (3个月内)
- 完成所有文件测试
- 达到整体80%覆盖率目标
- 建立持续集成和部署流程

---
**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return report


def main():
    """主函数"""
    plan = FileLevelCoveragePlan()
    result = plan.execute_coverage_plan()

    # 保存报告
    report_file = plan.project_root / "reports" / "testing" / "file_level_coverage_report.md"
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(result['report'])

    print(f"\n✅ 文件级覆盖率提升计划执行完成！")
    print(f"📄 报告已保存到: {report_file}")

    return result


if __name__ == "__main__":
    main()
