#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自适应测试执行器

根据历史表现动态调整测试策略：
- 智能测试选择和排序
- 基于风险的执行策略
- 动态资源分配
- 实时性能监控和调整
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import statistics

logger = logging.getLogger(__name__)


@dataclass
class TestExecutionContext:
    """测试执行上下文"""
    test_file: str
    priority: int
    risk_level: str
    predicted_time: float
    predicted_failure_prob: float
    historical_success_rate: float
    execution_count: int
    last_execution_time: Optional[datetime] = None
    consecutive_failures: int = 0
    tags: Set[str] = field(default_factory=set)


@dataclass
class ExecutionStrategy:
    """执行策略"""
    name: str
    description: str
    priority_weights: Dict[str, float]  # 不同优先级的权重
    risk_thresholds: Dict[str, float]   # 风险阈值
    batch_size: int                     # 批次大小
    parallel_limit: int                 # 并行限制
    timeout_multiplier: float           # 超时倍数
    failure_retry_limit: int            # 失败重试次数


@dataclass
class AdaptiveResult:
    """自适应执行结果"""
    strategy_used: str
    total_tests: int
    executed_tests: int
    successful_tests: int
    failed_tests: int
    skipped_tests: int
    total_time: float
    average_time_per_test: float
    efficiency_score: float
    adaptation_decisions: List[Dict[str, Any]]


class TestSelector:
    """智能测试选择器"""

    def __init__(self):
        self.historical_data = {}
        self.load_historical_data()

    def load_historical_data(self, data_path: str = "test_logs/performance_history.json"):
        """加载历史数据"""
        try:
            if Path(data_path).exists():
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 按测试文件组织历史数据
                    for record in data:
                        test_file = record.get('test_file', '')
                        if test_file not in self.historical_data:
                            self.historical_data[test_file] = []
                        self.historical_data[test_file].append(record)
                logger.info(f"加载了 {len(self.historical_data)} 个测试的历史数据")
        except Exception as e:
            logger.warning(f"加载历史数据失败: {e}")

    def build_execution_contexts(self, test_files: List[str]) -> List[TestExecutionContext]:
        """构建执行上下文"""
        contexts = []

        for test_file in test_files:
            context = self._build_context_for_test(test_file)
            contexts.append(context)

        return contexts

    def _build_context_for_test(self, test_file: str) -> TestExecutionContext:
        """为单个测试构建上下文"""
        # 获取历史统计
        history = self.historical_data.get(test_file, [])
        execution_count = len(history)

        if history:
            # 计算成功率
            successes = sum(1 for h in history if h.get('success', True))
            success_rate = successes / execution_count

            # 计算连续失败次数
            consecutive_failures = 0
            for h in reversed(history[-10:]):  # 检查最近10次
                if not h.get('success', True):
                    consecutive_failures += 1
                else:
                    break

            # 计算平均执行时间
            execution_times = [h.get('execution_time', 0) for h in history if h.get('execution_time', 0) > 0]
            avg_time = statistics.mean(execution_times) if execution_times else 0

            # 获取最后执行时间
            last_execution = max((h.get('timestamp') for h in history if h.get('timestamp')), default=None)
            last_execution_time = datetime.fromisoformat(last_execution) if last_execution else None

        else:
            success_rate = 0.5  # 默认50%成功率
            consecutive_failures = 0
            avg_time = 1.0  # 默认1秒
            last_execution_time = None

        # 基于历史数据计算优先级
        priority = self._calculate_priority(success_rate, consecutive_failures, execution_count, avg_time)

        # 确定风险等级
        risk_level = self._calculate_risk_level(success_rate, consecutive_failures, avg_time)

        # 提取标签
        tags = self._extract_tags(test_file)

        return TestExecutionContext(
            test_file=test_file,
            priority=priority,
            risk_level=risk_level,
            predicted_time=avg_time,
            predicted_failure_prob=1 - success_rate,
            historical_success_rate=success_rate,
            execution_count=execution_count,
            last_execution_time=last_execution_time,
            consecutive_failures=consecutive_failures,
            tags=tags
        )

    def _calculate_priority(self, success_rate: float, consecutive_failures: int,
                          execution_count: int, avg_time: float) -> int:
        """计算测试优先级 (1-10, 1最高)"""
        priority_score = 5  # 基础优先级

        # 成功率影响：失败率高的测试优先级更高
        if success_rate < 0.5:
            priority_score -= 2
        elif success_rate > 0.9:
            priority_score += 1

        # 连续失败影响：连续失败的测试优先级更高
        priority_score -= min(consecutive_failures, 3)

        # 执行次数影响：执行少的测试优先级稍高
        if execution_count < 5:
            priority_score -= 1

        # 执行时间影响：短时间测试优先级稍高
        if avg_time < 0.5:
            priority_score -= 1
        elif avg_time > 30:
            priority_score += 1

        # 确保优先级在1-10范围内
        return max(1, min(10, priority_score))

    def _calculate_risk_level(self, success_rate: float, consecutive_failures: int, avg_time: float) -> str:
        """计算风险等级"""
        risk_score = 0

        # 成功率风险
        if success_rate < 0.3:
            risk_score += 3
        elif success_rate < 0.7:
            risk_score += 1

        # 连续失败风险
        risk_score += min(consecutive_failures, 3)

        # 执行时间风险（过长的测试可能有问题）
        if avg_time > 60:
            risk_score += 1

        if risk_score >= 5:
            return "critical"
        elif risk_score >= 3:
            return "high"
        elif risk_score >= 1:
            return "medium"
        else:
            return "low"

    def _extract_tags(self, test_file: str) -> Set[str]:
        """提取测试标签"""
        tags = set()
        path_parts = Path(test_file).parts

        # 基于路径的标签
        if 'unit' in path_parts:
            tags.add('unit')
        if 'integration' in path_parts:
            tags.add('integration')
        if 'e2e' in path_parts or 'end_to_end' in str(path_parts):
            tags.add('e2e')
        if 'performance' in path_parts:
            tags.add('performance')
        if 'security' in path_parts:
            tags.add('security')

        # 基于文件名的标签
        filename = Path(test_file).name.lower()
        if 'smoke' in filename:
            tags.add('smoke')
        if 'critical' in filename:
            tags.add('critical')
        if 'slow' in filename:
            tags.add('slow')

        return tags


class StrategyEngine:
    """策略引擎"""

    def __init__(self):
        self.strategies = self._define_strategies()

    def _define_strategies(self) -> Dict[str, ExecutionStrategy]:
        """定义执行策略"""
        return {
            'conservative': ExecutionStrategy(
                name='conservative',
                description='保守策略：优先执行低风险测试，确保稳定性',
                priority_weights={'low': 1.0, 'medium': 0.7, 'high': 0.3, 'critical': 0.1},
                risk_thresholds={'low': 0.8, 'medium': 0.6, 'high': 0.4, 'critical': 0.2},
                batch_size=5,
                parallel_limit=2,
                timeout_multiplier=2.0,
                failure_retry_limit=2
            ),
            'balanced': ExecutionStrategy(
                name='balanced',
                description='平衡策略：综合考虑风险和效率',
                priority_weights={'low': 0.8, 'medium': 1.0, 'high': 1.2, 'critical': 1.5},
                risk_thresholds={'low': 0.7, 'medium': 0.5, 'high': 0.3, 'critical': 0.1},
                batch_size=10,
                parallel_limit=4,
                timeout_multiplier=1.5,
                failure_retry_limit=3
            ),
            'aggressive': ExecutionStrategy(
                name='aggressive',
                description='激进策略：优先执行高风险测试，快速发现问题',
                priority_weights={'low': 0.3, 'medium': 0.7, 'high': 1.0, 'critical': 1.5},
                risk_thresholds={'low': 0.6, 'medium': 0.4, 'high': 0.2, 'critical': 0.05},
                batch_size=15,
                parallel_limit=6,
                timeout_multiplier=1.2,
                failure_retry_limit=1
            ),
            'performance': ExecutionStrategy(
                name='performance',
                description='性能策略：优先执行快速测试，优化执行时间',
                priority_weights={'fast': 1.5, 'medium': 1.0, 'slow': 0.5},
                risk_thresholds={'low': 0.9, 'medium': 0.7, 'high': 0.5, 'critical': 0.3},
                batch_size=20,
                parallel_limit=8,
                timeout_multiplier=1.0,
                failure_retry_limit=1
            )
        }

    def select_strategy(self, contexts: List[TestExecutionContext],
                       execution_mode: str = 'balanced') -> ExecutionStrategy:
        """选择执行策略"""
        if execution_mode in self.strategies:
            return self.strategies[execution_mode]

        # 基于上下文智能选择策略
        risk_distribution = self._analyze_risk_distribution(contexts)

        # 如果有很多高风险测试，选择保守策略
        if risk_distribution.get('critical', 0) > 0.3:  # 30%以上是高风险
            return self.strategies['conservative']
        elif risk_distribution.get('high', 0) > 0.5:  # 50%以上是高风险
            return self.strategies['balanced']
        else:
            return self.strategies['aggressive']

    def _analyze_risk_distribution(self, contexts: List[TestExecutionContext]) -> Dict[str, float]:
        """分析风险分布"""
        total = len(contexts)
        if total == 0:
            return {}

        distribution = {}
        for context in contexts:
            risk_level = context.risk_level
            distribution[risk_level] = distribution.get(risk_level, 0) + 1

        # 转换为百分比
        return {k: v / total for k, v in distribution.items()}

    def adapt_strategy(self, current_strategy: ExecutionStrategy,
                      execution_results: List[Dict[str, Any]]) -> ExecutionStrategy:
        """根据执行结果调整策略"""
        # 分析执行结果
        success_rate = sum(1 for r in execution_results if r.get('success', False)) / len(execution_results)
        avg_time = statistics.mean([r.get('duration', 1) for r in execution_results])

        # 根据表现调整策略
        if success_rate < 0.5:  # 成功率太低，切换到更保守的策略
            if current_strategy.name == 'aggressive':
                return self.strategies['balanced']
            elif current_strategy.name == 'balanced':
                return self.strategies['conservative']
        elif success_rate > 0.9 and avg_time < 5:  # 表现很好，可以更激进
            if current_strategy.name == 'conservative':
                return self.strategies['balanced']
            elif current_strategy.name == 'balanced':
                return self.strategies['aggressive']

        return current_strategy


class AdaptiveTestExecutor:
    """自适应测试执行器"""

    def __init__(self):
        self.selector = TestSelector()
        self.strategy_engine = StrategyEngine()
        self.execution_history = []
        self.monitoring_active = False

    def execute_adaptive(self, test_files: List[str],
                        execution_mode: str = 'auto',
                        max_time: Optional[int] = None) -> AdaptiveResult:
        """执行自适应测试"""
        logger.info(f"开始自适应测试执行，模式: {execution_mode}")

        start_time = time.time()
        adaptation_decisions = []

        # 构建执行上下文
        contexts = self.selector.build_execution_contexts(test_files)

        # 选择初始策略
        current_strategy = self.strategy_engine.select_strategy(contexts, execution_mode)
        adaptation_decisions.append({
            'timestamp': datetime.now().isoformat(),
            'decision': 'initial_strategy',
            'strategy': current_strategy.name,
            'reason': f'基于{len(contexts)}个测试的分析'
        })

        logger.info(f"选择执行策略: {current_strategy.name} - {current_strategy.description}")

        # 按策略对测试排序
        sorted_contexts = self._sort_contexts_by_strategy(contexts, current_strategy)

        # 执行测试
        executed_tests = 0
        successful_tests = 0
        failed_tests = 0
        skipped_tests = 0
        batch_results = []

        # 分批执行
        for i in range(0, len(sorted_contexts), current_strategy.batch_size):
            batch = sorted_contexts[i:i + current_strategy.batch_size]

            if max_time and time.time() - start_time > max_time:
                logger.warning("达到最大执行时间，停止执行")
                skipped_tests += len(sorted_contexts) - executed_tests
                break

            # 执行批次
            batch_start_time = time.time()
            batch_result = self._execute_batch(batch, current_strategy)
            batch_time = time.time() - batch_start_time

            batch_results.extend(batch_result)
            executed_tests += len(batch_result)
            successful_tests += sum(1 for r in batch_result if r.get('success', False))
            failed_tests += sum(1 for r in batch_result if not r.get('success', False))

            # 记录批次结果
            self.execution_history.extend(batch_result)

            # 基于批次结果调整策略
            if len(batch_results) >= 10:  # 每10个测试检查一次
                new_strategy = self.strategy_engine.adapt_strategy(current_strategy, batch_results[-10:])
                if new_strategy.name != current_strategy.name:
                    adaptation_decisions.append({
                        'timestamp': datetime.now().isoformat(),
                        'decision': 'strategy_adaptation',
                        'from_strategy': current_strategy.name,
                        'to_strategy': new_strategy.name,
                        'reason': '基于最近10个测试的表现调整'
                    })
                    current_strategy = new_strategy
                    logger.info(f"调整执行策略到: {current_strategy.name}")

        total_time = time.time() - start_time
        average_time_per_test = total_time / executed_tests if executed_tests > 0 else 0

        # 计算效率评分
        efficiency_score = self._calculate_efficiency_score(
            successful_tests, executed_tests, total_time, len(contexts)
        )

        result = AdaptiveResult(
            strategy_used=current_strategy.name,
            total_tests=len(contexts),
            executed_tests=executed_tests,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            total_time=total_time,
            average_time_per_test=average_time_per_test,
            efficiency_score=efficiency_score,
            adaptation_decisions=adaptation_decisions
        )

        # 生成报告
        self._generate_adaptive_report(result, contexts)

        logger.info("自适应测试执行完成")
        return result

    def _sort_contexts_by_strategy(self, contexts: List[TestExecutionContext],
                                  strategy: ExecutionStrategy) -> List[TestExecutionContext]:
        """按策略排序上下文"""
        def sort_key(context):
            # 基于策略的权重计算排序键
            priority_weight = strategy.priority_weights.get(context.risk_level, 1.0)

            # 综合考虑优先级、风险和预测时间
            priority_score = context.priority * priority_weight
            risk_score = self._risk_level_to_score(context.risk_level)
            time_score = min(context.predicted_time / 60, 1)  # 归一化到0-1

            # 返回排序键（值越小优先级越高）
            return (priority_score, risk_score, time_score)

        return sorted(contexts, key=sort_key)

    def _risk_level_to_score(self, risk_level: str) -> int:
        """风险等级转换为分数"""
        mapping = {'critical': 1, 'high': 2, 'medium': 3, 'low': 4}
        return mapping.get(risk_level, 3)

    def _execute_batch(self, batch: List[TestExecutionContext],
                      strategy: ExecutionStrategy) -> List[Dict[str, Any]]:
        """执行测试批次"""
        results = []

        # 简单的顺序执行（可以扩展为并行执行）
        for context in batch:
            result = self._execute_single_test(context, strategy)
            results.append(result)

        return results

    def _execute_single_test(self, context: TestExecutionContext,
                           strategy: ExecutionStrategy) -> Dict[str, Any]:
        """执行单个测试"""
        start_time = time.time()

        try:
            # 模拟测试执行（实际应该调用真实的测试执行器）
            # 这里只是演示，实际项目中应该调用 pytest 或其他测试框架

            # 计算模拟的执行时间（基于预测时间）
            simulated_time = max(0.1, context.predicted_time * (0.5 + np.random.random()))

            time.sleep(min(simulated_time, 5))  # 限制最大等待时间

            # 基于历史成功率和风险等级模拟结果
            success_probability = context.historical_success_rate
            if context.risk_level == 'critical':
                success_probability *= 0.7
            elif context.risk_level == 'high':
                success_probability *= 0.8
            elif context.risk_level == 'low':
                success_probability *= 1.2

            success_probability = max(0.1, min(0.95, success_probability))
            success = np.random.random() < success_probability

            duration = time.time() - start_time

            return {
                'test_file': context.test_file,
                'success': success,
                'duration': duration,
                'risk_level': context.risk_level,
                'priority': context.priority,
                'tags': list(context.tags)
            }

        except Exception as e:
            return {
                'test_file': context.test_file,
                'success': False,
                'duration': time.time() - start_time,
                'error': str(e),
                'risk_level': context.risk_level,
                'priority': context.priority,
                'tags': list(context.tags)
            }

    def _calculate_efficiency_score(self, successful_tests: int, executed_tests: int,
                                  total_time: float, total_available: int) -> float:
        """计算效率评分"""
        if executed_tests == 0:
            return 0.0

        # 成功率评分（40%）
        success_rate = successful_tests / executed_tests
        success_score = success_rate * 0.4

        # 执行率评分（30%）
        execution_rate = executed_tests / total_available if total_available > 0 else 0
        execution_score = execution_rate * 0.3

        # 时间效率评分（30%）
        avg_time_per_test = total_time / executed_tests
        # 理想的平均测试时间是2秒
        time_efficiency = max(0, 1 - abs(avg_time_per_test - 2) / 2)
        time_score = time_efficiency * 0.3

        return success_score + execution_score + time_score

    def _generate_adaptive_report(self, result: AdaptiveResult, contexts: List[TestExecutionContext]):
        """生成自适应执行报告"""
        report_path = Path("test_logs/adaptive_test_report.md")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 自适应测试执行报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 📊 执行概览\n\n")
            f.write(f"- **使用的策略**: {result.strategy_used}\n")
            f.write(f"- **总可用测试**: {result.total_tests}\n")
            f.write(f"- **执行测试数**: {result.executed_tests}\n")
            f.write(f"- **成功测试**: {result.successful_tests}\n")
            f.write(f"- **失败测试**: {result.failed_tests}\n")
            f.write(f"- **跳过测试**: {result.skipped_tests}\n")
            f.write(".2")
            f.write(".2")
            f.write(".2")
            f.write("## 🎯 策略适应决策\n\n")
            for decision in result.adaptation_decisions:
                timestamp = datetime.fromisoformat(decision['timestamp']).strftime('%H:%M:%S')
                f.write(f"- **{timestamp}**: {decision['decision']} - {decision['reason']}\n")
                if 'from_strategy' in decision:
                    f.write(f"  从 `{decision['from_strategy']}` 调整到 `{decision['to_strategy']}`\n")

            f.write("\n## 📈 风险分布分析\n\n")

            # 分析上下文中的风险分布
            risk_counts = {}
            for context in contexts:
                risk_counts[context.risk_level] = risk_counts.get(context.risk_level, 0) + 1

            for risk_level, count in risk_counts.items():
                percentage = count / len(contexts) * 100
                f.write(f"- **{risk_level}**: {count} 个测试 ({percentage:.1f}%)\n")

            f.write("\n## 🏆 高优先级测试\n\n")
            high_priority = [c for c in contexts if c.priority <= 3][:10]

            f.write("| 测试文件 | 优先级 | 风险等级 | 预测时间 | 成功率 |\n")
            f.write("|----------|--------|----------|----------|--------|\n")

            for context in high_priority:
                f.write(f"| `{Path(context.test_file).name}` | {context.priority} | {context.risk_level} | {context.predicted_time:.2f}s | {context.historical_success_rate:.1f} |\n")

            f.write("\n## 🎯 自适应价值\n\n")
            f.write("### 对测试执行的价值\n")
            f.write("1. **智能排序**: 基于历史表现和风险评估的测试优先级排序\n")
            f.write("2. **动态调整**: 根据执行结果实时调整测试策略\n")
            f.write("3. **资源优化**: 平衡测试质量和执行效率\n")
            f.write("4. **风险控制**: 优先处理高风险测试，及早发现问题\n")
            f.write("\n### 对团队的价值\n")
            f.write("1. **反馈加速**: 高风险测试优先执行，快速获得重要反馈\n")
            f.write("2. **质量保障**: 自适应策略确保测试覆盖的关键场景\n")
            f.write("3. **效率提升**: 避免无效测试，专注有价值的测试执行\n")
            f.write("4. **持续改进**: 基于数据驱动的测试策略优化\n")

        logger.info(f"自适应测试报告已生成: {report_path}")


def main():
    """主函数"""
    executor = AdaptiveTestExecutor()

    print("🔄 自适应测试执行器启动")
    print("🎯 功能: 智能测试选择 + 动态策略调整 + 风险优先级排序")

    # 发现测试文件
    test_files = []
    for pattern in ["test_*.py", "*_test.py"]:
        test_files.extend([str(f) for f in Path("tests").rglob(pattern)])

    if not test_files:
        print("⚠️ 未发现测试文件")
        return

    # 限制测试数量用于演示
    test_files = test_files[:50]

    print(f"🎪 发现 {len(test_files)} 个测试文件，开始自适应执行...")

    # 执行自适应测试
    result = executor.execute_adaptive(test_files, execution_mode='balanced', max_time=120)

    print("\n📊 自适应测试结果:")
    print(f"  🎯 执行策略: {result.strategy_used}")
    print(f"  📋 总测试数: {result.total_tests}")
    print(f"  ▶️ 执行测试: {result.executed_tests}")
    print(f"  ✅ 成功测试: {result.successful_tests}")
    print(f"  ❌ 失败测试: {result.failed_tests}")
    print(f"  ⏭️ 跳过测试: {result.skipped_tests}")
    print(".2")
    print(".2")
    print(".2")
    # 显示策略适应决策
    if result.adaptation_decisions:
        print("\n🔄 策略适应决策:")
        for decision in result.adaptation_decisions[-3:]:  # 显示最后3个
            print(f"  • {decision['decision']}: {decision.get('reason', '')}")

    print("📄 详细报告已保存到: test_logs/adaptive_test_report.md")
    print("\n✅ 自适应测试执行器运行完成")


if __name__ == "__main__":
    main()
