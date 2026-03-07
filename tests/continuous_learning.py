#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
持续学习系统

从测试执行中学习并改进测试策略：
- 学习历史模式和趋势
- 预测性维护和预防
- 自动策略优化
- 知识积累和传承
- 智能决策支持
"""

import os
import json
import time
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import statistics
import numpy as np
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class LearningPattern:
    """学习模式"""
    pattern_id: str
    pattern_type: str  # 'success', 'failure', 'performance', 'resource'
    conditions: Dict[str, Any]
    outcomes: Dict[str, Any]
    confidence: float
    occurrences: int
    last_seen: datetime
    recommendations: List[str]


@dataclass
class StrategyOptimization:
    """策略优化建议"""
    strategy_name: str
    current_performance: Dict[str, float]
    recommended_changes: Dict[str, Any]
    expected_improvement: float
    confidence: float
    rationale: str


@dataclass
class PredictiveInsight:
    """预测性洞察"""
    insight_type: str
    description: str
    affected_components: List[str]
    risk_level: str
    preventive_actions: List[str]
    confidence: float
    timeframe: str


@dataclass
class LearningResult:
    """学习结果"""
    patterns_discovered: int
    strategies_optimized: int
    insights_generated: int
    knowledge_updated: bool
    recommendations: List[str]
    execution_time: float


class PatternLearner:
    """模式学习器"""

    def __init__(self):
        self.patterns: Dict[str, LearningPattern] = {}
        self.pattern_history: List[Dict[str, Any]] = []
        self.load_patterns()

    def analyze_execution_data(self, execution_data: List[Dict[str, Any]]) -> List[LearningPattern]:
        """分析执行数据发现模式"""
        logger.info(f"开始分析 {len(execution_data)} 条执行数据")

        new_patterns = []

        # 分析成功模式
        success_patterns = self._analyze_success_patterns(execution_data)
        new_patterns.extend(success_patterns)

        # 分析失败模式
        failure_patterns = self._analyze_failure_patterns(execution_data)
        new_patterns.extend(failure_patterns)

        # 分析性能模式
        performance_patterns = self._analyze_performance_patterns(execution_data)
        new_patterns.extend(performance_patterns)

        # 分析资源使用模式
        resource_patterns = self._analyze_resource_patterns(execution_data)
        new_patterns.extend(resource_patterns)

        # 更新现有模式或创建新模式
        for pattern in new_patterns:
            if pattern.pattern_id in self.patterns:
                # 更新现有模式
                existing = self.patterns[pattern.pattern_id]
                existing.occurrences += pattern.occurrences
                existing.confidence = (existing.confidence * existing.occurrences +
                                     pattern.confidence * pattern.occurrences) / (existing.occurrences + pattern.occurrences)
                existing.last_seen = pattern.last_seen
            else:
                # 添加新模式
                self.patterns[pattern.pattern_id] = pattern

        self._save_patterns()
        logger.info(f"发现 {len(new_patterns)} 个新模式，总共 {len(self.patterns)} 个模式")

        return new_patterns

    def _analyze_success_patterns(self, data: List[Dict[str, Any]]) -> List[LearningPattern]:
        """分析成功模式"""
        patterns = []

        # 按测试文件分组
        file_groups = defaultdict(list)
        for record in data:
            if record.get('success', False):
                file_groups[record.get('test_file', 'unknown')].append(record)

        for test_file, records in file_groups.items():
            if len(records) < 3:  # 需要至少3次成功记录
                continue

            # 计算成功率
            success_rate = len(records) / len([r for r in data if r.get('test_file') == test_file])

            if success_rate > 0.9:  # 成功率超过90%
                # 分析共同特征
                avg_time = statistics.mean([r.get('execution_time', 0) for r in records])
                common_tags = self._find_common_tags(records)

                pattern = LearningPattern(
                    pattern_id=f"success_pattern_{test_file}",
                    pattern_type="success",
                    conditions={
                        "test_file": test_file,
                        "success_rate_threshold": 0.9,
                        "avg_execution_time": avg_time,
                        "common_tags": common_tags
                    },
                    outcomes={
                        "expected_success": True,
                        "avg_execution_time": avg_time,
                        "stability_score": success_rate
                    },
                    confidence=min(success_rate * 1.2, 0.95),
                    occurrences=len(records),
                    last_seen=datetime.now(),
                    recommendations=[
                        f"为 {test_file} 维护稳定的执行环境",
                        "监控执行时间变化趋势",
                        f"确保相关标签保持: {', '.join(common_tags)}"
                    ]
                )
                patterns.append(pattern)

        return patterns

    def _analyze_failure_patterns(self, data: List[Dict[str, Any]]) -> List[LearningPattern]:
        """分析失败模式"""
        patterns = []

        # 按测试文件分组失败记录
        file_groups = defaultdict(list)
        for record in data:
            if not record.get('success', True):
                file_groups[record.get('test_file', 'unknown')].append(record)

        for test_file, records in file_groups.items():
            if len(records) < 2:  # 需要至少2次失败记录
                continue

            # 分析失败特征
            failure_times = [r.get('execution_time', 0) for r in records]
            avg_failure_time = statistics.mean(failure_times) if failure_times else 0

            # 查找失败时间模式
            failure_hours = [datetime.fromisoformat(r.get('timestamp', datetime.now().isoformat())).hour for r in records if r.get('timestamp')]
            common_failure_hour = Counter(failure_hours).most_common(1)[0][0] if failure_hours else None

            pattern = LearningPattern(
                pattern_id=f"failure_pattern_{test_file}",
                pattern_type="failure",
                conditions={
                    "test_file": test_file,
                    "failure_count": len(records),
                    "common_failure_hour": common_failure_hour
                },
                outcomes={
                    "expected_success": False,
                    "avg_failure_time": avg_failure_time,
                    "failure_frequency": len(records) / len([r for r in data if r.get('test_file') == test_file])
                },
                confidence=min(len(records) / 10, 0.9),  # 失败次数越多，置信度越高
                occurrences=len(records),
                last_seen=datetime.now(),
                recommendations=[
                    f"调查 {test_file} 的失败原因",
                    "检查测试依赖和环境设置",
                    f"关注在 {common_failure_hour}:00 附近的执行" if common_failure_hour else "随机监控执行时间",
                    "考虑隔离或重构该测试"
                ]
            )
            patterns.append(pattern)

        return patterns

    def _analyze_performance_patterns(self, data: List[Dict[str, Any]]) -> List[LearningPattern]:
        """分析性能模式"""
        patterns = []

        # 计算整体性能趋势
        execution_times = [r.get('execution_time', 0) for r in data if r.get('execution_time', 0) > 0]
        if len(execution_times) < 5:
            return patterns

        # 检测性能退化
        recent_times = execution_times[-10:] if len(execution_times) >= 10 else execution_times
        older_times = execution_times[:-10] if len(execution_times) >= 20 else execution_times[:len(execution_times)//2]

        if recent_times and older_times:
            recent_avg = statistics.mean(recent_times)
            older_avg = statistics.mean(older_times)
            degradation_rate = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0

            if degradation_rate > 0.2:  # 性能退化超过20%
                pattern = LearningPattern(
                    pattern_id="performance_degradation",
                    pattern_type="performance",
                    conditions={
                        "time_window": "recent_vs_older",
                        "degradation_threshold": 0.2
                    },
                    outcomes={
                        "degradation_rate": degradation_rate,
                        "recent_avg_time": recent_avg,
                        "older_avg_time": older_avg
                    },
                    confidence=min(abs(degradation_rate) * 2, 0.95),
                    occurrences=1,
                    last_seen=datetime.now(),
                    recommendations=[
                        "调查性能退化的根本原因",
                        "检查代码变更历史",
                        "优化慢速测试或重构代码",
                        "增加性能监控和基准测试"
                    ]
                )
                patterns.append(pattern)

        return patterns

    def _analyze_resource_patterns(self, data: List[Dict[str, Any]]) -> List[LearningPattern]:
        """分析资源使用模式"""
        patterns = []

        # 分析内存使用模式
        memory_usage = [r.get('memory_usage', 0) for r in data if r.get('memory_usage', 0) > 0]
        if len(memory_usage) >= 5:
            avg_memory = statistics.mean(memory_usage)
            memory_std = statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0

            if memory_std / avg_memory > 0.5:  # 内存使用波动大
                pattern = LearningPattern(
                    pattern_id="memory_usage_volatility",
                    pattern_type="resource",
                    conditions={
                        "volatility_threshold": 0.5,
                        "memory_measurements": len(memory_usage)
                    },
                    outcomes={
                        "avg_memory_usage": avg_memory,
                        "memory_volatility": memory_std / avg_memory,
                        "memory_range": f"{min(memory_usage)} - {max(memory_usage)}"
                    },
                    confidence=min(len(memory_usage) / 20, 0.9),
                    occurrences=1,
                    last_seen=datetime.now(),
                    recommendations=[
                        "调查内存使用波动的根本原因",
                        "检查是否存在内存泄漏",
                        "优化内存密集型测试",
                        "增加内存监控和告警"
                    ]
                )
                patterns.append(pattern)

        return patterns

    def _find_common_tags(self, records: List[Dict[str, Any]]) -> List[str]:
        """查找共同标签"""
        all_tags = []
        for record in records:
            tags = record.get('tags', [])
            if isinstance(tags, list):
                all_tags.extend(tags)

        if not all_tags:
            return []

        # 找出出现频率最高的标签
        tag_counts = Counter(all_tags)
        common_tags = [tag for tag, count in tag_counts.most_common(3) if count > len(records) * 0.5]

        return common_tags

    def get_patterns_by_type(self, pattern_type: str) -> List[LearningPattern]:
        """按类型获取模式"""
        return [p for p in self.patterns.values() if p.pattern_type == pattern_type]

    def get_recent_patterns(self, days: int = 7) -> List[LearningPattern]:
        """获取最近的模式"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [p for p in self.patterns.values() if p.last_seen > cutoff_date]

    def _save_patterns(self):
        """保存模式"""
        patterns_file = Path("test_logs/learning_patterns.json")

        patterns_data = {}
        for pattern_id, pattern in self.patterns.items():
            patterns_data[pattern_id] = {
                'pattern_id': pattern.pattern_id,
                'pattern_type': pattern.pattern_type,
                'conditions': pattern.conditions,
                'outcomes': pattern.outcomes,
                'confidence': pattern.confidence,
                'occurrences': pattern.occurrences,
                'last_seen': pattern.last_seen.isoformat(),
                'recommendations': pattern.recommendations
            }

        try:
            patterns_file.parent.mkdir(parents=True, exist_ok=True)
            with open(patterns_file, 'w', encoding='utf-8') as f:
                json.dump(patterns_data, f, indent=2, ensure_ascii=False)
            logger.info(f"保存了 {len(self.patterns)} 个学习模式")
        except Exception as e:
            logger.error(f"保存模式失败: {e}")

    def load_patterns(self):
        """加载模式"""
        patterns_file = Path("test_logs/learning_patterns.json")

        if not patterns_file.exists():
            return

        try:
            with open(patterns_file, 'r', encoding='utf-8') as f:
                patterns_data = json.load(f)

            for pattern_id, data in patterns_data.items():
                pattern = LearningPattern(
                    pattern_id=data['pattern_id'],
                    pattern_type=data['pattern_type'],
                    conditions=data['conditions'],
                    outcomes=data['outcomes'],
                    confidence=data['confidence'],
                    occurrences=data['occurrences'],
                    last_seen=datetime.fromisoformat(data['last_seen']),
                    recommendations=data['recommendations']
                )
                self.patterns[pattern_id] = pattern

            logger.info(f"加载了 {len(self.patterns)} 个学习模式")

        except Exception as e:
            logger.error(f"加载模式失败: {e}")


class StrategyOptimizer:
    """策略优化器"""

    def __init__(self):
        self.current_strategies = self._load_strategies()
        self.optimization_history = []

    def optimize_strategies(self, patterns: List[LearningPattern],
                          performance_data: List[Dict[str, Any]]) -> List[StrategyOptimization]:
        """优化策略"""
        logger.info("开始策略优化")

        optimizations = []

        # 基于成功模式优化
        success_optimizations = self._optimize_based_on_success_patterns(patterns)
        optimizations.extend(success_optimizations)

        # 基于失败模式优化
        failure_optimizations = self._optimize_based_on_failure_patterns(patterns)
        optimizations.extend(failure_optimizations)

        # 基于性能模式优化
        performance_optimizations = self._optimize_based_on_performance_patterns(patterns, performance_data)
        optimizations.extend(performance_optimizations)

        # 保存优化历史
        self.optimization_history.append({
            'timestamp': datetime.now().isoformat(),
            'optimizations': [opt.__dict__ for opt in optimizations]
        })

        logger.info(f"生成了 {len(optimizations)} 个策略优化建议")
        return optimizations

    def _optimize_based_on_success_patterns(self, patterns: List[LearningPattern]) -> List[StrategyOptimization]:
        """基于成功模式优化"""
        optimizations = []

        success_patterns = [p for p in patterns if p.pattern_type == 'success']

        for pattern in success_patterns:
            if pattern.confidence > 0.8:  # 高置信度成功模式
                optimization = StrategyOptimization(
                    strategy_name="adaptive_testing",
                    current_performance={
                        "success_rate": pattern.outcomes.get('stability_score', 0),
                        "avg_time": pattern.outcomes.get('avg_execution_time', 0)
                    },
                    recommended_changes={
                        "priority_boost": 2,
                        "frequency_increase": 1.5,
                        "resource_allocation": "standard"
                    },
                    expected_improvement=pattern.outcomes.get('stability_score', 0) * 0.1,
                    confidence=pattern.confidence,
                    rationale=f"基于稳定的成功模式 {pattern.pattern_id}"
                )
                optimizations.append(optimization)

        return optimizations

    def _optimize_based_on_failure_patterns(self, patterns: List[LearningPattern]) -> List[StrategyOptimization]:
        """基于失败模式优化"""
        optimizations = []

        failure_patterns = [p for p in patterns if p.pattern_type == 'failure']

        for pattern in failure_patterns:
            optimization = StrategyOptimization(
                strategy_name="fault_tolerance",
                current_performance={
                    "failure_rate": pattern.outcomes.get('failure_frequency', 0),
                    "avg_time": pattern.outcomes.get('avg_failure_time', 0)
                },
                recommended_changes={
                    "retry_attempts": min(pattern.occurrences, 5),
                    "isolation_level": "high",
                    "monitoring_intensity": "increased"
                },
                expected_improvement=pattern.outcomes.get('failure_frequency', 0) * 0.3,
                confidence=pattern.confidence,
                rationale=f"基于失败模式 {pattern.pattern_id} 优化容错策略"
            )
            optimizations.append(optimization)

        return optimizations

    def _optimize_based_on_performance_patterns(self, patterns: List[LearningPattern],
                                              performance_data: List[Dict[str, Any]]) -> List[StrategyOptimization]:
        """基于性能模式优化"""
        optimizations = []

        performance_patterns = [p for p in patterns if p.pattern_type == 'performance']

        for pattern in performance_patterns:
            if 'degradation' in pattern.pattern_id:
                degradation_rate = pattern.outcomes.get('degradation_rate', 0)

                optimization = StrategyOptimization(
                    strategy_name="performance_optimization",
                    current_performance={
                        "degradation_rate": degradation_rate,
                        "recent_avg_time": pattern.outcomes.get('recent_avg_time', 0)
                    },
                    recommended_changes={
                        "parallelization_increase": 2,
                        "resource_limits": "optimized",
                        "profiling_enabled": True
                    },
                    expected_improvement=min(abs(degradation_rate) * 0.5, 0.3),
                    confidence=pattern.confidence,
                    rationale="基于性能退化模式优化执行策略"
                )
                optimizations.append(optimization)

        return optimizations

    def _load_strategies(self) -> Dict[str, Any]:
        """加载当前策略"""
        # 这里应该从配置文件加载实际的策略设置
        return {
            'adaptive_testing': {'enabled': True, 'parameters': {}},
            'fault_tolerance': {'enabled': True, 'parameters': {}},
            'performance_optimization': {'enabled': True, 'parameters': {}}
        }


class PredictiveAnalyzer:
    """预测分析器"""

    def __init__(self):
        self.insights_history = []

    def generate_insights(self, patterns: List[LearningPattern],
                         performance_data: List[Dict[str, Any]]) -> List[PredictiveInsight]:
        """生成预测洞察"""
        logger.info("开始生成预测洞察")

        insights = []

        # 基于模式生成洞察
        pattern_insights = self._generate_pattern_based_insights(patterns)
        insights.extend(pattern_insights)

        # 基于趋势生成洞察
        trend_insights = self._generate_trend_based_insights(performance_data)
        insights.extend(trend_insights)

        # 基于异常检测生成洞察
        anomaly_insights = self._generate_anomaly_based_insights(performance_data)
        insights.extend(anomaly_insights)

        # 保存洞察历史
        self.insights_history.append({
            'timestamp': datetime.now().isoformat(),
            'insights': [insight.__dict__ for insight in insights]
        })

        logger.info(f"生成了 {len(insights)} 个预测洞察")
        return insights

    def _generate_pattern_based_insights(self, patterns: List[LearningPattern]) -> List[PredictiveInsight]:
        """基于模式生成洞察"""
        insights = []

        # 识别高风险模式
        high_risk_patterns = [p for p in patterns if p.confidence > 0.8 and
                            (p.pattern_type == 'failure' or 'degradation' in p.pattern_id)]

        for pattern in high_risk_patterns:
            affected_components = [pattern.conditions.get('test_file', 'unknown')]

            insight = PredictiveInsight(
                insight_type="risk_prediction",
                description=f"基于模式 {pattern.pattern_id} 预测潜在风险",
                affected_components=affected_components,
                risk_level="high" if pattern.confidence > 0.9 else "medium",
                preventive_actions=pattern.recommendations,
                confidence=pattern.confidence,
                timeframe="immediate"
            )
            insights.append(insight)

        # 识别改进机会
        success_patterns = [p for p in patterns if p.pattern_type == 'success' and p.confidence > 0.8]

        for pattern in success_patterns:
            insight = PredictiveInsight(
                insight_type="optimization_opportunity",
                description=f"发现稳定的成功模式 {pattern.pattern_id}，可用于优化",
                affected_components=[pattern.conditions.get('test_file', 'unknown')],
                risk_level="low",
                preventive_actions=[
                    "将成功模式扩展到相似测试",
                    "优化相关测试的执行策略",
                    "分享最佳实践经验"
                ],
                confidence=pattern.confidence,
                timeframe="short_term"
            )
            insights.append(insight)

        return insights

    def _generate_trend_based_insights(self, performance_data: List[Dict[str, Any]]) -> List[PredictiveInsight]:
        """基于趋势生成洞察"""
        insights = []

        if len(performance_data) < 10:
            return insights

        # 分析成功率趋势
        success_rates = []
        for i in range(0, len(performance_data), 10):  # 每10条记录计算一次
            batch = performance_data[i:i+10]
            success_count = sum(1 for r in batch if r.get('success', False))
            success_rate = success_count / len(batch) if batch else 0
            success_rates.append(success_rate)

        if len(success_rates) >= 3:
            # 计算趋势
            trend = np.polyfit(range(len(success_rates)), success_rates, 1)[0]

            if trend < -0.01:  # 成功率下降趋势
                insight = PredictiveInsight(
                    insight_type="trend_analysis",
                    description="检测到测试成功率下降趋势",
                    affected_components=["test_suite"],
                    risk_level="medium",
                    preventive_actions=[
                        "调查最近的代码变更",
                        "检查测试环境稳定性",
                        "增加测试覆盖率",
                        "实施回归测试"
                    ],
                    confidence=min(abs(trend) * 100, 0.8),
                    timeframe="short_term"
                )
                insights.append(insight)

        return insights

    def _generate_anomaly_based_insights(self, performance_data: List[Dict[str, Any]]) -> List[PredictiveInsight]:
        """基于异常检测生成洞察"""
        insights = []

        if len(performance_data) < 5:
            return insights

        # 检测异常长的执行时间
        execution_times = [r.get('execution_time', 0) for r in performance_data if r.get('execution_time', 0) > 0]
        if execution_times:
            mean_time = statistics.mean(execution_times)
            std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0

            # 查找异常值（超过3个标准差）
            threshold = mean_time + 3 * std_time
            anomalies = [r for r in performance_data if r.get('execution_time', 0) > threshold]

            if anomalies:
                affected_tests = list(set(r.get('test_file', 'unknown') for r in anomalies))

                insight = PredictiveInsight(
                    insight_type="anomaly_detection",
                    description=f"检测到 {len(anomalies)} 个异常慢速测试执行",
                    affected_components=affected_tests,
                    risk_level="medium",
                    preventive_actions=[
                        "分析慢速测试的根本原因",
                        "优化测试实现或数据",
                        "考虑将慢速测试分离执行",
                        "增加性能监控"
                    ],
                    confidence=min(len(anomalies) / len(performance_data) * 2, 0.9),
                    timeframe="immediate"
                )
                insights.append(insight)

        return insights


class ContinuousLearningSystem:
    """持续学习系统"""

    def __init__(self):
        self.pattern_learner = PatternLearner()
        self.strategy_optimizer = StrategyOptimizer()
        self.predictive_analyzer = PredictiveAnalyzer()
        self.knowledge_base = self._load_knowledge_base()

    def perform_continuous_learning(self, execution_data_path: str = "test_logs/performance_history.json",
                                  performance_data_path: str = "test_logs/performance_history.json") -> LearningResult:
        """执行持续学习"""
        logger.info("开始持续学习周期")

        start_time = time.time()

        # 加载数据
        execution_data = self._load_data(execution_data_path)
        performance_data = self._load_data(performance_data_path)

        if not execution_data:
            logger.warning("没有执行数据用于学习")
            return LearningResult(0, 0, 0, False, [], time.time() - start_time)

        # 1. 模式学习
        new_patterns = self.pattern_learner.analyze_execution_data(execution_data)

        # 2. 策略优化
        strategy_optimizations = self.strategy_optimizer.optimize_strategies(new_patterns, performance_data)

        # 3. 预测分析
        predictive_insights = self.predictive_analyzer.generate_insights(new_patterns, performance_data)

        # 4. 更新知识库
        knowledge_updated = self._update_knowledge_base(new_patterns, strategy_optimizations, predictive_insights)

        # 5. 生成建议
        recommendations = self._generate_learning_recommendations(new_patterns, strategy_optimizations, predictive_insights)

        # 6. 生成学习报告
        self._generate_learning_report(new_patterns, strategy_optimizations, predictive_insights, recommendations)

        execution_time = time.time() - start_time

        result = LearningResult(
            patterns_discovered=len(new_patterns),
            strategies_optimized=len(strategy_optimizations),
            insights_generated=len(predictive_insights),
            knowledge_updated=knowledge_updated,
            recommendations=recommendations,
            execution_time=execution_time
        )

        logger.info("持续学习周期完成")
        return result

    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载数据"""
        try:
            if Path(data_path).exists():
                with open(data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"加载数据失败 {data_path}: {e}")

        return []

    def _update_knowledge_base(self, patterns: List[LearningPattern],
                             optimizations: List[StrategyOptimization],
                             insights: List[PredictiveInsight]) -> bool:
        """更新知识库"""
        try:
            knowledge_update = {
                'timestamp': datetime.now().isoformat(),
                'patterns_learned': len(patterns),
                'strategies_optimized': len(optimizations),
                'insights_generated': len(insights),
                'new_knowledge': {
                    'patterns': [p.__dict__ for p in patterns],
                    'optimizations': [o.__dict__ for o in optimizations],
                    'insights': [i.__dict__ for i in insights]
                }
            }

            self.knowledge_base.append(knowledge_update)

            # 只保留最近100条知识记录
            if len(self.knowledge_base) > 100:
                self.knowledge_base = self.knowledge_base[-100:]

            self._save_knowledge_base()

            return True

        except Exception as e:
            logger.error(f"更新知识库失败: {e}")
            return False

    def _generate_learning_recommendations(self, patterns: List[LearningPattern],
                                         optimizations: List[StrategyOptimization],
                                         insights: List[PredictiveInsight]) -> List[str]:
        """生成学习建议"""
        recommendations = []

        # 基于模式的建议
        if patterns:
            recommendations.append(f"基于 {len(patterns)} 个新发现的模式优化测试策略")

        # 基于优化的建议
        if optimizations:
            high_impact_optimizations = [o for o in optimizations if o.expected_improvement > 0.1]
            if high_impact_optimizations:
                recommendations.append(f"实施 {len(high_impact_optimizations)} 个高影响力的策略优化")

        # 基于洞察的建议
        high_risk_insights = [i for i in insights if i.risk_level == 'high']
        if high_risk_insights:
            recommendations.append(f"关注 {len(high_risk_insights)} 个高风险预测洞察")

        # 通用建议
        recommendations.extend([
            "定期审查和更新学习模式",
            "监控策略优化效果",
            "关注预测洞察的准确性",
            "持续收集更多执行数据"
        ])

        return recommendations

    def _generate_learning_report(self, patterns: List[LearningPattern],
                                optimizations: List[StrategyOptimization],
                                insights: List[PredictiveInsight],
                                recommendations: List[str]):
        """生成学习报告"""
        report_path = Path("test_logs/continuous_learning_report.md")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 持续学习报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 📊 学习成果概览\n\n")
            f.write(f"- **新发现模式**: {len(patterns)}\n")
            f.write(f"- **策略优化建议**: {len(optimizations)}\n")
            f.write(f"- **预测洞察**: {len(insights)}\n")
            f.write(f"- **学习建议**: {len(recommendations)}\n\n")

            f.write("## 🧠 发现的模式\n\n")
            for pattern in patterns[:5]:  # 显示前5个
                f.write(f"### {pattern.pattern_type.title()}: {pattern.pattern_id}\n\n")
                f.write(f"- **置信度**: {pattern.confidence:.2f}\n")
                f.write(f"- **出现次数**: {pattern.occurrences}\n")
                f.write(f"- **建议**: {', '.join(pattern.recommendations[:2])}\n\n")

            f.write("## 🎯 策略优化建议\n\n")
            for opt in optimizations[:3]:  # 显示前3个
                f.write(f"### {opt.strategy_name}\n\n")
                f.write(f"- **预期改善**: {opt.expected_improvement:.1f}\n")
                f.write(f"- **置信度**: {opt.confidence:.2f}\n")
                f.write(f"- **理由**: {opt.rationale}\n\n")

            f.write("## 🔮 预测洞察\n\n")
            for insight in insights[:3]:  # 显示前3个
                f.write(f"### {insight.insight_type.replace('_', ' ').title()}\n\n")
                f.write(f"- **描述**: {insight.description}\n")
                f.write(f"- **风险等级**: {insight.risk_level}\n")
                f.write(f"- **时间范围**: {insight.timeframe}\n")
                f.write(f"- **置信度**: {insight.confidence:.2f}\n\n")

            f.write("## 💡 学习建议\n\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")

            f.write("\n## 📈 持续学习价值\n\n")
            f.write("### 对测试团队的价值\n")
            f.write("1. **模式识别**: 自动发现测试执行的规律和趋势\n")
            f.write("2. **策略优化**: 基于数据驱动的测试策略改进\n")
            f.write("3. **风险预测**: 提前识别和预防潜在问题\n")
            f.write("4. **知识积累**: 构建可重用的测试经验库\n")
            f.write("\n### 对产品质量的价值\n")
            f.write("1. **质量趋势**: 监控和预测产品质量变化\n")
            f.write("2. **问题预防**: 在问题发生前采取预防措施\n")
            f.write("3. **效率提升**: 优化测试执行和资源使用\n")
            f.write("4. **持续改进**: 基于学习的不断质量优化\n")

        logger.info(f"持续学习报告已生成: {report_path}")

    def _load_knowledge_base(self) -> List[Dict[str, Any]]:
        """加载知识库"""
        knowledge_file = Path("test_logs/knowledge_base.json")

        if not knowledge_file.exists():
            return []

        try:
            with open(knowledge_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载知识库失败: {e}")
            return []

    def _save_knowledge_base(self):
        """保存知识库"""
        knowledge_file = Path("test_logs/knowledge_base.json")

        try:
            knowledge_file.parent.mkdir(parents=True, exist_ok=True)
            with open(knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存知识库失败: {e}")

    def get_learning_insights(self) -> Dict[str, Any]:
        """获取学习洞察"""
        return {
            'total_patterns': len(self.pattern_learner.patterns),
            'recent_patterns': len(self.pattern_learner.get_recent_patterns(7)),
            'knowledge_entries': len(self.knowledge_base),
            'insights_generated': sum(len(entry.get('insights', [])) for entry in self.knowledge_base)
        }


def main():
    """主函数"""
    system = ContinuousLearningSystem()

    print("🧠 持续学习系统启动")
    print("🎯 功能: 模式学习 + 策略优化 + 预测洞察 + 知识积累")

    # 执行持续学习
    result = system.perform_continuous_learning()

    print("\n📊 学习结果:")
    print(f"  🧩 发现模式: {result.patterns_discovered}")
    print(f"  🎯 策略优化: {result.strategies_optimized}")
    print(f"  🔮 生成洞察: {result.insights_generated}")
    print(f"  📚 知识更新: {'✅' if result.knowledge_updated else '❌'}")
    print(".2")
    # 显示学习洞察
    insights = system.get_learning_insights()
    print("\n📈 学习洞察:")
    print(f"  📊 总模式数: {insights['total_patterns']}")
    print(f"  🆕 最近模式: {insights['recent_patterns']}")
    print(f"  📚 知识条目: {insights['knowledge_entries']}")
    print(f"  💡 生成洞察: {insights['insights_generated']}")

    # 显示建议
    if result.recommendations:
        print("\n💡 学习建议:")
        for i, rec in enumerate(result.recommendations[:3], 1):  # 显示前3个
            print(f"  {i}. {rec}")

    print("\n📄 详细报告已保存到: test_logs/continuous_learning_report.md")
    print("\n✅ 持续学习系统运行完成")


if __name__ == "__main__":
    main()
