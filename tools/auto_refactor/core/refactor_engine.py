#!/usr/bin/env python3
"""
自动化重构引擎

核心引擎负责协调整个重构过程，包括分析、执行、安全验证等。
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from tools.smart_code_analyzer import SmartCodeAnalyzer, AnalysisResult, RefactoringSuggestion
from .safety_manager import SafetyManager
from .config import RefactorConfig, SafetyLevel


@dataclass
class RefactorResult:
    """重构结果"""

    file_path: str
    refactor_type: str
    success: bool
    changes_made: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    backup_created: bool = False
    validation_passed: bool = True

    def add_change(self, change_type: str, description: str, **kwargs):
        """添加变更记录"""
        self.changes_made.append({
            'type': change_type,
            'description': description,
            'timestamp': time.time(),
            **kwargs
        })

    def add_error(self, error: str):
        """添加错误"""
        self.errors.append(error)

    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(warning)


@dataclass
class RefactorStats:
    """重构统计"""

    total_files_processed: int = 0
    total_refactors_attempted: int = 0
    successful_refactors: int = 0
    failed_refactors: int = 0
    skipped_refactors: int = 0
    total_execution_time: float = 0.0
    backups_created: int = 0
    validations_performed: int = 0

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_refactors_attempted == 0:
            return 0.0
        return self.successful_refactors / self.total_refactors_attempted

    @property
    def average_execution_time(self) -> float:
        """平均执行时间"""
        if self.successful_refactors == 0:
            return 0.0
        return self.total_execution_time / self.successful_refactors


class AutoRefactorEngine:
    """自动化重构引擎"""

    def __init__(self, config: Optional[RefactorConfig] = None):
        self.config = config or RefactorConfig()
        self.safety_manager = SafetyManager(self.config)
        self.analyzer = SmartCodeAnalyzer()

        # 统计信息
        self.stats = RefactorStats()

        # 执行器映射
        self.executors = self._load_executors()

    def _load_executors(self) -> Dict[str, Any]:
        """加载重构执行器"""
        # 这里会动态加载所有可用的执行器
        # 暂时返回空字典，实际实现中会从executors包中加载
        return {}

    def execute_auto_refactor(self, analysis_results: Dict[str, AnalysisResult],
                              safety_level: str = 'high') -> Dict[str, List[RefactorResult]]:
        """
        执行自动重构

        Args:
            analysis_results: 分析结果字典
            safety_level: 安全级别 ('low', 'medium', 'high')

        Returns:
            重构结果字典
        """
        self.config.safety_level = SafetyLevel(safety_level)

        results = {}

        # 按优先级排序重构建议
        all_suggestions = self._collect_all_suggestions(analysis_results)
        prioritized_suggestions = self._prioritize_suggestions(all_suggestions)

        # 执行重构
        if self.config.parallel_processing:
            results = self._execute_parallel(prioritized_suggestions)
        else:
            results = self._execute_sequential(prioritized_suggestions)

        return results

    def _collect_all_suggestions(self, analysis_results: Dict[str, AnalysisResult]) -> List[RefactoringSuggestion]:
        """收集所有重构建议"""
        all_suggestions = []
        for result in analysis_results.values():
            all_suggestions.extend(result.suggestions)
        return all_suggestions

    def _prioritize_suggestions(self, suggestions: List[RefactoringSuggestion]) -> List[RefactoringSuggestion]:
        """按优先级排序建议"""
        def priority_key(suggestion):
            severity_weights = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'info': 0}
            effort_weights = {'low': 3, 'medium': 2, 'high': 1}

            severity_score = severity_weights.get(suggestion.severity, 0)
            confidence_score = suggestion.confidence
            effort_score = effort_weights.get(suggestion.estimated_effort, 2)

            return (severity_score, confidence_score, effort_score)

        return sorted(suggestions, key=priority_key, reverse=True)

    def _execute_sequential(self, suggestions: List[RefactoringSuggestion]) -> Dict[str, List[RefactorResult]]:
        """顺序执行重构"""
        results = {}

        for suggestion in suggestions:
            if suggestion.file_path not in results:
                results[suggestion.file_path] = []

            result = self._execute_single_refactor(suggestion)
            results[suggestion.file_path].append(result)

            # 更新统计
            self._update_stats(result)

            # 检查是否需要停止
            if self.config.fail_fast and not result.success:
                break

        return results

    def _execute_parallel(self, suggestions: List[RefactoringSuggestion]) -> Dict[str, List[RefactorResult]]:
        """并行执行重构"""
        results = {}
        max_workers = min(self.config.max_workers, len(suggestions))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_suggestion = {
                executor.submit(self._execute_single_refactor, suggestion): suggestion
                for suggestion in suggestions
            }

            # 收集结果
            for future in as_completed(future_to_suggestion):
                suggestion = future_to_suggestion[future]
                try:
                    result = future.result()

                    if suggestion.file_path not in results:
                        results[suggestion.file_path] = []
                    results[suggestion.file_path].append(result)

                    # 更新统计
                    self._update_stats(result)

                except Exception as e:
                    # 创建失败结果
                    failed_result = RefactorResult(
                        file_path=suggestion.file_path,
                        refactor_type=suggestion.suggestion_type,
                        success=False,
                        errors=[f"Execution failed: {str(e)}"]
                    )

                    if suggestion.file_path not in results:
                        results[suggestion.file_path] = []
                    results[suggestion.file_path].append(failed_result)

                    self._update_stats(failed_result)

        return results

    def _execute_single_refactor(self, suggestion: RefactoringSuggestion) -> RefactorResult:
        """执行单个重构"""
        start_time = time.time()

        result = RefactorResult(
            file_path=suggestion.file_path,
            refactor_type=suggestion.suggestion_type,
            success=False
        )

        try:
            # 安全检查
            if not self.safety_manager.check_safety(suggestion):
                result.add_error("Safety check failed")
                result.success = False
                result.execution_time = time.time() - start_time
                return result

            # 创建备份
            if self.config.safety_level != SafetyLevel.LOW:
                backup_result = self.safety_manager.create_backup(suggestion.file_path)
                if backup_result.success:
                    result.backup_created = True
                else:
                    result.add_warning(f"Backup failed: {backup_result.error}")

            # 获取执行器
            executor = self.executors.get(suggestion.suggestion_type)
            if not executor:
                result.add_error(f"No executor found for type: {suggestion.suggestion_type}")
                result.execution_time = time.time() - start_time
                return result

            # 执行重构
            refactor_result = executor.execute(suggestion, self.config)

            # 验证结果
            if refactor_result.success and self.config.validation_enabled:
                validation_result = self.safety_manager.validate_refactor(
                    suggestion.file_path, refactor_result
                )

                if not validation_result.success:
                    result.add_error(f"Validation failed: {validation_result.error}")
                    result.validation_passed = False

                    # 回滚如果验证失败
                    if self.config.rollback_on_failure and result.backup_created:
                        rollback_result = self.safety_manager.rollback_backup(suggestion.file_path)
                        if rollback_result.success:
                            result.add_warning("Rolled back due to validation failure")
                        else:
                            result.add_error(f"Rollback failed: {rollback_result.error}")

                    result.success = False
                else:
                    result.success = True
                    result.changes_made.extend(refactor_result.changes)

            elif refactor_result.success:
                result.success = True
                result.changes_made.extend(refactor_result.changes)

            # 添加警告和错误
            result.warnings.extend(refactor_result.warnings)
            result.errors.extend(refactor_result.errors)

        except Exception as e:
            result.add_error(f"Unexpected error: {str(e)}")
            result.success = False

            # 尝试回滚
            if self.config.rollback_on_failure and result.backup_created:
                try:
                    rollback_result = self.safety_manager.rollback_backup(suggestion.file_path)
                    if rollback_result.success:
                        result.add_warning("Rolled back due to unexpected error")
                except Exception as rollback_error:
                    result.add_error(f"Rollback failed: {str(rollback_error)}")

        result.execution_time = time.time() - start_time
        return result

    def _update_stats(self, result: RefactorResult):
        """更新统计信息"""
        self.stats.total_refactors_attempted += 1

        if result.success:
            self.stats.successful_refactors += 1
        else:
            self.stats.failed_refactors += 1

        if result.backup_created:
            self.stats.backups_created += 1

        if hasattr(result, 'validation_passed') and result.validation_passed:
            self.stats.validations_performed += 1

        self.stats.total_execution_time += result.execution_time

    def get_execution_stats(self) -> RefactorStats:
        """获取执行统计"""
        return self.stats

    def generate_refactor_report(self, results: Dict[str, List[RefactorResult]]) -> Dict[str, Any]:
        """
        生成重构报告

        Args:
            results: 重构结果字典

        Returns:
            详细报告字典
        """
        total_files = len(results)
        total_refactors = sum(len(file_results) for file_results in results.values())
        successful_refactors = sum(
            sum(1 for r in file_results if r.success)
            for file_results in results.values()
        )

        # 按类型统计
        type_stats = {}
        for file_results in results.values():
            for result in file_results:
                refactor_type = result.refactor_type
                if refactor_type not in type_stats:
                    type_stats[refactor_type] = {'attempted': 0, 'successful': 0}
                type_stats[refactor_type]['attempted'] += 1
                if result.success:
                    type_stats[refactor_type]['successful'] += 1

        # 收集所有错误和警告
        all_errors = []
        all_warnings = []
        for file_results in results.values():
            for result in file_results:
                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)

        return {
            'summary': {
                'total_files_processed': total_files,
                'total_refactors_attempted': total_refactors,
                'successful_refactors': successful_refactors,
                'failed_refactors': total_refactors - successful_refactors,
                'success_rate': successful_refactors / total_refactors if total_refactors > 0 else 0,
                'total_execution_time': self.stats.total_execution_time,
                'backups_created': self.stats.backups_created
            },
            'by_type': type_stats,
            'errors': all_errors[:50],  # 限制数量
            'warnings': all_warnings[:50],  # 限制数量
            'file_results': {
                file_path: [
                    {
                        'type': r.refactor_type,
                        'success': r.success,
                        'execution_time': r.execution_time,
                        'changes_count': len(r.changes_made),
                        'errors_count': len(r.errors),
                        'warnings_count': len(r.warnings)
                    }
                    for r in file_results
                ]
                for file_path, file_results in results.items()
            },
            'generated_at': time.time(),
            'config_used': {
                'safety_level': self.config.safety_level.value,
                'parallel_processing': self.config.parallel_processing,
                'validation_enabled': self.config.validation_enabled,
                'fail_fast': self.config.fail_fast
            }
        }

    def analyze_refactor_impact(self, before_results: Dict[str, AnalysisResult],
                                after_results: Dict[str, AnalysisResult]) -> Dict[str, Any]:
        """
        分析重构影响

        Args:
            before_results: 重构前分析结果
            after_results: 重构后分析结果

        Returns:
            影响分析结果
        """
        impact = {
            'quality_improvement': {},
            'metrics_changes': {},
            'files_affected': set(),
            'overall_impact_score': 0.0
        }

        # 比较质量指标
        for file_path in set(before_results.keys()) | set(after_results.keys()):
            before = before_results.get(file_path)
            after = after_results.get(file_path)

            if before and after:
                impact['files_affected'].add(file_path)

                # 质量评分变化
                quality_change = after.quality_score - before.quality_score
                impact['quality_improvement'][file_path] = quality_change

                # 具体指标变化
                impact['metrics_changes'][file_path] = {
                    'complexity_change': after.metrics.cyclomatic_complexity - before.metrics.cyclomatic_complexity,
                    'lines_change': after.metrics.lines_of_code - before.metrics.lines_of_code,
                    'duplication_change': after.metrics.duplication_percentage - before.metrics.duplication_percentage,
                    'smell_count_change': (
                        (after.metrics.long_method_count + after.metrics.long_class_count +
                         after.metrics.duplicate_code_blocks + after.metrics.unused_imports +
                         after.metrics.magic_numbers) -
                        (before.metrics.long_method_count + before.metrics.long_class_count +
                         before.metrics.duplicate_code_blocks + before.metrics.unused_imports +
                         before.metrics.magic_numbers)
                    )
                }

        # 计算整体影响分数
        if impact['quality_improvement']:
            avg_quality_improvement = sum(
                impact['quality_improvement'].values()) / len(impact['quality_improvement'])
            impact['overall_impact_score'] = avg_quality_improvement

        return impact
