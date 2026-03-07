#!/usr/bin/env python3
"""
智能缺陷预测系统 - Phase 5智能化测试

基于历史测试数据和代码分析，智能预测潜在缺陷和风险：
1. 分析测试失败模式和趋势
2. 识别高风险代码区域
3. 预测潜在的缺陷类型
4. 提供预防性测试建议

作者: AI Assistant
创建时间: 2025年12月4日
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestFailurePattern:
    """测试失败模式"""
    pattern_id: str
    failure_type: str  # import_error, assertion_error, timeout, etc.
    affected_module: str
    affected_function: str
    error_message: str
    frequency: int
    last_occurrence: datetime
    risk_level: str  # high, medium, low
    predicted_impact: str


@dataclass
class DefectPrediction:
    """缺陷预测结果"""
    file_path: str
    line_number: Optional[int]
    defect_type: str
    confidence_score: float
    risk_level: str
    description: str
    recommended_tests: List[str]
    prevention_suggestions: List[str]


@dataclass
class RiskAssessment:
    """风险评估结果"""
    module_name: str
    overall_risk_score: float
    risk_factors: List[str]
    predicted_defects: List[DefectPrediction]
    mitigation_strategies: List[str]
    priority_level: str  # critical, high, medium, low


class IntelligentDefectPredictor:
    """
    智能缺陷预测器

    通过分析历史数据和代码特征，预测潜在的缺陷
    """

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent)
        self.test_logs_dir = self.project_root / "test_logs"
        self.failure_patterns = {}
        self.defect_predictions = []

    def analyze_historical_failures(self) -> Dict[str, TestFailurePattern]:
        """
        分析历史测试失败数据
        """
        print("🔍 分析历史测试失败模式...")

        failure_patterns = {}

        # 扫描test_logs目录中的日志文件
        if self.test_logs_dir.exists():
            for log_file in self.test_logs_dir.glob("*.json"):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                        self._extract_failure_patterns(log_data, failure_patterns)
                except Exception as e:
                    logger.warning(f"解析日志文件失败 {log_file}: {e}")

        # 分析pytest缓存和输出
        self._analyze_pytest_cache(failure_patterns)

        print(f"✅ 识别出 {len(failure_patterns)} 个失败模式")
        self.failure_patterns = failure_patterns
        return failure_patterns

    def _extract_failure_patterns(self, log_data: Dict[str, Any],
                                patterns: Dict[str, TestFailurePattern]):
        """从日志数据中提取失败模式"""
        # 这里可以根据实际日志格式进行解析
        # 简化实现，基于常见的错误模式

        if "error_summary" in log_data:
            error_text = log_data["error_summary"]
            self._categorize_error(error_text, patterns)

        # 分析层级结果
        if "layer_results" in log_data:
            for layer_name, layer_data in log_data["layer_results"].items():
                if layer_data.get("status") == "failed":
                    pattern_key = f"{layer_name}_execution_failure"
                    if pattern_key not in patterns:
                        patterns[pattern_key] = TestFailurePattern(
                            pattern_id=pattern_key,
                            failure_type="execution_failure",
                            affected_module=layer_name,
                            affected_function="multiple",
                            error_message="Layer execution failed",
                            frequency=1,
                            last_occurrence=datetime.now(),
                            risk_level="high",
                            predicted_impact="Layer functionality compromised"
                        )
                    else:
                        patterns[pattern_key].frequency += 1

    def _categorize_error(self, error_text: str, patterns: Dict[str, TestFailurePattern]):
        """对错误进行分类"""
        error_text = str(error_text).lower()

        # 导入错误
        if "importerror" in error_text or "no module named" in error_text:
            pattern_key = "import_errors"
            failure_type = "import_error"
            risk_level = "high"
            impact = "Module dependencies broken"

        # 断言错误
        elif "assertionerror" in error_text:
            pattern_key = "assertion_failures"
            failure_type = "assertion_error"
            risk_level = "medium"
            impact = "Logic errors in code"

        # 超时错误
        elif "timeout" in error_text:
            pattern_key = "timeout_errors"
            failure_type = "timeout"
            risk_level = "medium"
            impact = "Performance issues"

        # 语法错误
        elif "syntaxerror" in error_text:
            pattern_key = "syntax_errors"
            failure_type = "syntax_error"
            risk_level = "high"
            impact = "Code cannot execute"

        else:
            pattern_key = "other_errors"
            failure_type = "unknown_error"
            risk_level = "low"
            impact = "Unspecified errors"

        if pattern_key not in patterns:
            patterns[pattern_key] = TestFailurePattern(
                pattern_id=pattern_key,
                failure_type=failure_type,
                affected_module="multiple",
                affected_function="unknown",
                error_message=error_text[:200],
                frequency=1,
                last_occurrence=datetime.now(),
                risk_level=risk_level,
                predicted_impact=impact
            )
        else:
            patterns[pattern_key].frequency += 1

    def _analyze_pytest_cache(self, patterns: Dict[str, TestFailurePattern]):
        """分析pytest缓存中的失败信息"""
        # 这里可以分析.pytest_cache目录中的缓存文件
        # 简化实现，基于已有的错误模式进行推断
        pass

    def predict_defects(self) -> List[DefectPrediction]:
        """
        基于失败模式和代码分析预测缺陷
        """
        print("🔮 基于历史数据预测潜在缺陷...")

        predictions = []

        # 基于失败模式预测
        for pattern in self.failure_patterns.values():
            if pattern.failure_type == "import_error" and pattern.frequency > 2:
                # 频繁的导入错误可能表示模块依赖问题
                prediction = DefectPrediction(
                    file_path=f"src/{pattern.affected_module.replace('.', '/')}",
                    line_number=None,
                    defect_type="dependency_issue",
                    confidence_score=min(pattern.frequency * 0.1, 0.9),
                    risk_level="high",
                    description=f"频繁的导入错误表明{pattern.affected_module}模块存在依赖问题",
                    recommended_tests=[
                        f"test_{pattern.affected_module}_import_stability",
                        f"test_{pattern.affected_module}_dependency_chain"
                    ],
                    prevention_suggestions=[
                        "检查模块导入路径",
                        "验证依赖关系",
                        "添加导入错误处理"
                    ]
                )
                predictions.append(prediction)

            elif pattern.failure_type == "assertion_error":
                # 断言错误可能表示逻辑缺陷
                prediction = DefectPrediction(
                    file_path=f"src/{pattern.affected_module.replace('.', '/')}",
                    line_number=None,
                    defect_type="logic_error",
                    confidence_score=min(pattern.frequency * 0.15, 0.95),
                    risk_level="medium",
                    description=f"断言失败表明{pattern.affected_module}存在逻辑错误",
                    recommended_tests=[
                        f"test_{pattern.affected_module}_edge_cases",
                        f"test_{pattern.affected_module}_boundary_conditions"
                    ],
                    prevention_suggestions=[
                        "增加边界条件测试",
                        "添加数据验证",
                        "实现错误恢复机制"
                    ]
                )
                predictions.append(prediction)

        # 基于代码复杂度预测
        code_predictions = self._predict_from_code_complexity()
        predictions.extend(code_predictions)

        # 基于测试覆盖率预测
        coverage_predictions = self._predict_from_test_coverage()
        predictions.extend(coverage_predictions)

        print(f"✅ 生成 {len(predictions)} 个缺陷预测")
        self.defect_predictions = predictions
        return predictions

    def _predict_from_code_complexity(self) -> List[DefectPrediction]:
        """基于代码复杂度预测缺陷"""
        predictions = []

        # 扫描源代码文件
        for py_file in self.project_root.glob("src/**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 计算复杂度指标
                lines_of_code = len(content.split('\n'))
                function_count = len(re.findall(r'def \w+', content))
                class_count = len(re.findall(r'class \w+', content))
                import_count = len(re.findall(r'^(import|from)', content, re.MULTILINE))

                # 复杂度评分
                complexity_score = (function_count * 0.3 +
                                  class_count * 0.4 +
                                  import_count * 0.2 +
                                  lines_of_code * 0.001)

                if complexity_score > 10:  # 高复杂度文件
                    prediction = DefectPrediction(
                        file_path=str(py_file),
                        line_number=None,
                        defect_type="complexity_risk",
                        confidence_score=min(complexity_score * 0.05, 0.8),
                        risk_level="medium" if complexity_score < 20 else "high",
                        description=f"文件复杂度较高 ({complexity_score:.1f})，容易出现缺陷",
                        recommended_tests=[
                            f"test_{py_file.stem}_integration_scenarios",
                            f"test_{py_file.stem}_error_handling"
                        ],
                        prevention_suggestions=[
                            "考虑重构复杂函数",
                            "增加单元测试覆盖",
                            "添加代码审查"
                        ]
                    )
                    predictions.append(prediction)

            except Exception as e:
                logger.warning(f"分析文件复杂度失败 {py_file}: {e}")

        return predictions

    def _predict_from_test_coverage(self) -> List[DefectPrediction]:
        """基于测试覆盖率预测缺陷"""
        predictions = []

        # 分析测试覆盖率报告
        coverage_files = list(self.test_logs_dir.glob("*coverage*.html"))
        if coverage_files:
            # 这里可以解析coverage.py生成的HTML报告
            # 简化实现，基于文件名推断
            for coverage_file in coverage_files:
                if "ml" in coverage_file.name.lower():
                    prediction = DefectPrediction(
                        file_path="src/ml/",
                        line_number=None,
                        defect_type="coverage_gap",
                        confidence_score=0.7,
                        risk_level="high",
                        description="ML模块测试覆盖率不足，存在较高缺陷风险",
                        recommended_tests=[
                            "test_ml_model_validation",
                            "test_ml_feature_engineering",
                            "test_ml_prediction_accuracy"
                        ],
                        prevention_suggestions=[
                            "增加ML模块单元测试",
                            "添加集成测试覆盖",
                            "实现模型验证机制"
                        ]
                    )
                    predictions.append(prediction)

        return predictions

    def assess_module_risks(self) -> Dict[str, RiskAssessment]:
        """
        评估各模块的风险水平
        """
        print("📊 评估模块风险水平...")

        module_risks = {}

        # 按模块分组预测结果
        module_predictions = {}
        for prediction in self.defect_predictions:
            module = prediction.file_path.split('/')[1] if '/' in prediction.file_path else 'unknown'
            if module not in module_predictions:
                module_predictions[module] = []
            module_predictions[module].append(prediction)

        # 计算每个模块的风险评分
        for module_name, predictions in module_predictions.items():
            # 风险评分计算
            base_risk = len(predictions) * 0.1
            high_risk_count = sum(1 for p in predictions if p.risk_level == "high")
            avg_confidence = statistics.mean(p.confidence_score for p in predictions) if predictions else 0

            overall_risk = min(base_risk + high_risk_count * 0.2 + avg_confidence * 0.3, 1.0)

            # 确定优先级
            if overall_risk > 0.8:
                priority = "critical"
            elif overall_risk > 0.6:
                priority = "high"
            elif overall_risk > 0.4:
                priority = "medium"
            else:
                priority = "low"

            # 识别风险因素
            risk_factors = []
            if high_risk_count > 0:
                risk_factors.append(f"{high_risk_count}个高风险缺陷预测")
            if len(predictions) > 5:
                risk_factors.append("缺陷预测数量过多")
            if avg_confidence > 0.7:
                risk_factors.append("预测置信度较高")

            # 制定缓解策略
            mitigation_strategies = []
            if "import_errors" in [p.defect_type for p in predictions]:
                mitigation_strategies.append("修复模块依赖关系")
            if "complexity_risk" in [p.defect_type for p in predictions]:
                mitigation_strategies.append("重构复杂代码")
            if "coverage_gap" in [p.defect_type for p in predictions]:
                mitigation_strategies.append("增加测试覆盖率")

            module_risks[module_name] = RiskAssessment(
                module_name=module_name,
                overall_risk_score=overall_risk,
                risk_factors=risk_factors,
                predicted_defects=predictions,
                mitigation_strategies=mitigation_strategies,
                priority_level=priority
            )

        print(f"✅ 完成 {len(module_risks)} 个模块的风险评估")
        return module_risks

    def generate_intelligent_report(self) -> Dict[str, Any]:
        """
        生成智能缺陷预测报告
        """
        print("📋 生成智能缺陷预测报告...")

        # 分析历史失败
        failure_patterns = self.analyze_historical_failures()

        # 预测缺陷
        predictions = self.predict_defects()

        # 评估风险
        module_risks = self.assess_module_risks()

        # 生成报告
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "analysis_period": "recent_test_runs",
                "prediction_model": "pattern_based_ai",
                "confidence_level": "medium"
            },
            "failure_analysis": {
                "total_patterns": len(failure_patterns),
                "patterns_by_type": self._group_patterns_by_type(failure_patterns),
                "most_frequent_failures": self._get_most_frequent_failures(failure_patterns)
            },
            "defect_predictions": {
                "total_predictions": len(predictions),
                "predictions_by_type": self._group_predictions_by_type(predictions),
                "high_confidence_predictions": [p for p in predictions if p.confidence_score > 0.8],
                "critical_risk_predictions": [p for p in predictions if p.risk_level == "high"]
            },
            "risk_assessment": {
                "module_risks": {name: asdict(risk) for name, risk in module_risks.items()},
                "high_priority_modules": [name for name, risk in module_risks.items() if risk.priority_level in ["critical", "high"]],
                "overall_system_risk": self._calculate_system_risk(module_risks)
            },
            "defect_predictions_detail": [asdict(p) for p in predictions],
            "recommendations": {
                "immediate_actions": self._generate_immediate_actions(predictions, module_risks),
                "preventive_measures": self._generate_preventive_measures(failure_patterns),
                "testing_priorities": self._generate_testing_priorities(module_risks)
            }
        }

        return report

    def _group_patterns_by_type(self, patterns: Dict[str, TestFailurePattern]) -> Dict[str, int]:
        """按类型分组失败模式"""
        type_counts = {}
        for pattern in patterns.values():
            failure_type = pattern.failure_type
            type_counts[failure_type] = type_counts.get(failure_type, 0) + 1
        return type_counts

    def _get_most_frequent_failures(self, patterns: Dict[str, TestFailurePattern]) -> List[str]:
        """获取最频繁的失败模式"""
        sorted_patterns = sorted(patterns.values(), key=lambda p: p.frequency, reverse=True)
        return [p.pattern_id for p in sorted_patterns[:5]]

    def _group_predictions_by_type(self, predictions: List[DefectPrediction]) -> Dict[str, int]:
        """按类型分组缺陷预测"""
        type_counts = {}
        for prediction in predictions:
            defect_type = prediction.defect_type
            type_counts[defect_type] = type_counts.get(defect_type, 0) + 1
        return type_counts

    def _calculate_system_risk(self, module_risks: Dict[str, RiskAssessment]) -> float:
        """计算系统整体风险"""
        if not module_risks:
            return 0.0

        total_risk = sum(risk.overall_risk_score for risk in module_risks.values())
        return min(total_risk / len(module_risks), 1.0)

    def _generate_immediate_actions(self, predictions: List[DefectPrediction],
                                  module_risks: Dict[str, RiskAssessment]) -> List[str]:
        """生成立即行动建议"""
        actions = []

        # 高风险模块优先处理
        high_risk_modules = [name for name, risk in module_risks.items()
                           if risk.priority_level == "critical"]
        if high_risk_modules:
            actions.append(f"立即处理高风险模块: {', '.join(high_risk_modules)}")

        # 高置信度缺陷优先修复
        high_confidence_predictions = [p for p in predictions if p.confidence_score > 0.8]
        if high_confidence_predictions:
            actions.append(f"修复高置信度缺陷: {len(high_confidence_predictions)} 个预测")

        # 频繁失败模式处理
        frequent_failures = [p for p in self.failure_patterns.values() if p.frequency > 3]
        if frequent_failures:
            actions.append(f"解决频繁失败模式: {len(frequent_failures)} 个模式")

        return actions

    def _generate_preventive_measures(self, failure_patterns: Dict[str, TestFailurePattern]) -> List[str]:
        """生成预防措施建议"""
        measures = []

        # 基于失败模式建议预防措施
        import_errors = [p for p in failure_patterns.values() if p.failure_type == "import_error"]
        if import_errors:
            measures.append("加强模块依赖管理和导入验证")

        assertion_errors = [p for p in failure_patterns.values() if p.failure_type == "assertion_error"]
        if assertion_errors:
            measures.append("增加边界条件和异常情况测试")

        timeout_errors = [p for p in failure_patterns.values() if p.failure_type == "timeout"]
        if timeout_errors:
            measures.append("优化代码性能和添加性能测试")

        measures.extend([
            "实施代码审查和静态分析",
            "建立持续集成质量门禁",
            "完善错误监控和告警机制"
        ])

        return measures

    def _generate_testing_priorities(self, module_risks: Dict[str, RiskAssessment]) -> List[str]:
        """生成测试优先级建议"""
        priorities = []

        # 按风险等级排序模块
        sorted_modules = sorted(module_risks.items(),
                              key=lambda x: x[1].overall_risk_score, reverse=True)

        priorities.append("测试优先级排序:")
        for module_name, risk in sorted_modules[:5]:  # 前5个最高风险模块
            priorities.append(f"  1. {module_name} (风险: {risk.overall_risk_score:.2f}, 优先级: {risk.priority_level})")

        return priorities

    def save_report(self, report: Dict[str, Any]):
        """保存缺陷预测报告"""
        output_dir = Path("test_logs")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"defect_prediction_report_{timestamp}.json"

        output_file = output_dir / filename

        # 处理JSON序列化问题
        serializable_report = self._make_json_serializable(report)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)

        # 生成摘要报告
        summary_file = output_dir / f"defect_prediction_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("智能缺陷预测报告摘要\n")
            f.write("=" * 40 + "\n")
            f.write(f"生成时间: {report['report_metadata']['generated_at']}\n")
            f.write(f"失败模式数: {report['failure_analysis']['total_patterns']}\n")
            f.write(f"缺陷预测数: {report['defect_predictions']['total_predictions']}\n")
            f.write(f"系统整体风险: {report['risk_assessment']['overall_system_risk']:.2f}\n")
            f.write(f"高优先级模块数: {len(report['risk_assessment']['high_priority_modules'])}\n")

            f.write("\n立即行动建议:\n")
            for action in report['recommendations']['immediate_actions']:
                f.write(f"  • {action}\n")

            f.write("\n预防措施建议:\n")
            for measure in report['recommendations']['preventive_measures'][:5]:
                f.write(f"  • {measure}\n")

        print(f"💾 缺陷预测报告已保存: {output_file}")
        print(f"📄 摘要已保存: {summary_file}")

    def _make_json_serializable(self, obj: Any) -> Any:
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dataclass_fields__'):
            # dataclass对象转换为字典
            return asdict(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj


def main():
    """主函数"""
    predictor = IntelligentDefectPredictor()

    print("🚀 启动智能缺陷预测系统")
    print("=" * 50)

    # 生成智能报告
    report = predictor.generate_intelligent_report()

    # 保存报告
    predictor.save_report(report)

    # 输出关键发现
    failure_analysis = report['failure_analysis']
    defect_predictions = report['defect_predictions']
    risk_assessment = report['risk_assessment']

    print("\n🎯 关键发现:")
    print(f"  • 识别失败模式: {failure_analysis['total_patterns']} 个")
    print(f"  • 缺陷预测数量: {defect_predictions['total_predictions']} 个")
    print(f"  • 系统整体风险: {risk_assessment['overall_system_risk']:.2f}")
    print(f"  • 高优先级模块: {len(risk_assessment['high_priority_modules'])} 个")

    print("\n✅ 智能缺陷预测分析完成！")
    print("系统测试智能化水平显著提升！")


if __name__ == "__main__":
    main()
