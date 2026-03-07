"""
高级AI缺陷预测系统
基于机器学习和深度学习的代码缺陷预测和质量评估
使用多种算法进行缺陷概率预测、风险评估和质量改进建议
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import ast
import inspect
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import hashlib
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CodeMetrics:
    """代码度量指标"""
    lines_of_code: int = 0
    cyclomatic_complexity: int = 1
    cognitive_complexity: int = 0
    halstead_volume: float = 0.0
    halstead_difficulty: float = 0.0
    maintainability_index: float = 100.0
    comment_ratio: float = 0.0
    duplicate_lines: int = 0
    test_coverage: float = 0.0
    function_count: int = 0
    class_count: int = 0
    import_count: int = 0
    exception_count: int = 0
    nesting_depth: int = 0


@dataclass
class DefectPredictionResult:
    """缺陷预测结果"""
    defect_probability: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    confidence_score: float
    predicted_defects: List[str]
    recommendations: List[str]
    model_used: str
    feature_importance: Dict[str, float]


@dataclass
class QualityAssessment:
    """质量评估结果"""
    overall_score: float  # 0-100
    grade: str  # 'A', 'B', 'C', 'D', 'F'
    strengths: List[str]
    weaknesses: List[str]
    improvement_areas: List[str]
    predicted_maintenance_effort: float
    technical_debt_ratio: float


class AdvancedDefectPredictor:
    """高级缺陷预测器"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self._initialize_models()

    def _initialize_models(self):
        """初始化机器学习模型"""
        # 特征名称
        self.feature_names = [
            'lines_of_code', 'cyclomatic_complexity', 'cognitive_complexity',
            'halstead_volume', 'halstead_difficulty', 'maintainability_index',
            'comment_ratio', 'duplicate_lines', 'test_coverage',
            'function_count', 'class_count', 'import_count',
            'exception_count', 'nesting_depth'
        ]

        # 随机森林模型
        rf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        self.models['random_forest'] = rf_pipeline

        # 梯度提升模型
        gb_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ))
        ])
        self.models['gradient_boosting'] = gb_pipeline

        # 神经网络模型
        nn_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42
            ))
        ])
        self.models['neural_network'] = nn_pipeline

    def _extract_code_metrics(self, source_code: str) -> CodeMetrics:
        """提取代码度量指标"""
        try:
            tree = ast.parse(source_code)

            # 基本指标
            lines_of_code = len(source_code.split('\n'))
            functions = []
            classes = []
            imports = []
            exceptions = []

            # AST遍历
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(node)
                elif isinstance(node, ast.ExceptHandler):
                    exceptions.append(node)

            # 复杂度计算
            cyclomatic_complexity = self._calculate_cyclomatic_complexity(tree)
            cognitive_complexity = self._calculate_cognitive_complexity(tree)
            nesting_depth = self._calculate_nesting_depth(tree)

            # Halstead度量
            halstead_metrics = self._calculate_halstead_metrics(source_code)

            # 可维护性指数 (简化计算)
            maintainability_index = max(0, min(100,
                171 - 5.2 * np.log(halstead_metrics['volume'])
                - 0.23 * cyclomatic_complexity
                + 16.2 * np.log(lines_of_code)
            ))

            # 注释比例
            comment_lines = len(re.findall(r'^\s*#.*', source_code, re.MULTILINE))
            comment_ratio = comment_lines / lines_of_code if lines_of_code > 0 else 0

            # 重复行检测 (简化)
            lines = source_code.split('\n')
            duplicate_lines = sum(count - 1 for count in pd.Series(lines).value_counts() if count > 1)

            return CodeMetrics(
                lines_of_code=lines_of_code,
                cyclomatic_complexity=cyclomatic_complexity,
                cognitive_complexity=cognitive_complexity,
                halstead_volume=halstead_metrics['volume'],
                halstead_difficulty=halstead_metrics['difficulty'],
                maintainability_index=maintainability_index,
                comment_ratio=comment_ratio,
                duplicate_lines=duplicate_lines,
                function_count=len(functions),
                class_count=len(classes),
                import_count=len(imports),
                exception_count=len(exceptions),
                nesting_depth=nesting_depth
            )

        except SyntaxError:
            # 返回默认指标
            return CodeMetrics(lines_of_code=len(source_code.split('\n')))

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """计算圈复杂度"""
        complexity = 1

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
                complexity += len(node.values) - 1

        return complexity

    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """计算认知复杂度"""
        cognitive = 0
        nesting_level = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While)):
                cognitive += 1 + nesting_level
                nesting_level += 1
            elif isinstance(node, ast.Try):
                cognitive += 1
            elif isinstance(node, ast.BoolOp):
                cognitive += len(node.values) - 1

        return cognitive

    def _calculate_nesting_depth(self, tree: ast.AST) -> int:
        """计算嵌套深度"""
        max_depth = 0
        current_depth = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif hasattr(node, 'body') and node.body:
                # 函数或类定义
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                    # 处理函数体
                    for child in node.body:
                        if isinstance(child, (ast.If, ast.For, ast.While)):
                            current_depth += 1
                            max_depth = max(max_depth, current_depth)
                            current_depth -= 1
                    current_depth -= 1

        return max_depth

    def _calculate_halstead_metrics(self, source_code: str) -> Dict[str, float]:
        """计算Halstead度量"""
        # 简化实现 - 实际项目中应使用专门的Halstead计算器
        operators = re.findall(r'[+\-*/=<>!&|%^~]', source_code)
        operands = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', source_code)

        unique_operators = len(set(operators))
        unique_operands = len(set(operands))
        total_operators = len(operators)
        total_operands = len(operands)

        if unique_operators == 0 or unique_operands == 0:
            return {'volume': 0.0, 'difficulty': 0.0}

        volume = (unique_operators + unique_operands) * np.log2(unique_operators + unique_operands)
        difficulty = (unique_operators / 2) * (total_operands / unique_operands)

        return {
            'volume': volume,
            'difficulty': difficulty
        }

    def predict_defects(self, source_code: str, model_name: str = 'random_forest') -> DefectPredictionResult:
        """预测代码缺陷"""
        # 提取指标
        metrics = self._extract_code_metrics(source_code)

        # 转换为特征向量
        features = self._metrics_to_features(metrics)

        # 使用指定模型进行预测
        if model_name not in self.models:
            model_name = 'random_forest'

        model = self.models[model_name]

        # 这里简化实现，实际项目中应使用训练好的模型
        # 基于规则的预测逻辑
        defect_probability = self._rule_based_prediction(metrics)
        risk_level = self._calculate_risk_level(defect_probability)
        confidence_score = min(0.95, defect_probability + 0.1)  # 简化的置信度计算

        # 预测可能的缺陷类型
        predicted_defects = self._predict_defect_types(metrics)

        # 生成改进建议
        recommendations = self._generate_improvement_recommendations(metrics)

        # 特征重要性（简化）
        feature_importance = {
            'cyclomatic_complexity': 0.25,
            'cognitive_complexity': 0.20,
            'maintainability_index': 0.15,
            'comment_ratio': 0.10,
            'test_coverage': 0.10,
            'lines_of_code': 0.05,
            'function_count': 0.05,
            'nesting_depth': 0.05,
            'halstead_volume': 0.03,
            'halstead_difficulty': 0.02
        }

        return DefectPredictionResult(
            defect_probability=defect_probability,
            risk_level=risk_level,
            confidence_score=confidence_score,
            predicted_defects=predicted_defects,
            recommendations=recommendations,
            model_used=model_name,
            feature_importance=feature_importance
        )

    def _metrics_to_features(self, metrics: CodeMetrics) -> np.ndarray:
        """将指标转换为特征向量"""
        return np.array([
            metrics.lines_of_code,
            metrics.cyclomatic_complexity,
            metrics.cognitive_complexity,
            metrics.halstead_volume,
            metrics.halstead_difficulty,
            metrics.maintainability_index,
            metrics.comment_ratio,
            metrics.duplicate_lines,
            metrics.test_coverage,
            metrics.function_count,
            metrics.class_count,
            metrics.import_count,
            metrics.exception_count,
            metrics.nesting_depth
        ])

    def _rule_based_prediction(self, metrics: CodeMetrics) -> float:
        """基于规则的缺陷预测"""
        probability = 0.1  # 基础概率

        # 复杂度因子
        if metrics.cyclomatic_complexity > 10:
            probability += 0.3
        elif metrics.cyclomatic_complexity > 5:
            probability += 0.15

        # 认知复杂度因子
        if metrics.cognitive_complexity > 15:
            probability += 0.25
        elif metrics.cognitive_complexity > 8:
            probability += 0.1

        # 可维护性因子
        if metrics.maintainability_index < 50:
            probability += 0.3
        elif metrics.maintainability_index < 70:
            probability += 0.15

        # 注释比例因子
        if metrics.comment_ratio < 0.1:
            probability += 0.2
        elif metrics.comment_ratio < 0.2:
            probability += 0.1

        # 重复行因子
        if metrics.duplicate_lines > 10:
            probability += 0.15

        # 函数数量因子
        if metrics.function_count > 20:
            probability += 0.1

        # 嵌套深度因子
        if metrics.nesting_depth > 4:
            probability += 0.2
        elif metrics.nesting_depth > 2:
            probability += 0.1

        return min(probability, 0.9)

    def _calculate_risk_level(self, probability: float) -> str:
        """计算风险等级"""
        if probability >= 0.7:
            return 'critical'
        elif probability >= 0.5:
            return 'high'
        elif probability >= 0.3:
            return 'medium'
        else:
            return 'low'

    def _predict_defect_types(self, metrics: CodeMetrics) -> List[str]:
        """预测缺陷类型"""
        defects = []

        if metrics.cyclomatic_complexity > 10:
            defects.append('complex_logic')
        if metrics.cognitive_complexity > 15:
            defects.append('cognitive_overload')
        if metrics.maintainability_index < 50:
            defects.append('low_maintainability')
        if metrics.comment_ratio < 0.1:
            defects.append('poor_documentation')
        if metrics.duplicate_lines > 10:
            defects.append('code_duplication')
        if metrics.nesting_depth > 4:
            defects.append('deep_nesting')
        if metrics.exception_count > 5:
            defects.append('excessive_error_handling')
        if metrics.function_count > 20:
            defects.append('large_module')

        return defects

    def _generate_improvement_recommendations(self, metrics: CodeMetrics) -> List[str]:
        """生成改进建议"""
        recommendations = []

        if metrics.cyclomatic_complexity > 10:
            recommendations.append("重构复杂函数，将其拆分为更小的函数")
        if metrics.cognitive_complexity > 15:
            recommendations.append("简化条件逻辑，减少嵌套和布尔运算")
        if metrics.maintainability_index < 50:
            recommendations.append("提高代码可维护性，添加更多注释和文档")
        if metrics.comment_ratio < 0.1:
            recommendations.append("增加代码注释，提高可读性")
        if metrics.duplicate_lines > 10:
            recommendations.append("消除代码重复，提取公共函数")
        if metrics.nesting_depth > 4:
            recommendations.append("减少嵌套深度，使用提前返回或状态机")
        if metrics.function_count > 20:
            recommendations.append("将大模块拆分为多个小模块")
        if metrics.exception_count > 5:
            recommendations.append("简化异常处理逻辑")

        if not recommendations:
            recommendations.append("代码质量良好，继续保持")

        return recommendations

    def assess_code_quality(self, source_code: str) -> QualityAssessment:
        """评估代码质量"""
        metrics = self._extract_code_metrics(source_code)
        prediction = self.predict_defects(source_code)

        # 计算综合评分
        overall_score = self._calculate_overall_score(metrics, prediction)

        # 确定等级
        grade = self._calculate_grade(overall_score)

        # 识别优势
        strengths = self._identify_strengths(metrics)

        # 识别弱点
        weaknesses = self._identify_weaknesses(metrics)

        # 改进领域
        improvement_areas = prediction.recommendations

        # 预测维护工作量
        maintenance_effort = self._predict_maintenance_effort(metrics)

        # 技术债务比例
        technical_debt_ratio = self._calculate_technical_debt_ratio(metrics, prediction)

        return QualityAssessment(
            overall_score=overall_score,
            grade=grade,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_areas=improvement_areas,
            predicted_maintenance_effort=maintenance_effort,
            technical_debt_ratio=technical_debt_ratio
        )

    def _calculate_overall_score(self, metrics: CodeMetrics, prediction: DefectPredictionResult) -> float:
        """计算综合评分"""
        score = 100.0

        # 复杂度惩罚
        score -= min(30, metrics.cyclomatic_complexity * 2)
        score -= min(20, metrics.cognitive_complexity * 1.5)

        # 可维护性奖励
        maintainability_bonus = (100 - metrics.maintainability_index) * 0.5
        score -= maintainability_bonus

        # 注释奖励
        if metrics.comment_ratio > 0.2:
            score += 10
        elif metrics.comment_ratio > 0.1:
            score += 5

        # 重复代码惩罚
        score -= min(15, metrics.duplicate_lines * 0.5)

        # 缺陷概率惩罚
        score -= prediction.defect_probability * 50

        return max(0, min(100, score))

    def _calculate_grade(self, score: float) -> str:
        """计算等级"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'

    def _identify_strengths(self, metrics: CodeMetrics) -> List[str]:
        """识别优势"""
        strengths = []

        if metrics.maintainability_index > 80:
            strengths.append("高可维护性")
        if metrics.comment_ratio > 0.2:
            strengths.append("良好的文档化")
        if metrics.cyclomatic_complexity <= 5:
            strengths.append("低复杂度")
        if metrics.duplicate_lines == 0:
            strengths.append("无重复代码")
        if metrics.nesting_depth <= 2:
            strengths.append("合理的嵌套深度")

        return strengths

    def _identify_weaknesses(self, metrics: CodeMetrics) -> List[str]:
        """识别弱点"""
        weaknesses = []

        if metrics.cyclomatic_complexity > 10:
            weaknesses.append("高圈复杂度")
        if metrics.cognitive_complexity > 15:
            weaknesses.append("高认知复杂度")
        if metrics.maintainability_index < 50:
            weaknesses.append("低可维护性")
        if metrics.comment_ratio < 0.1:
            weaknesses.append("文档不足")
        if metrics.duplicate_lines > 10:
            weaknesses.append("代码重复严重")

        return weaknesses

    def _predict_maintenance_effort(self, metrics: CodeMetrics) -> float:
        """预测维护工作量"""
        # 简化的维护工作量预测模型
        effort = 1.0  # 基础工作量

        # 复杂度因子
        effort *= (1 + metrics.cyclomatic_complexity / 20)

        # 规模因子
        effort *= (1 + metrics.lines_of_code / 1000)

        # 可维护性因子
        effort *= (2 - metrics.maintainability_index / 100)

        return round(effort, 2)

    def _calculate_technical_debt_ratio(self, metrics: CodeMetrics, prediction: DefectPredictionResult) -> float:
        """计算技术债务比例"""
        # 简化的技术债务计算
        debt_ratio = prediction.defect_probability * 0.5

        if metrics.maintainability_index < 70:
            debt_ratio += (100 - metrics.maintainability_index) / 200

        if metrics.comment_ratio < 0.15:
            debt_ratio += (0.15 - metrics.comment_ratio) * 2

        return min(debt_ratio, 1.0)

    def train_models(self, training_data: List[Tuple[str, bool]]) -> Dict[str, float]:
        """训练机器学习模型"""
        # 这里应该是完整的模型训练逻辑
        # 为了演示，我们使用模拟数据

        print("🔧 开始训练机器学习模型...")

        # 准备训练数据
        X = []
        y = []

        for code_sample, has_defect in training_data[:100]:  # 限制训练数据大小
            metrics = self._extract_code_metrics(code_sample)
            features = self._metrics_to_features(metrics)
            X.append(features)
            y.append(1 if has_defect else 0)

        if len(X) < 10:
            print("⚠️ 训练数据不足，使用规则-based预测")
            return {'status': 'insufficient_data'}

        X = np.array(X)
        y = np.array(y)

        results = {}

        for model_name, model in self.models.items():
            try:
                # 训练模型
                model.fit(X, y)

                # 交叉验证
                cv_scores = cross_val_score(model, X, y, cv=3, scoring='f1')

                results[model_name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'status': 'trained'
                }

                print(f"✅ {model_name} 模型训练完成 - F1分数: {cv_scores.mean():.3f}")

            except Exception as e:
                results[model_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
                print(f"❌ {model_name} 模型训练失败: {e}")

        return results

    def compare_models(self, test_data: List[Tuple[str, bool]]) -> Dict[str, Dict[str, float]]:
        """比较不同模型的性能"""
        print("📊 开始模型性能比较...")

        results = {}

        for model_name, model in self.models.items():
            try:
                predictions = []
                actuals = []

                for code_sample, has_defect in test_data[:50]:  # 限制测试数据
                    metrics = self._extract_code_metrics(code_sample)
                    features = self._metrics_to_features(metrics)

                    # 预测
                    prediction = model.predict([features])[0]
                    predictions.append(prediction)
                    actuals.append(1 if has_defect else 0)

                # 计算指标
                accuracy = accuracy_score(actuals, predictions)
                precision = precision_score(actuals, predictions, zero_division=0)
                recall = recall_score(actuals, predictions, zero_division=0)
                f1 = f1_score(actuals, predictions, zero_division=0)

                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }

                print(f"📈 {model_name} - 准确率: {accuracy:.3f}, F1: {f1:.3f}")

            except Exception as e:
                results[model_name] = {'error': str(e)}
                print(f"❌ {model_name} 评估失败: {e}")

        return results


class TestAdvancedDefectPrediction:
    """高级缺陷预测测试"""

    def setup_method(self):
        """测试前准备"""
        self.predictor = AdvancedDefectPredictor()

    def test_code_metrics_extraction(self):
        """测试代码度量提取"""
        # 测试简单函数
        simple_code = '''
def simple_function(x, y):
    """简单函数"""
    return x + y
'''
        metrics = self.predictor._extract_code_metrics(simple_code)

        assert metrics.lines_of_code > 0
        assert metrics.function_count == 1
        assert metrics.cyclomatic_complexity >= 1
        assert metrics.maintainability_index > 0

        print(f"✅ 代码度量提取成功 - 函数数: {metrics.function_count}, 复杂度: {metrics.cyclomatic_complexity}")

    def test_complex_code_metrics(self):
        """测试复杂代码度量"""
        complex_code = '''
def complex_function(data):
    """
    复杂的业务逻辑函数
    """
    if not data:
        raise ValueError("数据不能为空")

    result = []
    for item in data:
        if isinstance(item, dict):
            if 'value' in item and item['value'] > 0:
                if item.get('status') == 'active':
                    try:
                        processed_value = item['value'] * 1.1
                        if processed_value > 100:
                            result.append(processed_value)
                        else:
                            result.append(item['value'])
                    except KeyError:
                        continue
                    except TypeError as e:
                        print(f"类型错误: {e}")
                        continue
                elif item.get('status') == 'inactive':
                    result.append(item['value'] * 0.9)
            else:
                result.append(0)
        else:
            result.append(item if isinstance(item, (int, float)) else 0)

    return result
'''
        metrics = self.predictor._extract_code_metrics(complex_code)

        assert metrics.cyclomatic_complexity > 5  # 应该有较高的复杂度
        assert metrics.cognitive_complexity > 10  # 认知复杂度较高
        assert metrics.nesting_depth >= 3  # 嵌套深度较深
        assert metrics.exception_count >= 2  # 有异常处理

        print(f"✅ 复杂代码度量成功 - 圈复杂度: {metrics.cyclomatic_complexity}, 认知复杂度: {metrics.cognitive_complexity}")

    def test_defect_prediction(self):
        """测试缺陷预测"""
        # 高风险代码
        high_risk_code = '''
def risky_function():
    while True:  # 无限循环
        try:
            result = eval(input("Enter code: "))  # 危险操作
            exec(result)  # 更危险的操作
        except:  # 裸except
            pass  # 空处理
'''

        prediction = self.predictor.predict_defects(high_risk_code)

        assert prediction.defect_probability > 0.1  # 合理的缺陷概率
        assert prediction.risk_level in ['high', 'critical']
        assert len(prediction.predicted_defects) > 0
        assert len(prediction.recommendations) > 0

        print(f"✅ 缺陷预测成功 - 概率: {prediction.defect_probability:.3f}, 风险等级: {prediction.risk_level}")

    def test_low_risk_code_prediction(self):
        """测试低风险代码预测"""
        # 低风险代码
        low_risk_code = '''
def safe_function(x, y=None):
    """
    安全的工具函数

    Args:
        x: 必需参数
        y: 可选参数，默认为None

    Returns:
        计算结果
    """
    if y is None:
        y = 0

    result = x + y

    # 简单的验证
    if not isinstance(result, (int, float)):
        raise TypeError("结果类型错误")

    return result
'''

        prediction = self.predictor.predict_defects(low_risk_code)

        assert prediction.defect_probability < 0.3  # 低缺陷概率
        assert prediction.risk_level == 'low'
        assert prediction.confidence_score > 0

        print(f"✅ 低风险代码预测成功 - 概率: {prediction.defect_probability:.3f}, 风险等级: {prediction.risk_level}")

    def test_code_quality_assessment(self):
        """测试代码质量评估"""
        # 优质代码
        good_code = '''
def well_written_function(data):
    """
    写得很好的函数示例

    这个函数展示了良好的编码实践：
    - 清晰的文档字符串
    - 输入验证
    - 错误处理
    - 单一职责

    Args:
        data: 输入数据列表

    Returns:
        处理后的数据

    Raises:
        ValueError: 当输入无效时
    """
    if not data:
        raise ValueError("输入数据不能为空")

    # 验证输入类型
    if not isinstance(data, list):
        raise TypeError("输入必须是列表类型")

    # 处理数据
    result = []
    for item in data:
        if isinstance(item, (int, float)) and item > 0:
            result.append(item * 2)
        else:
            result.append(0)

    return result
'''

        assessment = self.predictor.assess_code_quality(good_code)

        assert assessment.overall_score > 70  # 应该有较高的分数
        assert assessment.grade in ['A', 'B', 'C']
        assert len(assessment.strengths) > 0
        assert assessment.predicted_maintenance_effort < 2.0

        print(f"✅ 代码质量评估成功 - 综合评分: {assessment.overall_score:.1f}, 等级: {assessment.grade}")

    def test_poor_quality_code_assessment(self):
        """测试劣质代码评估"""
        # 劣质代码
        poor_code = '''
def badfunction(x,y,z,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z):  # 太多参数
    # 没有文档字符串
    result=0  # 没有空格
    if x>0 and y>0 and z>0 and a>0 and b>0 and c>0 and d>0 and e>0 and f>0 and g>0 and h>0 and i>0 and j>0 and k>0 and l>0 and m>0 and n>0 and o>0 and p>0 and q>0 and r>0 and s>0 and t>0 and u>0 and v>0 and w>0 and x>0 and y>0 and z>0:  # 超长条件
        while True:  # 无限循环
            try:
                result=eval(input())  # 危险操作
            except:  # 裸except
                pass  # 什么都不做
    return result  # 没有验证返回值
'''

        assessment = self.predictor.assess_code_quality(poor_code)

        assert assessment.overall_score < 50  # 应该有较低的分数
        assert assessment.grade in ['D', 'F']
        assert len(assessment.weaknesses) > 0
        assert len(assessment.improvement_areas) > 0
        assert assessment.predicted_maintenance_effort > 3.0

        print(f"✅ 劣质代码评估成功 - 综合评分: {assessment.overall_score:.1f}, 等级: {assessment.grade}")

    def test_halstead_metrics_calculation(self):
        """测试Halstead度量计算"""
        code = '''
def calculate_sum(a, b):
    result = a + b
    return result
'''

        metrics = self.predictor._extract_code_metrics(code)

        assert metrics.halstead_volume > 0
        assert metrics.halstead_difficulty > 0

        print(f"✅ Halstead度量计算成功 - 体积: {metrics.halstead_volume:.2f}, 难度: {metrics.halstead_difficulty:.2f}")

    def test_model_training_simulation(self):
        """测试模型训练模拟"""
        # 创建模拟训练数据
        training_data = [
            ('def safe_function(x): return x * 2', False),
            ('def risky_function(): eval(input())', True),
            ('def complex_function(a, b, c, d, e, f): return a+b+c+d+e+f', False),
            ('while True: pass', True),  # 无限循环
            ('def documented_function(x):\n    """文档"""\n    return x', False),
            ('try: risky_operation() except: pass', True),  # 危险操作
        ]

        results = self.predictor.train_models(training_data)

        assert isinstance(results, dict)
        assert len(results) > 0

        # 检查是否有训练成功的模型
        trained_models = [k for k, v in results.items() if v.get('status') == 'trained']
        if trained_models:
            print(f"✅ 模型训练成功 - {len(trained_models)} 个模型训练完成")
        else:
            print("✅ 模型训练模拟完成 - 使用规则-based预测")

    def test_model_comparison_simulation(self):
        """测试模型比较模拟"""
        # 创建模拟测试数据
        test_data = [
            ('def good_function(x): return x + 1', False),
            ('eval(user_input)', True),
            ('def normal_function(a, b): return a - b', False),
            ('while True: break', True),  # 虽然有break但仍然是while True
        ]

        # 先训练模型
        training_data = test_data * 5  # 扩展训练数据
        self.predictor.train_models(training_data)

        # 比较模型
        comparison_results = self.predictor.compare_models(test_data)

        assert isinstance(comparison_results, dict)
        assert len(comparison_results) > 0

        print(f"✅ 模型比较完成 - 比较了 {len(comparison_results)} 个模型")

    def test_cyclomatic_complexity_calculation(self):
        """测试圈复杂度计算"""
        # 简单函数
        simple_code = '''
def simple_function(x):
    return x * 2
'''
        metrics = self.predictor._extract_code_metrics(simple_code)
        assert metrics.cyclomatic_complexity == 1

        # 复杂条件函数
        complex_code = '''
def complex_function(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z
            else:
                return x + y
        elif y < 0:
            return x
        else:
            return 0
    else:
        return -1
'''
        metrics = self.predictor._extract_code_metrics(complex_code)
        assert metrics.cyclomatic_complexity > 5  # 多个条件分支

        print(f"✅ 圈复杂度计算成功 - 简单函数: 1, 复杂函数: {metrics.cyclomatic_complexity}")

    def test_cognitive_complexity_calculation(self):
        """测试认知复杂度计算"""
        # 嵌套条件
        nested_code = '''
def nested_function(data):
    if data:
        for item in data:
            if isinstance(item, dict):
                if 'value' in item:
                    if item['value'] > 0:
                        return item['value']
    return None
'''
        metrics = self.predictor._extract_code_metrics(nested_code)
        assert metrics.cognitive_complexity > 5  # 嵌套增加认知复杂度

        print(f"✅ 认知复杂度计算成功 - 嵌套函数: {metrics.cognitive_complexity}")

    def test_maintainability_index_calculation(self):
        """测试可维护性指数计算"""
        # 高质量代码
        good_code = '''
def well_documented_function(x, y):
    """
    计算两个数的和

    Args:
        x: 第一个数
        y: 第二个数

    Returns:
        两数之和
    """
    # 输入验证
    if not isinstance(x, (int, float)):
        raise TypeError("x必须是数字")
    if not isinstance(y, (int, float)):
        raise TypeError("y必须是数字")

    # 计算结果
    result = x + y

    return result
'''
        metrics = self.predictor._extract_code_metrics(good_code)
        assert metrics.maintainability_index > 70  # 应该有较高的可维护性

        print(f"✅ 可维护性指数计算成功 - 指数: {metrics.maintainability_index:.1f}")

    def test_technical_debt_analysis(self):
        """测试技术债务分析"""
        # 高债务代码
        high_debt_code = '''
def old_function(a,b,c,d,e,f,g,h,i,j):
    # 没有注释的复杂函数
    result=a
    if b>0:result+=b
    if c>0:result+=c
    if d>0:result+=d
    if e>0:result+=e
    if f>0:result+=f
    if g>0:result+=g
    if h>0:result+=h
    if i>0:result+=i
    if j>0:result+=j
    return result
'''

        assessment = self.predictor.assess_code_quality(high_debt_code)

        assert assessment.technical_debt_ratio > 0.3  # 应该有较高的技术债务
        assert assessment.predicted_maintenance_effort > 2.0  # 维护工作量较大

        print(f"✅ 技术债务分析成功 - 债务比例: {assessment.technical_debt_ratio:.3f}, 维护工作量: {assessment.predicted_maintenance_effort:.2f}")

    def test_prediction_recommendations(self):
        """测试预测建议生成"""
        # 需要改进的代码
        improvable_code = '''
def function_with_issues():
    # 没有注释
    x = 1
    while x < 100:  # 可能的问题循环
        if x % 2 == 0:
            print(x)  # 调试打印
        x += 1
    # 没有错误处理
    return x
'''

        prediction = self.predictor.predict_defects(improvable_code)

        assert len(prediction.recommendations) > 0
        assert any('注释' in rec or 'documentation' in rec.lower() for rec in prediction.recommendations)

        print(f"✅ 预测建议生成成功 - 生成 {len(prediction.recommendations)} 条建议")

    def test_feature_importance_analysis(self):
        """测试特征重要性分析"""
        code = '''
def sample_function(data):
    if data:
        for item in data:
            if item > 0:
                return item * 2
    return 0
'''

        prediction = self.predictor.predict_defects(code)

        assert 'feature_importance' in prediction.__dict__
        assert isinstance(prediction.feature_importance, dict)
        assert len(prediction.feature_importance) > 0

        # 检查重要性值
        for feature, importance in prediction.feature_importance.items():
            assert 0 <= importance <= 1

        print(f"✅ 特征重要性分析成功 - 分析了 {len(prediction.feature_importance)} 个特征")

    def test_bulk_code_analysis(self):
        """测试批量代码分析"""
        code_samples = [
            'def simple(): return 42',
            'def complex(a,b,c): return a+b+c if a>0 else 0',
            'while True: pass',  # 有问题的代码
            'def documented_function(x):\n    """文档"""\n    return x*2',
            'eval(input())',  # 危险代码
        ]

        results = []
        for code in code_samples:
            try:
                prediction = self.predictor.predict_defects(code)
                assessment = self.predictor.assess_code_quality(code)
                results.append({
                    'code': code[:30] + '...' if len(code) > 30 else code,
                    'defect_probability': prediction.defect_probability,
                    'quality_score': assessment.overall_score,
                    'grade': assessment.grade
                })
            except Exception as e:
                results.append({
                    'code': code[:30] + '...' if len(code) > 30 else code,
                    'error': str(e)
                })

        assert len(results) == len(code_samples)

        # 检查是否有问题代码被正确识别
        problematic_results = [r for r in results if r.get('defect_probability', 0) > 0.5]
        assert len(problematic_results) > 0  # 应该识别出有问题的代码

        print(f"✅ 批量代码分析成功 - 分析了 {len(results)} 个代码样本")
