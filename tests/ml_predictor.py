#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机器学习测试预测器

使用机器学习算法预测：
- 测试执行时间
- 测试失败概率
- 性能瓶颈识别
- 智能测试排序
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class TestPrediction:
    """测试预测结果"""
    test_file: str
    predicted_time: float
    predicted_failure_prob: float
    confidence: float
    risk_level: str
    recommended_priority: int
    features_used: Dict[str, Any]


@dataclass
class MLModelMetrics:
    """模型性能指标"""
    time_prediction_mae: float
    failure_prediction_accuracy: float
    feature_importance: Dict[str, float]
    training_samples: int
    model_version: str


class FeatureExtractor:
    """特征提取器"""

    def __init__(self):
        self.label_encoders = {}

    def extract_features(self, test_file: str, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """提取测试文件的特征"""
        features = {}

        # 基本文件特征
        file_path = Path(test_file)
        features['file_size'] = file_path.stat().st_size if file_path.exists() else 0
        features['file_lines'] = self._count_lines(file_path)
        features['file_name_length'] = len(file_path.name)
        features['has_fixture'] = 'fixture' in self._read_file_content(file_path)
        features['has_mock'] = 'mock' in self._read_file_content(file_path)
        features['has_patch'] = 'patch' in self._read_file_content(file_path)
        features['test_functions'] = self._count_test_functions(file_path)

        # 历史性能特征
        historical_stats = self._get_historical_stats(test_file, historical_data)
        features.update(historical_stats)

        # 时间特征
        now = datetime.now()
        features['hour_of_day'] = now.hour
        features['day_of_week'] = now.weekday()
        features['is_weekend'] = 1 if now.weekday() >= 5 else 0

        # 路径特征
        path_parts = str(file_path).split(os.sep)
        features['path_depth'] = len(path_parts)
        features['in_unit_tests'] = 1 if 'unit' in path_parts else 0
        features['in_integration_tests'] = 1 if 'integration' in path_parts else 0

        return features

    def _count_lines(self, file_path: Path) -> int:
        """统计文件行数"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return len(f.readlines())
        except Exception:
            return 0

    def _read_file_content(self, file_path: Path) -> str:
        """读取文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read().lower()
        except Exception:
            return ""

    def _count_test_functions(self, file_path: Path) -> int:
        """统计测试函数数量"""
        content = self._read_file_content(file_path)
        return content.count('def test_')

    def _get_historical_stats(self, test_file: str, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取历史统计数据"""
        if not historical_data:
            return {
                'avg_execution_time': 0.0,
                'std_execution_time': 0.0,
                'failure_rate': 0.0,
                'recent_failures': 0,
                'total_runs': 0,
                'last_run_days': 999
            }

        # 过滤当前测试文件的历史数据
        file_history = [h for h in historical_data if h.get('test_file') == test_file]

        if not file_history:
            return {
                'avg_execution_time': np.mean([h.get('execution_time', 0) for h in historical_data]),
                'std_execution_time': np.std([h.get('execution_time', 0) for h in historical_data]),
                'failure_rate': np.mean([0 if h.get('success', True) else 1 for h in historical_data]),
                'recent_failures': 0,
                'total_runs': 0,
                'last_run_days': 999
            }

        execution_times = [h.get('execution_time', 0) for h in file_history]
        failures = [0 if h.get('success', True) else 1 for h in file_history]

        # 最近7次运行的失败情况
        recent_history = file_history[-7:] if len(file_history) >= 7 else file_history
        recent_failures = sum(0 if h.get('success', True) else 1 for h in recent_history)

        # 最后运行时间
        last_run = max((h.get('timestamp') for h in file_history if h.get('timestamp')), default=None)
        last_run_days = (datetime.now() - datetime.fromisoformat(last_run)).days if last_run else 999

        return {
            'avg_execution_time': np.mean(execution_times),
            'std_execution_time': np.std(execution_times),
            'failure_rate': np.mean(failures),
            'recent_failures': recent_failures,
            'total_runs': len(file_history),
            'last_run_days': last_run_days
        }


class TestTimePredictor:
    """测试时间预测器"""

    def __init__(self, model_path: str = "models/test_time_predictor.pkl"):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False

    def train(self, historical_data: List[Dict[str, Any]], feature_extractor: FeatureExtractor) -> MLModelMetrics:
        """训练时间预测模型"""
        logger.info("开始训练测试时间预测模型...")

        # 准备训练数据
        X, y, feature_names = self._prepare_training_data(historical_data, feature_extractor)

        if len(X) < 10:
            logger.warning("训练数据不足，无法训练模型")
            return MLModelMetrics(0, 0, {}, 0, "insufficient_data")

        # 分割训练和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 标准化特征
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 训练随机森林模型
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # 评估模型
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)

        # 特征重要性
        feature_importance = dict(zip(feature_names, self.model.feature_importances_))

        # 保存模型
        self.feature_names = feature_names
        self._save_model()

        self.is_trained = True

        metrics = MLModelMetrics(
            time_prediction_mae=mae,
            failure_prediction_accuracy=0,  # 时间预测器不预测失败
            feature_importance=feature_importance,
            training_samples=len(X_train),
            model_version=datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        logger.info(f"时间预测模型训练完成，MAE: {mae:.3f}s")
        return metrics

    def predict(self, test_file: str, historical_data: List[Dict[str, Any]],
            feature_extractor: FeatureExtractor) -> Tuple[float, float]:
        """预测测试执行时间"""
        if not self.is_trained:
            return 0.0, 0.0

        # 提取特征
        raw_features = feature_extractor.extract_features(test_file, historical_data)

        # 转换为模型输入格式
        feature_vector = [raw_features.get(name, 0) for name in self.feature_names]

        # 标准化
        if self.scaler:
            feature_vector = self.scaler.transform([feature_vector])

        # 预测
        prediction = self.model.predict(feature_vector)[0]

        # 计算置信度（基于特征重要性和方差）
        confidence = self._calculate_confidence(features)

        return max(0.1, prediction), confidence

    def _prepare_training_data(self, historical_data: List[Dict[str, Any]],
                            feature_extractor: FeatureExtractor) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """准备训练数据"""
        X = []
        y = []
        feature_names = None

        for record in historical_data:
            test_file = record.get('test_file', '')
            execution_time = record.get('execution_time', 0)

            if execution_time <= 0:
                continue

            # 提取特征
            features = feature_extractor.extract_features(test_file, historical_data)

            if feature_names is None:
                feature_names = list(features.keys())

            # 转换为数值向量
            feature_vector = [features[name] for name in feature_names]
            X.append(feature_vector)
            y.append(execution_time)

        return np.array(X), np.array(y), feature_names

    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """计算预测置信度"""
        # 基于历史数据丰富程度计算置信度
        total_runs = features.get('total_runs', 0)
        base_confidence = min(total_runs / 10, 1.0)  # 运行次数越多，置信度越高

        # 考虑数据新鲜度
        last_run_days = features.get('last_run_days', 999)
        recency_factor = max(0, 1 - last_run_days / 30)  # 30天内的数据更可信

        return base_confidence * recency_factor

    def _save_model(self):
        """保存模型"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'trained_at': datetime.now().isoformat()
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"时间预测模型已保存: {self.model_path}")

        except Exception as e:
            logger.error(f"保存模型失败: {e}")

    def _load_model(self):
        """加载模型"""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.is_trained = True

                logger.info("时间预测模型已加载")
                return True

        except Exception as e:
            logger.error(f"加载模型失败: {e}")

        return False


class TestFailurePredictor:
    """测试失败预测器"""

    def __init__(self, model_path: str = "models/test_failure_predictor.pkl"):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False

    def train(self, historical_data: List[Dict[str, Any]], feature_extractor: FeatureExtractor) -> MLModelMetrics:
        """训练失败预测模型"""
        logger.info("开始训练测试失败预测模型...")

        # 准备训练数据
        X, y, feature_names = self._prepare_training_data(historical_data, feature_extractor)

        if len(X) < 10:
            logger.warning("训练数据不足，无法训练模型")
            return MLModelMetrics(0, 0, {}, 0, "insufficient_data")

        # 分割训练和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 标准化特征
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 训练随机森林分类器
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # 评估模型
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # 特征重要性
        feature_importance = dict(zip(feature_names, self.model.feature_importances_))

        # 保存模型
        self.feature_names = feature_names
        self._save_model()

        self.is_trained = True

        metrics = MLModelMetrics(
            time_prediction_mae=0,  # 失败预测器不预测时间
            failure_prediction_accuracy=accuracy,
            feature_importance=feature_importance,
            training_samples=len(X_train),
            model_version=datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        logger.info(f"失败预测模型训练完成，准确率: {accuracy:.3f}")
        return metrics

    def predict(self, test_file: str, historical_data: List[Dict[str, Any]]) -> Tuple[float, float]:
        """预测测试失败概率"""
        if not self.is_trained:
            return 0.0, 0.0

        # 提取特征
        raw_features = feature_extractor.extract_features(test_file, historical_data)

        # 转换为模型输入格式
        feature_vector = [raw_features.get(name, 0) for name in self.feature_names]

        # 标准化
        if self.scaler:
            feature_vector = self.scaler.transform([feature_vector])

        # 预测失败概率
        failure_prob = self.model.predict_proba(feature_vector)[0][1]

        # 计算置信度
        confidence = self._calculate_confidence(features)

        return failure_prob, confidence

    def _prepare_training_data(self, historical_data: List[Dict[str, Any]],
                            feature_extractor: FeatureExtractor) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """准备训练数据"""
        X = []
        y = []
        feature_names = None

        for record in historical_data:
            test_file = record.get('test_file', '')
            success = record.get('success', True)

            # 提取特征
            features = feature_extractor.extract_features(test_file, historical_data)

            if feature_names is None:
                feature_names = list(features.keys())

            # 转换为数值向量
            feature_vector = [features[name] for name in feature_names]
            X.append(feature_vector)
            y.append(0 if success else 1)  # 0: 成功, 1: 失败

        return np.array(X), np.array(y), feature_names

    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """计算预测置信度"""
        # 基于历史失败数据计算置信度
        total_runs = features.get('total_runs', 0)
        recent_failures = features.get('recent_failures', 0)

        if total_runs == 0:
            return 0.0

        # 失败历史越丰富，置信度越高
        failure_history_confidence = min(recent_failures / 5, 1.0)

        # 总体运行次数的影响
        run_count_confidence = min(total_runs / 20, 1.0)

        return (failure_history_confidence + run_count_confidence) / 2

    def _save_model(self):
        """保存模型"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'trained_at': datetime.now().isoformat()
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"失败预测模型已保存: {self.model_path}")

        except Exception as e:
            logger.error(f"保存模型失败: {e}")

    def _load_model(self):
        """加载模型"""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.is_trained = True

                logger.info("失败预测模型已加载")
                return True

        except Exception as e:
            logger.error(f"加载模型失败: {e}")

        return False


class MLPredictor:
    """机器学习预测器"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.feature_extractor = FeatureExtractor()
        self.time_predictor = TestTimePredictor(self.models_dir / "test_time_predictor.pkl")
        self.failure_predictor = TestFailurePredictor(self.models_dir / "test_failure_predictor.pkl")

        # 尝试加载现有模型
        self.time_predictor._load_model()
        self.failure_predictor._load_model()

    def train_models(self, historical_data_path: str = "test_logs/performance_history.json") -> Dict[str, MLModelMetrics]:
        """训练所有模型"""
        logger.info("开始训练机器学习模型...")

        # 加载历史数据
        historical_data = self._load_historical_data(historical_data_path)

        if not historical_data:
            logger.warning("没有历史数据用于训练")
            return {}

        # 训练时间预测模型
        time_metrics = self.time_predictor.train(historical_data, self.feature_extractor)

        # 训练失败预测模型
        failure_metrics = self.failure_predictor.train(historical_data, self.feature_extractor)

        metrics = {
            'time_predictor': time_metrics,
            'failure_predictor': failure_metrics
        }

        logger.info("机器学习模型训练完成")
        return metrics

    def predict_test(self, test_file: str, historical_data_path: str = "test_logs/performance_history.json") -> TestPrediction:
        """预测单个测试"""
        # 加载历史数据
        historical_data = self._load_historical_data(historical_data_path)

        # 预测执行时间
        predicted_time, time_confidence = self.time_predictor.predict(test_file, historical_data, self.feature_extractor)

        # 预测失败概率
        failure_prob, failure_confidence = self.failure_predictor.predict(test_file, historical_data, self.feature_extractor)

        # 计算综合置信度
        overall_confidence = (time_confidence + failure_confidence) / 2

        # 确定风险等级
        risk_level = self._calculate_risk_level(failure_prob, overall_confidence)

        # 确定推荐优先级
        priority = self._calculate_priority(predicted_time, failure_prob, risk_level)

        # 提取使用的特征
        features = self.feature_extractor.extract_features(test_file, historical_data)

        return TestPrediction(
            test_file=test_file,
            predicted_time=predicted_time,
            predicted_failure_prob=failure_prob,
            confidence=overall_confidence,
            risk_level=risk_level,
            recommended_priority=priority,
            features_used=features
        )

    def predict_test_suite(self, test_files: List[str], historical_data_path: str = "test_logs/performance_history.json") -> List[TestPrediction]:
        """预测测试套件"""
        logger.info(f"开始预测测试套件，共 {len(test_files)} 个测试")

        predictions = []
        historical_data = self._load_historical_data(historical_data_path)

        for test_file in test_files:
            try:
                prediction = self.predict_test(test_file, historical_data)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"预测测试失败 {test_file}: {e}")

        # 按推荐优先级排序
        predictions.sort(key=lambda x: x.recommended_priority)

        logger.info("测试套件预测完成")
        return predictions

    def optimize_test_order(self, test_files: List[str]) -> List[str]:
        """优化测试执行顺序"""
        predictions = self.predict_test_suite(test_files)

        # 按优先级排序：高风险、高失败概率、短时间测试优先
        optimized_order = [p.test_file for p in predictions]

        return optimized_order

    def get_risky_tests(self, test_files: List[str], risk_threshold: float = 0.7) -> List[TestPrediction]:
        """获取高风险测试"""
        predictions = self.predict_test_suite(test_files)

        risky_tests = [p for p in predictions if p.predicted_failure_prob > risk_threshold or p.risk_level in ['high', 'critical']]

        return risky_tests

    def _load_historical_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载历史数据"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载历史数据失败: {e}")
            return []

    def _calculate_risk_level(self, failure_prob: float, confidence: float) -> str:
        """计算风险等级"""
        risk_score = failure_prob * confidence

        if risk_score > 0.8:
            return "critical"
        elif risk_score > 0.6:
            return "high"
        elif risk_score > 0.4:
            return "medium"
        elif risk_score > 0.2:
            return "low"
        else:
            return "minimal"

    def _calculate_priority(self, predicted_time: float, failure_prob: float, risk_level: str) -> int:
        """计算优先级（1-10，1最高优先级）"""
        # 基础优先级
        if risk_level == "critical":
            base_priority = 1
        elif risk_level == "high":
            base_priority = 2
        elif risk_level == "medium":
            base_priority = 4
        elif risk_level == "low":
            base_priority = 7
        else:
            base_priority = 10

        # 失败概率调整
        prob_adjustment = int(failure_prob * 3)

        # 执行时间调整（短时间测试优先）
        time_adjustment = 0
        if predicted_time < 1:
            time_adjustment = -1
        elif predicted_time > 30:
            time_adjustment = 1

        final_priority = max(1, min(10, base_priority + prob_adjustment + time_adjustment))

        return final_priority

    def generate_prediction_report(self, predictions: List[TestPrediction]):
        """生成预测报告"""
        report_path = Path("test_logs/ml_prediction_report.md")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 机器学习测试预测报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 📊 预测概览\n\n")
            if predictions:
                f.write(f"- **预测测试数**: {len(predictions)}\n")
                f.write(".2")
                f.write(".1")
                f.write(".1f")
                # 风险分布
                risk_counts = {}
                for p in predictions:
                    risk_counts[p.risk_level] = risk_counts.get(p.risk_level, 0) + 1

                f.write(f"- **风险分布**: {risk_counts}\n")

            f.write("\n## 🏆 高风险测试Top 10\n\n")
            high_risk = sorted(predictions, key=lambda x: (x.predicted_failure_prob, -x.confidence), reverse=True)[:10]

            f.write("| 测试文件 | 失败概率 | 风险等级 | 预测时间 | 置信度 |\n")
            f.write("|----------|----------|----------|----------|--------|\n")

            for p in high_risk:
                f.write(f"| `{Path(p.test_file).name}` | {p.predicted_failure_prob:.1f} | {p.risk_level} | {p.predicted_time:.2f}s | {p.confidence:.1f} |\n")

            f.write("\n## ⏱️ 最慢测试Top 10\n\n")
            slow_tests = sorted(predictions, key=lambda x: x.predicted_time, reverse=True)[:10]

            f.write("| 测试文件 | 预测时间 | 失败概率 | 优先级 |\n")
            f.write("|----------|----------|----------|--------|\n")

            for p in slow_tests:
                f.write(f"| `{Path(p.test_file).name}` | {p.predicted_time:.2f}s | {p.predicted_failure_prob:.1f} | {p.recommended_priority} |\n")

            f.write("\n## 🎯 推荐执行顺序\n\n")
            priority_order = sorted(predictions, key=lambda x: x.recommended_priority)[:15]

            f.write("| 优先级 | 测试文件 | 风险等级 | 失败概率 |\n")
            f.write("|--------|----------|----------|----------|\n")

            for p in priority_order:
                f.write(f"| {p.recommended_priority} | `{Path(p.test_file).name}` | {p.risk_level} | {p.predicted_failure_prob:.1f} |\n")

            f.write("\n## 📈 机器学习价值\n\n")
            f.write("### 对测试执行的价值\n")
            f.write("1. **智能排序**: 基于预测结果优化测试执行顺序\n")
            f.write("2. **风险预警**: 提前识别可能失败的测试\n")
            f.write("3. **时间预测**: 准确预估测试执行时间\n")
            f.write("4. **资源优化**: 优先执行高价值测试\n")
            f.write("\n### 对质量保障的价值\n")
            f.write("1. **缺陷预测**: 使用历史数据预测测试失败概率\n")
            f.write("2. **性能监控**: 持续学习和改进预测模型\n")
            f.write("3. **决策支持**: 为测试策略调整提供数据支持\n")
            f.write("4. **持续改进**: 基于反馈的模型优化\n")

        logger.info(f"机器学习预测报告已生成: {report_path}")


def main():
    """主函数"""
    predictor = MLPredictor()

    print("🧠 机器学习测试预测器启动")
    print("🎯 功能: 测试时间预测 + 失败概率预测 + 智能排序")

    # 训练模型
    print("📚 正在训练模型...")
    metrics = predictor.train_models()

    if metrics:
        print("✅ 模型训练完成:")
        if 'time_predictor' in metrics:
            time_metrics = metrics['time_predictor']
            print(".3")
            print(f"  📊 时间预测训练样本: {time_metrics.training_samples}")

        if 'failure_predictor' in metrics:
            failure_metrics = metrics['failure_predictor']
            print(".3")
            print(f"  📊 失败预测训练样本: {failure_metrics.training_samples}")

    # 发现测试文件
    test_files = []
    for pattern in ["test_*.py", "*_test.py"]:
        test_files.extend([str(f) for f in Path("tests").rglob(pattern)])

    if test_files:
        print(f"\n🎪 发现 {len(test_files)} 个测试文件，开始预测...")

        # 预测测试套件
        predictions = predictor.predict_test_suite(test_files[:20])  # 限制数量用于演示

        # 生成报告
        predictor.generate_prediction_report(predictions)

        print("✅ 预测完成，结果已保存到报告")
    else:
        print("⚠️ 未发现测试文件")

    print("\n📄 详细报告已保存到: test_logs/ml_prediction_report.md")
    print("\n✅ 机器学习测试预测器运行完成")


if __name__ == "__main__":
    main()
