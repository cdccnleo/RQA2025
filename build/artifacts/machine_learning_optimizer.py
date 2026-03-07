#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习优化器
引入智能参数调整算法
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import random

@dataclass
class MLConfig:
    """机器学习配置"""
    model_type: str = "gradient_boosting"
    learning_rate: float = 0.1
    max_iterations: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    feature_importance_threshold: float = 0.05
    optimization_interval: int = 3600  # 1小时
    min_data_points: int = 50

@dataclass
class MarketFeature:
    """市场特征"""
    volatility: float
    volume: float
    price_change: float
    market_sentiment: float
    time_of_day: float
    day_of_week: int

@dataclass
class OptimizationTarget:
    """优化目标"""
    price_limit_threshold: float
    after_hours_threshold: float
    circuit_breaker_threshold: float
    cache_ttl: int
    monitoring_interval: int

class MarketDataSimulator:
    """市场数据模拟器"""
    
    def __init__(self):
        self.current_time = datetime.now()
        self.market_state = "normal"
    
    def generate_market_features(self) -> MarketFeature:
        """生成市场特征"""
        # 模拟市场数据
        volatility = random.uniform(0.01, 0.05)
        volume = random.uniform(1000000, 5000000)
        price_change = random.uniform(-0.1, 0.1)
        market_sentiment = random.uniform(-1.0, 1.0)
        time_of_day = self.current_time.hour + self.current_time.minute / 60.0
        day_of_week = self.current_time.weekday()
        
        return MarketFeature(
            volatility=volatility,
            volume=volume,
            price_change=price_change,
            market_sentiment=market_sentiment,
            time_of_day=time_of_day,
            day_of_week=day_of_week
        )
    
    def update_market_state(self):
        """更新市场状态"""
        # 模拟市场状态变化
        if random.random() < 0.1:  # 10%概率状态变化
            states = ["normal", "volatile", "crisis"]
            self.market_state = random.choice(states)
        
        # 更新时间
        self.current_time = datetime.now()

class GradientBoostingOptimizer:
    """梯度提升优化器"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.model = None
        self.feature_importance = {}
        self.optimization_history = []
        self.current_iteration = 0
    
    def initialize_model(self):
        """初始化模型"""
        print("🤖 初始化梯度提升优化器...")
        
        # 模拟模型初始化
        self.model = {
            "type": self.config.model_type,
            "learning_rate": self.config.learning_rate,
            "max_iterations": self.config.max_iterations,
            "initialized": True,
            "last_update": time.time()
        }
        
        # 初始化特征重要性
        self.feature_importance = {
            "volatility": 0.3,
            "volume": 0.25,
            "price_change": 0.2,
            "market_sentiment": 0.15,
            "time_of_day": 0.05,
            "day_of_week": 0.05
        }
        
        print("✅ 梯度提升优化器初始化完成")
    
    def extract_features(self, market_feature: MarketFeature) -> List[float]:
        """提取特征"""
        return [
            market_feature.volatility,
            market_feature.volume,
            market_feature.price_change,
            market_feature.market_sentiment,
            market_feature.time_of_day,
            market_feature.day_of_week
        ]
    
    def predict_optimal_parameters(self, market_feature: MarketFeature) -> OptimizationTarget:
        """预测最优参数"""
        if not self.model or not self.model.get("initialized"):
            self.initialize_model()
        
        # 提取特征
        features = self.extract_features(market_feature)
        
        # 模拟梯度提升预测
        # 基于特征重要性进行加权预测
        base_price_limit = 0.1
        base_after_hours = 0.05
        base_circuit_breaker = 0.15
        base_cache_ttl = 3600
        base_monitoring_interval = 30
        
        # 根据市场特征调整参数
        volatility_factor = 1 + market_feature.volatility * 2
        volume_factor = 1 + (market_feature.volume - 3000000) / 10000000
        sentiment_factor = 1 + market_feature.market_sentiment * 0.5
        
        # 预测最优参数
        optimal_price_limit = base_price_limit * volatility_factor
        optimal_after_hours = base_after_hours * volume_factor
        optimal_circuit_breaker = base_circuit_breaker * sentiment_factor
        optimal_cache_ttl = int(base_cache_ttl * (1 + market_feature.volatility))
        optimal_monitoring_interval = int(base_monitoring_interval * (1 + abs(market_feature.price_change)))
        
        return OptimizationTarget(
            price_limit_threshold=optimal_price_limit,
            after_hours_threshold=optimal_after_hours,
            circuit_breaker_threshold=optimal_circuit_breaker,
            cache_ttl=optimal_cache_ttl,
            monitoring_interval=optimal_monitoring_interval
        )
    
    def update_model(self, market_feature: MarketFeature, actual_performance: Dict[str, float]):
        """更新模型"""
        self.current_iteration += 1
        
        # 模拟模型更新
        update_info = {
            "iteration": self.current_iteration,
            "timestamp": time.time(),
            "market_features": asdict(market_feature),
            "actual_performance": actual_performance,
            "model_improvement": random.uniform(0.01, 0.05)
        }
        
        self.optimization_history.append(update_info)
        
        # 更新特征重要性
        for feature in self.feature_importance:
            self.feature_importance[feature] += random.uniform(-0.01, 0.01)
            self.feature_importance[feature] = max(0, min(1, self.feature_importance[feature]))
        
        print(f"🔄 模型更新完成 (迭代 {self.current_iteration})")
    
    def get_model_status(self) -> Dict[str, Any]:
        """获取模型状态"""
        return {
            "model_initialized": self.model is not None and self.model.get("initialized", False),
            "current_iteration": self.current_iteration,
            "feature_importance": self.feature_importance,
            "optimization_history_count": len(self.optimization_history),
            "last_update": self.model.get("last_update") if self.model else None
        }

class PerformanceEvaluator:
    """性能评估器"""
    
    def __init__(self):
        self.evaluation_history = []
    
    def evaluate_performance(self, target: OptimizationTarget, market_feature: MarketFeature) -> Dict[str, float]:
        """评估性能"""
        # 模拟性能评估
        performance_metrics = {
            "risk_control_accuracy": self._calculate_risk_accuracy(target, market_feature),
            "system_response_time": self._calculate_response_time(target),
            "cache_hit_rate": self._calculate_cache_hit_rate(target),
            "monitoring_efficiency": self._calculate_monitoring_efficiency(target),
            "overall_score": 0.0
        }
        
        # 计算综合得分
        performance_metrics["overall_score"] = (
            performance_metrics["risk_control_accuracy"] * 0.4 +
            performance_metrics["system_response_time"] * 0.3 +
            performance_metrics["cache_hit_rate"] * 0.2 +
            performance_metrics["monitoring_efficiency"] * 0.1
        )
        
        return performance_metrics
    
    def _calculate_risk_accuracy(self, target: OptimizationTarget, market_feature: MarketFeature) -> float:
        """计算风控准确性"""
        # 基于市场波动率和参数设置计算准确性
        volatility_factor = 1 - market_feature.volatility
        threshold_factor = 1 - abs(target.price_limit_threshold - 0.1) / 0.1
        return min(1.0, max(0.0, volatility_factor * threshold_factor))
    
    def _calculate_response_time(self, target: OptimizationTarget) -> float:
        """计算响应时间得分"""
        # 基于缓存TTL和监控间隔计算响应时间
        cache_factor = 1 - (target.cache_ttl - 3600) / 3600
        monitoring_factor = 1 - (target.monitoring_interval - 30) / 30
        return min(1.0, max(0.0, (cache_factor + monitoring_factor) / 2))
    
    def _calculate_cache_hit_rate(self, target: OptimizationTarget) -> float:
        """计算缓存命中率"""
        # 基于缓存TTL计算命中率
        ttl_factor = target.cache_ttl / 7200  # 标准化到2小时
        return min(1.0, max(0.5, ttl_factor))
    
    def _calculate_monitoring_efficiency(self, target: OptimizationTarget) -> float:
        """计算监控效率"""
        # 基于监控间隔计算效率
        interval_factor = 1 - (target.monitoring_interval - 15) / 60
        return min(1.0, max(0.0, interval_factor))

class MLIntegrationManager:
    """机器学习集成管理器"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.optimizer = GradientBoostingOptimizer(config)
        self.evaluator = PerformanceEvaluator()
        self.market_simulator = MarketDataSimulator()
        self.integration_status = {}
        self.optimization_results = []
    
    def start_ml_integration(self) -> Dict[str, Any]:
        """启动机器学习集成"""
        print("🚀 启动机器学习集成...")
        
        try:
            # 1. 初始化优化器
            self.optimizer.initialize_model()
            
            # 2. 执行初始优化
            initial_optimization = self._perform_optimization_cycle()
            
            # 3. 启动持续优化
            continuous_optimization = self._start_continuous_optimization()
            
            # 4. 生成集成报告
            integration_report = self._generate_integration_report()
            
            return {
                "status": "success",
                "initial_optimization": initial_optimization,
                "continuous_optimization": continuous_optimization,
                "integration_report": integration_report
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": "机器学习集成失败"
            }
    
    def _perform_optimization_cycle(self) -> Dict[str, Any]:
        """执行优化周期"""
        print("🔄 执行优化周期...")
        
        # 生成市场特征
        market_feature = self.market_simulator.generate_market_features()
        
        # 预测最优参数
        optimal_target = self.optimizer.predict_optimal_parameters(market_feature)
        
        # 评估性能
        performance = self.evaluator.evaluate_performance(optimal_target, market_feature)
        
        # 更新模型
        self.optimizer.update_model(market_feature, performance)
        
        # 记录结果
        optimization_result = {
            "timestamp": time.time(),
            "market_feature": asdict(market_feature),
            "optimal_target": asdict(optimal_target),
            "performance": performance,
            "model_status": self.optimizer.get_model_status()
        }
        
        self.optimization_results.append(optimization_result)
        
        return optimization_result
    
    def _start_continuous_optimization(self) -> Dict[str, Any]:
        """启动持续优化"""
        print("🔄 启动持续优化...")
        
        # 模拟持续优化过程
        optimization_cycles = []
        
        for i in range(5):  # 模拟5个优化周期
            # 更新市场状态
            self.market_simulator.update_market_state()
            
            # 执行优化周期
            cycle_result = self._perform_optimization_cycle()
            optimization_cycles.append(cycle_result)
            
            # 模拟时间间隔
            time.sleep(0.1)
        
        return {
            "cycles_completed": len(optimization_cycles),
            "average_performance": self._calculate_average_performance(optimization_cycles),
            "optimization_trend": self._analyze_optimization_trend(optimization_cycles)
        }
    
    def _calculate_average_performance(self, cycles: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算平均性能"""
        if not cycles:
            return {}
        
        avg_performance = {}
        performance_keys = ["risk_control_accuracy", "system_response_time", "cache_hit_rate", "monitoring_efficiency", "overall_score"]
        
        for key in performance_keys:
            values = [cycle["performance"][key] for cycle in cycles if key in cycle["performance"]]
            if values:
                avg_performance[key] = sum(values) / len(values)
        
        return avg_performance
    
    def _analyze_optimization_trend(self, cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析优化趋势"""
        if len(cycles) < 2:
            return {"trend": "insufficient_data"}
        
        # 分析整体得分趋势
        scores = [cycle["performance"]["overall_score"] for cycle in cycles]
        trend = "improving" if scores[-1] > scores[0] else "declining" if scores[-1] < scores[0] else "stable"
        
        return {
            "trend": trend,
            "score_change": scores[-1] - scores[0],
            "volatility": np.std(scores) if len(scores) > 1 else 0
        }
    
    def _generate_integration_report(self) -> Dict[str, Any]:
        """生成集成报告"""
        return {
            "timestamp": time.time(),
            "config": asdict(self.config),
            "model_status": self.optimizer.get_model_status(),
            "optimization_results_count": len(self.optimization_results),
            "feature_importance": self.optimizer.feature_importance,
            "performance_summary": self._calculate_average_performance(self.optimization_results)
        }

class MLIntegrationReporter:
    """机器学习集成报告器"""
    
    def generate_integration_report(self, integration_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成集成报告"""
        report = {
            "timestamp": time.time(),
            "integration_result": integration_result,
            "summary": self._generate_summary(integration_result),
            "recommendations": self._generate_recommendations(integration_result)
        }
        
        return report
    
    def _generate_summary(self, integration_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成摘要"""
        if integration_result["status"] == "error":
            return {
                "integration_status": "failed",
                "error": integration_result["error"]
            }
        
        initial_opt = integration_result["initial_optimization"]
        continuous_opt = integration_result["continuous_optimization"]
        
        return {
            "integration_status": "success",
            "initial_performance": initial_opt["performance"]["overall_score"],
            "cycles_completed": continuous_opt["cycles_completed"],
            "average_performance": continuous_opt["average_performance"]["overall_score"],
            "optimization_trend": continuous_opt["optimization_trend"]["trend"]
        }
    
    def _generate_recommendations(self, integration_result: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if integration_result["status"] == "error":
            recommendations.append("集成失败，建议检查错误信息并修复问题")
            return recommendations
        
        summary = self._generate_summary(integration_result)
        
        if summary["optimization_trend"] == "improving":
            recommendations.append("优化趋势良好，建议继续监控和调整")
        elif summary["optimization_trend"] == "declining":
            recommendations.append("优化趋势下降，建议检查模型参数和特征")
        else:
            recommendations.append("优化趋势稳定，建议增加更多训练数据")
        
        recommendations.append("建议定期评估模型性能，确保优化效果")
        recommendations.append("建议监控特征重要性变化，识别关键影响因素")
        
        return recommendations

def main():
    """主函数"""
    print("🤖 启动机器学习集成...")
    
    # 创建ML配置
    config = MLConfig(
        model_type="gradient_boosting",
        learning_rate=0.1,
        max_iterations=100,
        early_stopping_patience=10,
        validation_split=0.2,
        feature_importance_threshold=0.05,
        optimization_interval=3600,
        min_data_points=50
    )
    
    # 创建集成管理器
    manager = MLIntegrationManager(config)
    
    # 执行集成
    integration_result = manager.start_ml_integration()
    
    # 生成报告
    reporter = MLIntegrationReporter()
    report = reporter.generate_integration_report(integration_result)
    
    print("✅ 机器学习集成完成!")
    
    # 打印结果
    print("\n" + "="*50)
    print("🎯 集成结果:")
    print("="*50)
    
    summary = report["summary"]
    print(f"集成状态: {summary['integration_status']}")
    
    if summary["integration_status"] == "success":
        print(f"初始性能: {summary['initial_performance']:.3f}")
        print(f"完成周期: {summary['cycles_completed']}")
        print(f"平均性能: {summary['average_performance']:.3f}")
        print(f"优化趋势: {summary['optimization_trend']}")
    
    print("\n📊 详细结果:")
    if integration_result["status"] == "success":
        initial_opt = integration_result["initial_optimization"]
        print(f"初始优化性能:")
        for metric, value in initial_opt["performance"].items():
            print(f"  {metric}: {value:.3f}")
        
        continuous_opt = integration_result["continuous_optimization"]
        print(f"\n持续优化结果:")
        print(f"  完成周期: {continuous_opt['cycles_completed']}")
        print(f"  平均性能: {continuous_opt['average_performance']['overall_score']:.3f}")
        print(f"  优化趋势: {continuous_opt['optimization_trend']['trend']}")
    
    print("\n💡 建议:")
    for recommendation in report["recommendations"]:
        print(f"  - {recommendation}")
    
    print("="*50)
    
    # 保存集成报告
    output_dir = Path("reports/ml_integration/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / "ml_integration_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"📄 集成报告已保存: {report_file}")

if __name__ == "__main__":
    main() 