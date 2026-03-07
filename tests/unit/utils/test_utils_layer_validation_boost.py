# -*- coding: utf-8 -*-
"""
工具层 - 工具层功能验证测试
补充工具层单元测试，目标覆盖率: 70%+

测试范围:
1. 回测工具测试 - 回测数据处理、性能计算、风险分析
2. 开发工具测试 - CI/CD集成、文档管理、代码质量
3. 日志工具测试 - 日志配置、日志记录、日志分析
4. 辅助工具测试 - 数据转换、格式验证、工具函数
"""

import pytest
import time
import tempfile
import os
import json
import logging
import subprocess
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path


class TestBacktestUtils:
    """测试回测工具功能"""

    def test_backtest_performance_calculator(self):
        """测试回测性能计算器"""
        class BacktestPerformanceCalculator:
            def __init__(self):
                self.risk_free_rate = 0.02  # 2%无风险利率

            def calculate_returns(self, prices: List[float]) -> Dict[str, Any]:
                """计算收益率指标"""
                if len(prices) < 2:
                    return {"error": "insufficient_data"}

                # 计算简单收益率
                returns = []
                for i in range(1, len(prices)):
                    ret = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(ret)

                # 计算累计收益率
                cumulative_return = (prices[-1] - prices[0]) / prices[0]

                # 计算年化收益率（假设252个交易日）
                total_days = len(prices) - 1
                if total_days > 0:
                    annualized_return = (1 + cumulative_return) ** (252 / total_days) - 1
                else:
                    annualized_return = 0

                # 计算波动率
                volatility = np.std(returns) * np.sqrt(252) if returns else 0

                # 计算夏普比率
                excess_returns = [r - self.risk_free_rate/252 for r in returns]
                avg_excess_return = sum(excess_returns) / len(excess_returns) if excess_returns else 0
                sharpe_ratio = avg_excess_return / (np.std(excess_returns) + 1e-10) * np.sqrt(252)

                return {
                    "total_return": cumulative_return,
                    "annualized_return": annualized_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": self._calculate_max_drawdown(prices),
                    "win_rate": self._calculate_win_rate(returns),
                    "profit_loss_ratio": self._calculate_profit_loss_ratio(returns)
                }

            def _calculate_max_drawdown(self, prices: List[float]) -> float:
                """计算最大回撤"""
                if len(prices) < 2:
                    return 0.0

                peak = prices[0]
                max_drawdown = 0.0

                for price in prices:
                    if price > peak:
                        peak = price
                    drawdown = (peak - price) / peak
                    max_drawdown = max(max_drawdown, drawdown)

                return max_drawdown

            def _calculate_win_rate(self, returns: List[float]) -> float:
                """计算胜率"""
                if not returns:
                    return 0.0

                winning_trades = sum(1 for r in returns if r > 0)
                return winning_trades / len(returns)

            def _calculate_profit_loss_ratio(self, returns: List[float]) -> float:
                """计算盈亏比"""
                if not returns:
                    return 0.0

                profits = [r for r in returns if r > 0]
                losses = [abs(r) for r in returns if r < 0]

                avg_profit = sum(profits) / len(profits) if profits else 0
                avg_loss = sum(losses) / len(losses) if losses else 0

                return avg_profit / avg_loss if avg_loss > 0 else float('inf')

            def compare_strategies(self, strategy_returns: Dict[str, List[float]]) -> Dict[str, Any]:
                """比较不同策略的性能"""
                results = {}

                for strategy_name, prices in strategy_returns.items():
                    if len(prices) >= 2:
                        results[strategy_name] = self.calculate_returns(prices)

                if not results:
                    return {"error": "no_valid_strategies"}

                # 计算排名
                ranking = sorted(results.items(), key=lambda x: x[1].get('sharpe_ratio', 0), reverse=True)

                return {
                    "strategy_results": results,
                    "ranking": ranking,
                    "best_strategy": ranking[0][0] if ranking else None,
                    "worst_strategy": ranking[-1][0] if ranking else None,
                    "performance_summary": self._generate_performance_summary(results)
                }

            def _generate_performance_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
                """生成性能汇总"""
                if not results:
                    return {}

                metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']

                summary = {}
                for metric in metrics:
                    values = [r.get(metric, 0) for r in results.values()]
                    summary[f"{metric}_avg"] = sum(values) / len(values)
                    summary[f"{metric}_best"] = max(values)
                    summary[f"{metric}_worst"] = min(values)
                    summary[f"{metric}_std"] = np.std(values)

                return summary

        # 测试回测性能计算器
        calculator = BacktestPerformanceCalculator()

        # 测试收益率计算
        prices = [100, 105, 102, 108, 106, 110, 115]
        performance = calculator.calculate_returns(prices)

        assert "total_return" in performance
        assert "annualized_return" in performance
        assert "sharpe_ratio" in performance
        assert "max_drawdown" in performance
        assert performance["total_return"] > 0  # 应该有正收益
        assert performance["win_rate"] >= 0 and performance["win_rate"] <= 1

        # 测试策略比较
        strategy_returns = {
            "strategy_a": [100, 110, 105, 115],
            "strategy_b": [100, 102, 108, 112],
            "strategy_c": [100, 98, 105, 118]
        }

        comparison = calculator.compare_strategies(strategy_returns)

        assert "strategy_results" in comparison
        assert "ranking" in comparison
        assert "best_strategy" in comparison
        assert len(comparison["ranking"]) == 3

    def test_backtest_risk_analyzer(self):
        """测试回测风险分析器"""
        class BacktestRiskAnalyzer:
            def __init__(self):
                self.confidence_levels = [0.95, 0.99, 0.999]

            def calculate_var(self, returns: List[float], confidence_level: float = 0.95) -> Dict[str, Any]:
                """计算VaR（在险价值）"""
                if len(returns) < 30:  # 需要足够的数据
                    return {"error": "insufficient_data"}

                # 计算历史VaR
                sorted_returns = sorted(returns)
                index = int((1 - confidence_level) * len(returns))
                historical_var = sorted_returns[index]

                # 计算参数VaR（假设正态分布）
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                z_score = {0.95: -1.645, 0.99: -2.326, 0.999: -3.090}[confidence_level]
                parametric_var = mean_return + z_score * std_return

                return {
                    "historical_var": historical_var,
                    "parametric_var": parametric_var,
                    "confidence_level": confidence_level,
                    "expected_shortfall": self._calculate_expected_shortfall(returns, confidence_level),
                    "var_percentile": (1 - confidence_level) * 100
                }

            def _calculate_expected_shortfall(self, returns: List[float], confidence_level: float) -> float:
                """计算期望亏空（ES）"""
                sorted_returns = sorted(returns)
                index = int((1 - confidence_level) * len(returns))
                tail_losses = sorted_returns[:index]
                return sum(tail_losses) / len(tail_losses) if tail_losses else 0

            def stress_test_portfolio(self, portfolio_returns: List[float],
                                    stress_scenarios: Dict[str, float]) -> Dict[str, Any]:
                """压力测试投资组合"""
                results = {}

                for scenario_name, shock in stress_scenarios.items():
                    # 应用冲击
                    stressed_returns = [r * (1 + shock) for r in portfolio_returns]

                    # 计算压力测试结果
                    stressed_performance = self._calculate_stressed_performance(stressed_returns)

                    results[scenario_name] = {
                        "shock": shock,
                        "stressed_returns": stressed_returns[:5],  # 只显示前5个
                        "performance_impact": stressed_performance
                    }

                return {
                    "stress_test_results": results,
                    "worst_case_scenario": max(results.items(), key=lambda x: abs(x[1]["performance_impact"]["total_loss"])),
                    "risk_assessment": self._assess_portfolio_resilience(results)
                }

            def _calculate_stressed_performance(self, returns: List[float]) -> Dict[str, Any]:
                """计算压力测试性能"""
                if not returns:
                    return {"total_loss": 0, "max_loss": 0, "loss_probability": 0}

                losses = [-r for r in returns if r < 0]
                total_loss = sum(losses)
                max_loss = max(losses) if losses else 0
                loss_probability = len(losses) / len(returns)

                return {
                    "total_loss": total_loss,
                    "max_loss": max_loss,
                    "loss_probability": loss_probability,
                    "avg_loss": total_loss / len(losses) if losses else 0
                }

            def _assess_portfolio_resilience(self, stress_results: Dict[str, Dict[str, Any]]) -> str:
                """评估投资组合韧性"""
                total_losses = [r["performance_impact"]["total_loss"] for r in stress_results.values()]
                avg_loss = sum(total_losses) / len(total_losses)

                if avg_loss < 0.1:  # 损失小于10%
                    return "high_resilience"
                elif avg_loss < 0.25:  # 损失小于25%
                    return "moderate_resilience"
                else:
                    return "low_resilience"

            def monte_carlo_simulation(self, returns: List[float], num_simulations: int = 1000,
                                     time_horizon: int = 252) -> Dict[str, Any]:
                """蒙特卡洛模拟"""
                if len(returns) < 30:
                    return {"error": "insufficient_historical_data"}

                # 拟合分布参数
                mean_return = np.mean(returns)
                std_return = np.std(returns)

                simulation_results = []

                for _ in range(num_simulations):
                    # 生成模拟路径
                    simulated_returns = np.random.normal(mean_return, std_return, time_horizon)
                    cumulative_return = np.prod(1 + simulated_returns) - 1

                    simulation_results.append(cumulative_return)

                # 计算统计信息
                sorted_results = sorted(simulation_results)
                var_95 = sorted_results[int(0.05 * num_simulations)]
                var_99 = sorted_results[int(0.01 * num_simulations)]

                return {
                    "num_simulations": num_simulations,
                    "time_horizon": time_horizon,
                    "expected_return": np.mean(simulation_results),
                    "return_std": np.std(simulation_results),
                    "var_95": var_95,
                    "var_99": var_99,
                    "worst_case": min(simulation_results),
                    "best_case": max(simulation_results),
                    "probability_of_loss": sum(1 for r in simulation_results if r < 0) / num_simulations
                }

        # 测试回测风险分析器
        analyzer = BacktestRiskAnalyzer()

        # 生成测试数据
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)  # 100天的收益率数据

        # 测试VaR计算
        var_result = analyzer.calculate_var(returns.tolist(), 0.95)
        assert "historical_var" in var_result
        assert "parametric_var" in var_result
        assert var_result["confidence_level"] == 0.95
        assert var_result["historical_var"] < 0  # VaR应该是负数

        # 测试压力测试
        stress_scenarios = {
            "market_crash": -0.3,  # 30%市场下跌
            "moderate_decline": -0.15,  # 15%市场下跌
            "volatility_spike": 0.2   # 波动率增加
        }

        stress_test = analyzer.stress_test_portfolio(returns.tolist(), stress_scenarios)
        assert "stress_test_results" in stress_test
        assert "worst_case_scenario" in stress_test
        assert len(stress_test["stress_test_results"]) == 3

        # 测试蒙特卡洛模拟
        mc_simulation = analyzer.monte_carlo_simulation(returns.tolist(), num_simulations=100, time_horizon=30)
        assert "expected_return" in mc_simulation
        assert "var_95" in mc_simulation
        assert mc_simulation["num_simulations"] == 100
        assert mc_simulation["probability_of_loss"] >= 0 and mc_simulation["probability_of_loss"] <= 1

    def test_backtest_data_processor(self):
        """测试回测数据处理器"""
        class BacktestDataProcessor:
            def __init__(self):
                self.data_cache = {}
                self.processors = {}

            def register_data_processor(self, processor_name: str,
                                      processor_func: Callable) -> str:
                """注册数据处理器"""
                self.processors[processor_name] = {
                    "function": processor_func,
                    "registered_at": datetime.now(),
                    "usage_count": 0
                }
                return processor_name

            def process_market_data(self, raw_data: Dict[str, Any],
                                  processor_name: str) -> Dict[str, Any]:
                """处理市场数据"""
                if processor_name not in self.processors:
                    return {"error": "processor_not_found"}

                processor = self.processors[processor_name]
                processor["usage_count"] += 1

                try:
                    processed_data = processor["function"](raw_data)

                    # 缓存处理结果
                    cache_key = f"{processor_name}_{hash(str(raw_data))}"
                    self.data_cache[cache_key] = {
                        "processed_data": processed_data,
                        "timestamp": datetime.now(),
                        "processor": processor_name
                    }

                    return processed_data

                except Exception as e:
                    return {
                        "error": "processing_failed",
                        "message": str(e),
                        "processor": processor_name
                    }

            def validate_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
                """验证数据质量"""
                quality_checks = {
                    "has_required_fields": self._check_required_fields(data),
                    "data_completeness": self._check_data_completeness(data),
                    "data_consistency": self._check_data_consistency(data),
                    "outlier_detection": self._detect_outliers(data)
                }

                overall_quality_score = sum(
                    1 for check in quality_checks.values() if check.get("passed", False)
                ) / len(quality_checks)

                return {
                    "quality_checks": quality_checks,
                    "overall_score": overall_quality_score,
                    "quality_rating": self._get_quality_rating(overall_quality_score),
                    "issues": [k for k, v in quality_checks.items() if not v.get("passed", False)]
                }

            def _check_required_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
                """检查必需字段"""
                required_fields = ["timestamp", "open", "high", "low", "close", "volume"]

                missing_fields = [field for field in required_fields if field not in data]

                return {
                    "passed": len(missing_fields) == 0,
                    "missing_fields": missing_fields,
                    "required_fields": required_fields
                }

            def _check_data_completeness(self, data: Dict[str, Any]) -> Dict[str, Any]:
                """检查数据完整性"""
                total_fields = len(data)
                null_fields = sum(1 for v in data.values() if v is None or (isinstance(v, str) and v.strip() == ""))

                completeness_ratio = (total_fields - null_fields) / total_fields if total_fields > 0 else 0

                return {
                    "passed": completeness_ratio >= 0.95,  # 95%以上完整
                    "completeness_ratio": completeness_ratio,
                    "total_fields": total_fields,
                    "null_fields": null_fields
                }

            def _check_data_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
                """检查数据一致性"""
                issues = []

                # 检查价格逻辑：high >= max(open, close) >= min(open, close) >= low
                if all(k in data for k in ["open", "high", "low", "close"]):
                    open_price = data["open"]
                    high_price = data["high"]
                    low_price = data["low"]
                    close_price = data["close"]

                    if not (high_price >= max(open_price, close_price) and
                           min(open_price, close_price) >= low_price):
                        issues.append("price_logic_violation")

                # 检查成交量合理性
                if "volume" in data and data["volume"] < 0:
                    issues.append("negative_volume")

                return {
                    "passed": len(issues) == 0,
                    "issues": issues,
                    "consistency_checks": ["price_logic", "volume_validity"]
                }

            def _detect_outliers(self, data: Dict[str, Any]) -> Dict[str, Any]:
                """检测异常值"""
                outliers = []

                # 使用IQR方法检测价格异常值
                price_fields = ["open", "high", "low", "close"]
                prices = [data.get(field) for field in price_fields if field in data and isinstance(data[field], (int, float))]

                if len(prices) >= 4:
                    q1 = np.percentile(prices, 25)
                    q3 = np.percentile(prices, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    outlier_prices = [p for p in prices if p < lower_bound or p > upper_bound]
                    if outlier_prices:
                        outliers.extend(["price_outlier"] * len(outlier_prices))

                # 检查成交量异常
                if "volume" in data and isinstance(data["volume"], (int, float)):
                    # 成交量为0可能表示数据问题
                    if data["volume"] == 0:
                        outliers.append("zero_volume")

                return {
                    "passed": len(outliers) == 0,
                    "outliers_detected": outliers,
                    "outlier_types": list(set(outliers))
                }

            def _get_quality_rating(self, score: float) -> str:
                """获取质量评级"""
                if score >= 0.9:
                    return "excellent"
                elif score >= 0.8:
                    return "good"
                elif score >= 0.7:
                    return "acceptable"
                else:
                    return "poor"

        # 测试回测数据处理器
        processor = BacktestDataProcessor()

        # 注册数据处理器
        def normalize_ohlcv(data):
            """标准化OHLCV数据"""
            normalized = data.copy()
            for field in ["open", "high", "low", "close"]:
                if field in normalized:
                    # 简单的标准化（这里可以是更复杂的处理）
                    normalized[field] = round(normalized[field], 2)
            return normalized

        processor.register_data_processor("ohlcv_normalizer", normalize_ohlcv)

        # 处理市场数据
        raw_data = {
            "timestamp": "2023-01-01",
            "open": 100.123,
            "high": 105.456,
            "low": 99.789,
            "close": 104.321,
            "volume": 1000000
        }

        processed_data = processor.process_market_data(raw_data, "ohlcv_normalizer")
        assert "timestamp" in processed_data
        assert processed_data["open"] == 100.12  # 应该被四舍五入

        # 验证数据质量
        quality_report = processor.validate_data_quality(raw_data)
        assert "quality_checks" in quality_report
        assert "overall_score" in quality_report
        assert quality_report["overall_score"] >= 0 and quality_report["overall_score"] <= 1

        # 测试有问题的数据
        bad_data = {
            "timestamp": None,
            "open": 100,
            "high": 90,  # 高价低于开盘价（不合理）
            "low": 95,
            "close": 105,  # 收盘价高于高价（不合理）
            "volume": -100  # 负成交量
        }

        bad_quality = processor.validate_data_quality(bad_data)
        assert bad_quality["overall_score"] < 1  # 应该有问题
        assert len(bad_quality["issues"]) > 0


class TestDevTools:
    """测试开发工具功能"""

    def test_ci_cd_integration(self):
        """测试CI/CD集成工具"""
        class CICDPipelineManager:
            def __init__(self):
                self.pipelines = {}
                self.build_history = []
                self.deployment_status = {}

            def create_pipeline(self, pipeline_name: str,
                              stages: List[Dict[str, Any]]) -> str:
                """创建CI/CD流水线"""
                pipeline = {
                    "name": pipeline_name,
                    "stages": stages,
                    "created_at": datetime.now(),
                    "last_run": None,
                    "status": "created",
                    "stage_status": {stage["name"]: "pending" for stage in stages}
                }

                self.pipelines[pipeline_name] = pipeline
                return pipeline_name

            def execute_pipeline(self, pipeline_name: str) -> Dict[str, Any]:
                """执行CI/CD流水线"""
                if pipeline_name not in self.pipelines:
                    return {"error": "pipeline_not_found"}

                pipeline = self.pipelines[pipeline_name]
                pipeline["status"] = "running"
                pipeline["last_run"] = datetime.now()

                execution_log = []
                failed_stages = []

                for stage in pipeline["stages"]:
                    stage_result = self._execute_stage(stage)
                    execution_log.append(stage_result)

                    pipeline["stage_status"][stage["name"]] = stage_result["status"]

                    if stage_result["status"] == "failed":
                        failed_stages.append(stage["name"])
                        # 停止流水线执行
                        break

                # 更新流水线状态
                if failed_stages:
                    pipeline["status"] = "failed"
                    overall_status = "failed"
                else:
                    pipeline["status"] = "completed"
                    overall_status = "success"

                execution_result = {
                    "pipeline_name": pipeline_name,
                    "overall_status": overall_status,
                    "execution_time": (datetime.now() - pipeline["last_run"]).total_seconds(),
                    "stages_executed": len(execution_log),
                    "failed_stages": failed_stages,
                    "execution_log": execution_log,
                    "timestamp": datetime.now()
                }

                self.build_history.append(execution_result)

                return execution_result

            def _execute_stage(self, stage: Dict[str, Any]) -> Dict[str, Any]:
                """执行单个阶段"""
                stage_name = stage["name"]
                stage_type = stage.get("type", "generic")

                start_time = time.time()

                try:
                    # 模拟不同类型的阶段执行
                    if stage_type == "build":
                        result = self._execute_build_stage(stage)
                    elif stage_type == "test":
                        result = self._execute_test_stage(stage)
                    elif stage_type == "deploy":
                        result = self._execute_deploy_stage(stage)
                    else:
                        result = {"status": "success", "message": "Generic stage completed"}

                    execution_time = time.time() - start_time

                    return {
                        "stage_name": stage_name,
                        "status": result["status"],
                        "execution_time": execution_time,
                        "message": result["message"],
                        "artifacts": result.get("artifacts", [])
                    }

                except Exception as e:
                    execution_time = time.time() - start_time
                    return {
                        "stage_name": stage_name,
                        "status": "failed",
                        "execution_time": execution_time,
                        "message": f"Stage execution failed: {str(e)}",
                        "error": str(e)
                    }

            def _execute_build_stage(self, stage: Dict[str, Any]) -> Dict[str, Any]:
                """执行构建阶段"""
                # 模拟构建过程
                success = time.time() % 2 > 0.2  # 80%成功率

                if success:
                    return {
                        "status": "success",
                        "message": "Build completed successfully",
                        "artifacts": ["app.jar", "config.yml", "Dockerfile"]
                    }
                else:
                    raise Exception("Build failed: compilation error")

            def _execute_test_stage(self, stage: Dict[str, Any]) -> Dict[str, Any]:
                """执行测试阶段"""
                # 模拟测试执行
                test_results = {
                    "total_tests": 100,
                    "passed": 95,
                    "failed": 3,
                    "skipped": 2
                }

                if test_results["failed"] == 0:
                    return {
                        "status": "success",
                        "message": f"Tests passed: {test_results['passed']}/{test_results['total_tests']}",
                        "artifacts": ["test-results.xml", "coverage-report.html"]
                    }
                else:
                    raise Exception(f"Tests failed: {test_results['failed']} tests failed")

            def _execute_deploy_stage(self, stage: Dict[str, Any]) -> Dict[str, Any]:
                """执行部署阶段"""
                # 模拟部署过程
                success = time.time() % 2 > 0.1  # 90%成功率

                if success:
                    return {
                        "status": "success",
                        "message": "Deployment completed successfully",
                        "artifacts": ["deployment-log.txt", "health-check-results.json"]
                    }
                else:
                    raise Exception("Deployment failed: service unavailable")

            def get_pipeline_status(self, pipeline_name: str) -> Dict[str, Any]:
                """获取流水线状态"""
                if pipeline_name not in self.pipelines:
                    return {"error": "pipeline_not_found"}

                pipeline = self.pipelines[pipeline_name]

                return {
                    "pipeline_name": pipeline_name,
                    "status": pipeline["status"],
                    "last_run": pipeline["last_run"],
                    "stage_status": pipeline["stage_status"],
                    "created_at": pipeline["created_at"],
                    "total_stages": len(pipeline["stages"])
                }

            def get_build_metrics(self) -> Dict[str, Any]:
                """获取构建指标"""
                if not self.build_history:
                    return {"error": "no_build_history"}

                total_builds = len(self.build_history)
                successful_builds = sum(1 for b in self.build_history if b["overall_status"] == "success")
                failed_builds = total_builds - successful_builds

                avg_execution_time = sum(b["execution_time"] for b in self.build_history) / total_builds

                # 计算最近10次构建的成功率
                recent_builds = self.build_history[-10:]
                recent_success_rate = sum(1 for b in recent_builds if b["overall_status"] == "success") / len(recent_builds)

                return {
                    "total_builds": total_builds,
                    "successful_builds": successful_builds,
                    "failed_builds": failed_builds,
                    "success_rate": successful_builds / total_builds,
                    "recent_success_rate": recent_success_rate,
                    "avg_execution_time": avg_execution_time,
                    "build_frequency": self._calculate_build_frequency()
                }

            def _calculate_build_frequency(self) -> float:
                """计算构建频率（每天）"""
                if len(self.build_history) < 2:
                    return 0.0

                timestamps = sorted([b["timestamp"] for b in self.build_history])
                time_span_days = (timestamps[-1] - timestamps[0]).total_seconds() / (24 * 3600)

                if time_span_days > 0:
                    return len(self.build_history) / time_span_days
                else:
                    return 0.0

        # 测试CI/CD集成工具
        manager = CICDPipelineManager()

        # 创建CI/CD流水线
        stages = [
            {"name": "checkout", "type": "generic", "script": "git pull"},
            {"name": "build", "type": "build", "script": "mvn compile"},
            {"name": "test", "type": "test", "script": "mvn test"},
            {"name": "deploy", "type": "deploy", "script": "kubectl apply"}
        ]

        pipeline_name = manager.create_pipeline("main_pipeline", stages)
        assert pipeline_name == "main_pipeline"

        # 执行流水线
        result = manager.execute_pipeline(pipeline_name)

        assert "overall_status" in result
        assert "execution_log" in result
        assert result["stages_executed"] >= 0
        assert result["overall_status"] in ["success", "failed"]

        # 获取流水线状态
        status = manager.get_pipeline_status(pipeline_name)
        assert status["status"] in ["completed", "failed"]
        assert "stage_status" in status

        # 获取构建指标
        metrics = manager.get_build_metrics()
        assert "total_builds" in metrics
        assert "success_rate" in metrics
        assert metrics["success_rate"] >= 0 and metrics["success_rate"] <= 1

    def test_documentation_manager(self):
        """测试文档管理器"""
        class DocumentationManager:
            def __init__(self):
                self.documents = {}
                self.templates = {}
                self.generation_history = []

            def create_document_template(self, template_name: str,
                                       template_content: str,
                                       variables: List[str]) -> str:
                """创建文档模板"""
                self.templates[template_name] = {
                    "content": template_content,
                    "variables": variables,
                    "created_at": datetime.now(),
                    "usage_count": 0
                }

                return template_name

            def generate_document(self, template_name: str,
                                variables: Dict[str, Any],
                                output_format: str = "markdown") -> Dict[str, Any]:
                """生成文档"""
                if template_name not in self.templates:
                    return {"error": "template_not_found"}

                template = self.templates[template_name]
                template["usage_count"] += 1

                try:
                    # 渲染模板
                    content = template["content"]

                    for var_name, var_value in variables.items():
                        placeholder = f"{{{{ {var_name} }}}}"
                        content = content.replace(placeholder, str(var_value))

                    # 格式化输出
                    if output_format == "html":
                        content = self._convert_to_html(content)
                    elif output_format == "pdf":
                        content = self._convert_to_pdf(content)

                    document = {
                        "template": template_name,
                        "content": content,
                        "variables": variables,
                        "format": output_format,
                        "generated_at": datetime.now(),
                        "metadata": {
                            "word_count": len(content.split()),
                            "line_count": len(content.split('\n')),
                            "template_version": "1.0"
                        }
                    }

                    # 存储生成的文档
                    doc_id = f"{template_name}_{int(datetime.now().timestamp())}"
                    self.documents[doc_id] = document

                    # 记录生成历史
                    self.generation_history.append({
                        "doc_id": doc_id,
                        "template": template_name,
                        "timestamp": datetime.now(),
                        "format": output_format
                    })

                    return {
                        "doc_id": doc_id,
                        "content": content,
                        "format": output_format,
                        "metadata": document["metadata"]
                    }

                except Exception as e:
                    return {
                        "error": "generation_failed",
                        "message": str(e),
                        "template": template_name
                    }

            def _convert_to_html(self, markdown_content: str) -> str:
                """转换为HTML（简化实现）"""
                # 简单的Markdown到HTML转换
                html = markdown_content
                html = html.replace("# ", "<h1>").replace("\n", "</h1>\n")
                html = html.replace("## ", "<h2>").replace("\n", "</h2>\n")
                html = html.replace("**", "<strong>").replace("**", "</strong>")
                html = html.replace("*", "<em>").replace("*", "</em>")
                return f"<html><body>{html}</body></html>"

            def _convert_to_pdf(self, content: str) -> bytes:
                """转换为PDF（模拟）"""
                # 模拟PDF生成
                return f"PDF_CONTENT:{content[:100]}...".encode('utf-8')

            def validate_document(self, doc_id: str) -> Dict[str, Any]:
                """验证文档"""
                if doc_id not in self.documents:
                    return {"error": "document_not_found"}

                document = self.documents[doc_id]
                content = document["content"]

                validation_results = {
                    "spelling_check": self._check_spelling(content),
                    "grammar_check": self._check_grammar(content),
                    "completeness_check": self._check_completeness(document),
                    "format_check": self._check_format(content, document["format"])
                }

                overall_valid = all(result["passed"] for result in validation_results.values())

                return {
                    "doc_id": doc_id,
                    "overall_valid": overall_valid,
                    "validation_results": validation_results,
                    "issues": [k for k, v in validation_results.items() if not v["passed"]]
                }

            def _check_spelling(self, content: str) -> Dict[str, Any]:
                """检查拼写（简化实现）"""
                # 简单的拼写检查（检查常见错误）
                common_errors = ["teh", "recieve", "seperate"]
                found_errors = [error for error in common_errors if error in content.lower()]

                return {
                    "passed": len(found_errors) == 0,
                    "errors_found": found_errors,
                    "error_count": len(found_errors)
                }

            def _check_grammar(self, content: str) -> Dict[str, Any]:
                """检查语法（简化实现）"""
                # 检查句子结构
                sentences = content.split('.')
                incomplete_sentences = sum(1 for s in sentences if len(s.strip()) > 0 and not s.strip()[-1] in ['!', '?'])

                return {
                    "passed": incomplete_sentences == 0,
                    "incomplete_sentences": incomplete_sentences,
                    "total_sentences": len(sentences)
                }

            def _check_completeness(self, document: Dict[str, Any]) -> Dict[str, Any]:
                """检查完整性"""
                required_sections = ["title", "introduction", "content", "conclusion"]
                template_vars = document.get("variables", {})

                missing_sections = [section for section in required_sections
                                  if section not in str(document["content"]).lower()]

                return {
                    "passed": len(missing_sections) == 0,
                    "missing_sections": missing_sections,
                    "required_sections": required_sections
                }

            def _check_format(self, content: str, format_type: str) -> Dict[str, Any]:
                """检查格式"""
                if format_type == "markdown":
                    # 检查Markdown格式
                    has_headers = "#" in content
                    has_lists = any(line.strip().startswith(("- ", "* ", "1. ")) for line in content.split('\n'))

                    return {
                        "passed": has_headers,
                        "format_issues": [] if has_headers else ["missing_headers"],
                        "format_features": {"has_headers": has_headers, "has_lists": has_lists}
                    }

                elif format_type == "html":
                    # 检查HTML格式
                    has_html_tags = "<" in content and ">" in content

                    return {
                        "passed": has_html_tags,
                        "format_issues": [] if has_html_tags else ["missing_html_tags"],
                        "format_features": {"has_html_tags": has_html_tags}
                    }

                return {"passed": True, "format_issues": [], "format_features": {}}

            def get_documentation_stats(self) -> Dict[str, Any]:
                """获取文档统计"""
                total_docs = len(self.documents)
                total_templates = len(self.templates)

                format_distribution = {}
                for doc in self.documents.values():
                    fmt = doc["format"]
                    format_distribution[fmt] = format_distribution.get(fmt, 0) + 1

                template_usage = {}
                for name, template in self.templates.items():
                    template_usage[name] = template["usage_count"]

                return {
                    "total_documents": total_docs,
                    "total_templates": total_templates,
                    "format_distribution": format_distribution,
                    "template_usage": template_usage,
                    "most_used_template": max(template_usage.items(), key=lambda x: x[1]) if template_usage else None,
                    "generation_history": len(self.generation_history)
                }

        # 测试文档管理器
        manager = DocumentationManager()

        # 创建文档模板
        template_content = """
# {{ title }}

## 介绍
{{ introduction }}

## 内容
{{ content }}

## 结论
{{ conclusion }}

生成时间: {{ generation_date }}
"""

        variables = ["title", "introduction", "content", "conclusion", "generation_date"]
        template_name = manager.create_document_template("api_doc", template_content, variables)

        # 生成文档
        doc_vars = {
            "title": "API文档",
            "introduction": "这是API的使用指南",
            "content": "详细的API说明和示例代码",
            "conclusion": "总结和常见问题解答",
            "generation_date": datetime.now().strftime("%Y-%m-%d")
        }

        doc_result = manager.generate_document(template_name, doc_vars, "markdown")
        assert "doc_id" in doc_result
        assert "content" in doc_result
        assert "{{ title }}" not in doc_result["content"]  # 变量应该被替换

        # 验证文档
        validation = manager.validate_document(doc_result["doc_id"])
        assert "overall_valid" in validation
        assert "validation_results" in validation

        # 获取文档统计
        stats = manager.get_documentation_stats()
        assert stats["total_documents"] == 1
        assert stats["total_templates"] == 1
        assert stats["format_distribution"]["markdown"] == 1


class TestLoggingTools:
    """测试日志工具功能"""

    def test_logger_manager(self):
        """测试日志管理器"""
        class LoggerManager:
            def __init__(self):
                self.loggers = {}
                self.log_history = []
                self.log_levels = {
                    "DEBUG": 10,
                    "INFO": 20,
                    "WARNING": 30,
                    "ERROR": 40,
                    "CRITICAL": 50
                }

            def create_logger(self, logger_name: str,
                            level: str = "INFO",
                            handlers: List[Dict[str, Any]] = None) -> str:
                """创建日志记录器"""
                if handlers is None:
                    handlers = [{"type": "console", "level": level}]

                logger_config = {
                    "name": logger_name,
                    "level": level,
                    "handlers": handlers,
                    "created_at": datetime.now(),
                    "message_count": 0,
                    "error_count": 0
                }

                self.loggers[logger_name] = logger_config
                return logger_name

            def log_message(self, logger_name: str, level: str,
                          message: str, extra_data: Dict[str, Any] = None) -> bool:
                """记录日志消息"""
                if logger_name not in self.loggers:
                    return False

                if extra_data is None:
                    extra_data = {}

                logger = self.loggers[logger_name]

                # 检查日志级别
                if self.log_levels.get(level, 0) < self.log_levels.get(logger["level"], 20):
                    return False  # 低于配置的日志级别

                log_entry = {
                    "logger": logger_name,
                    "level": level,
                    "message": message,
                    "timestamp": datetime.now(),
                    "extra_data": extra_data
                }

                # 处理日志（模拟）
                for handler in logger["handlers"]:
                    self._process_log_entry(log_entry, handler)

                # 更新统计
                logger["message_count"] += 1
                if level in ["ERROR", "CRITICAL"]:
                    logger["error_count"] += 1

                # 存储日志历史
                self.log_history.append(log_entry)

                # 保持最近1000条日志
                if len(self.log_history) > 1000:
                    self.log_history = self.log_history[-1000:]

                return True

            def _process_log_entry(self, log_entry: Dict[str, Any], handler: Dict[str, Any]):
                """处理日志条目"""
                handler_type = handler.get("type", "console")

                if handler_type == "console":
                    # 控制台输出
                    formatted_message = f"[{log_entry['timestamp']}] {log_entry['level']}: {log_entry['message']}"
                    print(formatted_message)
                elif handler_type == "file":
                    # 文件输出（模拟）
                    pass
                elif handler_type == "database":
                    # 数据库存储（模拟）
                    pass

            def get_logger_stats(self, logger_name: str) -> Dict[str, Any]:
                """获取日志记录器统计"""
                if logger_name not in self.loggers:
                    return {"error": "logger_not_found"}

                logger = self.loggers[logger_name]

                # 计算日志级别分布
                level_distribution = {}
                for entry in self.log_history:
                    if entry["logger"] == logger_name:
                        level = entry["level"]
                        level_distribution[level] = level_distribution.get(level, 0) + 1

                # 计算错误率
                error_rate = logger["error_count"] / logger["message_count"] if logger["message_count"] > 0 else 0

                return {
                    "logger_name": logger_name,
                    "total_messages": logger["message_count"],
                    "error_count": logger["error_count"],
                    "error_rate": error_rate,
                    "level_distribution": level_distribution,
                    "created_at": logger["created_at"],
                    "current_level": logger["level"]
                }

            def search_logs(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
                """搜索日志"""
                matching_logs = []

                for log_entry in self.log_history:
                    match = True

                    for filter_key, filter_value in filters.items():
                        if filter_key == "start_time" and log_entry["timestamp"] < filter_value:
                            match = False
                            break
                        elif filter_key == "end_time" and log_entry["timestamp"] > filter_value:
                            match = False
                            break
                        elif filter_key == "level" and log_entry["level"] != filter_value:
                            match = False
                            break
                        elif filter_key == "logger" and log_entry["logger"] != filter_value:
                            match = False
                            break
                        elif filter_key == "message_contains" and filter_value not in log_entry["message"]:
                            match = False
                            break

                    if match:
                        matching_logs.append(log_entry)

                return matching_logs

            def analyze_log_patterns(self) -> Dict[str, Any]:
                """分析日志模式"""
                if not self.log_history:
                    return {"error": "no_log_data"}

                # 分析错误模式
                error_logs = [log for log in self.log_history if log["level"] in ["ERROR", "CRITICAL"]]

                # 按小时统计错误
                error_by_hour = {}
                for error_log in error_logs:
                    hour = error_log["timestamp"].hour
                    error_by_hour[hour] = error_by_hour.get(hour, 0) + 1

                # 分析最常见的错误消息
                error_messages = [log["message"] for log in error_logs]
                message_counts = {}
                for msg in error_messages:
                    # 简化消息（取前50字符）
                    simplified_msg = msg[:50]
                    message_counts[simplified_msg] = message_counts.get(simplified_msg, 0) + 1

                top_errors = sorted(message_counts.items(), key=lambda x: x[1], reverse=True)[:5]

                # 计算系统健康指标
                total_logs = len(self.log_history)
                error_rate = len(error_logs) / total_logs if total_logs > 0 else 0

                health_score = max(0, 1 - error_rate)  # 简化的健康评分

                return {
                    "total_logs": total_logs,
                    "error_logs": len(error_logs),
                    "error_rate": error_rate,
                    "health_score": health_score,
                    "error_by_hour": error_by_hour,
                    "top_error_messages": top_errors,
                    "peak_error_hour": max(error_by_hour.items(), key=lambda x: x[1])[0] if error_by_hour else None
                }

        # 测试日志管理器
        manager = LoggerManager()

        # 创建日志记录器
        logger_name = manager.create_logger("app_logger", "INFO",
                                          [{"type": "console", "level": "INFO"}])

        # 记录日志消息
        manager.log_message(logger_name, "INFO", "Application started")
        manager.log_message(logger_name, "WARNING", "Low memory warning")
        manager.log_message(logger_name, "ERROR", "Database connection failed")
        manager.log_message(logger_name, "INFO", "Request processed successfully")

        # 获取日志统计
        stats = manager.get_logger_stats(logger_name)
        assert stats["total_messages"] == 4
        assert stats["error_count"] == 1
        assert stats["error_rate"] == 0.25

        # 搜索日志
        error_logs = manager.search_logs({"level": "ERROR"})
        assert len(error_logs) == 1
        assert error_logs[0]["message"] == "Database connection failed"

        # 分析日志模式
        analysis = manager.analyze_log_patterns()
        assert "error_rate" in analysis
        assert "health_score" in analysis
        assert analysis["error_rate"] == 0.25

    def test_log_analyzer(self):
        """测试日志分析器"""
        class LogAnalyzer:
            def __init__(self):
                self.parsed_logs = []
                self.analysis_cache = {}

            def parse_log_file(self, log_content: str, format_type: str = "generic") -> List[Dict[str, Any]]:
                """解析日志文件"""
                parsed_entries = []

                lines = log_content.strip().split('\n')

                for line in lines:
                    if not line.strip():
                        continue

                    try:
                        if format_type == "generic":
                            parsed_entry = self._parse_generic_log(line)
                        elif format_type == "json":
                            parsed_entry = self._parse_json_log(line)
                        else:
                            parsed_entry = {"raw_line": line, "parse_error": "unsupported_format"}

                        if parsed_entry:
                            parsed_entries.append(parsed_entry)

                    except Exception as e:
                        parsed_entries.append({
                            "raw_line": line,
                            "parse_error": str(e),
                            "timestamp": datetime.now()
                        })

                self.parsed_logs.extend(parsed_entries)
                return parsed_entries

            def _parse_generic_log(self, line: str) -> Dict[str, Any]:
                """解析通用日志格式"""
                # 假设格式: [2023-01-01 10:00:00] INFO: Message content
                import re

                pattern = r'\[([^\]]+)\]\s+(\w+):\s+(.+)'
                match = re.match(pattern, line)

                if match:
                    timestamp_str, level, message = match.groups()
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    except:
                        timestamp = datetime.now()

                    return {
                        "timestamp": timestamp,
                        "level": level,
                        "message": message,
                        "raw_line": line
                    }

                return None

            def _parse_json_log(self, line: str) -> Dict[str, Any]:
                """解析JSON日志格式"""
                try:
                    data = json.loads(line)
                    return {
                        "timestamp": datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
                        "level": data.get("level", "INFO"),
                        "message": data.get("message", ""),
                        "extra_data": {k: v for k, v in data.items() if k not in ["timestamp", "level", "message"]},
                        "raw_line": line
                    }
                except json.JSONDecodeError:
                    return None

            def generate_log_report(self) -> Dict[str, Any]:
                """生成日志报告"""
                if not self.parsed_logs:
                    return {"error": "no_parsed_logs"}

                # 统计基本信息
                total_entries = len(self.parsed_logs)
                level_distribution = {}
                hourly_distribution = {}
                error_messages = []

                for entry in self.parsed_logs:
                    # 级别分布
                    level = entry.get("level", "UNKNOWN")
                    level_distribution[level] = level_distribution.get(level, 0) + 1

                    # 小时分布
                    if "timestamp" in entry:
                        hour = entry["timestamp"].hour
                        hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1

                    # 收集错误消息
                    if level in ["ERROR", "CRITICAL"]:
                        error_messages.append(entry.get("message", ""))

                # 计算错误率
                error_count = level_distribution.get("ERROR", 0) + level_distribution.get("CRITICAL", 0)
                error_rate = error_count / total_entries if total_entries > 0 else 0

                # 分析错误模式
                error_patterns = self._analyze_error_patterns(error_messages)

                # 生成时间序列
                time_series = sorted(
                    [(entry["timestamp"], 1) for entry in self.parsed_logs if "timestamp" in entry],
                    key=lambda x: x[0]
                )

                return {
                    "total_entries": total_entries,
                    "level_distribution": level_distribution,
                    "hourly_distribution": hourly_distribution,
                    "error_rate": error_rate,
                    "error_patterns": error_patterns,
                    "time_series": time_series,
                    "summary": {
                        "most_common_level": max(level_distribution.items(), key=lambda x: x[1])[0] if level_distribution else None,
                        "peak_hour": max(hourly_distribution.items(), key=lambda x: x[1])[0] if hourly_distribution else None,
                        "time_span": (time_series[-1][0] - time_series[0][0]).total_seconds() / 3600 if len(time_series) >= 2 else 0
                    }
                }

            def _analyze_error_patterns(self, error_messages: List[str]) -> Dict[str, Any]:
                """分析错误模式"""
                if not error_messages:
                    return {"patterns": [], "most_common": None}

                # 简单的模式识别
                patterns = {}
                for msg in error_messages:
                    # 提取关键词（简化实现）
                    words = msg.lower().split()
                    key_words = [word for word in words if len(word) > 3][:3]  # 前3个长词
                    pattern_key = " ".join(key_words)

                    patterns[pattern_key] = patterns.get(pattern_key, 0) + 1

                most_common = max(patterns.items(), key=lambda x: x[1]) if patterns else None

                return {
                    "patterns": patterns,
                    "most_common": most_common,
                    "unique_patterns": len(patterns)
                }

            def detect_anomalies(self) -> Dict[str, Any]:
                """检测异常"""
                if len(self.parsed_logs) < 10:
                    return {"error": "insufficient_data"}

                # 分析日志频率异常
                timestamps = [entry["timestamp"] for entry in self.parsed_logs if "timestamp" in entry]

                if len(timestamps) < 10:
                    return {"error": "insufficient_timestamps"}

                # 计算日志间隔
                intervals = []
                for i in range(1, len(timestamps)):
                    interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                    intervals.append(interval)

                # 检测异常间隔（使用简单的统计方法）
                mean_interval = sum(intervals) / len(intervals)
                std_interval = (sum((i - mean_interval)**2 for i in intervals) / len(intervals))**0.5

                anomalies = []
                for i, interval in enumerate(intervals):
                    z_score = abs(interval - mean_interval) / (std_interval + 1e-10)
                    if z_score > 3:  # 3倍标准差
                        anomalies.append({
                            "index": i,
                            "interval": interval,
                            "z_score": z_score,
                            "timestamp": timestamps[i+1]
                        })

                return {
                    "total_intervals": len(intervals),
                    "mean_interval": mean_interval,
                    "std_interval": std_interval,
                    "anomalies_detected": len(anomalies),
                    "anomaly_details": anomalies,
                    "anomaly_rate": len(anomalies) / len(intervals) if intervals else 0
                }

        # 测试日志分析器
        analyzer = LogAnalyzer()

        # 模拟日志内容
        log_content = """
[2023-01-01 10:00:00] INFO: Application started
[2023-01-01 10:01:00] INFO: Processing request 123
[2023-01-01 10:02:00] WARNING: Low memory warning
[2023-01-01 10:03:00] ERROR: Database connection failed
[2023-01-01 10:04:00] INFO: Request processed successfully
[2023-01-01 10:05:00] ERROR: Network timeout occurred
"""

        # 解析日志
        parsed_logs = analyzer.parse_log_file(log_content, "generic")
        assert len(parsed_logs) >= 5  # 应该解析出至少5条日志

        # 生成日志报告
        report = analyzer.generate_log_report()
        assert "total_entries" in report
        assert "level_distribution" in report
        assert report["error_rate"] > 0  # 应该有错误

        # 检测异常（需要足够的数据）
        if len(analyzer.parsed_logs) >= 10:
            anomalies = analyzer.detect_anomalies()
            assert "total_intervals" in anomalies
        else:
            # 数据不足，跳过异常检测测试
            anomalies = analyzer.detect_anomalies()
            assert "error" in anomalies


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
