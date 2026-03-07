#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试业务流程优化器

测试目标：提升business_process/optimizer/optimizer_refactored.py的覆盖率到100%
"""

import pytest

# 尝试导入所需模块
try:
    from src.core.business_process.optimizer.optimizer import ProcessOptimizer
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

from src.core.business_process.optimizer.optimizer_refactored import BusinessProcessOptimizer


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestBusinessProcessOptimizer:
    """测试业务流程优化器"""

    @pytest.fixture
    def optimizer(self):
        """创建业务流程优化器实例"""
        return BusinessProcessOptimizer()

    def test_optimizer_initialization(self, optimizer):
        """测试优化器初始化"""
        assert optimizer.name == "BusinessProcessOptimizer"
        assert optimizer.version == "2.0"
        assert optimizer.description == "智能业务流程优化器 (重构版 v2.0)"

        # 检查组件是否初始化
        assert hasattr(optimizer, 'performance_analyzer')
        assert hasattr(optimizer, 'decision_engine')
        assert hasattr(optimizer, 'process_executor')
        assert hasattr(optimizer, 'recommendation_generator')
        assert hasattr(optimizer, 'process_monitor')

    def test_optimize_process(self, optimizer):
        """测试流程优化"""
        process_data = {
            "process_id": "test_process_001",
            "current_metrics": {"throughput": 100, "latency": 50},
            "target_metrics": {"throughput": 120, "latency": 40}
        }

        with patch.object(optimizer.decision_engine, 'analyze_and_decide') as mock_decide, \
             patch.object(optimizer.process_executor, 'execute_optimization') as mock_execute, \
             patch.object(optimizer.recommendation_generator, 'generate_recommendations') as mock_recommend:

            mock_decision = Mock()
            mock_decision.should_optimize = True
            mock_decision.confidence_score = 0.85
            mock_decide.return_value = mock_decision

            mock_result = Mock()
            mock_result.success = True
            mock_result.improvement_metrics = {"throughput": 15, "latency": -8}
            mock_execute.return_value = mock_result

            mock_recommendations = ["Increase thread pool", "Optimize database queries"]
            mock_recommend.return_value = mock_recommendations

            result = optimizer.optimize_process(process_data)

            assert result["success"] == True
            assert result["process_id"] == "test_process_001"
            assert "improvement_metrics" in result
            assert "recommendations" in result

    def test_optimize_process_no_optimization_needed(self, optimizer):
        """测试不需要优化的流程"""
        process_data = {"process_id": "test_process_002"}

        with patch.object(optimizer.decision_engine, 'analyze_and_decide') as mock_decide:
            mock_decision = Mock()
            mock_decision.should_optimize = False
            mock_decision.reason = "Process already optimized"
            mock_decide.return_value = mock_decision

            result = optimizer.optimize_process(process_data)

            assert result["success"] == True
            assert result["optimized"] == False
            assert result["reason"] == "Process already optimized"

    def test_optimize_process_execution_failure(self, optimizer):
        """测试优化执行失败的情况"""
        process_data = {"process_id": "test_process_003"}

        with patch.object(optimizer.decision_engine, 'analyze_and_decide') as mock_decide, \
             patch.object(optimizer.process_executor, 'execute_optimization') as mock_execute:

            mock_decision = Mock()
            mock_decision.should_optimize = True
            mock_decide.return_value = mock_decision

            mock_execute.side_effect = Exception("Execution failed")

            result = optimizer.optimize_process(process_data)

            assert result["success"] == False
            assert "error" in result

    def test_analyze_performance(self, optimizer):
        """测试性能分析"""
        process_data = {
            "process_id": "test_process_004",
            "metrics": {"cpu": 75, "memory": 60, "throughput": 100}
        }

        with patch.object(optimizer.performance_analyzer, 'analyze_performance') as mock_analyze:
            mock_analysis = {
                "bottlenecks": ["CPU usage high", "Memory pressure"],
                "recommendations": ["Scale up CPU", "Optimize memory usage"],
                "predicted_improvement": {"throughput": 25}
            }
            mock_analyze.return_value = mock_analysis

            result = optimizer.analyze_performance(process_data)

            assert result == mock_analysis

    def test_generate_optimization_plan(self, optimizer):
        """测试生成优化计划"""
        process_data = {"process_id": "test_process_005"}
        analysis_result = {"bottlenecks": ["CPU", "Memory"]}

        with patch.object(optimizer.decision_engine, 'create_optimization_plan') as mock_plan:
            mock_optimization_plan = {
                "steps": ["Increase CPU cores", "Optimize memory", "Tune database"],
                "estimated_improvement": 30,
                "risk_level": "medium"
            }
            mock_plan.return_value = mock_optimization_plan

            result = optimizer.generate_optimization_plan(process_data, analysis_result)

            assert result == mock_optimization_plan

    def test_execute_optimization_plan(self, optimizer):
        """测试执行优化计划"""
        optimization_plan = {
            "process_id": "test_process_006",
            "steps": ["Step 1", "Step 2", "Step 3"]
        }

        with patch.object(optimizer.process_executor, 'execute_plan') as mock_execute:
            mock_execution_result = {
                "success": True,
                "completed_steps": 3,
                "actual_improvement": {"throughput": 20}
            }
            mock_execute.return_value = mock_execution_result

            result = optimizer.execute_optimization_plan(optimization_plan)

            assert result == mock_execution_result

    def test_monitor_optimization_results(self, optimizer):
        """测试监控优化结果"""
        process_id = "test_process_007"

        with patch.object(optimizer.process_monitor, 'monitor_results') as mock_monitor:
            mock_monitoring_data = {
                "current_metrics": {"throughput": 120, "latency": 35},
                "improvement_percentage": 25.5,
                "stability_score": 0.9
            }
            mock_monitor.return_value = mock_monitoring_data

            result = optimizer.monitor_optimization_results(process_id)

            assert result == mock_monitoring_data

    def test_get_optimization_history(self, optimizer):
        """测试获取优化历史"""
        process_id = "test_process_008"

        with patch.object(optimizer.process_monitor, 'get_history') as mock_history:
            mock_history_data = [
                {"timestamp": "2024-01-01", "improvement": 15},
                {"timestamp": "2024-01-02", "improvement": 22}
            ]
            mock_history.return_value = mock_history_data

            result = optimizer.get_optimization_history(process_id)

            assert result == mock_history_data

    def test_get_optimization_statistics(self, optimizer):
        """测试获取优化统计"""
        with patch.object(optimizer.process_monitor, 'get_statistics') as mock_stats:
            mock_statistics = {
                "total_optimizations": 50,
                "successful_optimizations": 45,
                "average_improvement": 18.5,
                "top_bottlenecks": ["CPU", "Memory", "Database"]
            }
            mock_stats.return_value = mock_statistics

            result = optimizer.get_optimization_statistics()

            assert result == mock_statistics

    def test_validate_optimization_config(self, optimizer):
        """测试验证优化配置"""
        valid_config = {
            "max_iterations": 100,
            "convergence_threshold": 0.01,
            "timeout_seconds": 300
        }

        result = optimizer.validate_optimization_config(valid_config)
        assert result == True

    def test_validate_optimization_config_invalid(self, optimizer):
        """测试验证无效的优化配置"""
        invalid_config = {
            "max_iterations": -1,  # 无效值
            "timeout_seconds": 0
        }

        result = optimizer.validate_optimization_config(invalid_config)
        assert result == False

    def test_rollback_optimization(self, optimizer):
        """测试回滚优化"""
        process_id = "test_process_009"

        with patch.object(optimizer.process_executor, 'rollback_changes') as mock_rollback:
            mock_rollback.return_value = True

            result = optimizer.rollback_optimization(process_id)

            assert result == True

    def test_rollback_optimization_failure(self, optimizer):
        """测试回滚优化失败"""
        process_id = "test_process_010"

        with patch.object(optimizer.process_executor, 'rollback_changes') as mock_rollback:
            mock_rollback.return_value = False

            result = optimizer.rollback_optimization(process_id)

            assert result == False

    def test_get_supported_optimization_types(self, optimizer):
        """测试获取支持的优化类型"""
        types = optimizer.get_supported_optimization_types()

        assert isinstance(types, list)
        assert len(types) > 0
        assert "performance" in types

    def test_estimate_optimization_time(self, optimizer):
        """测试估算优化时间"""
        process_data = {"complexity": "high", "data_size": 10000}

        with patch.object(optimizer.performance_analyzer, 'estimate_time') as mock_estimate:
            mock_estimate.return_value = 45.5  # 45.5 seconds

            result = optimizer.estimate_optimization_time(process_data)

            assert result == 45.5

    def test_get_optimization_recommendations(self, optimizer):
        """测试获取优化建议"""
        process_id = "test_process_011"

        with patch.object(optimizer.recommendation_generator, 'get_recommendations') as mock_recommend:
            mock_recommendations = [
                {"type": "scaling", "description": "Increase CPU cores", "priority": "high"},
                {"type": "caching", "description": "Add Redis cache", "priority": "medium"}
            ]
            mock_recommend.return_value = mock_recommendations

            result = optimizer.get_optimization_recommendations(process_id)

            assert result == mock_recommendations

    def test_export_optimization_report(self, optimizer):
        """测试导出优化报告"""
        process_id = "test_process_012"
        report_format = "pdf"

        with patch.object(optimizer.process_monitor, 'export_report') as mock_export:
            mock_export.return_value = "/path/to/report.pdf"

            result = optimizer.export_optimization_report(process_id, report_format)

            assert result == "/path/to/report.pdf"

    def test_get_component_status(self, optimizer):
        """测试获取组件状态"""
        status = optimizer.get_component_status()

        assert isinstance(status, dict)
        assert "performance_analyzer" in status
        assert "decision_engine" in status
        assert "process_executor" in status
        assert "recommendation_generator" in status
        assert "process_monitor" in status

    def test_health_check(self, optimizer):
        """测试健康检查"""
        health = optimizer.health_check()

        assert isinstance(health, dict)
        assert "status" in health
        assert "components" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

    def test_shutdown(self, optimizer):
        """测试关闭"""
        result = optimizer.shutdown()

        assert result == True

    def test_get_version_info(self, optimizer):
        """测试获取版本信息"""
        version_info = optimizer.get_version_info()

        assert isinstance(version_info, dict)
        assert "version" in version_info
        assert "build_date" in version_info
        assert version_info["version"] == "2.0"


class TestBusinessProcessOptimizerIntegration:
    """测试业务流程优化器集成场景"""

    @pytest.fixture
    def optimizer(self):
        """创建完整的优化器"""
        return BusinessProcessOptimizer()

    def test_complete_optimization_workflow(self, optimizer):
        """测试完整的优化工作流程"""
        # 1. 分析性能
        process_data = {
            "process_id": "workflow_test_001",
            "metrics": {"throughput": 80, "latency": 60, "error_rate": 0.02}
        }

        with patch.object(optimizer.performance_analyzer, 'analyze_performance') as mock_analyze:
            mock_analyze.return_value = {
                "bottlenecks": ["High latency", "Low throughput"],
                "severity": "high",
                "estimated_improvement": {"latency": -20, "throughput": 30}
            }

            analysis_result = optimizer.analyze_performance(process_data)

            assert analysis_result["severity"] == "high"
            assert "bottlenecks" in analysis_result

        # 2. 生成优化计划
        with patch.object(optimizer.decision_engine, 'create_optimization_plan') as mock_plan:
            mock_plan.return_value = {
                "steps": ["Optimize database queries", "Add caching layer", "Scale resources"],
                "estimated_time": 30,
                "risk_assessment": "medium"
            }

            plan = optimizer.generate_optimization_plan(process_data, analysis_result)

            assert len(plan["steps"]) == 3
            assert plan["risk_assessment"] == "medium"

        # 3. 执行优化
        with patch.object(optimizer.process_executor, 'execute_plan') as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "steps_completed": 3,
                "actual_improvements": {"latency": -18, "throughput": 25}
            }

            execution_result = optimizer.execute_optimization_plan(plan)

            assert execution_result["success"] == True
            assert execution_result["steps_completed"] == 3

        # 4. 监控结果
        with patch.object(optimizer.process_monitor, 'monitor_results') as mock_monitor:
            mock_monitor.return_value = {
                "current_performance": {"latency": 42, "throughput": 105},
                "improvement_percentage": 22.5,
                "stability_score": 0.95
            }

            monitoring_result = optimizer.monitor_optimization_results("workflow_test_001")

            assert monitoring_result["improvement_percentage"] == 22.5
            assert monitoring_result["stability_score"] == 0.95

    def test_error_handling_and_recovery(self, optimizer):
        """测试错误处理和恢复"""
        process_data = {"process_id": "error_test_001"}

        # 测试分析失败的情况
        with patch.object(optimizer.performance_analyzer, 'analyze_performance') as mock_analyze:
            mock_analyze.side_effect = Exception("Analysis failed")

            result = optimizer.analyze_performance(process_data)

            assert result["success"] == False
            assert "error" in result

        # 测试执行失败的情况
        optimization_plan = {"process_id": "error_test_001", "steps": ["step1"]}

        with patch.object(optimizer.process_executor, 'execute_plan') as mock_execute:
            mock_execute.side_effect = Exception("Execution failed")

            result = optimizer.execute_optimization_plan(optimization_plan)

            assert result["success"] == False
            assert "error" in result

        # 验证优化器仍然可用
        status = optimizer.get_component_status()
        assert isinstance(status, dict)

    def test_concurrent_optimization_requests(self, optimizer):
        """测试并发优化请求"""
        import threading
        import time

        results = []
        errors = []

        def run_optimization(request_id):
            try:
                process_data = {"process_id": f"concurrent_test_{request_id}"}

                with patch.object(optimizer.decision_engine, 'analyze_and_decide') as mock_decide:
                    mock_decision = Mock()
                    mock_decision.should_optimize = True
                    mock_decide.return_value = mock_decision

                    with patch.object(optimizer.process_executor, 'execute_optimization') as mock_execute:
                        mock_result = Mock()
                        mock_result.success = True
                        mock_execute.return_value = mock_result

                        result = optimizer.optimize_process(process_data)
                        results.append(f"request_{request_id}_success")

            except Exception as e:
                errors.append(f"request_{request_id}_error: {str(e)}")

        # 创建多个并发请求
        threads = []
        for i in range(5):
            thread = threading.Thread(target=run_optimization, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误发生
        assert len(errors) == 0

        # 验证所有请求都成功处理
        assert len(results) == 5
        assert all("success" in r for r in results)

    def test_optimization_configuration_management(self, optimizer):
        """测试优化配置管理"""
        # 测试默认配置
        config = optimizer.get_optimization_config()
        assert isinstance(config, dict)
        assert "max_iterations" in config
        assert "timeout_seconds" in config

        # 测试配置更新
        new_config = {
            "max_iterations": 200,
            "timeout_seconds": 600,
            "convergence_threshold": 0.005
        }

        optimizer.update_optimization_config(new_config)

        updated_config = optimizer.get_optimization_config()
        assert updated_config["max_iterations"] == 200
        assert updated_config["timeout_seconds"] == 600

    def test_optimization_metrics_and_analytics(self, optimizer):
        """测试优化指标和分析"""
        # 获取统计信息
        with patch.object(optimizer.process_monitor, 'get_statistics') as mock_stats:
            mock_stats.return_value = {
                "total_processes": 100,
                "optimized_processes": 85,
                "average_improvement": 24.5,
                "success_rate": 0.85
            }

            stats = optimizer.get_optimization_statistics()

            assert stats["total_processes"] == 100
            assert stats["success_rate"] == 0.85

        # 获取性能指标
        with patch.object(optimizer.performance_analyzer, 'get_performance_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "analysis_time_avg": 2.5,
                "optimization_time_avg": 15.8,
                "success_rate": 0.92
            }

            metrics = optimizer.get_performance_metrics()

            assert metrics["analysis_time_avg"] == 2.5
            assert metrics["success_rate"] == 0.92

    def test_optimization_learning_and_adaptation(self, optimizer):
        """测试优化学习和适应"""
        # 模拟学习过程
        historical_data = [
            {"process_type": "data_processing", "improvement": 15, "technique": "parallelization"},
            {"process_type": "data_processing", "improvement": 22, "technique": "caching"},
            {"process_type": "web_service", "improvement": 18, "technique": "load_balancing"}
        ]

        with patch.object(optimizer.decision_engine, 'learn_from_history') as mock_learn:
            mock_learn.return_value = {
                "learned_patterns": ["parallelization effective for data_processing"],
                "recommended_techniques": {"data_processing": "parallelization"}
            }

            learning_result = optimizer.learn_from_historical_data(historical_data)

            assert "learned_patterns" in learning_result
            assert "recommended_techniques" in learning_result

        # 测试基于学习的优化建议
        new_process_data = {"process_type": "data_processing", "current_metrics": {"throughput": 50}}

        with patch.object(optimizer.recommendation_generator, 'generate_smart_recommendations') as mock_smart:
            mock_smart.return_value = ["Use parallelization", "Add caching layer"]

            recommendations = optimizer.get_smart_recommendations(new_process_data)

            assert len(recommendations) == 2
            assert "parallelization" in recommendations[0].lower()

    def test_resource_management_and_limits(self, optimizer):
        """测试资源管理和限制"""
        # 测试资源使用情况
        resource_usage = optimizer.get_resource_usage()

        assert isinstance(resource_usage, dict)
        assert "cpu_usage" in resource_usage
        assert "memory_usage" in resource_usage

        # 测试容量限制
        with patch.object(optimizer, 'get_active_optimizations_count') as mock_count:
            mock_count.return_value = 10  # 达到限制

            can_accept = optimizer.can_accept_new_optimization()

            # 应该拒绝新的优化请求
            assert can_accept == False

        # 测试队列管理
        queued_result = optimizer.queue_optimization_request({"process_id": "queued_process"})
        assert "queued" in queued_result or "accepted" in queued_result

    def test_optimization_scheduling_and_prioritization(self, optimizer):
        """测试优化调度和优先级"""
        # 创建不同优先级的优化请求
        high_priority_request = {
            "process_id": "critical_process",
            "priority": "high",
            "business_impact": "revenue_critical"
        }

        normal_priority_request = {
            "process_id": "normal_process",
            "priority": "normal",
            "business_impact": "standard"
        }

        # 添加到队列
        optimizer.queue_optimization_request(high_priority_request)
        optimizer.queue_optimization_request(normal_priority_request)

        # 检查调度顺序
        next_request = optimizer.get_next_scheduled_request()

        # 高优先级请求应该先被处理
        assert next_request["priority"] == "high"

    def test_optimization_rollback_and_undo(self, optimizer):
        """测试优化回滚和撤销"""
        process_id = "rollback_test"

        # 执行优化
        with patch.object(optimizer.process_executor, 'execute_optimization') as mock_execute:
            mock_result = Mock()
            mock_result.success = True
            mock_result.backup_data = {"original_config": {"threads": 4}}
            mock_execute.return_value = mock_result

            optimizer.optimize_process({"process_id": process_id})

        # 测试回滚
        rollback_result = optimizer.rollback_optimization(process_id)

        assert rollback_result == True

        # 验证回滚后的状态
        with patch.object(optimizer.process_monitor, 'get_current_state') as mock_state:
            mock_state.return_value = {"config": {"threads": 4}}  # 应该恢复到原始配置

            current_state = optimizer.get_process_state(process_id)
            assert current_state["config"]["threads"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
