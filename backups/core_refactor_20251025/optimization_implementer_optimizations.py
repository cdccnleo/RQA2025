#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
优化实现器
基于核心层优化完成报告的优化建议实现
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ..base import BaseComponent
from .short_term_optimizations import (
    UserFeedbackCollector,
    FeedbackAnalyzer,
    PerformanceMonitor,
    DocumentationEnhancer,
    TestingEnhancer,
    MemoryOptimizer,
)
from .medium_term_optimizations import (
    DistributedSupport,
    MultiLevelCache,
    MonitoringEnhancer,
    PerformanceTuner,
)
from .long_term_optimizations import (
    MicroserviceMigration,
    CloudNativeSupport,
    AIIntegration,
    EcosystemBuilding,
)

logger = logging.getLogger(__name__)


class OptimizationPhase(Enum):
    """优化阶段"""

    SHORT_TERM = "short_term"  # 1 - 2周
    MEDIUM_TERM = "medium_term"  # 1 - 2个月
    LONG_TERM = "long_term"  # 3 - 6个月


class OptimizationType(Enum):
    """优化类型"""

    PERFORMANCE = "performance"
    MEMORY = "memory"
    ARCHITECTURE = "architecture"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    MONITORING = "monitoring"


@dataclass
class OptimizationTask:
    """优化任务"""

    task_id: str
    name: str
    description: str
    phase: OptimizationPhase
    type: OptimizationType
    priority: int  # 1 - 5, 5为最高优先级
    estimated_duration: int  # 小时
    dependencies: List[str] = None
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0  # 0.0 - 1.0
    results: Dict[str, Any] = None
    created_at: float = None
    started_at: float = None
    completed_at: float = None


class OptimizationImplementer(BaseComponent):
    """优化实现器"""

    def __init__(self):

        super().__init__("OptimizationImplementer")
        self.tasks: Dict[str, OptimizationTask] = {}
        self.optimization_history: List[Dict[str, Any]] = []

        # 初始化短期优化组件
        self.feedback_collector = UserFeedbackCollector()
        self.feedback_analyzer = FeedbackAnalyzer()
        self.performance_monitor = PerformanceMonitor()
        self.documentation_enhancer = DocumentationEnhancer()
        self.testing_enhancer = TestingEnhancer()
        self.memory_optimizer = MemoryOptimizer()

        # 初始化中期优化组件
        self.distributed_support = DistributedSupport()
        self.multi_level_cache = MultiLevelCache()
        self.monitoring_enhancer = MonitoringEnhancer()
        self.performance_tuner = PerformanceTuner()

        # 初始化长期优化组件
        self.microservice_migration = MicroserviceMigration()
        self.cloud_native_support = CloudNativeSupport()
        self.ai_integration = AIIntegration()
        self.ecosystem_building = EcosystemBuilding()

        # 添加优化任务
        self._add_short_term_tasks()
        self._add_medium_term_tasks()
        self._add_long_term_tasks()

        logger.info("优化实现器初始化完成")

    def shutdown(self) -> bool:
        """关闭优化实现器

        优雅关闭所有组件和服务，确保资源正确释放。
        """
        try:
            logger.info("开始关闭优化实现器")

            # 按优先级顺序关闭组件
            self._shutdown_performance_monitoring()
            self._shutdown_short_term_components()
            self._shutdown_medium_term_components()
            self._shutdown_long_term_components()
            self._cleanup_resources()

            logger.info("优化实现器关闭完成")
            return True
        except Exception as e:
            logger.error(f"关闭优化实现器失败: {e}")
            return False

    def _shutdown_performance_monitoring(self):
        """关闭性能监控组件"""
        if hasattr(self, "performance_monitor") and self.performance_monitor:
            self.performance_monitor.stop_monitoring()

    def _shutdown_short_term_components(self):
        """关闭短期优化组件"""
        short_term_components = [
            "feedback_collector", "feedback_analyzer", "documentation_enhancer",
            "testing_enhancer", "memory_optimizer"
        ]
        self._shutdown_components(short_term_components)

    def _shutdown_medium_term_components(self):
        """关闭中期优化组件"""
        medium_term_components = [
            "distributed_support", "multi_level_cache", "monitoring_enhancer",
            "performance_tuner"
        ]
        self._shutdown_components(medium_term_components)

    def _shutdown_long_term_components(self):
        """关闭长期优化组件"""
        long_term_components = [
            "microservice_migration", "cloud_native_support", "ai_integration",
            "ecosystem_building"
        ]
        self._shutdown_components(long_term_components)

    def _shutdown_components(self, component_names: list):
        """关闭指定的组件列表"""
        for component_name in component_names:
            if hasattr(self, component_name):
                component = getattr(self, component_name)
                if component:
                    try:
                        component.shutdown()
                    except Exception as e:
                        logger.warning(f"关闭组件 {component_name} 失败: {e}")

    def _cleanup_resources(self):
        """清理资源"""
        try:
            self.tasks.clear()
            self.optimization_history.clear()
        except Exception as e:
            logger.warning(f"清理资源失败: {e}")

    def _add_short_term_tasks(self):
        """添加短期优化任务"""
        short_term_tasks = [
            OptimizationTask(
                task_id="ST001",
                name="收集用户反馈",
                description="收集开发团队对优化成果的反馈",
                phase=OptimizationPhase.SHORT_TERM,
                type=OptimizationType.MONITORING,
                priority=5,
                estimated_duration=8,
                created_at=time.time(),
            ),
            OptimizationTask(
                task_id="ST002",
                name="建立性能监控体系",
                description="建立实时性能监控体系",
                phase=OptimizationPhase.SHORT_TERM,
                type=OptimizationType.MONITORING,
                priority=4,
                estimated_duration=16,
                created_at=time.time(),
            ),
            OptimizationTask(
                task_id="ST003",
                name="完善文档",
                description="补充使用示例和最佳实践",
                phase=OptimizationPhase.SHORT_TERM,
                type=OptimizationType.DOCUMENTATION,
                priority=3,
                estimated_duration=12,
                created_at=time.time(),
            ),
            OptimizationTask(
                task_id="ST004",
                name="增强测试",
                description="增加更多边界条件测试",
                phase=OptimizationPhase.SHORT_TERM,
                type=OptimizationType.TESTING,
                priority=4,
                estimated_duration=20,
                created_at=time.time(),
            ),
            OptimizationTask(
                task_id="ST005",
                name="内存优化",
                description="优化内存使用和垃圾回收",
                phase=OptimizationPhase.SHORT_TERM,
                type=OptimizationType.MEMORY,
                priority=3,
                estimated_duration=16,
                created_at=time.time(),
            ),
        ]

        for task in short_term_tasks:
            self.tasks[task.task_id] = task

    def _add_medium_term_tasks(self):
        """添加中期优化任务"""
        medium_term_tasks = [
            OptimizationTask(
                task_id="MT001",
                name="分布式支持",
                description="考虑分布式架构支持",
                phase=OptimizationPhase.MEDIUM_TERM,
                type=OptimizationType.ARCHITECTURE,
                priority=4,
                estimated_duration=80,
                dependencies=["ST001", "ST002"],
                created_at=time.time(),
            ),
            OptimizationTask(
                task_id="MT002",
                name="多级缓存机制",
                description="实现多级缓存机制",
                phase=OptimizationPhase.MEDIUM_TERM,
                type=OptimizationType.PERFORMANCE,
                priority=3,
                estimated_duration=60,
                dependencies=["ST005"],
                created_at=time.time(),
            ),
            OptimizationTask(
                task_id="MT003",
                name="监控增强",
                description="增加更多监控指标",
                phase=OptimizationPhase.MEDIUM_TERM,
                type=OptimizationType.MONITORING,
                priority=3,
                estimated_duration=40,
                dependencies=["ST002"],
                created_at=time.time(),
            ),
            OptimizationTask(
                task_id="MT004",
                name="性能调优",
                description="进一步优化性能瓶颈",
                phase=OptimizationPhase.MEDIUM_TERM,
                type=OptimizationType.PERFORMANCE,
                priority=4,
                estimated_duration=100,
                dependencies=["ST002", "ST005"],
                created_at=time.time(),
            ),
        ]

        for task in medium_term_tasks:
            self.tasks[task.task_id] = task

    def _add_long_term_tasks(self):
        """添加长期优化任务"""
        long_term_tasks = [
            OptimizationTask(
                task_id="LT001",
                name="微服务化",
                description="考虑微服务化改造",
                phase=OptimizationPhase.LONG_TERM,
                type=OptimizationType.ARCHITECTURE,
                priority=3,
                estimated_duration=200,
                dependencies=["MT001"],
                created_at=time.time(),
            ),
            OptimizationTask(
                task_id="LT002",
                name="云原生支持",
                description="支持云原生部署",
                phase=OptimizationPhase.LONG_TERM,
                type=OptimizationType.ARCHITECTURE,
                priority=3,
                estimated_duration=160,
                dependencies=["MT001"],
                created_at=time.time(),
            ),
            OptimizationTask(
                task_id="LT003",
                name="AI集成",
                description="集成AI驱动的优化",
                phase=OptimizationPhase.LONG_TERM,
                type=OptimizationType.PERFORMANCE,
                priority=2,
                estimated_duration=240,
                dependencies=["MT004"],
                created_at=time.time(),
            ),
            OptimizationTask(
                task_id="LT004",
                name="生态建设",
                description="建立开发者生态",
                phase=OptimizationPhase.LONG_TERM,
                type=OptimizationType.DOCUMENTATION,
                priority=2,
                estimated_duration=180,
                dependencies=["ST003"],
                created_at=time.time(),
            ),
        ]

        for task in long_term_tasks:
            self.tasks[task.task_id] = task

    def execute_optimizations(
        self, phase: OptimizationPhase = None, timeout: int = 3600
    ) -> Dict[str, Any]:
        """执行优化任务 - 增强超时控制"""
        logger.info(f"开始执行优化任务，阶段: {phase}，超时时间: {timeout}秒")

        if phase:
            tasks_to_execute = [
                task for task in self.tasks.values() if task.phase == phase
            ]
        else:
            tasks_to_execute = list(self.tasks.values())

        # 按优先级排序
        tasks_to_execute.sort(key=lambda x: x.priority, reverse=True)

        results = {
            "phase": phase.value if phase else "all",
            "total_tasks": len(tasks_to_execute),
            "completed_tasks": 0,
            "failed_tasks": 0,
            "timeout_tasks": 0,
            "results": [],
            "execution_time": 0.0,
            "start_time": time.time(),
        }

        start_time = time.time()

        for task in tasks_to_execute:
            try:
                # 检查整体执行时间是否超过超时限制
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    logger.warning(f"整体执行时间已超过超时限制 {timeout}秒，停止执行")
                    break

                # 检查单个任务的剩余执行时间
                remaining_time = timeout - elapsed_time
                if remaining_time <= 0:
                    logger.warning(f"任务 {task.task_id} 无剩余执行时间，跳过")
                    results["timeout_tasks"] += 1
                    continue

                # 检查依赖
                if not self._check_dependencies(task):
                    logger.warning(f"任务 {task.task_id} 依赖未满足，跳过")
                    continue

                # 执行任务（带超时控制）
                task_result = self._execute_task_with_timeout(task, remaining_time)
                results["results"].append(task_result)

                if task_result["status"] == "completed":
                    results["completed_tasks"] += 1
                elif task_result["status"] == "timeout":
                    results["timeout_tasks"] += 1
                else:
                    results["failed_tasks"] += 1

            except Exception as e:
                logger.error(f"执行任务 {task.task_id} 失败: {e}")
                task.status = "failed"
                task.results = {"error": str(e)}
                results["failed_tasks"] += 1
                results["results"].append(
                    {"task_id": task.task_id, "status": "failed", "error": str(e)}
                )

        results["execution_time"] = time.time() - start_time
        self.optimization_history.append(results)
        logger.info(
            f"优化任务执行完成: {results['completed_tasks']}/{results['total_tasks']} 成功，执行时间: {results['execution_time']:.2f}秒"
        )

        return results

    def _check_dependencies(self, task: OptimizationTask) -> bool:
        """检查任务依赖"""
        if not task.dependencies:
            return True

        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                logger.warning(f"依赖任务 {dep_id} 不存在")
                return False

            dep_task = self.tasks[dep_id]
            if dep_task.status != "completed":
                logger.warning(f"依赖任务 {dep_id} 未完成")
                return False

        return True

    def _execute_task_with_timeout(
        self, task: OptimizationTask, timeout: float
    ) -> Dict[str, Any]:
        """执行任务带超时控制"""
        import concurrent.futures

        def execute_with_timeout():
            """在单独线程中执行任务"""
            try:
                return self._execute_task(task)
            except Exception as e:
                raise e

        try:
            # 使用ThreadPoolExecutor执行任务，带超时控制
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(execute_with_timeout)

                try:
                    # 等待任务完成，最多等待timeout秒
                    result = future.result(timeout=timeout)
                    return result

                except concurrent.futures.TimeoutError:
                    logger.warning(
                        f"任务 {task.task_id} 执行超时 ({timeout:.1f}秒)，取消执行"
                    )
                    future.cancel()

                    # 更新任务状态
                    task.status = "timeout"
                    task.results = {"error": f"执行超时: {timeout:.1f}秒"}

                    return {
                        "task_id": task.task_id,
                        "status": "timeout",
                        "error": f"执行超时: {timeout:.1f}秒",
                        "timeout_seconds": timeout,
                    }

        except Exception as e:
            logger.error(f"执行任务 {task.task_id} 时发生异常: {e}")
            task.status = "failed"
            task.results = {"error": str(e)}

            return {"task_id": task.task_id, "status": "failed", "error": str(e)}

    def _execute_task(self, task: OptimizationTask) -> Dict[str, Any]:
        """执行单个任务"""
        logger.info(f"开始执行任务: {task.task_id} - {task.name}")

        task.status = "running"
        task.started_at = time.time()
        task.progress = 0.0

        try:
            if task.phase == OptimizationPhase.SHORT_TERM:
                result = self._execute_short_term_task(task)
            elif task.phase == OptimizationPhase.MEDIUM_TERM:
                result = self._execute_medium_term_task(task)
            elif task.phase == OptimizationPhase.LONG_TERM:
                result = self._execute_long_term_task(task)
            else:
                raise ValueError(f"未知的优化阶段: {task.phase}")

            task.status = "completed"
            task.progress = 1.0
            task.completed_at = time.time()
            task.results = result

            logger.info(f"任务 {task.task_id} 执行完成")
            return {"task_id": task.task_id, "status": "completed", "results": result}

        except Exception as e:
            task.status = "failed"
            task.results = {"error": str(e)}
            logger.error(f"任务 {task.task_id} 执行失败: {e}")
            return {"task_id": task.task_id, "status": "failed", "error": str(e)}

    def _execute_short_term_task(self, task: OptimizationTask) -> Dict[str, Any]:
        """执行短期优化任务"""
        if task.task_id == "ST001":
            return self._execute_feedback_collection()
        elif task.task_id == "ST002":
            return self._execute_performance_monitoring()
        elif task.task_id == "ST003":
            return self._execute_documentation_enhancement()
        elif task.task_id == "ST004":
            return self._execute_testing_enhancement()
        elif task.task_id == "ST005":
            return self._execute_memory_optimization()
        else:
            raise ValueError(f"未知的短期任务: {task.task_id}")

    def _execute_medium_term_task(self, task: OptimizationTask) -> Dict[str, Any]:
        """执行中期优化任务"""
        if task.task_id == "MT001":
            return self._execute_distributed_support()
        elif task.task_id == "MT002":
            return self._execute_multi_level_cache()
        elif task.task_id == "MT003":
            return self._execute_monitoring_enhancement()
        elif task.task_id == "MT004":
            return self._execute_performance_tuning()
        else:
            raise ValueError(f"未知的中期任务: {task.task_id}")

    def _execute_long_term_task(self, task: OptimizationTask) -> Dict[str, Any]:
        """执行长期优化任务"""
        if task.task_id == "LT001":
            return self._execute_microservice_migration()
        elif task.task_id == "LT002":
            return self._execute_cloud_native_support()
        elif task.task_id == "LT003":
            return self._execute_ai_integration()
        elif task.task_id == "LT004":
            return self._execute_ecosystem_building()
        else:
            raise ValueError(f"未知的长期任务: {task.task_id}")

    def _execute_feedback_collection(self) -> Dict[str, Any]:
        """执行反馈收集"""
        feedback = self.feedback_collector.collect_feedback()
        analysis = self.feedback_analyzer.analyze_feedback(feedback)
        suggestions = self.feedback_analyzer.generate_suggestions(analysis)

        return {
            "feedback_collected": len(feedback),
            "analysis": analysis,
            "suggestions": suggestions,
        }

    def _execute_performance_monitoring(self) -> Dict[str, Any]:
        """执行性能监控"""
        self.performance_monitor.start_monitoring()
        time.sleep(5)  # 收集一些初始数据
        metrics_summary = self.performance_monitor.get_metrics_summary()

        return {"monitoring_started": True, "metrics_summary": metrics_summary}

    def _execute_documentation_enhancement(self) -> Dict[str, Any]:
        """执行文档增强"""
        examples = self.documentation_enhancer.generate_examples()
        best_practices = self.documentation_enhancer.generate_best_practices()
        updated_docs = self.documentation_enhancer.update_documentation(
            examples, best_practices
        )

        return {
            "examples_generated": len(examples),
            "best_practices_generated": len(best_practices),
            "docs_updated": len(updated_docs),
        }

    def _execute_testing_enhancement(self) -> Dict[str, Any]:
        """执行测试增强"""
        boundary_tests = self.testing_enhancer.add_boundary_tests()
        performance_tests = self.testing_enhancer.add_performance_tests()
        integration_tests = self.testing_enhancer.add_integration_tests()

        return {
            "boundary_tests_added": len(boundary_tests),
            "performance_tests_added": len(performance_tests),
            "integration_tests_added": len(integration_tests),
        }

    def _execute_memory_optimization(self) -> Dict[str, Any]:
        """执行内存优化"""
        analysis = self.memory_optimizer.analyze_memory_usage()
        allocation_optimization = self.memory_optimizer.optimize_memory_allocation()
        gc_optimization = self.memory_optimizer.optimize_garbage_collection()
        summary = self.memory_optimizer.get_memory_optimization_summary()

        return {
            "analysis": analysis,
            "allocation_optimization": allocation_optimization,
            "gc_optimization": gc_optimization,
            "summary": summary,
        }

    def _execute_distributed_support(self) -> Dict[str, Any]:
        """执行分布式支持"""
        logger.info("开始实现分布式支持")

        # 注册示例节点
        nodes_registered = []
        nodes_registered.append(
            self.distributed_support.register_node(
                "node_001", "192.168.1.100", 8080, ["event_bus", "cache"]
            )
        )
        nodes_registered.append(
            self.distributed_support.register_node(
                "node_002", "192.168.1.101", 8080, ["event_bus", "storage"]
            )
        )

        # 启动心跳检测
        self.distributed_support.start_heartbeat()

        # 服务发现
        services = self.distributed_support.discover_services()

        return {
            "status": "completed",
            "nodes_registered": sum(nodes_registered),
            "services_discovered": len(services),
            "heartbeat_started": True,
            "description": "分布式支持功能已实现",
        }

    def _execute_multi_level_cache(self) -> Dict[str, Any]:
        """执行多级缓存机制"""
        logger.info("开始实现多级缓存机制")

        # 测试缓存功能
        test_data = {"test_key": "test_value", "timestamp": time.time()}

        # 设置缓存数据
        l1_success = self.multi_level_cache.set("test_key", test_data, "L1")
        l2_success = self.multi_level_cache.set("test_key_2", test_data, "L2")

        # 获取缓存数据
        l1_data = self.multi_level_cache.get("test_key")
        l2_data = self.multi_level_cache.get("test_key_2")

        # 获取缓存统计
        cache_stats = self.multi_level_cache.get_cache_stats()

        return {
            "status": "completed",
            "l1_cache_working": l1_success and l1_data is not None,
            "l2_cache_working": l2_success and l2_data is not None,
            "cache_stats": cache_stats,
            "description": "多级缓存机制已实现",
        }

    def _execute_monitoring_enhancement(self) -> Dict[str, Any]:
        """执行监控增强"""
        logger.info("开始实现监控增强")

        # 添加示例指标
        self.monitoring_enhancer.add_metric("cpu_usage", 45.2, "system")
        self.monitoring_enhancer.add_metric("memory_usage", 62.8, "system")
        self.monitoring_enhancer.add_metric("error_rate", 2.1, "application")

        # 创建示例仪表板
        dashboard_created = self.monitoring_enhancer.create_dashboard(
            "system_overview", ["cpu_usage", "memory_usage", "error_rate"]
        )

        # 获取指标摘要和告警
        metrics_summary = self.monitoring_enhancer.get_metrics_summary()
        active_alerts = self.monitoring_enhancer.get_active_alerts()

        return {
            "status": "completed",
            "metrics_added": len(self.monitoring_enhancer.metrics),
            "dashboard_created": dashboard_created,
            "active_alerts": len(active_alerts),
            "metrics_summary": metrics_summary,
            "description": "监控增强功能已实现",
        }

    def _execute_performance_tuning(self) -> Dict[str, Any]:
        """执行性能调优"""
        logger.info("开始实现性能调优")

        # 分析性能
        performance_analysis = self.performance_tuner.analyze_performance()

        # 优化关键路径
        critical_path_optimization = self.performance_tuner.optimize_critical_path()

        # 获取性能摘要
        performance_summary = self.performance_tuner.get_performance_summary()

        return {
            "status": "completed",
            "performance_analysis": performance_analysis,
            "critical_path_optimization": critical_path_optimization,
            "performance_summary": performance_summary,
            "description": "性能调优功能已实现",
        }

    def _execute_microservice_migration(self) -> Dict[str, Any]:
        """执行微服务化迁移"""
        logger.info("开始执行微服务化迁移")

        # 分析当前架构
        current_analysis = self.microservice_migration.analyze_current_architecture()

        # 设计微服务架构
        microservices = self.microservice_migration.design_microservices()

        # 创建迁移计划
        migration_plan = self.microservice_migration.create_migration_plan()

        # 生成服务配置
        service_configs = self.microservice_migration.generate_service_configs()

        return {
            "status": "completed",
            "current_architecture": current_analysis,
            "microservices_designed": len(microservices),
            "migration_plan": migration_plan,
            "service_configs": service_configs,
            "description": "微服务化迁移功能已实现",
        }

    def _execute_cloud_native_support(self) -> Dict[str, Any]:
        """执行云原生支持"""
        logger.info("开始执行云原生支持")

        # 分析云原生需求
        requirements = self.cloud_native_support.analyze_cloud_requirements()

        # 设计云原生架构
        architecture = self.cloud_native_support.design_cloud_architecture()

        # 创建部署配置
        deployment_configs = self.cloud_native_support.create_deployment_configs()

        # 生成云资源
        cloud_resources = self.cloud_native_support.generate_cloud_resources()

        return {
            "status": "completed",
            "requirements": requirements,
            "architecture": architecture,
            "deployment_configs": deployment_configs,
            "cloud_resources": len(cloud_resources),
            "description": "云原生支持功能已实现",
        }

    def _execute_ai_integration(self) -> Dict[str, Any]:
        """执行AI集成"""
        logger.info("开始执行AI集成")

        # 分析AI需求
        requirements = self.ai_integration.analyze_ai_requirements()

        # 设计AI架构
        architecture = self.ai_integration.design_ai_architecture()

        # 创建AI模型
        ai_models = self.ai_integration.create_ai_models()

        # 设置AI流水线
        pipeline = self.ai_integration.setup_ai_pipeline()

        return {
            "status": "completed",
            "requirements": requirements,
            "architecture": architecture,
            "ai_models": len(ai_models),
            "pipeline": pipeline,
            "description": "AI集成功能已实现",
        }

    def _execute_ecosystem_building(self) -> Dict[str, Any]:
        """执行生态建设"""
        logger.info("开始执行生态建设")

        # 分析生态需求
        needs = self.ecosystem_building.analyze_ecosystem_needs()

        # 设计生态架构
        architecture = self.ecosystem_building.design_ecosystem_architecture()

        # 创建开发者资源
        developer_resources = self.ecosystem_building.create_developer_resources()

        # 设置社区平台
        community_platforms = self.ecosystem_building.setup_community_platforms()

        return {
            "status": "completed",
            "needs": needs,
            "architecture": architecture,
            "developer_resources": developer_resources,
            "community_platforms": community_platforms,
            "description": "生态建设功能已实现",
        }

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]
        return {
            "task_id": task.task_id,
            "name": task.name,
            "status": task.status,
            "progress": task.progress,
            "results": task.results,
        }

    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        total_tasks = len(self.tasks)
        completed_tasks = sum(
            1 for task in self.tasks.values() if task.status == "completed"
        )
        failed_tasks = sum(1 for task in self.tasks.values() if task.status == "failed")
        running_tasks = sum(
            1 for task in self.tasks.values() if task.status == "running"
        )

        phase_summary = {}
        for phase in OptimizationPhase:
            phase_tasks = [task for task in self.tasks.values() if task.phase == phase]
            phase_summary[phase.value] = {
                "total": len(phase_tasks),
                "completed": sum(
                    1 for task in phase_tasks if task.status == "completed"
                ),
                "failed": sum(1 for task in phase_tasks if task.status == "failed"),
                "running": sum(1 for task in phase_tasks if task.status == "running"),
            }

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "running_tasks": running_tasks,
            "completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "phase_summary": phase_summary,
            "recent_history": (
                self.optimization_history[-5:] if self.optimization_history else []
            ),
        }
