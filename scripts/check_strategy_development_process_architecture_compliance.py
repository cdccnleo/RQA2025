#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
量化策略开发流程架构符合性检查脚本
检查8个环节是否符合业务流程驱动架构设计和各层架构设计要求
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArchitectureComplianceChecker:
    """架构符合性检查器"""
    
    def __init__(self):
        self.results = {
            "check_time": datetime.now().isoformat(),
            "processes": {},
            "summary": {
                "total_checks": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
        }
        
    def check_strategy_management(self) -> Dict[str, Any]:
        """检查策略管理环节"""
        logger.info("检查策略管理环节...")
        result = {
            "process_name": "策略管理",
            "business_process_step": "策略构思",
            "checks": []
        }
        
        # 1.1 业务流程符合性检查
        checks = [
            self._check_file_exists("src/strategy/core/strategy_service.py", "策略服务核心"),
            self._check_file_exists("src/strategy/lifecycle/strategy_lifecycle_manager.py", "生命周期管理"),
            self._check_file_exists("src/gateway/web/strategy_routes.py", "API路由"),
            self._check_uses_adapter("src/strategy/core/strategy_service.py", "统一适配器"),
            self._check_has_lifecycle_management("src/strategy/lifecycle/strategy_lifecycle_manager.py"),
        ]
        result["checks"].extend(checks)
        
        # 1.2 架构层符合性检查
        checks = [
            self._check_uses_service_container("src/strategy/core/strategy_service.py"),
            self._check_uses_event_bus("src/strategy/core/strategy_service.py"),
            self._check_uses_orchestrator("src/strategy/core/strategy_service.py"),
            self._check_restful_api("src/gateway/web/strategy_routes.py"),
        ]
        result["checks"].extend(checks)
        
        self._update_summary(result["checks"])
        return result
    
    def check_data_collection(self) -> Dict[str, Any]:
        """检查数据采集环节"""
        logger.info("检查数据采集环节...")
        result = {
            "process_name": "数据采集",
            "business_process_step": "数据收集",
            "checks": []
        }
        
        # 2.1 业务流程符合性检查
        checks = [
            self._check_file_exists("src/core/orchestration/business_process/data_collection_orchestrator.py", "数据采集编排器"),
            self._check_file_exists("src/gateway/web/data_collectors.py", "数据采集器"),
            self._check_uses_adapter("src/core/orchestration/business_process/data_collection_orchestrator.py", "数据适配器"),
        ]
        result["checks"].extend(checks)
        
        # 2.2 架构层符合性检查
        checks = [
            self._check_uses_event_bus("src/core/orchestration/business_process/data_collection_orchestrator.py"),
            self._check_uses_orchestrator("src/core/orchestration/business_process/data_collection_orchestrator.py"),
        ]
        result["checks"].extend(checks)
        
        self._update_summary(result["checks"])
        return result
    
    def check_feature_analysis(self) -> Dict[str, Any]:
        """检查特征分析环节"""
        logger.info("检查特征分析环节...")
        result = {
            "process_name": "特征分析",
            "business_process_step": "特征工程",
            "checks": []
        }
        
        # 3.1 业务流程符合性检查
        checks = [
            self._check_file_exists("src/gateway/web/feature_engineering_service.py", "特征工程服务"),
            self._check_directory_exists("src/features", "特征处理模块"),
        ]
        result["checks"].extend(checks)
        
        # 3.2 架构层符合性检查
        checks = [
            self._check_uses_adapter("src/gateway/web/feature_engineering_service.py", "特征适配器"),
        ]
        result["checks"].extend(checks)
        
        self._update_summary(result["checks"])
        return result
    
    def check_model_training(self) -> Dict[str, Any]:
        """检查模型训练环节"""
        logger.info("检查模型训练环节...")
        result = {
            "process_name": "模型训练",
            "business_process_step": "模型训练",
            "checks": []
        }
        
        # 4.1 业务流程符合性检查
        checks = [
            self._check_file_exists("src/ml/core/ml_core.py", "ML核心服务"),
            self._check_directory_exists("src/ml/training", "训练模块"),
        ]
        result["checks"].extend(checks)
        
        # 4.2 架构层符合性检查
        checks = [
            self._check_uses_orchestrator("src/ml/core/ml_core.py"),
            self._check_uses_adapter("src/ml/core/ml_core.py", "ML适配器"),
        ]
        result["checks"].extend(checks)
        
        self._update_summary(result["checks"])
        return result
    
    def check_strategy_backtest(self) -> Dict[str, Any]:
        """检查策略回测环节"""
        logger.info("检查策略回测环节...")
        result = {
            "process_name": "策略回测",
            "business_process_step": "策略回测",
            "checks": []
        }
        
        # 5.1 业务流程符合性检查
        checks = [
            self._check_file_exists("src/strategy/backtest/backtest_service.py", "回测服务"),
            self._check_file_exists("src/gateway/web/backtest_service.py", "回测API服务"),
        ]
        result["checks"].extend(checks)
        
        # 5.2 架构层符合性检查
        checks = [
            self._check_uses_adapter("src/strategy/backtest/backtest_service.py", "数据适配器"),
        ]
        result["checks"].extend(checks)
        
        self._update_summary(result["checks"])
        return result
    
    def check_strategy_optimization(self) -> Dict[str, Any]:
        """检查策略优化环节"""
        logger.info("检查策略优化环节...")
        result = {
            "process_name": "策略优化",
            "business_process_step": "性能评估",
            "checks": []
        }
        
        # 6.1 业务流程符合性检查
        checks = [
            self._check_file_exists("src/strategy/core/performance_optimizer.py", "性能优化器"),
            self._check_file_exists("src/gateway/web/strategy_optimization_service.py", "优化服务"),
        ]
        result["checks"].extend(checks)
        
        # 6.2 架构层符合性检查
        checks = [
            self._check_uses_adapter("src/strategy/core/performance_optimizer.py", "优化适配器"),
        ]
        result["checks"].extend(checks)
        
        self._update_summary(result["checks"])
        return result
    
    def check_strategy_deployment(self) -> Dict[str, Any]:
        """检查策略部署环节"""
        logger.info("检查策略部署环节...")
        result = {
            "process_name": "策略部署",
            "business_process_step": "策略部署",
            "checks": []
        }
        
        # 7.1 业务流程符合性检查
        checks = [
            self._check_file_exists("src/strategy/lifecycle/strategy_lifecycle_manager.py", "生命周期管理"),
            self._check_file_exists("src/strategy/cloud_native/kubernetes_deployment.py", "K8s部署"),
        ]
        result["checks"].extend(checks)
        
        # 7.2 架构层符合性检查
        checks = [
            self._check_has_deployment_mechanism("src/strategy/lifecycle/strategy_lifecycle_manager.py"),
        ]
        result["checks"].extend(checks)
        
        self._update_summary(result["checks"])
        return result
    
    def check_execution_monitoring(self) -> Dict[str, Any]:
        """检查执行监控环节"""
        logger.info("检查执行监控环节...")
        result = {
            "process_name": "执行监控",
            "business_process_step": "监控优化",
            "checks": []
        }
        
        # 8.1 业务流程符合性检查
        checks = [
            self._check_file_exists("src/gateway/web/strategy_execution_service.py", "执行监控服务"),
            self._check_directory_exists("src/monitoring", "监控模块"),
        ]
        result["checks"].extend(checks)
        
        # 8.2 架构层符合性检查
        checks = [
            self._check_uses_event_bus("src/gateway/web/strategy_execution_service.py"),
        ]
        result["checks"].extend(checks)
        
        self._update_summary(result["checks"])
        return result
    
    def _check_file_exists(self, filepath: str, description: str) -> Dict[str, Any]:
        """检查文件是否存在"""
        full_path = project_root / filepath
        exists = full_path.exists()
        return {
            "check": f"文件存在性: {description}",
            "file": filepath,
            "status": "passed" if exists else "failed",
            "message": f"文件存在" if exists else f"文件不存在: {filepath}"
        }
    
    def _check_directory_exists(self, dirpath: str, description: str) -> Dict[str, Any]:
        """检查目录是否存在"""
        full_path = project_root / dirpath
        exists = full_path.exists() and full_path.is_dir()
        return {
            "check": f"目录存在性: {description}",
            "file": dirpath,
            "status": "passed" if exists else "failed",
            "message": f"目录存在" if exists else f"目录不存在: {dirpath}"
        }
    
    def _check_uses_adapter(self, filepath: str, adapter_type: str) -> Dict[str, Any]:
        """检查是否使用适配器"""
        full_path = project_root / filepath
        if not full_path.exists():
            return {
                "check": f"使用适配器: {adapter_type}",
                "file": filepath,
                "status": "failed",
                "message": f"文件不存在"
            }
        
        try:
            content = full_path.read_text(encoding='utf-8')
            uses_adapter = (
                "get_unified_adapter_factory" in content or
                "adapter_factory" in content or
                "get_adapter" in content or
                "UnifiedBusinessAdapter" in content
            )
            return {
                "check": f"使用适配器: {adapter_type}",
                "file": filepath,
                "status": "passed" if uses_adapter else "warning",
                "message": f"使用适配器" if uses_adapter else f"未使用适配器模式"
            }
        except Exception as e:
            return {
                "check": f"使用适配器: {adapter_type}",
                "file": filepath,
                "status": "failed",
                "message": f"读取文件失败: {e}"
            }
    
    def _check_uses_service_container(self, filepath: str) -> Dict[str, Any]:
        """检查是否使用服务容器"""
        full_path = project_root / filepath
        if not full_path.exists():
            return {
                "check": "使用ServiceContainer",
                "file": filepath,
                "status": "failed",
                "message": "文件不存在"
            }
        
        try:
            content = full_path.read_text(encoding='utf-8')
            uses_container = (
                "ServiceContainer" in content or
                "DependencyContainer" in content or
                "container" in content.lower()
            )
            return {
                "check": "使用ServiceContainer",
                "file": filepath,
                "status": "passed" if uses_container else "warning",
                "message": "使用服务容器" if uses_container else "未使用服务容器"
            }
        except Exception as e:
            return {
                "check": "使用ServiceContainer",
                "file": filepath,
                "status": "failed",
                "message": f"读取文件失败: {e}"
            }
    
    def _check_uses_event_bus(self, filepath: str) -> Dict[str, Any]:
        """检查是否使用事件总线"""
        full_path = project_root / filepath
        if not full_path.exists():
            return {
                "check": "使用EventBus",
                "file": filepath,
                "status": "failed",
                "message": "文件不存在"
            }
        
        try:
            content = full_path.read_text(encoding='utf-8')
            uses_event_bus = (
                "EventBus" in content or
                "event_bus" in content or
                "publish" in content or
                "subscribe" in content
            )
            return {
                "check": "使用EventBus",
                "file": filepath,
                "status": "passed" if uses_event_bus else "warning",
                "message": "使用事件总线" if uses_event_bus else "未使用事件总线"
            }
        except Exception as e:
            return {
                "check": "使用EventBus",
                "file": filepath,
                "status": "failed",
                "message": f"读取文件失败: {e}"
            }
    
    def _check_uses_orchestrator(self, filepath: str) -> Dict[str, Any]:
        """检查是否使用业务流程编排器"""
        full_path = project_root / filepath
        if not full_path.exists():
            return {
                "check": "使用BusinessProcessOrchestrator",
                "file": filepath,
                "status": "failed",
                "message": "文件不存在"
            }
        
        try:
            content = full_path.read_text(encoding='utf-8')
            uses_orchestrator = (
                "BusinessProcessOrchestrator" in content or
                "orchestrator" in content.lower() or
                "orchestration" in content.lower()
            )
            return {
                "check": "使用BusinessProcessOrchestrator",
                "file": filepath,
                "status": "passed" if uses_orchestrator else "warning",
                "message": "使用业务流程编排器" if uses_orchestrator else "未使用业务流程编排器"
            }
        except Exception as e:
            return {
                "check": "使用BusinessProcessOrchestrator",
                "file": filepath,
                "status": "failed",
                "message": f"读取文件失败: {e}"
            }
    
    def _check_restful_api(self, filepath: str) -> Dict[str, Any]:
        """检查RESTful API设计"""
        full_path = project_root / filepath
        if not full_path.exists():
            return {
                "check": "RESTful API设计",
                "file": filepath,
                "status": "failed",
                "message": "文件不存在"
            }
        
        try:
            content = full_path.read_text(encoding='utf-8')
            has_restful = (
                "@router.get" in content or
                "@router.post" in content or
                "@router.put" in content or
                "@router.delete" in content or
                "APIRouter" in content
            )
            return {
                "check": "RESTful API设计",
                "file": filepath,
                "status": "passed" if has_restful else "warning",
                "message": "符合RESTful设计" if has_restful else "未完全符合RESTful设计"
            }
        except Exception as e:
            return {
                "check": "RESTful API设计",
                "file": filepath,
                "status": "failed",
                "message": f"读取文件失败: {e}"
            }
    
    def _check_has_lifecycle_management(self, filepath: str) -> Dict[str, Any]:
        """检查是否有生命周期管理"""
        full_path = project_root / filepath
        if not full_path.exists():
            return {
                "check": "生命周期管理",
                "file": filepath,
                "status": "failed",
                "message": "文件不存在"
            }
        
        try:
            content = full_path.read_text(encoding='utf-8')
            has_lifecycle = (
                "lifecycle" in content.lower() or
                "LifecycleStage" in content or
                "LifecycleEvent" in content
            )
            return {
                "check": "生命周期管理",
                "file": filepath,
                "status": "passed" if has_lifecycle else "warning",
                "message": "有生命周期管理" if has_lifecycle else "缺少生命周期管理"
            }
        except Exception as e:
            return {
                "check": "生命周期管理",
                "file": filepath,
                "status": "failed",
                "message": f"读取文件失败: {e}"
            }
    
    def _check_has_deployment_mechanism(self, filepath: str) -> Dict[str, Any]:
        """检查是否有部署机制"""
        full_path = project_root / filepath
        if not full_path.exists():
            return {
                "check": "部署机制",
                "file": filepath,
                "status": "failed",
                "message": "文件不存在"
            }
        
        try:
            content = full_path.read_text(encoding='utf-8')
            has_deployment = (
                "deploy" in content.lower() or
                "deployment" in content.lower() or
                "DEPLOYING" in content
            )
            return {
                "check": "部署机制",
                "file": filepath,
                "status": "passed" if has_deployment else "warning",
                "message": "有部署机制" if has_deployment else "缺少部署机制"
            }
        except Exception as e:
            return {
                "check": "部署机制",
                "file": filepath,
                "status": "failed",
                "message": f"读取文件失败: {e}"
            }
    
    def _update_summary(self, checks: List[Dict[str, Any]]):
        """更新摘要统计"""
        self.results["summary"]["total_checks"] += len(checks)
        for check in checks:
            if check["status"] == "passed":
                self.results["summary"]["passed"] += 1
            elif check["status"] == "failed":
                self.results["summary"]["failed"] += 1
            elif check["status"] == "warning":
                self.results["summary"]["warnings"] += 1
    
    def run_all_checks(self):
        """运行所有检查"""
        logger.info("开始架构符合性检查...")
        
        self.results["processes"]["strategy_management"] = self.check_strategy_management()
        self.results["processes"]["data_collection"] = self.check_data_collection()
        self.results["processes"]["feature_analysis"] = self.check_feature_analysis()
        self.results["processes"]["model_training"] = self.check_model_training()
        self.results["processes"]["strategy_backtest"] = self.check_strategy_backtest()
        self.results["processes"]["strategy_optimization"] = self.check_strategy_optimization()
        self.results["processes"]["strategy_deployment"] = self.check_strategy_deployment()
        self.results["processes"]["execution_monitoring"] = self.check_execution_monitoring()
        
        logger.info("架构符合性检查完成")
        return self.results
    
    def save_results(self, output_file: str = "docs/strategy_development_process_architecture_compliance_check_results.json"):
        """保存检查结果"""
        output_path = project_root / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"检查结果已保存到: {output_path}")


def main():
    """主函数"""
    checker = ArchitectureComplianceChecker()
    results = checker.run_all_checks()
    checker.save_results()
    
    # 打印摘要
    summary = results["summary"]
    print("\n" + "="*60)
    print("架构符合性检查摘要")
    print("="*60)
    print(f"总检查项: {summary['total_checks']}")
    print(f"通过: {summary['passed']} ({summary['passed']/summary['total_checks']*100:.1f}%)")
    print(f"失败: {summary['failed']}")
    print(f"警告: {summary['warnings']}")
    print("="*60)
    
    return 0 if summary['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

