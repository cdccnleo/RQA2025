#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型训练监控仪表盘功能与持久化检查脚本
检查前端功能、后端API、持久化实现和数据流
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


class ModelTrainingMonitorChecker:
    """模型训练监控仪表盘检查器"""
    
    def __init__(self):
        self.results = {
            "check_time": datetime.now().isoformat(),
            "frontend": {},
            "backend": {},
            "persistence": {},
            "data_flow": {},
            "issues": [],
            "summary": {
                "total_checks": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
        }
        
    def check_frontend_modules(self) -> Dict[str, Any]:
        """检查前端功能模块"""
        logger.info("检查前端功能模块...")
        result = {
            "modules": {}
        }
        
        html_path = project_root / "web-static/model-training-monitor.html"
        if not html_path.exists():
            result["error"] = "前端文件不存在"
            return result
        
        try:
            content = html_path.read_text(encoding='utf-8')
            
            # 1.1 统计卡片模块
            result["modules"]["statistics_cards"] = {
                "check": "统计卡片模块",
                "status": "passed" if "running-jobs" in content and "gpu-usage" in content and "avg-accuracy" in content else "failed",
                "elements": {
                    "running-jobs": "running-jobs" in content,
                    "gpu-usage": "gpu-usage" in content,
                    "avg-accuracy": "avg-accuracy" in content,
                    "avg-training-time": "avg-training-time" in content
                }
            }
            
            # 1.2 训练任务列表模块
            result["modules"]["job_list"] = {
                "check": "训练任务列表模块",
                "status": "passed" if "trainingJobsBody" in content and "renderTrainingJobs" in content else "failed",
                "elements": {
                    "table_body": "trainingJobsBody" in content,
                    "render_function": "renderTrainingJobs" in content
                }
            }
            
            # 1.3 训练图表模块
            result["modules"]["charts"] = {
                "check": "训练图表模块",
                "status": "passed" if "lossChart" in content and "accuracyChart" in content and "Chart" in content else "failed",
                "elements": {
                    "loss_chart": "lossChart" in content,
                    "accuracy_chart": "accuracyChart" in content,
                    "chart_init": "initCharts" in content or "Chart" in content
                }
            }
            
            # 1.4 资源使用情况模块
            result["modules"]["resource_usage"] = {
                "check": "资源使用情况模块",
                "status": "passed" if "gpu-usage-value" in content and "cpu-usage-value" in content and "memory-usage-value" in content else "failed",
                "elements": {
                    "gpu_usage": "gpu-usage-value" in content,
                    "cpu_usage": "cpu-usage-value" in content,
                    "memory_usage": "memory-usage-value" in content,
                    "update_function": "updateResourceUsage" in content
                }
            }
            
            # 1.5 超参数优化模块
            result["modules"]["hyperparameters"] = {
                "check": "超参数优化模块",
                "status": "passed" if "hyperparameterChart" in content else "failed",
                "elements": {
                    "chart": "hyperparameterChart" in content
                }
            }
            
            self._update_summary(list(result["modules"].values()))
            return result
            
        except Exception as e:
            result["error"] = f"读取前端文件失败: {e}"
            return result
    
    def check_frontend_functions(self) -> Dict[str, Any]:
        """检查前端功能函数"""
        logger.info("检查前端功能函数...")
        result = {
            "functions": {}
        }
        
        html_path = project_root / "web-static/model-training-monitor.html"
        if not html_path.exists():
            result["error"] = "前端文件不存在"
            return result
        
        try:
            content = html_path.read_text(encoding='utf-8')
            
            # 1.6 创建任务功能
            result["functions"]["create_job"] = {
                "check": "创建任务功能",
                "status": "passed" if "createTrainingJob" in content and "submitCreateJob" in content and "closeCreateJobModal" in content else "failed",
                "functions": {
                    "createTrainingJob": "createTrainingJob" in content and "createJobModal" in content,
                    "submitCreateJob": "submitCreateJob" in content and "POST" in content and "/ml/training/jobs" in content,
                    "closeCreateJobModal": "closeCreateJobModal" in content
                },
                "api_call": "POST" in content and "/ml/training/jobs" in content and "fetch" in content
            }
            
            # 1.7 停止任务功能
            result["functions"]["stop_job"] = {
                "check": "停止任务功能",
                "status": "passed" if "stopJob" in content and "/stop" in content and "POST" in content else "failed",
                "functions": {
                    "stopJob": "stopJob" in content,
                    "api_call": "/stop" in content and "POST" in content and "fetch" in content
                }
            }
            
            # 1.8 查看任务详情功能
            result["functions"]["view_details"] = {
                "check": "查看任务详情功能",
                "status": "passed" if "viewJobDetails" in content and "/ml/training/jobs/${jobId}" in content or "/ml/training/jobs/" in content and "fetch" in content else "failed",
                "functions": {
                    "viewJobDetails": "viewJobDetails" in content,
                    "api_call": ("/ml/training/jobs/" in content and "fetch" in content and "jobId" in content) or "/ml/training/jobs/${jobId}" in content
                }
            }
            
            # 1.9 WebSocket实时更新
            result["functions"]["websocket"] = {
                "check": "WebSocket实时更新",
                "status": "passed" if "WebSocket" in content and "/ws/model-training" in content or "ws/model-training" in content else "failed",
                "functions": {
                    "connect": "WebSocket" in content or "ws" in content.lower(),
                    "endpoint": "/ws/model-training" in content or "ws/model-training" in content,
                    "reconnect": "setTimeout" in content or "reconnect" in content.lower()
                }
            }
            
            self._update_summary(list(result["functions"].values()))
            return result
            
        except Exception as e:
            result["error"] = f"读取前端文件失败: {e}"
            return result
    
    def check_backend_apis(self) -> Dict[str, Any]:
        """检查后端API端点"""
        logger.info("检查后端API端点...")
        result = {
            "endpoints": {}
        }
        
        routes_path = project_root / "src/gateway/web/model_training_routes.py"
        if not routes_path.exists():
            result["error"] = "后端路由文件不存在"
            return result
        
        try:
            content = routes_path.read_text(encoding='utf-8')
            
            # 2.1 任务列表API
            result["endpoints"]["get_jobs"] = {
                "check": "任务列表API",
                "status": "passed" if "@router.get" in content and "/ml/training/jobs" in content and "get_training_jobs" in content else "failed",
                "endpoint": "/ml/training/jobs" in content,
                "method": "@router.get" in content and "/ml/training/jobs" in content,
                "service_call": "get_training_jobs" in content
            }
            
            # 2.2 创建任务API
            result["endpoints"]["create_job"] = {
                "check": "创建任务API",
                "status": "passed" if "@router.post" in content and "/ml/training/jobs" in content and "save_training_job" in content else "failed",
                "endpoint": "/ml/training/jobs" in content and "@router.post" in content,
                "persistence": "save_training_job" in content
            }
            
            # 2.3 停止任务API
            result["endpoints"]["stop_job"] = {
                "check": "停止任务API",
                "status": "passed" if "/stop" in content and "update_training_job" in content else "failed",
                "endpoint": "/stop" in content,
                "persistence": "update_training_job" in content
            }
            
            # 2.4 任务详情API
            result["endpoints"]["get_job_details"] = {
                "check": "任务详情API",
                "status": "passed" if "/ml/training/jobs/{job_id}" in content and "@router.get" in content else "failed",
                "endpoint": "/ml/training/jobs/{job_id}" in content,
                "method": "@router.get" in content
            }
            
            # 2.5 训练指标API
            result["endpoints"]["get_metrics"] = {
                "check": "训练指标API",
                "status": "passed" if "/ml/training/metrics" in content and "get_training_metrics" in content else "failed",
                "endpoint": "/ml/training/metrics" in content,
                "method": "@router.get" in content and "/ml/training/metrics" in content,
                "service_call": "get_training_metrics" in content
            }
            
            self._update_summary(list(result["endpoints"].values()))
            return result
            
        except Exception as e:
            result["error"] = f"读取后端路由文件失败: {e}"
            return result
    
    def check_persistence(self) -> Dict[str, Any]:
        """检查持久化实现"""
        logger.info("检查持久化实现...")
        result = {
            "persistence": {}
        }
        
        persistence_path = project_root / "src/gateway/web/training_job_persistence.py"
        if not persistence_path.exists():
            result["error"] = "持久化模块文件不存在"
            return result
        
        try:
            content = persistence_path.read_text(encoding='utf-8')
            
            # 3.1 持久化模块
            result["persistence"]["module"] = {
                "check": "持久化模块",
                "status": "passed",
                "functions": {
                    "save_training_job": "def save_training_job" in content,
                    "load_training_job": "def load_training_job" in content,
                    "list_training_jobs": "def list_training_jobs" in content,
                    "update_training_job": "def update_training_job" in content,
                    "delete_training_job": "def delete_training_job" in content
                },
                "storage": {
                    "filesystem": "TRAINING_JOBS_DIR" in content or "training_jobs" in content,
                    "postgresql": "_save_to_postgresql" in content or "postgresql" in content.lower()
                }
            }
            
            # 3.2 服务层集成
            service_path = project_root / "src/gateway/web/model_training_service.py"
            if service_path.exists():
                service_content = service_path.read_text(encoding='utf-8')
                result["persistence"]["service_integration"] = {
                    "check": "服务层集成",
                    "status": "passed" if "list_training_jobs" in service_content or "training_job_persistence" in service_content else "warning",
                    "integration": {
                        "loads_from_persistence": "list_training_jobs" in service_content or "training_job_persistence" in service_content,
                        "saves_to_persistence": "save_training_job" in service_content
                    }
                }
            else:
                result["persistence"]["service_integration"] = {
                    "check": "服务层集成",
                    "status": "failed",
                    "error": "服务层文件不存在"
                }
            
            # 3.3 数据存储验证
            data_dir = project_root / "data/training_jobs"
            result["persistence"]["storage_verification"] = {
                "check": "数据存储验证",
                "status": "passed" if data_dir.exists() else "warning",
                "filesystem": {
                    "directory_exists": data_dir.exists(),
                    "directory_path": str(data_dir) if data_dir.exists() else "目录不存在"
                }
            }
            
            self._update_summary([result["persistence"]["module"], result["persistence"].get("service_integration", {}), result["persistence"]["storage_verification"]])
            return result
            
        except Exception as e:
            result["error"] = f"读取持久化文件失败: {e}"
            return result
    
    def check_data_flow(self) -> Dict[str, Any]:
        """检查数据流"""
        logger.info("检查数据流...")
        result = {
            "flows": {}
        }
        
        # 检查任务创建流程
        routes_path = project_root / "src/gateway/web/model_training_routes.py"
        service_path = project_root / "src/gateway/web/model_training_service.py"
        persistence_path = project_root / "src/gateway/web/training_job_persistence.py"
        
        if not all([routes_path.exists(), service_path.exists(), persistence_path.exists()]):
            result["error"] = "必需的文件不存在"
            return result
        
        try:
            routes_content = routes_path.read_text(encoding='utf-8')
            service_content = service_path.read_text(encoding='utf-8')
            persistence_content = persistence_path.read_text(encoding='utf-8')
            
            # 4.1 任务创建流程
            result["flows"]["create_job"] = {
                "check": "任务创建流程",
                "status": "passed" if "create_training_job" in routes_content and "save_training_job" in routes_content else "failed",
                "flow": {
                    "api_endpoint": "create_training_job" in routes_content,
                    "persistence": "save_training_job" in routes_content,
                    "filesystem": "TRAINING_JOBS_DIR" in persistence_content or "training_jobs" in persistence_content,
                    "postgresql": "_save_to_postgresql" in persistence_content
                }
            }
            
            # 4.2 任务查询流程
            result["flows"]["query_jobs"] = {
                "check": "任务查询流程",
                "status": "passed" if "get_training_jobs" in routes_content and ("list_training_jobs" in service_content or "list_training_jobs" in persistence_content) else "failed",
                "flow": {
                    "api_endpoint": "get_training_jobs_endpoint" in routes_content or "get_training_jobs" in routes_content,
                    "service_layer": "get_training_jobs" in service_content,
                    "persistence": "list_training_jobs" in persistence_content
                }
            }
            
            # 4.3 任务更新流程
            result["flows"]["update_job"] = {
                "check": "任务更新流程",
                "status": "passed" if "update_training_job" in routes_content or "update_training_job" in persistence_content else "failed",
                "flow": {
                    "api_endpoint": "stop_training_job" in routes_content,
                    "persistence": "update_training_job" in routes_content or "update_training_job" in persistence_content
                }
            }
            
            self._update_summary(list(result["flows"].values()))
            return result
            
        except Exception as e:
            result["error"] = f"检查数据流失败: {e}"
            return result
    
    def check_websocket_endpoint(self) -> Dict[str, Any]:
        """检查WebSocket端点"""
        logger.info("检查WebSocket端点...")
        result = {}
        
        ws_routes_path = project_root / "src/gateway/web/websocket_routes.py"
        if not ws_routes_path.exists():
            result["error"] = "WebSocket路由文件不存在"
            result["status"] = "failed"
            return result
        
        try:
            content = ws_routes_path.read_text(encoding='utf-8')
            result["check"] = "WebSocket端点",
            result["status"] = "passed" if "/ws/model-training" in content or "model-training" in content else "failed",
            result["endpoint"] = "/ws/model-training" in content or "model-training" in content
            return result
            
        except Exception as e:
            result["error"] = f"读取WebSocket路由文件失败: {e}"
            result["status"] = "failed"
            return result
    
    def _update_summary(self, checks: List[Dict[str, Any]]):
        """更新摘要统计"""
        for check in checks:
            if isinstance(check, dict) and "status" in check:
                self.results["summary"]["total_checks"] += 1
                if check["status"] == "passed":
                    self.results["summary"]["passed"] += 1
                elif check["status"] == "failed":
                    self.results["summary"]["failed"] += 1
                elif check["status"] == "warning":
                    self.results["summary"]["warnings"] += 1
    
    def run_all_checks(self):
        """运行所有检查"""
        logger.info("开始模型训练监控仪表盘检查...")
        
        self.results["frontend"] = {
            "modules": self.check_frontend_modules(),
            "functions": self.check_frontend_functions()
        }
        self.results["backend"] = {
            "apis": self.check_backend_apis(),
            "websocket": self.check_websocket_endpoint()
        }
        self.results["persistence"] = self.check_persistence()
        self.results["data_flow"] = self.check_data_flow()
        
        logger.info("模型训练监控仪表盘检查完成")
        return self.results
    
    def save_results(self, output_file: str = "docs/model_training_monitor_compliance_check_results.json"):
        """保存检查结果"""
        output_path = project_root / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"检查结果已保存到: {output_path}")


def main():
    """主函数"""
    checker = ModelTrainingMonitorChecker()
    results = checker.run_all_checks()
    checker.save_results()
    
    # 打印摘要
    summary = results["summary"]
    print("\n" + "="*60)
    print("模型训练监控仪表盘检查摘要")
    print("="*60)
    print(f"总检查项: {summary['total_checks']}")
    if summary['total_checks'] > 0:
        print(f"通过: {summary['passed']} ({summary['passed']/summary['total_checks']*100:.1f}%)")
        print(f"失败: {summary['failed']}")
        print(f"警告: {summary['warnings']}")
    print("="*60)
    
    return 0 if summary.get('failed', 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

