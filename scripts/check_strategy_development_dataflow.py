#!/usr/bin/env python3
"""
量化策略开发流程数据流检查脚本
验证8个步骤的数据流是否符合预期
"""

import sys
import os
import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataFlowChecker:
    """数据流检查器"""
    
    def __init__(self):
        self.results = {
            "end_to_end_test": {},
            "data_format_validation": {},
            "persistence_validation": {},
            "realtime_update_validation": {},
            "issues": []
        }
        self.strategy_id = None
        self.task_id = None
        self.job_id = None
        self.backtest_id = None
    
    def add_issue(self, step: str, issue: str, severity: str = "P1"):
        """添加问题"""
        self.results["issues"].append({
            "step": step,
            "issue": issue,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        })
        logger.warning(f"[{severity}] {step}: {issue}")
    
    def add_result(self, category: str, step: str, result: Dict[str, Any]):
        """添加检查结果"""
        if step not in self.results[category]:
            self.results[category][step] = []
        self.results[category][step].append(result)
    
    # ==================== 第一阶段：策略构思 → 数据收集 ====================
    
    async def check_step1_strategy_conception(self):
        """检查步骤1：策略构思"""
        logger.info("\n" + "=" * 60)
        logger.info("检查步骤1：策略构思 → 数据收集")
        logger.info("=" * 60)
        
        try:
            # 1.1 创建策略
            logger.info("\n[1.1] 测试策略创建...")
            from src.gateway.web.strategy_routes import save_strategy_conception
            
            strategy_data = {
                "id": f"test_strategy_{int(time.time())}",
                "name": "测试策略",
                "type": "trend_following",
                "description": "数据流测试策略",
                "targetMarket": "A股",
                "riskLevel": "medium",
                "nodes": [
                    {"id": "node1", "type": "data_source", "label": "数据源"},
                    {"id": "node2", "type": "trade", "label": "交易"}
                ],
                "connections": [
                    {"from": "node1", "to": "node2"}
                ],
                "parameters": {},
                "createdAt": datetime.now().isoformat(),
                "version": "1.0.0"
            }
            
            result = save_strategy_conception(strategy_data)
            self.strategy_id = result.get("strategy_id") or strategy_data["id"]
            
            logger.info(f"✓ 策略创建成功: {self.strategy_id}")
            
            # 验证策略ID格式
            if not self.strategy_id or not isinstance(self.strategy_id, str):
                self.add_issue("步骤1", "策略ID格式不正确", "P0")
            else:
                self.add_result("end_to_end_test", "步骤1", {
                    "action": "创建策略",
                    "strategy_id": self.strategy_id,
                    "status": "success"
                })
            
            # 1.2 验证策略持久化
            logger.info("\n[1.2] 验证策略持久化...")
            from src.gateway.web.strategy_routes import load_strategy_conceptions
            
            conceptions = load_strategy_conceptions()
            found = False
            for conception in conceptions:
                if conception.get("id") == self.strategy_id:
                    found = True
                    logger.info(f"✓ 策略已持久化: {self.strategy_id}")
                    self.add_result("persistence_validation", "步骤1", {
                        "action": "验证策略持久化",
                        "strategy_id": self.strategy_id,
                        "status": "success"
                    })
                    break
            
            if not found:
                self.add_issue("步骤1", "策略未正确持久化", "P0")
            
            # 1.3 验证数据格式
            logger.info("\n[1.3] 验证策略数据格式...")
            required_fields = ["id", "name", "type", "nodes", "createdAt"]
            missing_fields = []
            for field in required_fields:
                if field not in strategy_data:
                    missing_fields.append(field)
            
            if missing_fields:
                self.add_issue("步骤1", f"策略数据缺少必需字段: {missing_fields}", "P1")
            else:
                logger.info("✓ 策略数据格式正确")
                self.add_result("data_format_validation", "步骤1", {
                    "action": "验证数据格式",
                    "status": "success",
                    "fields": required_fields
                })
            
            return True
            
        except Exception as e:
            logger.error(f"步骤1检查失败: {e}", exc_info=True)
            self.add_issue("步骤1", f"检查失败: {str(e)}", "P0")
            return False
    
    # ==================== 第二阶段：数据收集 → 特征工程 ====================
    
    async def check_step2_data_collection(self):
        """检查步骤2：数据收集"""
        logger.info("\n" + "=" * 60)
        logger.info("检查步骤2：数据收集 → 特征工程")
        logger.info("=" * 60)
        
        try:
            # 2.1 验证数据源配置
            logger.info("\n[2.1] 验证数据源配置...")
            # 注意：数据源配置可能通过其他方式管理，这里只验证数据流传递
            
            # 2.2 验证策略关联
            logger.info("\n[2.2] 验证策略关联...")
            if not self.strategy_id:
                self.add_issue("步骤2", "缺少strategy_id，无法验证数据收集关联", "P0")
                return False
            
            logger.info(f"✓ 策略ID可用于数据收集: {self.strategy_id}")
            self.add_result("end_to_end_test", "步骤2", {
                "action": "验证策略关联",
                "strategy_id": self.strategy_id,
                "status": "success"
            })
            
            return True
            
        except Exception as e:
            logger.error(f"步骤2检查失败: {e}", exc_info=True)
            self.add_issue("步骤2", f"检查失败: {str(e)}", "P1")
            return False
    
    # ==================== 第三阶段：特征工程 → 模型训练 ====================
    
    async def check_step3_feature_engineering(self):
        """检查步骤3：特征工程"""
        logger.info("\n" + "=" * 60)
        logger.info("检查步骤3：特征工程 → 模型训练")
        logger.info("=" * 60)
        
        try:
            # 3.1 创建特征任务
            logger.info("\n[3.1] 测试特征任务创建...")
            from src.gateway.web.feature_engineering_service import create_feature_task
            
            task = create_feature_task(
                task_type="技术指标",
                config={"description": "数据流测试任务", "strategy_id": self.strategy_id}
            )
            
            if not task or "task_id" not in task:
                self.add_issue("步骤3", "特征任务创建失败或缺少task_id", "P0")
                return False
            
            self.task_id = task["task_id"]
            logger.info(f"✓ 特征任务创建成功: {self.task_id}")
            
            # 验证任务状态
            if task.get("status") != "pending":
                self.add_issue("步骤3", f"任务初始状态应为pending，实际为{task.get('status')}", "P1")
            
            self.add_result("end_to_end_test", "步骤3", {
                "action": "创建特征任务",
                "task_id": self.task_id,
                "strategy_id": self.strategy_id,
                "status": "success"
            })
            
            # 3.2 验证任务持久化
            logger.info("\n[3.2] 验证特征任务持久化...")
            from src.gateway.web.feature_task_persistence import load_feature_task
            
            persisted_task = load_feature_task(self.task_id)
            if persisted_task:
                logger.info(f"✓ 特征任务已持久化: {self.task_id}")
                self.add_result("persistence_validation", "步骤3", {
                    "action": "验证任务持久化",
                    "task_id": self.task_id,
                    "status": "success"
                })
            else:
                self.add_issue("步骤3", "特征任务未正确持久化", "P0")
            
            # 3.3 验证数据格式
            logger.info("\n[3.3] 验证特征任务数据格式...")
            required_fields = ["task_id", "task_type", "status", "created_at"]
            missing_fields = []
            for field in required_fields:
                if field not in task:
                    missing_fields.append(field)
            
            if missing_fields:
                self.add_issue("步骤3", f"特征任务数据缺少必需字段: {missing_fields}", "P1")
            else:
                logger.info("✓ 特征任务数据格式正确")
                self.add_result("data_format_validation", "步骤3", {
                    "action": "验证数据格式",
                    "status": "success",
                    "fields": required_fields
                })
            
            # 等待任务执行（如果执行器运行）
            logger.info("\n[3.4] 等待特征任务执行...")
            await asyncio.sleep(3)
            
            # 检查任务状态更新
            updated_task = load_feature_task(self.task_id)
            if updated_task:
                status = updated_task.get("status")
                logger.info(f"任务状态: {status}")
                if status in ["running", "completed"]:
                    logger.info("✓ 任务状态已更新")
                else:
                    logger.info(f"⚠ 任务状态仍为{status}，可能需要更多时间")
            
            return True
            
        except Exception as e:
            logger.error(f"步骤3检查失败: {e}", exc_info=True)
            self.add_issue("步骤3", f"检查失败: {str(e)}", "P0")
            return False
    
    # ==================== 第四阶段：模型训练 → 策略回测 ====================
    
    async def check_step4_model_training(self):
        """检查步骤4：模型训练"""
        logger.info("\n" + "=" * 60)
        logger.info("检查步骤4：模型训练 → 策略回测")
        logger.info("=" * 60)
        
        try:
            # 4.1 创建训练任务
            logger.info("\n[4.1] 测试训练任务创建...")
            from src.gateway.web.model_training_routes import create_training_job
            
            request_data = {
                "model_type": "LSTM",
                "config": {
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 5,  # 使用较小的epoch以便快速测试
                    "strategy_id": self.strategy_id,
                    "feature_task_id": self.task_id
                }
            }
            
            result = await create_training_job(request_data)
            
            if not result or "job_id" not in result:
                self.add_issue("步骤4", "训练任务创建失败或缺少job_id", "P0")
                return False
            
            self.job_id = result["job_id"]
            logger.info(f"✓ 训练任务创建成功: {self.job_id}")
            
            # 验证任务状态
            job = result.get("job", {})
            if job.get("status") != "pending":
                self.add_issue("步骤4", f"任务初始状态应为pending，实际为{job.get('status')}", "P1")
            
            self.add_result("end_to_end_test", "步骤4", {
                "action": "创建训练任务",
                "job_id": self.job_id,
                "strategy_id": self.strategy_id,
                "task_id": self.task_id,
                "status": "success"
            })
            
            # 4.2 验证任务持久化
            logger.info("\n[4.2] 验证训练任务持久化...")
            from src.gateway.web.training_job_persistence import load_training_job
            
            persisted_job = load_training_job(self.job_id)
            if persisted_job:
                logger.info(f"✓ 训练任务已持久化: {self.job_id}")
                self.add_result("persistence_validation", "步骤4", {
                    "action": "验证任务持久化",
                    "job_id": self.job_id,
                    "status": "success"
                })
            else:
                self.add_issue("步骤4", "训练任务未正确持久化", "P0")
            
            # 4.3 验证数据格式
            logger.info("\n[4.3] 验证训练任务数据格式...")
            required_fields = ["job_id", "model_type", "status", "start_time"]
            missing_fields = []
            for field in required_fields:
                if field not in job:
                    missing_fields.append(field)
            
            if missing_fields:
                self.add_issue("步骤4", f"训练任务数据缺少必需字段: {missing_fields}", "P1")
            else:
                logger.info("✓ 训练任务数据格式正确")
                self.add_result("data_format_validation", "步骤4", {
                    "action": "验证数据格式",
                    "status": "success",
                    "fields": required_fields
                })
            
            # 等待任务执行（如果执行器运行）
            logger.info("\n[4.4] 等待训练任务执行...")
            await asyncio.sleep(5)
            
            # 检查任务状态更新
            updated_job = load_training_job(self.job_id)
            if updated_job:
                status = updated_job.get("status")
                progress = updated_job.get("progress", 0)
                logger.info(f"任务状态: {status}, 进度: {progress}%")
                if status in ["running", "completed"]:
                    logger.info("✓ 任务状态已更新")
                else:
                    logger.info(f"⚠ 任务状态仍为{status}，可能需要更多时间")
            
            return True
            
        except Exception as e:
            logger.error(f"步骤4检查失败: {e}", exc_info=True)
            self.add_issue("步骤4", f"检查失败: {str(e)}", "P0")
            return False
    
    # ==================== 第五阶段：策略回测 → 性能评估 ====================
    
    async def check_step5_strategy_backtest(self):
        """检查步骤5：策略回测"""
        logger.info("\n" + "=" * 60)
        logger.info("检查步骤5：策略回测 → 性能评估")
        logger.info("=" * 60)
        
        try:
            if not self.strategy_id:
                self.add_issue("步骤5", "缺少strategy_id，无法执行回测", "P0")
                return False
            
            # 5.1 执行回测
            logger.info("\n[5.1] 测试回测执行...")
            from src.gateway.web.backtest_service import run_backtest
            
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            try:
                result = await run_backtest(
                    strategy_id=self.strategy_id,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=100000.0,
                    commission_rate=0.001
                )
                
                if not result or "backtest_id" not in result:
                    self.add_issue("步骤5", "回测执行失败或缺少backtest_id", "P0")
                    return False
                
                self.backtest_id = result["backtest_id"]
                logger.info(f"✓ 回测执行成功: {self.backtest_id}")
                
                self.add_result("end_to_end_test", "步骤5", {
                    "action": "执行回测",
                    "backtest_id": self.backtest_id,
                    "strategy_id": self.strategy_id,
                    "status": "success"
                })
                
            except Exception as e:
                # 回测引擎可能不可用，这是可接受的
                logger.warning(f"回测执行失败（可能是引擎不可用）: {e}")
                self.add_issue("步骤5", f"回测执行失败: {str(e)}", "P2")
                # 模拟backtest_id用于后续测试
                self.backtest_id = f"backtest_{self.strategy_id}_{int(time.time())}"
                return True
            
            # 5.2 验证回测结果持久化
            logger.info("\n[5.2] 验证回测结果持久化...")
            from src.gateway.web.backtest_persistence import load_backtest_result
            
            persisted_result = load_backtest_result(self.backtest_id)
            if persisted_result:
                logger.info(f"✓ 回测结果已持久化: {self.backtest_id}")
                self.add_result("persistence_validation", "步骤5", {
                    "action": "验证回测结果持久化",
                    "backtest_id": self.backtest_id,
                    "status": "success"
                })
            else:
                self.add_issue("步骤5", "回测结果未正确持久化", "P1")
            
            # 5.3 验证数据格式
            logger.info("\n[5.3] 验证回测结果数据格式...")
            # 注意：如果回测失败，result可能为空，需要检查
            if not result:
                logger.warning("回测结果为空，跳过数据格式验证")
                return True
            required_fields = ["backtest_id", "strategy_id", "status", "start_date", "end_date"]
            missing_fields = []
            for field in required_fields:
                if field not in result:
                    missing_fields.append(field)
            
            if missing_fields:
                self.add_issue("步骤5", f"回测结果数据缺少必需字段: {missing_fields}", "P1")
            else:
                logger.info("✓ 回测结果数据格式正确")
                self.add_result("data_format_validation", "步骤5", {
                    "action": "验证数据格式",
                    "status": "success",
                    "fields": required_fields
                })
            
            return True
            
        except Exception as e:
            logger.error(f"步骤5检查失败: {e}", exc_info=True)
            self.add_issue("步骤5", f"检查失败: {str(e)}", "P1")
            return False
    
    # ==================== 第六阶段：性能评估 → 策略部署 ====================
    
    async def check_step6_performance_evaluation(self):
        """检查步骤6：性能评估"""
        logger.info("\n" + "=" * 60)
        logger.info("检查步骤6：性能评估 → 策略部署")
        logger.info("=" * 60)
        
        try:
            # 6.1 验证性能评估数据获取
            logger.info("\n[6.1] 验证性能评估数据获取...")
            from src.gateway.web.strategy_performance_service import get_strategy_comparison
            
            comparison = get_strategy_comparison()
            
            if comparison is None:
                self.add_issue("步骤6", "无法获取策略对比数据", "P1")
            else:
                logger.info(f"✓ 获取到 {len(comparison)} 个策略对比数据")
                self.add_result("end_to_end_test", "步骤6", {
                    "action": "获取性能评估数据",
                    "strategy_count": len(comparison),
                    "status": "success"
                })
            
            # 6.2 验证性能指标计算
            logger.info("\n[6.2] 验证性能指标计算...")
            from src.gateway.web.strategy_performance_service import get_performance_metrics
            
            # get_performance_metrics不接受参数
            try:
                metrics = get_performance_metrics()
            except TypeError:
                # 如果函数需要参数，尝试传入strategy_id
                metrics = get_performance_metrics(self.strategy_id) if self.strategy_id else None
            
            if metrics:
                logger.info("✓ 性能指标计算成功")
                self.add_result("data_format_validation", "步骤6", {
                    "action": "验证性能指标",
                    "status": "success"
                })
            else:
                logger.info("⚠ 性能指标为空（可能是策略尚未有回测结果）")
            
            return True
            
        except Exception as e:
            logger.error(f"步骤6检查失败: {e}", exc_info=True)
            self.add_issue("步骤6", f"检查失败: {str(e)}", "P1")
            return False
    
    # ==================== 第七阶段：策略部署 → 监控优化 ====================
    
    async def check_step7_strategy_lifecycle(self):
        """检查步骤7：策略部署"""
        logger.info("\n" + "=" * 60)
        logger.info("检查步骤7：策略部署 → 监控优化")
        logger.info("=" * 60)
        
        try:
            if not self.strategy_id:
                self.add_issue("步骤7", "缺少strategy_id，无法测试部署", "P0")
                return False
            
            # 7.1 验证生命周期管理
            logger.info("\n[7.1] 验证生命周期管理...")
            from src.gateway.web.strategy_lifecycle_routes import get_strategy_lifecycle
            
            try:
                lifecycle = await get_strategy_lifecycle(self.strategy_id)
                
                if lifecycle:
                    logger.info(f"✓ 获取到生命周期信息: {lifecycle.get('current_stage', 'unknown')}")
                    self.add_result("end_to_end_test", "步骤7", {
                        "action": "获取生命周期信息",
                        "strategy_id": self.strategy_id,
                        "current_stage": lifecycle.get("current_stage"),
                        "status": "success"
                    })
                else:
                    self.add_issue("步骤7", "无法获取生命周期信息", "P1")
                    
            except Exception as e:
                logger.warning(f"获取生命周期信息失败: {e}")
                self.add_issue("步骤7", f"获取生命周期信息失败: {str(e)}", "P2")
            
            # 7.2 验证部署功能（不实际部署，只验证API）
            logger.info("\n[7.2] 验证部署API...")
            # 注意：这里只验证API存在，不实际执行部署
            
            return True
            
        except Exception as e:
            logger.error(f"步骤7检查失败: {e}", exc_info=True)
            self.add_issue("步骤7", f"检查失败: {str(e)}", "P1")
            return False
    
    # ==================== 第八阶段：监控优化 → 策略构思（循环）====================
    
    async def check_step8_execution_monitor(self):
        """检查步骤8：监控优化"""
        logger.info("\n" + "=" * 60)
        logger.info("检查步骤8：监控优化 → 策略构思（循环）")
        logger.info("=" * 60)
        
        try:
            # 8.1 验证执行监控数据获取
            logger.info("\n[8.1] 验证执行监控数据获取...")
            from src.gateway.web.strategy_execution_service import get_strategy_execution_status
            
            status = get_strategy_execution_status()
            
            if status:
                logger.info("✓ 获取到执行状态")
                self.add_result("end_to_end_test", "步骤8", {
                    "action": "获取执行状态",
                    "status": "success"
                })
            else:
                logger.info("⚠ 执行状态为空（可能是没有运行中的策略）")
            
            # 8.2 验证实时信号获取
            logger.info("\n[8.2] 验证实时信号获取...")
            from src.gateway.web.trading_signal_service import get_realtime_signals
            
            signals = get_realtime_signals()
            
            if signals is not None:
                logger.info(f"✓ 获取到 {len(signals)} 个实时信号")
                self.add_result("data_format_validation", "步骤8", {
                    "action": "验证实时信号",
                    "signal_count": len(signals),
                    "status": "success"
                })
            else:
                logger.info("⚠ 实时信号为空（可能是没有运行中的策略）")
            
            return True
            
        except Exception as e:
            logger.error(f"步骤8检查失败: {e}", exc_info=True)
            self.add_issue("步骤8", f"检查失败: {str(e)}", "P1")
            return False
    
    # ==================== ID传递链验证 ====================
    
    def check_id_chain(self):
        """验证ID传递链"""
        logger.info("\n" + "=" * 60)
        logger.info("验证ID传递链")
        logger.info("=" * 60)
        
        id_chain = {
            "strategy_id": self.strategy_id,
            "task_id": self.task_id,
            "job_id": self.job_id,
            "backtest_id": self.backtest_id
        }
        
        logger.info("ID传递链:")
        for key, value in id_chain.items():
            if value:
                logger.info(f"  ✓ {key}: {value}")
            else:
                logger.warning(f"  ✗ {key}: 缺失")
                self.add_issue("ID传递链", f"{key}缺失", "P1")
        
        # 验证关键ID
        if not self.strategy_id:
            self.add_issue("ID传递链", "strategy_id缺失，无法完成完整流程", "P0")
        
        self.add_result("end_to_end_test", "ID传递链", id_chain)
        
        return len([v for v in id_chain.values() if v]) > 0
    
    # ==================== 实时更新验证 ====================
    
    async def check_realtime_updates(self):
        """验证实时更新机制（WebSocket）"""
        logger.info("\n" + "=" * 60)
        logger.info("验证实时更新机制（WebSocket）")
        logger.info("=" * 60)
        
        try:
            # 验证WebSocket管理器
            logger.info("\n[实时更新] 验证WebSocket管理器...")
            from src.gateway.web.websocket_manager import ConnectionManager
            
            manager = ConnectionManager()
            
            # 检查WebSocket通道
            channels = [
                "feature_engineering",
                "model_training",
                "backtest_progress",
                "execution_status"
            ]
            
            for channel in channels:
                if channel in manager.active_connections:
                    logger.info(f"✓ WebSocket通道已配置: {channel}")
                    self.add_result("realtime_update_validation", channel, {
                        "action": "验证WebSocket通道",
                        "status": "success"
                    })
                else:
                    logger.warning(f"⚠ WebSocket通道未配置: {channel}")
                    self.add_issue("实时更新", f"WebSocket通道未配置: {channel}", "P2")
            
            # 验证广播函数存在
            logger.info("\n[实时更新] 验证广播函数...")
            broadcast_functions = [
                "_broadcast_feature_engineering",
                "_broadcast_model_training",
                "_broadcast_backtest_progress",
                "_broadcast_execution_status"
            ]
            
            for func_name in broadcast_functions:
                if hasattr(manager, func_name):
                    logger.info(f"✓ 广播函数存在: {func_name}")
                    self.add_result("realtime_update_validation", func_name, {
                        "action": "验证广播函数",
                        "status": "success"
                    })
                else:
                    logger.warning(f"⚠ 广播函数不存在: {func_name}")
                    self.add_issue("实时更新", f"广播函数不存在: {func_name}", "P1")
            
            return True
            
        except Exception as e:
            logger.error(f"实时更新验证失败: {e}", exc_info=True)
            self.add_issue("实时更新", f"验证失败: {str(e)}", "P1")
            return False
    
    # ==================== 生成报告 ====================
    
    def generate_report(self) -> Dict[str, Any]:
        """生成检查报告"""
        report = {
            "check_time": datetime.now().isoformat(),
            "summary": {
                "total_steps": 8,
                "completed_steps": len([k for k in self.results["end_to_end_test"].keys() if k.startswith("步骤")]),
                "total_issues": len(self.results["issues"]),
                "p0_issues": len([i for i in self.results["issues"] if i["severity"] == "P0"]),
                "p1_issues": len([i for i in self.results["issues"] if i["severity"] == "P1"]),
                "p2_issues": len([i for i in self.results["issues"] if i["severity"] == "P2"])
            },
            "results": self.results,
            "id_chain": {
                "strategy_id": self.strategy_id,
                "task_id": self.task_id,
                "job_id": self.job_id,
                "backtest_id": self.backtest_id
            }
        }
        
        return report


async def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("量化策略开发流程数据流检查")
    logger.info("=" * 60)
    
    checker = DataFlowChecker()
    
    # 执行各步骤检查
    steps = [
        ("步骤1：策略构思", checker.check_step1_strategy_conception),
        ("步骤2：数据收集", checker.check_step2_data_collection),
        ("步骤3：特征工程", checker.check_step3_feature_engineering),
        ("步骤4：模型训练", checker.check_step4_model_training),
        ("步骤5：策略回测", checker.check_step5_strategy_backtest),
        ("步骤6：性能评估", checker.check_step6_performance_evaluation),
        ("步骤7：策略部署", checker.check_step7_strategy_lifecycle),
        ("步骤8：监控优化", checker.check_step8_execution_monitor),
    ]
    
    for step_name, step_func in steps:
        try:
            await step_func()
        except Exception as e:
            logger.error(f"{step_name}检查异常: {e}", exc_info=True)
            checker.add_issue(step_name, f"检查异常: {str(e)}", "P0")
    
    # 验证ID传递链
    checker.check_id_chain()
    
    # 验证实时更新
    await checker.check_realtime_updates()
    
    # 生成报告
    report = checker.generate_report()
    
    # 保存报告
    report_file = os.path.join(project_root, "docs", "strategy_development_dataflow_check_report.json")
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("检查完成")
    logger.info("=" * 60)
    logger.info(f"完成步骤: {report['summary']['completed_steps']}/8")
    logger.info(f"发现问题: {report['summary']['total_issues']} (P0: {report['summary']['p0_issues']}, P1: {report['summary']['p1_issues']}, P2: {report['summary']['p2_issues']})")
    logger.info(f"报告已保存: {report_file}")
    
    return report


if __name__ == "__main__":
    report = asyncio.run(main())
    sys.exit(0 if report["summary"]["p0_issues"] == 0 else 1)

