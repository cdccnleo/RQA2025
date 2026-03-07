"""
全面部署阶段模块

负责100%流量切换、蓝绿部署和旧模型归档
"""

from typing import Dict, Any, Optional
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import shutil

from .base import PipelineStage
from ..exceptions import DeploymentException, StageExecutionException
from ..config import StageConfig


@dataclass
class DeploymentStatus:
    """部署状态"""
    deployment_id: str
    status: str  # deploying, active, rolling_back, archived
    traffic_percentage: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "status": self.status,
            "traffic_percentage": self.traffic_percentage,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error_message": self.error_message
        }


class FullDeploymentStage(PipelineStage):
    """
    全面部署阶段
    
    功能：
    - 100%流量切换到新模型
    - 蓝绿部署支持
    - 部署验证
    - 旧模型归档
    """
    
    def __init__(self, config: Optional[StageConfig] = None):
        super().__init__("full_deployment", config)
        self._deployment_status: Optional[DeploymentStatus] = None
        self._deployment_info: Dict[str, Any] = {}
        self._previous_model_path: Optional[str] = None
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行全面部署
        
        Args:
            context: 包含deployment_id, model_path, canary_passed的上下文
            
        Returns:
            包含deployment_info, deployment_status的输出
        """
        self.logger.info("开始全面部署阶段")
        
        # 检查金丝雀结果
        canary_passed = context.get("canary_passed", False)
        if not canary_passed:
            raise StageExecutionException(
                message="金丝雀部署未通过，无法进行全面部署",
                stage_name=self.name
            )
        
        # 获取配置
        deployment_strategy = self.config.config.get("strategy", "rolling")  # rolling, blue_green
        
        model_path = context.get("model_path")
        deployment_id = context.get("deployment_id")
        model_info = context.get("model_info", {})
        
        self.logger.info(f"全面部署策略: {deployment_strategy}")
        
        # 1. 保存当前模型（用于回滚）
        self._previous_model_path = self._backup_current_model()
        
        # 2. 执行部署
        if deployment_strategy == "blue_green":
            new_deployment_id = self._deploy_blue_green(model_path, deployment_id)
        else:
            new_deployment_id = self._deploy_rolling(model_path, deployment_id)
        
        # 3. 验证部署
        verified = self._verify_deployment(new_deployment_id)
        
        if not verified:
            raise DeploymentException(
                message="全面部署验证失败",
                deployment_type="full",
                model_version=model_info.get("training_timestamp")
            )
        
        # 4. 归档旧模型
        self._archive_old_model()
        
        # 5. 记录部署信息
        self._deployment_status = DeploymentStatus(
            deployment_id=new_deployment_id,
            status="active",
            traffic_percentage=100,
            start_time=datetime.now()
        )
        
        self._deployment_info = {
            "deployment_id": new_deployment_id,
            "previous_deployment_id": deployment_id,
            "model_path": model_path,
            "model_version": model_info.get("training_timestamp"),
            "strategy": deployment_strategy,
            "deployment_timestamp": datetime.now().isoformat(),
            "traffic_percentage": 100,
            "previous_model_backup": self._previous_model_path
        }
        
        self.logger.info(f"全面部署完成，部署ID: {new_deployment_id}")
        
        return {
            "deployment_info": self._deployment_info,
            "deployment_status": self._deployment_status.to_dict(),
            "deployment_id": new_deployment_id
        }
    
    def _backup_current_model(self) -> Optional[str]:
        """备份当前模型"""
        try:
            model_dir = Path("models")
            if not model_dir.exists():
                return None
            
            # 查找当前活跃的模型
            model_files = list(model_dir.glob("model_*.joblib"))
            if not model_files:
                return None
            
            # 最新的模型作为当前模型
            current_model = sorted(model_files)[-1]
            
            # 创建备份
            backup_dir = Path("models/backup")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"model_backup_{timestamp}.joblib"
            
            shutil.copy2(current_model, backup_path)
            self.logger.info(f"当前模型已备份: {backup_path}")
            
            return str(backup_path)
            
        except Exception as e:
            self.logger.warning(f"备份当前模型失败: {e}")
            return None
    
    def _deploy_rolling(self, model_path: str, canary_deployment_id: Optional[str]) -> str:
        """
        执行滚动部署
        
        逐步将流量从金丝雀切换到全面部署
        """
        import uuid
        deployment_id = f"full_{uuid.uuid4().hex[:8]}"
        
        self.logger.info(f"开始滚动部署: {deployment_id}")
        
        # 模拟渐进式流量切换
        traffic_steps = [25, 50, 75, 100]
        
        for percentage in traffic_steps:
            self.logger.info(f"切换流量到 {percentage}%")
            time.sleep(1)  # 模拟部署延迟
            
            # 验证当前流量比例的健康状况
            if not self._check_health(deployment_id):
                raise DeploymentException(
                    message=f"滚动部署在 {percentage}% 流量时健康检查失败",
                    deployment_type="rolling"
                )
        
        self.logger.info(f"滚动部署完成: {deployment_id}")
        return deployment_id
    
    def _deploy_blue_green(self, model_path: str, canary_deployment_id: Optional[str]) -> str:
        """
        执行蓝绿部署
        
        部署新版本（绿环境），验证通过后切换流量，保留旧版本（蓝环境）
        """
        import uuid
        deployment_id = f"bluegreen_{uuid.uuid4().hex[:8]}"
        
        self.logger.info(f"开始蓝绿部署: {deployment_id}")
        
        # 1. 部署绿环境（新版本）
        self.logger.info("部署绿环境（新版本）")
        time.sleep(2)
        
        # 2. 验证绿环境
        self.logger.info("验证绿环境")
        if not self._check_health(deployment_id):
            raise DeploymentException(
                message="绿环境健康检查失败",
                deployment_type="blue_green"
            )
        
        # 3. 切换流量到绿环境
        self.logger.info("切换流量到绿环境")
        time.sleep(1)
        
        # 4. 验证流量切换
        if not self._check_traffic_switch(deployment_id):
            # 回滚到蓝环境
            self.logger.error("流量切换验证失败，回滚到蓝环境")
            self._rollback_to_blue()
            raise DeploymentException(
                message="流量切换验证失败，已回滚",
                deployment_type="blue_green"
            )
        
        self.logger.info(f"蓝绿部署完成: {deployment_id}")
        return deployment_id
    
    def _check_health(self, deployment_id: str) -> bool:
        """健康检查"""
        # 模拟健康检查
        import random
        health_score = random.random()
        return health_score > 0.1  # 90%成功率
    
    def _check_traffic_switch(self, deployment_id: str) -> bool:
        """验证流量切换"""
        # 模拟流量切换验证
        time.sleep(1)
        return True
    
    def _rollback_to_blue(self) -> None:
        """回滚到蓝环境"""
        self.logger.info("回滚到蓝环境")
        time.sleep(1)
    
    def _verify_deployment(self, deployment_id: str) -> bool:
        """验证部署"""
        self.logger.info(f"验证部署: {deployment_id}")
        
        # 1. 健康检查
        if not self._check_health(deployment_id):
            self.logger.error("健康检查失败")
            return False
        
        # 2. 功能测试
        if not self._functional_test(deployment_id):
            self.logger.error("功能测试失败")
            return False
        
        # 3. 性能测试
        if not self._performance_test(deployment_id):
            self.logger.error("性能测试失败")
            return False
        
        self.logger.info("部署验证通过")
        return True
    
    def _functional_test(self, deployment_id: str) -> bool:
        """功能测试"""
        # 模拟功能测试
        time.sleep(1)
        return True
    
    def _performance_test(self, deployment_id: str) -> bool:
        """性能测试"""
        # 模拟性能测试
        time.sleep(1)
        return True
    
    def _archive_old_model(self) -> None:
        """归档旧模型"""
        try:
            model_dir = Path("models")
            archive_dir = Path("models/archive")
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # 移动旧模型到归档目录（保留最近5个）
            model_files = sorted(model_dir.glob("model_*.joblib"))
            if len(model_files) > 5:
                for old_model in model_files[:-5]:
                    if "backup" not in str(old_model):
                        shutil.move(str(old_model), str(archive_dir / old_model.name))
                        self.logger.debug(f"归档旧模型: {old_model.name}")
            
        except Exception as e:
            self.logger.warning(f"归档旧模型失败: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取阶段指标"""
        return {
            "deployment_id": self._deployment_info.get("deployment_id"),
            "strategy": self._deployment_info.get("strategy"),
            "traffic_percentage": self._deployment_status.traffic_percentage if self._deployment_status else 0,
            "status": self._deployment_status.status if self._deployment_status else "unknown"
        }
    
    def rollback(self, context: Dict[str, Any]) -> bool:
        """回滚全面部署"""
        self.logger.info("回滚全面部署")
        
        # 如果有备份，恢复到之前的模型
        if self._previous_model_path and Path(self._previous_model_path).exists():
            try:
                model_dir = Path("models")
                backup_path = Path(self._previous_model_path)
                
                # 恢复备份
                restored_path = model_dir / f"model_restored_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                shutil.copy2(backup_path, restored_path)
                
                self.logger.info(f"已恢复到之前的模型: {restored_path}")
                
            except Exception as e:
                self.logger.error(f"恢复模型失败: {e}")
                return False
        
        if self._deployment_status:
            self._deployment_status.status = "rolling_back"
            self._deployment_status.end_time = datetime.now()
        
        return True
