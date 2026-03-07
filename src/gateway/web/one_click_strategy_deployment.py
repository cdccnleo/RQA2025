"""
一键式策略部署模块

功能：
- 策略打包与验证
- 环境自动配置
- 依赖自动安装
- 一键部署到生产环境
- 部署历史记录
- 回滚机制

技术栈：
- subprocess: 执行部署命令
- json/yaml: 配置管理
- hashlib: 版本控制

作者: Claude
创建日期: 2026-02-21
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """部署状态"""
    PENDING = "pending"
    VALIDATING = "validating"
    PACKAGING = "packaging"
    DEPLOYING = "deploying"
    TESTING = "testing"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class DeploymentEnvironment(Enum):
    """部署环境"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class StrategyPackage:
    """策略包"""
    package_id: str
    strategy_id: str
    strategy_name: str
    version: str
    files: List[str]
    dependencies: List[str]
    config: Dict[str, Any]
    checksum: str
    created_at: datetime
    size_bytes: int


@dataclass
class DeploymentRecord:
    """部署记录"""
    deployment_id: str
    package_id: str
    strategy_id: str
    environment: DeploymentEnvironment
    status: DeploymentStatus
    started_at: datetime
    completed_at: Optional[datetime]
    logs: List[str]
    error_message: Optional[str]
    deployed_by: str
    rollback_to: Optional[str]
    strategy_version: Optional[str] = None
    optimization_history: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class StrategyPackager:
    """策略打包器"""
    
    def __init__(self, output_dir: str = "strategy_packages"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def package_strategy(self, strategy_path: str, strategy_name: str,
                        version: str, strategy_id: str) -> StrategyPackage:
        """
        打包策略
        
        Args:
            strategy_path: 策略代码路径
            strategy_name: 策略名称
            version: 版本号
            strategy_id: 策略ID
            
        Returns:
            策略包
        """
        package_id = f"{strategy_id}_{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        package_path = self.output_dir / f"{package_id}.zip"
        
        strategy_path = Path(strategy_path)
        files = []
        dependencies = []
        
        # 创建ZIP包
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 添加策略代码
            for file_path in strategy_path.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(strategy_path)
                    zf.write(file_path, arcname)
                    files.append(str(arcname))
            
            # 读取依赖
            requirements_file = strategy_path / "requirements.txt"
            if requirements_file.exists():
                with open(requirements_file, 'r') as f:
                    dependencies = [line.strip() for line in f if line.strip()]
            
            # 读取配置
            config_file = strategy_path / "strategy_config.json"
            config = {}
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
            
            # 添加元数据
            metadata = {
                'strategy_id': strategy_id,
                'strategy_name': strategy_name,
                'version': version,
                'created_at': datetime.now().isoformat(),
                'files': files,
                'dependencies': dependencies
            }
            zf.writestr('metadata.json', json.dumps(metadata, indent=2))
        
        # 计算校验和
        checksum = self._calculate_checksum(package_path)
        
        package = StrategyPackage(
            package_id=package_id,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            version=version,
            files=files,
            dependencies=dependencies,
            config=config,
            checksum=checksum,
            created_at=datetime.now(),
            size_bytes=package_path.stat().st_size
        )
        
        logger.info(f"策略打包完成: {package_id}")
        return package
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """计算文件校验和"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def validate_package(self, package_path: str) -> Tuple[bool, str]:
        """
        验证策略包
        
        Returns:
            (是否有效, 错误信息)
        """
        try:
            with zipfile.ZipFile(package_path, 'r') as zf:
                # 检查必要文件
                if 'metadata.json' not in zf.namelist():
                    return False, "缺少metadata.json"
                
                # 读取元数据
                metadata = json.loads(zf.read('metadata.json'))
                required_fields = ['strategy_id', 'strategy_name', 'version']
                for field in required_fields:
                    if field not in metadata:
                        return False, f"元数据缺少{field}"
                
                # 检查文件完整性
                for file in metadata.get('files', []):
                    if file not in zf.namelist():
                        return False, f"缺少文件: {file}"
            
            return True, ""
        except Exception as e:
            return False, str(e)


class DeploymentManager:
    """部署管理器"""
    
    def __init__(self, deploy_root: str = "deployed_strategies"):
        self.deploy_root = Path(deploy_root)
        self.deploy_root.mkdir(exist_ok=True)
        self.packager = StrategyPackager()
        self.deployments: Dict[str, DeploymentRecord] = {}
        self._load_deployment_history()
    
    def _load_deployment_history(self) -> None:
        """加载部署历史"""
        history_file = self.deploy_root / "deployment_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for dep_data in data:
                        record = DeploymentRecord(
                            deployment_id=dep_data['deployment_id'],
                            package_id=dep_data['package_id'],
                            strategy_id=dep_data['strategy_id'],
                            environment=DeploymentEnvironment(dep_data['environment']),
                            status=DeploymentStatus(dep_data['status']),
                            started_at=datetime.fromisoformat(dep_data['started_at']),
                            completed_at=datetime.fromisoformat(dep_data['completed_at']) if dep_data.get('completed_at') else None,
                            logs=dep_data.get('logs', []),
                            error_message=dep_data.get('error_message'),
                            deployed_by=dep_data.get('deployed_by', 'unknown'),
                            rollback_to=dep_data.get('rollback_to')
                        )
                        self.deployments[record.deployment_id] = record
            except Exception as e:
                logger.error(f"加载部署历史失败: {e}")
    
    def _save_deployment_history(self) -> None:
        """保存部署历史"""
        history_file = self.deploy_root / "deployment_history.json"
        try:
            data = []
            for record in self.deployments.values():
                data.append({
                    'deployment_id': record.deployment_id,
                    'package_id': record.package_id,
                    'strategy_id': record.strategy_id,
                    'environment': record.environment.value,
                    'status': record.status.value,
                    'started_at': record.started_at.isoformat(),
                    'completed_at': record.completed_at.isoformat() if record.completed_at else None,
                    'logs': record.logs,
                    'error_message': record.error_message,
                    'deployed_by': record.deployed_by,
                    'rollback_to': record.rollback_to,
                    'strategy_version': record.strategy_version,
                    'optimization_history': record.optimization_history,
                    'validation_results': record.validation_results,
                    'notes': record.notes
                })
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"保存部署历史失败: {e}")
    
    def deploy(self, package_path: str, environment: DeploymentEnvironment,
              deployed_by: str = "system") -> DeploymentRecord:
        """
        部署策略
        
        Args:
            package_path: 策略包路径
            environment: 部署环境
            deployed_by: 部署人
            
        Returns:
            部署记录
        """
        deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        record = DeploymentRecord(
            deployment_id=deployment_id,
            package_id=Path(package_path).stem,
            strategy_id="",
            environment=environment,
            status=DeploymentStatus.PENDING,
            started_at=datetime.now(),
            completed_at=None,
            logs=[],
            error_message=None,
            deployed_by=deployed_by,
            rollback_to=None
        )
        
        self.deployments[deployment_id] = record
        
        try:
            # 1. 验证
            self._update_status(record, DeploymentStatus.VALIDATING)
            is_valid, error = self.packager.validate_package(package_path)
            if not is_valid:
                raise Exception(f"验证失败: {error}")
            record.logs.append("✓ 验证通过")
            
            # 2. 解压
            self._update_status(record, DeploymentStatus.PACKAGING)
            deploy_path = self._extract_package(package_path, environment)
            record.logs.append(f"✓ 解压到: {deploy_path}")
            
            # 3. 安装依赖
            self._update_status(record, DeploymentStatus.DEPLOYING)
            self._install_dependencies(deploy_path)
            record.logs.append("✓ 依赖安装完成")
            
            # 4. 测试
            self._update_status(record, DeploymentStatus.TESTING)
            self._run_tests(deploy_path)
            record.logs.append("✓ 测试通过")
            
            # 5. 完成
            self._update_status(record, DeploymentStatus.SUCCESS)
            record.completed_at = datetime.now()
            record.logs.append("✓ 部署成功")
            
            logger.info(f"部署成功: {deployment_id}")
            
        except Exception as e:
            record.status = DeploymentStatus.FAILED
            record.error_message = str(e)
            record.completed_at = datetime.now()
            record.logs.append(f"✗ 部署失败: {str(e)}")
            logger.error(f"部署失败: {deployment_id} - {e}")
        
        self._save_deployment_history()
        return record
    
    def _update_status(self, record: DeploymentRecord, status: DeploymentStatus) -> None:
        """更新状态"""
        record.status = status
        record.logs.append(f"状态更新: {status.value}")
        self._save_deployment_history()
    
    def _extract_package(self, package_path: str, environment: DeploymentEnvironment) -> str:
        """解压策略包"""
        deploy_dir = self.deploy_root / environment.value / Path(package_path).stem
        
        if deploy_dir.exists():
            shutil.rmtree(deploy_dir)
        
        with zipfile.ZipFile(package_path, 'r') as zf:
            zf.extractall(deploy_dir)
        
        return str(deploy_dir)
    
    def _install_dependencies(self, deploy_path: str) -> None:
        """安装依赖"""
        requirements_file = Path(deploy_path) / "requirements.txt"
        if requirements_file.exists():
            subprocess.run(
                ["pip", "install", "-r", str(requirements_file)],
                check=True,
                capture_output=True
            )
    
    def _run_tests(self, deploy_path: str) -> None:
        """运行测试"""
        test_file = Path(deploy_path) / "test_strategy.py"
        if test_file.exists():
            result = subprocess.run(
                ["python", str(test_file)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise Exception(f"测试失败: {result.stderr}")
    
    def rollback(self, deployment_id: str) -> bool:
        """
        回滚部署
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            是否成功
        """
        record = self.deployments.get(deployment_id)
        if not record:
            return False
        
        try:
            record.status = DeploymentStatus.ROLLING_BACK
            self._save_deployment_history()
            
            # 删除部署目录
            deploy_dir = self.deploy_root / record.environment.value / record.package_id
            if deploy_dir.exists():
                shutil.rmtree(deploy_dir)
            
            record.status = DeploymentStatus.ROLLED_BACK
            record.completed_at = datetime.now()
            record.logs.append("✓ 回滚完成")
            
            self._save_deployment_history()
            logger.info(f"回滚成功: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"回滚失败: {deployment_id} - {e}")
            return False
    
    def get_deployment_history(self, strategy_id: Optional[str] = None) -> List[DeploymentRecord]:
        """获取部署历史"""
        records = list(self.deployments.values())
        if strategy_id:
            records = [r for r in records if r.strategy_id == strategy_id]
        return sorted(records, key=lambda x: x.started_at, reverse=True)
    
    def get_deployment(self, deployment_id: str) -> Optional[DeploymentRecord]:
        """获取部署记录"""
        return self.deployments.get(deployment_id)
    
    def one_click_deploy(self, strategy_path: str, strategy_name: str,
                        version: str, strategy_id: str,
                        environment: DeploymentEnvironment = DeploymentEnvironment.STAGING,
                        deployed_by: str = "system",
                        strategy_info: Optional[Dict[str, Any]] = None,
                        notes: Optional[str] = None) -> DeploymentRecord:
        """
        一键部署
        
        Args:
            strategy_path: 策略代码路径
            strategy_name: 策略名称
            version: 版本号
            strategy_id: 策略ID
            environment: 部署环境
            deployed_by: 部署人
            strategy_info: 策略信息（包含优化历史等）
            notes: 部署备注
            
        Returns:
            部署记录
        """
        # 1. 打包
        package = self.packager.package_strategy(
            strategy_path, strategy_name, version, strategy_id
        )
        
        # 2. 部署
        package_path = self.packager.output_dir / f"{package.package_id}.zip"
        record = self.deploy(str(package_path), environment, deployed_by)
        record.strategy_id = strategy_id
        record.strategy_version = version
        record.notes = notes
        
        # 3. 记录策略信息和优化历史
        if strategy_info:
            # 提取优化历史
            stats = strategy_info.get('stats') or strategy_info.get('parameters', {}).get('_stats', {})
            if stats:
                record.optimization_history = {
                    'backtest_count': stats.get('backtest_count', 0),
                    'optimization_count': stats.get('optimization_count', 0),
                    'last_backtest_at': stats.get('last_backtest_at'),
                    'last_optimization_at': stats.get('last_optimization_at')
                }
            
            # 提取验证结果（如果有）
            if 'backtest_result' in strategy_info:
                record.validation_results = {
                    'backtest_result': strategy_info['backtest_result']
                }
        
        # 保存更新后的记录
        self._save_deployment_history()
        
        return record
    
    def deploy_with_validation(self, strategy_path: str, strategy_name: str,
                              version: str, strategy_id: str,
                              environment: DeploymentEnvironment = DeploymentEnvironment.STAGING,
                              deployed_by: str = "system",
                              strategy_info: Optional[Dict[str, Any]] = None,
                              notes: Optional[str] = None) -> DeploymentRecord:
        """
        带验证的部署（预留接口，可集成回测和优化流程）
        
        Args:
            strategy_path: 策略代码路径
            strategy_name: 策略名称
            version: 版本号
            strategy_id: 策略ID
            environment: 部署环境
            deployed_by: 部署人
            strategy_info: 策略信息
            notes: 部署备注
            
        Returns:
            部署记录
        """
        # TODO: 在这里集成回测和优化验证流程
        # 可以调用 strategy_execution_routes 和 strategy_optimization_routes 中的功能
        
        logger.info(f"策略 {strategy_id} 开始带验证的部署流程")
        
        # 目前直接调用一键部署
        return self.one_click_deploy(
            strategy_path=strategy_path,
            strategy_name=strategy_name,
            version=version,
            strategy_id=strategy_id,
            environment=environment,
            deployed_by=deployed_by,
            strategy_info=strategy_info,
            notes=notes
        )


# 便捷函数
def get_deployment_manager(deploy_root: str = "deployed_strategies") -> DeploymentManager:
    """获取部署管理器实例"""
    return DeploymentManager(deploy_root)
