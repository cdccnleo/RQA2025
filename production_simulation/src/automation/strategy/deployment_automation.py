"""
Deployment Automation Module
部署自动化模块

This module provides automated deployment capabilities for quantitative trading strategies
此模块为量化交易策略提供自动化部署能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import shutil
import threading
import time
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):

    """Deployment status enumeration"""
    PENDING = "pending"
    BUILDING = "building"
    TESTING = "testing"
    DEPLOYING = "deploying"
    RUNNING = "running"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    STOPPED = "stopped"


class DeploymentType(Enum):

    """Deployment type enumeration"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    IMMEDIATE = "immediate"


class Environment(Enum):

    """Deployment environment enumeration"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DR = "dr"  # Disaster Recovery


@dataclass
class DeploymentConfig:

    """
    Deployment configuration data class
    部署配置数据类
    """
    environment: str
    deployment_type: str
    auto_rollback: bool = True
    health_check_timeout: int = 300
    traffic_distribution: Optional[Dict[str, float]] = None
    resource_limits: Optional[Dict[str, Any]] = None
    monitoring_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class DeploymentResult:

    """
    Deployment result data class
    部署结果数据类
    """
    success: bool
    start_time: datetime
    end_time: Optional[datetime] = None
    deployment_time: float = 0.0
    health_check_passed: bool = False
    rollback_triggered: bool = False
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


@dataclass
class DeploymentJob:

    """
    Deployment job data class
    部署作业数据类
    """
    job_id: str
    strategy_id: str
    version: str
    environment: str
    deployment_type: str
    config: DeploymentConfig
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[DeploymentResult] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        if self.result:
            data['result'] = self.result.to_dict()
        return data


class DeploymentEngine:

    """
    Deployment Engine Class
    部署引擎类

    Automated deployment engine for trading strategies
    交易策略的自动化部署引擎
    """

    def __init__(self, engine_name: str = "default_deployment_engine"):
        """
        Initialize deployment engine
        初始化部署引擎

        Args:
            engine_name: Name of the deployment engine
                        部署引擎名称
        """
        self.engine_name = engine_name
        self.deployment_jobs: Dict[str, DeploymentJob] = {}
        self.active_deployments: Dict[str, threading.Thread] = {}
        self.environment_status: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Engine configuration
        self.max_concurrent_deployments = 3
        self.deployment_timeout = 1800  # 30 minutes
        self.health_check_interval = 30  # seconds

        # Deployment paths
        self.base_deployment_path = Path("/opt / quant_trading / deployments")
        self.backup_path = Path("/opt / quant_trading / backups")

        # Performance tracking
        self.stats = {
            'total_deployments': 0,
            'successful_deployments': 0,
            'failed_deployments': 0,
            'average_deployment_time': 0.0,
            'rollback_rate': 0.0
        }

        logger.info(f"Deployment engine {engine_name} initialized")

    def create_deployment_job(self,


                              job_id: str,
                              strategy_id: str,
                              version: str,
                              environment: Environment,
                              deployment_type: DeploymentType,
                              config: Optional[DeploymentConfig] = None) -> str:
        """
        Create a deployment job
        创建部署作业

        Args:
            job_id: Unique job identifier
                   唯一作业标识符
            strategy_id: Strategy identifier
                        策略标识符
            version: Strategy version to deploy
                    要部署的策略版本
            environment: Target environment
                        目标环境
            deployment_type: Type of deployment
                           部署类型
            config: Deployment configuration
                   部署配置

        Returns:
            str: Created job ID
                 创建的作业ID
        """
        if config is None:
            config = DeploymentConfig(
                environment=environment.value,
                deployment_type=deployment_type.value
            )

        job = DeploymentJob(
            job_id=job_id,
            strategy_id=strategy_id,
            version=version,
            environment=environment.value,
            deployment_type=deployment_type.value,
            config=config,
            status=DeploymentStatus.PENDING.value,
            created_at=datetime.now(),
            metadata={}
        )

        self.deployment_jobs[job_id] = job
        logger.info(f"Created deployment job: {job_id} for strategy {strategy_id} v{version}")
        return job_id

    def execute_deployment(self, job_id: str, async_execution: bool = True) -> Dict[str, Any]:
        """
        Execute a deployment job
        执行部署作业

        Args:
            job_id: Job identifier
                   作业标识符
            async_execution: Whether to execute asynchronously
                           是否异步执行

        Returns:
            dict: Execution result
                  执行结果
        """
        if job_id not in self.deployment_jobs:
            return {'success': False, 'error': f'Deployment job {job_id} not found'}

        job = self.deployment_jobs[job_id]

        # Check concurrent deployment limit
        if len(self.active_deployments) >= self.max_concurrent_deployments:
            return {
                'success': False,
                'error': 'Maximum concurrent deployments reached'
            }

        # Check if environment is locked
        if self._is_environment_locked(job.environment):
            return {
                'success': False,
                'error': f'Environment {job.environment} is currently locked'
            }

        if async_execution:
            # Start async execution
            execution_thread = threading.Thread(
                target=self._execute_deployment_sync,
                args=(job_id,),
                daemon=True
            )
            self.active_deployments[job_id] = execution_thread
            execution_thread.start()

            return {
                'success': True,
                'execution_mode': 'async',
                'job_id': job_id
            }
        else:
            # Execute synchronously
            return self._execute_deployment_sync(job_id)

    def _execute_deployment_sync(self, job_id: str) -> Dict[str, Any]:
        """
        Execute deployment job synchronously
        同步执行部署作业

        Args:
            job_id: Job identifier
                   作业标识符

        Returns:
            dict: Execution result
                  执行结果
        """
        job = self.deployment_jobs[job_id]
        job.status = DeploymentStatus.BUILDING.value
        job.started_at = datetime.now()

        result = {
            'job_id': job_id,
            'success': False,
            'start_time': job.started_at,
            'execution_time': 0.0
        }

        start_time = time.time()

        try:
            # Execute deployment pipeline
            self._lock_environment(job.environment)

            # Phase 1: Build
            job.status = DeploymentStatus.BUILDING.value
            self._execute_build_phase(job)

            # Phase 2: Test
            job.status = DeploymentStatus.TESTING.value
            self._execute_test_phase(job)

            # Phase 3: Deploy
            job.status = DeploymentStatus.DEPLOYING.value
            deployment_result = self._execute_deploy_phase(job)

            # Phase 4: Health Check
            job.status = DeploymentStatus.RUNNING.value
            health_result = self._execute_health_check_phase(job)

            # Determine final result
            if health_result['passed']:
                job.status = DeploymentStatus.RUNNING.value
                success = True
            else:
                if job.config.auto_rollback:
                    self._execute_rollback(job)
                    job.status = DeploymentStatus.ROLLED_BACK.value
                else:
                    job.status = DeploymentStatus.FAILED.value
                success = False

            # Update job with results
            job.result = DeploymentResult(
                success=success,
                start_time=job.started_at,
                end_time=datetime.now(),
                deployment_time=time.time() - start_time,
                health_check_passed=health_result['passed'],
                rollback_triggered=(job.status == DeploymentStatus.ROLLED_BACK.value),
                metrics={
                    'build_time': deployment_result.get('build_time', 0),
                    'test_time': deployment_result.get('test_time', 0),
                    'deploy_time': deployment_result.get('deploy_time', 0),
                    'health_check_time': health_result.get('check_time', 0)
                }
            )

            job.completed_at = datetime.now()
            job.execution_time = time.time() - start_time

            result.update({
                'success': success,
                'end_time': job.completed_at,
                'execution_time': job.execution_time,
                'deployment_result': job.result.to_dict(),
                'health_check': health_result
            })

            # Update statistics
            self._update_deployment_stats(job, success)

            logger.info(f"Deployment job {job_id} completed with status: {job.status}")

        except Exception as e:
            execution_time = time.time() - start_time
            job.execution_time = execution_time
            job.completed_at = datetime.now()
            job.status = DeploymentStatus.FAILED.value
            job.error_message = str(e)

            result.update({
                'success': False,
                'end_time': job.completed_at,
                'execution_time': execution_time,
                'error': str(e)
            })

            # Update statistics
            self._update_deployment_stats(job, False)

            logger.error(f"Deployment job {job_id} failed: {str(e)}")

        finally:
            # Unlock environment
            self._unlock_environment(job.environment)

        # Clean up
        if job_id in self.active_deployments:
            del self.active_deployments[job_id]

        return result

    def _execute_build_phase(self, job: DeploymentJob) -> Dict[str, Any]:
        """
        Execute build phase of deployment
        执行部署的构建阶段

        Args:
            job: Deployment job
                部署作业

        Returns:
            dict: Build result
                  构建结果
        """
        logger.info(f"Executing build phase for job {job.job_id}")

        build_result = {
            'success': True,
            'build_time': 0.0,
            'artifacts': []
        }

        start_time = time.time()

        try:
            # Create deployment directory
            deployment_dir = self.base_deployment_path / job.environment / job.strategy_id / job.version
            deployment_dir.mkdir(parents=True, exist_ok=True)

            # Copy strategy files (placeholder - in real implementation, this would copy from source)
            source_dir = Path(f"/opt / quant_trading / strategies/{job.strategy_id}/{job.version}")
            if source_dir.exists():
                shutil.copytree(source_dir, deployment_dir, dirs_exist_ok=True)

            # Build artifacts
            artifacts = self._build_artifacts(deployment_dir, job)
            build_result['artifacts'] = artifacts

            build_result['build_time'] = time.time() - start_time

        except Exception as e:
            build_result['success'] = False
            build_result['error'] = str(e)
            logger.error(f"Build phase failed: {str(e)}")
            raise

        return build_result

    def _execute_test_phase(self, job: DeploymentJob) -> Dict[str, Any]:
        """
        Execute test phase of deployment
        执行部署的测试阶段

        Args:
            job: Deployment job
                部署作业

        Returns:
            dict: Test result
                  测试结果
        """
        logger.info(f"Executing test phase for job {job.job_id}")

        test_result = {
            'success': True,
            'test_time': 0.0,
            'tests_passed': 0,
            'tests_failed': 0
        }

        start_time = time.time()

        try:
            # Run automated tests
            deployment_dir = self.base_deployment_path / job.environment / job.strategy_id / job.version

            # Run unit tests
            unit_test_result = self._run_unit_tests(deployment_dir)
            test_result.update(unit_test_result)

            # Run integration tests
            integration_test_result = self._run_integration_tests(deployment_dir)
            test_result.update(integration_test_result)

            # Check test results
            if test_result['tests_failed'] > 0:
                test_result['success'] = False
                raise ValueError(
                    f"Tests failed: {test_result['tests_failed']} failed, {test_result['tests_passed']} passed")

            test_result['test_time'] = time.time() - start_time

        except Exception as e:
            test_result['success'] = False
            test_result['error'] = str(e)
            logger.error(f"Test phase failed: {str(e)}")
            raise

        return test_result

    def _execute_deploy_phase(self, job: DeploymentJob) -> Dict[str, Any]:
        """
        Execute deploy phase of deployment
        执行部署的部署阶段

        Args:
            job: Deployment job
                部署作业

        Returns:
            dict: Deploy result
                  部署结果
        """
        logger.info(f"Executing deploy phase for job {job.job_id}")

        deploy_result = {
            'success': True,
            'deploy_time': 0.0,
            'deployment_type': job.deployment_type
        }

        start_time = time.time()

        try:
            if job.deployment_type == DeploymentType.BLUE_GREEN.value:
                deploy_result.update(self._execute_blue_green_deployment(job))
            elif job.deployment_type == DeploymentType.CANARY.value:
                deploy_result.update(self._execute_canary_deployment(job))
            elif job.deployment_type == DeploymentType.ROLLING.value:
                deploy_result.update(self._execute_rolling_deployment(job))
            elif job.deployment_type == DeploymentType.IMMEDIATE.value:
                deploy_result.update(self._execute_immediate_deployment(job))
            else:
                raise ValueError(f"Unknown deployment type: {job.deployment_type}")

            deploy_result['deploy_time'] = time.time() - start_time

        except Exception as e:
            deploy_result['success'] = False
            deploy_result['error'] = str(e)
            logger.error(f"Deploy phase failed: {str(e)}")
            raise

        return deploy_result

    def _execute_health_check_phase(self, job: DeploymentJob) -> Dict[str, Any]:
        """
        Execute health check phase of deployment
        执行部署的健康检查阶段

        Args:
            job: Deployment job
                部署作业

        Returns:
            dict: Health check result
                  健康检查结果
        """
        logger.info(f"Executing health check phase for job {job.job_id}")

        health_result = {
            'passed': False,
            'check_time': 0.0,
            'checks_performed': [],
            'failures': []
        }

        start_time = time.time()
        timeout_time = start_time + job.config.health_check_timeout

        try:
            while time.time() < timeout_time:
                health_result = self._perform_health_checks(job)

                if health_result['passed']:
                    break

                time.sleep(self.health_check_interval)

            health_result['check_time'] = time.time() - start_time

            if not health_result['passed']:
                raise TimeoutError("Health checks did not pass within timeout period")

        except Exception as e:
            health_result['error'] = str(e)
            logger.error(f"Health check phase failed: {str(e)}")
            raise

        return health_result

    def _execute_blue_green_deployment(self, job: DeploymentJob) -> Dict[str, Any]:
        """
        Execute blue - green deployment
        执行蓝绿部署

        Args:
            job: Deployment job
                部署作业

        Returns:
            dict: Deployment result
                  部署结果
        """
        # Create blue environment
        blue_env = f"{job.environment}_blue"
        green_env = job.environment

        # Deploy to blue environment
        self._deploy_to_environment(job, blue_env)

        # Switch traffic to blue environment
        self._switch_traffic(blue_env, green_env)

        # Keep green as backup
        self.environment_status[green_env]['backup'] = True

        return {
            'blue_environment': blue_env,
            'green_environment': green_env,
            'traffic_switched': True
        }

    def _execute_canary_deployment(self, job: DeploymentJob) -> Dict[str, Any]:
        """
        Execute canary deployment
        执行金丝雀部署

        Args:
            job: Deployment job
                部署作业

        Returns:
            dict: Deployment result
                  部署结果
        """
        traffic_distribution = job.config.traffic_distribution or {'canary': 0.1, 'stable': 0.9}

        # Deploy canary version
        canary_env = f"{job.environment}_canary"
        self._deploy_to_environment(job, canary_env)

        # Gradually increase traffic to canary
        self._gradual_traffic_shift(canary_env, job.environment, traffic_distribution)

        return {
            'canary_environment': canary_env,
            'traffic_distribution': traffic_distribution,
            'gradual_rollout': True
        }

    def _execute_rolling_deployment(self, job: DeploymentJob) -> Dict[str, Any]:
        """
        Execute rolling deployment
        执行滚动部署

        Args:
            job: Deployment job
                部署作业

        Returns:
            dict: Deployment result
                  部署结果
        """
        # Deploy in batches
        batch_size = job.config.resource_limits.get('batch_size', 1)
        total_instances = job.config.resource_limits.get('total_instances', 3)

        deployed_instances = 0

        for batch in range(0, total_instances, batch_size):
            batch_end = min(batch + batch_size, total_instances)

            # Deploy batch
            self._deploy_batch(job, batch, batch_end)
            deployed_instances += (batch_end - batch)

            # Health check batch
            if not self._health_check_batch(job, batch, batch_end):
                raise RuntimeError(f"Batch {batch}-{batch_end} health check failed")

        return {
            'total_instances': total_instances,
            'batch_size': batch_size,
            'batches_deployed': deployed_instances
        }

    def _execute_immediate_deployment(self, job: DeploymentJob) -> Dict[str, Any]:
        """
        Execute immediate deployment
        执行立即部署

        Args:
            job: Deployment job
                部署作业

        Returns:
            dict: Deployment result
                  部署结果
        """
        # Stop current version
        self._stop_current_deployment(job.environment)

        # Deploy new version
        self._deploy_to_environment(job, job.environment)

        # Start new version
        self._start_deployment(job.environment)

        return {
            'immediate_shutdown': True,
            'downtime_expected': True
        }

    def _deploy_to_environment(self, job: DeploymentJob, target_env: str) -> None:
        """
        Deploy job to target environment
        将作业部署到目标环境

        Args:
            job: Deployment job
                部署作业
            target_env: Target environment
                       目标环境
        """
        source_dir = self.base_deployment_path / job.environment / job.strategy_id / job.version
        target_dir = self.base_deployment_path / target_env / job.strategy_id / job.version

        if source_dir.exists():
            shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

        # Update environment status
        self.environment_status[target_env].update({
            'strategy_id': job.strategy_id,
            'version': job.version,
            'status': 'deployed',
            'deployed_at': datetime.now()
        })

    def _switch_traffic(self, from_env: str, to_env: str) -> None:
        """
        Switch traffic between environments
        在环境之间切换流量

        Args:
            from_env: Source environment
                     源环境
            to_env: Target environment
                   目标环境
        """
        # Placeholder for traffic switching logic
        # In real implementation, this would update load balancers, DNS, etc.
        logger.info(f"Switching traffic from {from_env} to {to_env}")

    def _gradual_traffic_shift(self, from_env: str, to_env: str, distribution: Dict[str, float]) -> None:
        """
        Gradually shift traffic between environments
        在环境之间逐渐转移流量

        Args:
            from_env: Source environment
                     源环境
            to_env: Target environment
                   目标环境
            distribution: Traffic distribution
                         流量分配
        """
        # Placeholder for gradual traffic shifting
        logger.info(f"Gradually shifting traffic: {distribution}")

    def _deploy_batch(self, job: DeploymentJob, start_idx: int, end_idx: int) -> None:
        """
        Deploy a batch of instances
        部署一批实例

        Args:
            job: Deployment job
                部署作业
            start_idx: Start index
                      开始索引
            end_idx: End index
                    结束索引
        """
        logger.info(f"Deploying batch {start_idx}-{end_idx}")

    def _health_check_batch(self, job: DeploymentJob, start_idx: int, end_idx: int) -> bool:
        """
        Health check a batch of instances
        对一批实例进行健康检查

        Args:
            job: Deployment job
                部署作业
            start_idx: Start index
                      开始索引
            end_idx: End index
                    结束索引

        Returns:
            bool: True if healthy
                  健康则返回True
        """
        # Placeholder health check
        return True

    def _stop_current_deployment(self, environment: str) -> None:
        """
        Stop current deployment in environment
        停止环境中的当前部署

        Args:
            environment: Target environment
                        目标环境
        """
        # Placeholder for stopping deployment
        logger.info(f"Stopping current deployment in {environment}")

    def _start_deployment(self, environment: str) -> None:
        """
        Start deployment in environment
        在环境中启动部署

        Args:
            environment: Target environment
                        目标环境
        """
        # Placeholder for starting deployment
        logger.info(f"Starting deployment in {environment}")

    def _build_artifacts(self, deployment_dir: Path, job: DeploymentJob) -> List[str]:
        """
        Build deployment artifacts
        构建部署工件

        Args:
            deployment_dir: Deployment directory
                          部署目录
            job: Deployment job
                部署作业

        Returns:
            list: List of built artifacts
                  构建的工件列表
        """
        artifacts = []

        # Build Python package
        if (deployment_dir / "setup.py").exists():
            try:
                subprocess.run(
                    ["python", "setup.py", "bdist_wheel"],
                    cwd=deployment_dir,
                    check=True,
                    capture_output=True
                )
                artifacts.append("wheel")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to build wheel: {e}")

        # Build Docker image (placeholder)
        if (deployment_dir / "Dockerfile").exists():
            artifacts.append("docker_image")

        return artifacts

    def _run_unit_tests(self, deployment_dir: Path) -> Dict[str, Any]:
        """
        Run unit tests
        运行单元测试

        Args:
            deployment_dir: Deployment directory
                          部署目录

        Returns:
            dict: Test results
                  测试结果
        """
        result = {'tests_passed': 0, 'tests_failed': 0}

        # Run pytest if available
        try:
            if (deployment_dir / "tests").exists():
                result = subprocess.run(
                    ["python", "-m", "pytest", "tests/", "--tb=short"],
                    cwd=deployment_dir,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                # Parse results (simplified)
                if result.returncode == 0:
                    result = {'tests_passed': 10, 'tests_failed': 0}  # Placeholder
                else:
                    result = {'tests_passed': 8, 'tests_failed': 2}   # Placeholder

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            result = {'tests_passed': 0, 'tests_failed': 1}

        return result

    def _run_integration_tests(self, deployment_dir: Path) -> Dict[str, Any]:
        """
        Run integration tests
        运行集成测试

        Args:
            deployment_dir: Deployment directory
                          部署目录

        Returns:
            dict: Test results
                  测试结果
        """
        # Placeholder for integration tests
        return {'integration_tests_passed': 5, 'integration_tests_failed': 0}

    def _perform_health_checks(self, job: DeploymentJob) -> Dict[str, Any]:
        """
        Perform health checks on deployment
        对部署执行健康检查

        Args:
            job: Deployment job
                部署作业

        Returns:
            dict: Health check results
                  健康检查结果
        """
        checks = []
        failures = []

        # Check if service is running
        if not self._check_service_running(job.environment):
            failures.append("Service not running")

        # Check response time
        response_time = self._check_response_time(job.environment)
        if response_time > 5.0:  # 5 seconds threshold
            failures.append(f"Response time too slow: {response_time}s")

        # Check error rate
        error_rate = self._check_error_rate(job.environment)
        if error_rate > 0.05:  # 5% threshold
            failures.append(f"Error rate too high: {error_rate * 100}%")

        checks.extend([
            "service_running",
            "response_time",
            "error_rate"
        ])

        return {
            'passed': len(failures) == 0,
            'checks_performed': checks,
            'failures': failures,
            'check_time': time.time()
        }

    def _check_service_running(self, environment: str) -> bool:
        """
        Check if service is running
        检查服务是否正在运行

        Args:
            environment: Target environment
                        目标环境

        Returns:
            bool: True if running
                  运行中则返回True
        """
        # Placeholder check
        return True

    def _check_response_time(self, environment: str) -> float:
        """
        Check service response time
        检查服务响应时间

        Args:
            environment: Target environment
                        目标环境

        Returns:
            float: Response time in seconds
                   响应时间（秒）
        """
        # Placeholder check
        return 2.5

    def _check_error_rate(self, environment: str) -> float:
        """
        Check service error rate
        检查服务错误率

        Args:
            environment: Target environment
                        目标环境

        Returns:
            float: Error rate (0.0 to 1.0)
                   错误率（0.0到1.0）
        """
        # Placeholder check
        return 0.02

    def _execute_rollback(self, job: DeploymentJob) -> None:
        """
        Execute rollback to previous version
        执行回滚到以前版本

        Args:
            job: Deployment job
                部署作业
        """
        logger.info(f"Executing rollback for job {job.job_id}")

        # Find previous version
        prev_version = self._get_previous_version(job.environment, job.strategy_id)

        if prev_version:
            # Deploy previous version
            rollback_job = DeploymentJob(
                job_id=f"rollback_{job.job_id}",
                strategy_id=job.strategy_id,
                version=prev_version,
                environment=job.environment,
                deployment_type=DeploymentType.IMMEDIATE.value,
                config=job.config,
                status=DeploymentStatus.DEPLOYING.value,
                created_at=datetime.now(),
                metadata={'rollback_from': job.version}
            )

            self._execute_immediate_deployment(rollback_job)
            logger.info(f"Rolled back to version {prev_version}")
        else:
            logger.error("No previous version found for rollback")

    def _get_previous_version(self, environment: str, strategy_id: str) -> Optional[str]:
        """
        Get previous version for environment
        获取环境的以前版本

        Args:
            environment: Target environment
                        目标环境
            strategy_id: Strategy identifier
                        策略标识符

        Returns:
            str: Previous version or None
                 以前版本或None
        """
        env_status = self.environment_status.get(environment, {})
        return env_status.get('previous_version')

    def _lock_environment(self, environment: str) -> None:
        """
        Lock environment for deployment
        锁定环境以进行部署

        Args:
            environment: Environment to lock
                        要锁定的环境
        """
        self.environment_status[environment]['locked'] = True
        self.environment_status[environment]['locked_at'] = datetime.now()

    def _unlock_environment(self, environment: str) -> None:
        """
        Unlock environment after deployment
        部署后解锁环境

        Args:
            environment: Environment to unlock
                        要解锁的环境
        """
        self.environment_status[environment]['locked'] = False

    def _is_environment_locked(self, environment: str) -> bool:
        """
        Check if environment is locked
        检查环境是否被锁定

        Args:
            environment: Environment to check
                        要检查的环境

        Returns:
            bool: True if locked
                  锁定则返回True
        """
        return self.environment_status.get(environment, {}).get('locked', False)

    def get_deployment_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get deployment job status
        获取部署作业状态

        Args:
            job_id: Job identifier
                   作业标识符

        Returns:
            dict: Job status or None if not found
                  作业状态，如果未找到则返回None
        """
        if job_id in self.deployment_jobs:
            return self.deployment_jobs[job_id].to_dict()
        return None

    def cancel_deployment(self, job_id: str) -> bool:
        """
        Cancel a running deployment
        取消正在运行的部署

        Args:
            job_id: Job identifier
                   作业标识符

        Returns:
            bool: True if cancelled successfully
                  取消成功返回True
        """
        if job_id in self.deployment_jobs:
            job = self.deployment_jobs[job_id]
            if job.status in [DeploymentStatus.BUILDING.value,
                              DeploymentStatus.TESTING.value,
                              DeploymentStatus.DEPLOYING.value]:
                job.status = DeploymentStatus.CANCELLED.value
                job.completed_at = datetime.now()
                logger.info(f"Cancelled deployment job: {job_id}")
                return True
        return False

    def list_deployment_jobs(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List deployment jobs with optional status filter
        列出部署作业，可选状态过滤

        Args:
            status_filter: Status to filter by (optional)
                          要过滤的状态（可选）

        Returns:
            list: List of deployment jobs
                  部署作业列表
        """
        jobs = []
        for job in self.deployment_jobs.values():
            if status_filter is None or job.status == status_filter:
                jobs.append(job.to_dict())
        return jobs

    def get_environment_status(self, environment: str) -> Dict[str, Any]:
        """
        Get environment status
        获取环境状态

        Args:
            environment: Environment name
                        环境名称

        Returns:
            dict: Environment status
                  环境状态
        """
        return self.environment_status.get(environment, {})

    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get deployment engine statistics
        获取部署引擎统计信息

        Returns:
            dict: Engine statistics
                  引擎统计信息
        """
        return {
            'engine_name': self.engine_name,
            'total_jobs': len(self.deployment_jobs),
            'active_deployments': len(self.active_deployments),
            'stats': self.stats
        }

    def _update_deployment_stats(self, job: DeploymentJob, success: bool) -> None:
        """
        Update deployment statistics
        更新部署统计信息

        Args:
            job: Deployment job
                部署作业
            success: Whether deployment was successful
                    部署是否成功
        """
        self.stats['total_deployments'] += 1

        if success:
            self.stats['successful_deployments'] += 1
        else:
            self.stats['failed_deployments'] += 1

        # Update average deployment time
        total_completed = self.stats['successful_deployments'] + self.stats['failed_deployments']
        current_avg = self.stats['average_deployment_time']
        new_time = job.execution_time
        self.stats['average_deployment_time'] = (
            (current_avg * (total_completed - 1)) + new_time
        ) / total_completed

        # Update rollback rate
        if job.result and job.result.rollback_triggered:
            rollback_count = sum(1 for j in self.deployment_jobs.values()
                                 if j.result and j.result.rollback_triggered)
            self.stats['rollback_rate'] = rollback_count / self.stats['total_deployments']


# Global deployment engine instance
# 全局部署引擎实例
deployment_engine = DeploymentEngine()

__all__ = [
    'DeploymentStatus',
    'DeploymentType',
    'Environment',
    'DeploymentConfig',
    'DeploymentResult',
    'DeploymentJob',
    'DeploymentEngine',
    'deployment_engine'
]
