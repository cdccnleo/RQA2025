"""
Data Pipeline Automation Module
数据管道自动化模块

This module provides automated data pipeline capabilities for quantitative trading
此模块为量化交易提供自动化数据管道能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import threading
import time

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):

    """Data pipeline status"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class PipelineStep(Enum):

    """Pipeline step types"""
    EXTRACT = "extract"
    TRANSFORM = "transform"
    LOAD = "load"
    VALIDATE = "validate"
    CLEAN = "clean"


@dataclass
class PipelineMetrics:

    """
    Pipeline metrics data class
    管道指标数据类
    """
    pipeline_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    records_processed: int = 0
    records_successful: int = 0
    records_failed: int = 0
    processing_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


class DataPipelineStep:

    """
    Data Pipeline Step Class
    数据管道步骤类

    Represents a single step in a data pipeline
    表示数据管道中的单个步骤
    """

    def __init__(self,


                 step_id: str,
                 step_type: PipelineStep,
                 name: str,
                 config: Dict[str, Any],
                 dependencies: Optional[List[str]] = None):
        """
        Initialize pipeline step
        初始化管道步骤

        Args:
            step_id: Unique step identifier
                    唯一步骤标识符
            step_type: Type of pipeline step
                      管道步骤类型
            name: Human - readable step name
                 人类可读的步骤名称
            config: Step configuration
                   步骤配置
            dependencies: List of step IDs this step depends on
                        此步骤依赖的步骤ID列表
        """
        self.step_id = step_id
        self.step_type = step_type
        self.name = name
        self.config = config
        self.dependencies = dependencies or []

        # Runtime state
        self.status = PipelineStatus.IDLE
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.result: Any = None
        self.error: Optional[Exception] = None
        self.records_processed = 0
        self.records_successful = 0
        self.records_failed = 0

        # Performance tracking
        self.processing_time = 0.0

    def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute the pipeline step
        执行管道步骤

        Args:
            input_data: Input data for the step
                       步骤的输入数据

        Returns:
            dict: Execution result
                  执行结果
        """
        self.status = PipelineStatus.RUNNING
        self.start_time = datetime.now()

        result = {
            'step_id': self.step_id,
            'step_type': self.step_type.value,
            'step_name': self.name,
            'status': 'running',
            'start_time': self.start_time,
            'records_processed': 0,
            'records_successful': 0,
            'records_failed': 0,
            'processing_time': 0.0
        }

        start_time = time.time()

        try:
            if self.step_type == PipelineStep.EXTRACT:
                output_data = self._execute_extract()
            elif self.step_type == PipelineStep.TRANSFORM:
                output_data = self._execute_transform(input_data)
            elif self.step_type == PipelineStep.LOAD:
                output_data = self._execute_load(input_data)
            elif self.step_type == PipelineStep.VALIDATE:
                output_data = self._execute_validate(input_data)
            elif self.step_type == PipelineStep.CLEAN:
                output_data = self._execute_clean(input_data)
            else:
                raise ValueError(f"Unknown step type: {self.step_type}")

            execution_time = time.time() - start_time
            self.processing_time = execution_time

            self.status = PipelineStatus.COMPLETED
            self.end_time = datetime.now()

            result.update({
                'status': 'completed',
                'end_time': self.end_time,
                'processing_time': execution_time,
                'output_data': output_data
            })

            logger.info(f"Pipeline step {self.step_id} completed successfully")

        except Exception as e:
            execution_time = time.time() - start_time
            self.processing_time = execution_time

            self.status = PipelineStatus.FAILED
            self.error = e
            self.end_time = datetime.now()

            result.update({
                'status': 'failed',
                'end_time': self.end_time,
                'processing_time': execution_time,
                'error': str(e)
            })

            logger.error(f"Pipeline step {self.step_id} failed: {str(e)}")

        return result

    def _execute_extract(self) -> Any:
        """
        Execute data extraction
        执行数据提取

        Returns:
            Extracted data
            提取的数据
        """
        source_type = self.config.get('source_type', 'database')
        source_config = self.config.get('source_config', {})

        if source_type == 'database':
            return self._extract_from_database(source_config)
        elif source_type == 'api':
            return self._extract_from_api(source_config)
        elif source_type == 'file':
            return self._extract_from_file(source_config)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    def _execute_transform(self, input_data: Any) -> Any:
        """
        Execute data transformation
        执行数据转换

        Args:
            input_data: Input data to transform
                       要转换的输入数据

        Returns:
            Transformed data
            转换后的数据
        """
        transformations = self.config.get('transformations', [])

        transformed_data = input_data

        for transformation in transformations:
            transform_type = transformation.get('type', '')

            if transform_type == 'filter':
                transformed_data = self._apply_filter(transformed_data, transformation)
            elif transform_type == 'aggregate':
                transformed_data = self._apply_aggregation(transformed_data, transformation)
            elif transform_type == 'normalize':
                transformed_data = self._apply_normalization(transformed_data, transformation)
            elif transform_type == 'enrich':
                transformed_data = self._apply_enrichment(transformed_data, transformation)

        return transformed_data

    def _execute_load(self, input_data: Any) -> Any:
        """
        Execute data loading
        执行数据加载

        Args:
            input_data: Input data to load
                       要加载的输入数据

        Returns:
            Load result
            加载结果
        """
        destination_type = self.config.get('destination_type', 'database')
        destination_config = self.config.get('destination_config', {})

        if destination_type == 'database':
            return self._load_to_database(input_data, destination_config)
        elif destination_type == 'file':
            return self._load_to_file(input_data, destination_config)
        elif destination_type == 'stream':
            return self._load_to_stream(input_data, destination_config)
        else:
            raise ValueError(f"Unsupported destination type: {destination_type}")

    def _execute_validate(self, input_data: Any) -> Any:
        """
        Execute data validation
        执行数据验证

        Args:
            input_data: Input data to validate
                       要验证的输入数据

        Returns:
            Validation result
            验证结果
        """
        validation_rules = self.config.get('validation_rules', [])

        validation_result = {
            'is_valid': True,
            'total_records': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'validation_errors': []
        }

        # Implement validation logic based on rules
        for rule in validation_rules:
            rule_type = rule.get('type', '')
            field = rule.get('field', '')

            # Apply validation rule
            if rule_type == 'not_null':
                self._validate_not_null(input_data, field, validation_result)
            elif rule_type == 'range':
                self._validate_range(input_data, field, rule, validation_result)
            elif rule_type == 'format':
                self._validate_format(input_data, field, rule, validation_result)

        return validation_result

    def _execute_clean(self, input_data: Any) -> Any:
        """
        Execute data cleaning
        执行数据清理

        Args:
            input_data: Input data to clean
                       要清理的输入数据

        Returns:
            Cleaned data
            清理后的数据
        """
        cleaning_rules = self.config.get('cleaning_rules', [])

        cleaned_data = input_data

        for rule in cleaning_rules:
            rule_type = rule.get('type', '')

            if rule_type == 'remove_duplicates':
                cleaned_data = self._remove_duplicates(cleaned_data, rule)
            elif rule_type == 'fill_missing':
                cleaned_data = self._fill_missing_values(cleaned_data, rule)
            elif rule_type == 'remove_outliers':
                cleaned_data = self._remove_outliers(cleaned_data, rule)

        return cleaned_data

    def _extract_from_database(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Extract data from database"""
        # Placeholder implementation
        logger.info("Extracting data from database")
        return pd.DataFrame()

    def _extract_from_api(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from API"""
        # Placeholder implementation
        logger.info("Extracting data from API")
        return {}

    def _extract_from_file(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Extract data from file"""
        # Placeholder implementation
        logger.info("Extracting data from file")
        return pd.DataFrame()

    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering transformation"""
        # Placeholder implementation
        logger.info("Applying filter transformation")
        return data

    def _apply_aggregation(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply aggregation transformation"""
        # Placeholder implementation
        logger.info("Applying aggregation transformation")
        return data

    def _apply_normalization(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply normalization transformation"""
        # Placeholder implementation
        logger.info("Applying normalization transformation")
        return data

    def _apply_enrichment(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply enrichment transformation"""
        # Placeholder implementation
        logger.info("Applying enrichment transformation")
        return data

    def _load_to_database(self, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load data to database"""
        # Placeholder implementation
        logger.info("Loading data to database")
        return {'status': 'success'}

    def _load_to_file(self, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load data to file"""
        # Placeholder implementation
        logger.info("Loading data to file")
        return {'status': 'success'}

    def _load_to_stream(self, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load data to stream"""
        # Placeholder implementation
        logger.info("Loading data to stream")
        return {'status': 'success'}

    def _validate_not_null(self, data: Any, field: str, result: Dict[str, Any]) -> None:
        """Validate not null constraint"""
        # Placeholder implementation
        logger.info(f"Validating not null for field: {field}")

    def _validate_range(self, data: Any, field: str, rule: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Validate range constraint"""
        # Placeholder implementation
        logger.info(f"Validating range for field: {field}")

    def _validate_format(self, data: Any, field: str, rule: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Validate format constraint"""
        # Placeholder implementation
        logger.info(f"Validating format for field: {field}")

    def _remove_duplicates(self, data: Any, config: Dict[str, Any]) -> Any:
        """Remove duplicate records"""
        # Placeholder implementation
        logger.info("Removing duplicates")
        return data

    def _fill_missing_values(self, data: Any, config: Dict[str, Any]) -> Any:
        """Fill missing values"""
        # Placeholder implementation
        logger.info("Filling missing values")
        return data

    def _remove_outliers(self, data: Any, config: Dict[str, Any]) -> Any:
        """Remove outlier records"""
        # Placeholder implementation
        logger.info("Removing outliers")
        return data

    def get_step_status(self) -> Dict[str, Any]:
        """
        Get step execution status
        获取步骤执行状态

        Returns:
            dict: Step status
                  步骤状态
        """
        return {
            'step_id': self.step_id,
            'step_type': self.step_type.value,
            'name': self.name,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'processing_time': self.processing_time,
            'records_processed': self.records_processed,
            'records_successful': self.records_successful,
            'records_failed': self.records_failed,
            'success_rate': self.records_successful / max(self.records_processed, 1) * 100
        }


class DataPipeline:

    """
    Data Pipeline Class
    数据管道类

    Represents a complete data processing pipeline
    表示完整的数据处理管道
    """

    def __init__(self,


                 pipeline_id: str,
                 name: str,
                 description: str = "",
                 steps: Optional[Dict[str, DataPipelineStep]] = None):
        """
        Initialize data pipeline
        初始化数据管道

        Args:
            pipeline_id: Unique pipeline identifier
                        唯一管道标识符
            name: Human - readable pipeline name
                 人类可读的管道名称
            description: Pipeline description
                        管道描述
            steps: Dictionary of pipeline steps
                  管道步骤字典
        """
        self.pipeline_id = pipeline_id
        self.name = name
        self.description = description
        self.steps = steps or {}

        # Pipeline state
        self.status = PipelineStatus.IDLE
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.execution_time = 0.0

        # Execution tracking
        self.completed_steps = set()
        self.failed_steps = set()
        self.running_steps = set()

        # Results storage
        self.step_results: Dict[str, Dict[str, Any]] = {}

    def add_step(self, step: DataPipelineStep) -> None:
        """
        Add a step to the pipeline
        将步骤添加到管道中

        Args:
            step: Step to add
                 要添加的步骤
        """
        self.steps[step.step_id] = step
        logger.info(f"Added step {step.step_id} to pipeline {self.pipeline_id}")

    def remove_step(self, step_id: str) -> bool:
        """
        Remove a step from the pipeline
        从管道中移除步骤

        Args:
            step_id: Step identifier
                    步骤标识符

        Returns:
            bool: True if removed successfully, False otherwise
                  移除成功返回True，否则返回False
        """
        if step_id in self.steps:
            del self.steps[step_id]
            logger.info(f"Removed step {step_id} from pipeline {self.pipeline_id}")
            return True
        return False

    def get_executable_steps(self) -> List[str]:
        """
        Get steps that are ready to execute
        获取准备执行的步骤

        Returns:
            list: List of executable step IDs
                  可执行步骤ID列表
        """
        executable = []

        for step_id, step in self.steps.items():
            if step.status == PipelineStatus.IDLE:
                # Check if all dependencies are completed
                deps_satisfied = all(
                    dep_step_id in self.completed_steps
                    for dep_step_id in step.dependencies
                )

                if deps_satisfied:
                    executable.append(step_id)

        return executable

    def validate_pipeline(self) -> Dict[str, Any]:
        """
        Validate pipeline structure and dependencies
        验证管道结构和依赖关系

        Returns:
            dict: Validation results
                  验证结果
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Check for missing dependencies
        for step_id, step in self.steps.items():
            for dep in step.dependencies:
                if dep not in self.steps:
                    validation_result['errors'].append(
                        f"Step {step_id} depends on non - existent step {dep}"
                    )
                    validation_result['valid'] = False

        # Check for circular dependencies
        try:
            import networkx as nx
            graph = nx.DiGraph()

            for step_id in self.steps:
                graph.add_node(step_id)

            for step_id, step in self.steps.items():
                for dep in step.dependencies:
                    if dep in self.steps:
                        graph.add_edge(dep, step_id)

            cycles = list(nx.simple_cycles(graph))
            if cycles:
                validation_result['errors'].append(f"Circular dependencies detected: {cycles}")
                validation_result['valid'] = False
        except ImportError:
            validation_result['warnings'].append("NetworkX not available for dependency validation")

        return validation_result

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline status
        获取全面的管道状态

        Returns:
            dict: Pipeline status information
                  管道状态信息
        """
        total_steps = len(self.steps)
        completed_steps = sum(1 for s in self.steps.values()
                              if s.status == PipelineStatus.COMPLETED)
        failed_steps = sum(1 for s in self.steps.values() if s.status == PipelineStatus.FAILED)
        running_steps = sum(1 for s in self.steps.values() if s.status == PipelineStatus.RUNNING)

        return {
            'pipeline_id': self.pipeline_id,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'execution_time': self.execution_time,
            'total_steps': total_steps,
            'completed_steps': completed_steps,
            'failed_steps': failed_steps,
            'running_steps': running_steps,
            'progress_percentage': (completed_steps / max(total_steps, 1)) * 100
        }


class DataPipelineManager:

    """
    Data Pipeline Manager Class
    数据管道管理器类

    Manages the execution of data pipelines
    管理数据管道的执行
    """

    def __init__(self, manager_name: str = "default_data_pipeline_manager"):
        """
        Initialize data pipeline manager
        初始化数据管道管理器

        Args:
            manager_name: Name of the manager
                        管理器名称
        """
        self.manager_name = manager_name
        self.pipelines: Dict[str, DataPipeline] = {}
        self.active_pipelines: Dict[str, threading.Thread] = {}

        # Manager settings
        self.max_concurrent_pipelines = 5
        self.pipeline_check_interval = 10  # seconds

        # Statistics
        self.stats = {
            'total_pipelines': 0,
            'completed_pipelines': 0,
            'failed_pipelines': 0,
            'average_execution_time': 0.0
        }

        logger.info(f"Data pipeline manager {manager_name} initialized")

    def create_pipeline(self,


                        pipeline_id: str,
                        name: str,
                        description: str = "") -> str:
        """
        Create a new data pipeline
        创建新的数据管道

        Args:
            pipeline_id: Unique pipeline identifier
                        唯一管道标识符
            name: Pipeline name
                 管道名称
            description: Pipeline description
                        管道描述

        Returns:
            str: Created pipeline ID
                 创建的管道ID
        """
        pipeline = DataPipeline(pipeline_id, name, description)
        self.pipelines[pipeline_id] = pipeline
        self.stats['total_pipelines'] = len(self.pipelines)
        logger.info(f"Created pipeline: {name} ({pipeline_id})")
        return pipeline_id

    def add_step_to_pipeline(self,


                             pipeline_id: str,
                             step: DataPipelineStep) -> bool:
        """
        Add a step to an existing pipeline
        将步骤添加到现有管道中

        Args:
            pipeline_id: Pipeline identifier
                        管道标识符
            step: Step to add
                 要添加的步骤

        Returns:
            bool: True if added successfully, False otherwise
                  添加成功返回True，否则返回False
        """
        if pipeline_id not in self.pipelines:
            logger.error(f"Pipeline {pipeline_id} not found")
            return False

        self.pipelines[pipeline_id].add_step(step)
        return True

    def execute_pipeline(self,


                         pipeline_id: str,
                         async_execution: bool = True) -> Dict[str, Any]:
        """
        Execute a data pipeline
        执行数据管道

        Args:
            pipeline_id: Pipeline identifier
                        管道标识符
            async_execution: Whether to execute asynchronously
                           是否异步执行

        Returns:
            dict: Execution result
                  执行结果
        """
        if pipeline_id not in self.pipelines:
            return {'success': False, 'error': f'Pipeline {pipeline_id} not found'}

        pipeline = self.pipelines[pipeline_id]

        # Validate pipeline before execution
        validation = pipeline.validate_pipeline()
        if not validation['valid']:
            return {
                'success': False,
                'error': 'Pipeline validation failed',
                'validation_errors': validation['errors']
            }

        # Check concurrent pipeline limit
        if len(self.active_pipelines) >= self.max_concurrent_pipelines:
            return {
                'success': False,
                'error': 'Maximum concurrent pipelines reached'
            }

        if async_execution:
            # Start async execution
            execution_thread = threading.Thread(
                target=self._execute_pipeline_sync,
                args=(pipeline_id,),
                daemon=True
            )
            self.active_pipelines[pipeline_id] = execution_thread
            execution_thread.start()

            return {
                'success': True,
                'execution_mode': 'async',
                'pipeline_id': pipeline_id
            }
        else:
            # Execute synchronously
            return self._execute_pipeline_sync(pipeline_id)

    def _execute_pipeline_sync(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Execute pipeline synchronously
        同步执行管道

        Args:
            pipeline_id: Pipeline identifier
                    管道标识符

        Returns:
            dict: Execution result
                  执行结果
        """
        pipeline = self.pipelines[pipeline_id]
        pipeline.status = PipelineStatus.RUNNING
        pipeline.started_at = datetime.now()

        execution_result = {
            'pipeline_id': pipeline_id,
            'success': True,
            'start_time': pipeline.started_at,
            'executed_steps': [],
            'failed_steps': [],
            'execution_time': 0.0
        }

        start_time = time.time()
        step_outputs = {}  # Store outputs for dependent steps

        try:
            while True:
                # Get executable steps
                executable_steps = pipeline.get_executable_steps()

                if not executable_steps:
                    # Check if pipeline is complete
                    if all(step.status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED]
                           for step in pipeline.steps.values()):
                        break

                    # Wait for running steps to complete
                    time.sleep(1)
                    continue

                # Execute steps
                for step_id in executable_steps:
                    step = pipeline.steps[step_id]

                    # Get input data from dependencies
                    input_data = None
                    if step.dependencies:
                        # Use output from first dependency (for now)
                        dep_id = step.dependencies[0]
                        if dep_id in step_outputs:
                            input_data = step_outputs[dep_id]

                    # Execute step
                    step_result = step.execute(input_data)

                    # Store output for dependent steps
                    if step_result.get('status') == 'completed':
                        step_outputs[step_id] = step_result.get('output_data')
                        pipeline.completed_steps.add(step_id)
                        execution_result['executed_steps'].append(step_result)
                    else:
                        pipeline.failed_steps.add(step_id)
                        execution_result['failed_steps'].append(step_result)

            # Determine final pipeline status
            if pipeline.failed_steps:
                pipeline.status = PipelineStatus.FAILED
                execution_result['success'] = False
            else:
                pipeline.status = PipelineStatus.COMPLETED

            pipeline.completed_at = datetime.now()
            pipeline.execution_time = (pipeline.completed_at - pipeline.started_at).total_seconds()
            execution_result['execution_time'] = pipeline.execution_time
            execution_result['end_time'] = pipeline.completed_at

        except Exception as e:
            pipeline.status = PipelineStatus.FAILED
            execution_result['success'] = False
            execution_result['error'] = str(e)
            logger.error(f"Pipeline execution failed: {str(e)}")

        # Update statistics
        self._update_execution_stats(
            execution_result['success'], execution_result['execution_time'])

        # Clean up
        if pipeline_id in self.active_pipelines:
            del self.active_pipelines[pipeline_id]

        return execution_result

    def cancel_pipeline(self, pipeline_id: str) -> bool:
        """
        Cancel a running pipeline
        取消正在运行的管道

        Args:
            pipeline_id: Pipeline identifier
                    管道标识符

        Returns:
            bool: True if cancelled successfully, False otherwise
                  取消成功返回True，否则返回False
        """
        if pipeline_id not in self.pipelines:
            return False

        pipeline = self.pipelines[pipeline_id]

        if pipeline.status == PipelineStatus.RUNNING:
            pipeline.status = PipelineStatus.CANCELLED

            # Cancel running steps
            for step in pipeline.steps.values():
                if step.status == PipelineStatus.RUNNING:
                    step.status = PipelineStatus.CANCELLED

            logger.info(f"Cancelled pipeline: {pipeline_id}")
            return True

        return False

    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific pipeline
        获取特定管道的状态

        Args:
            pipeline_id: Pipeline identifier
                        管道标识符

        Returns:
            dict: Pipeline status or None if not found
                  管道状态，如果未找到则返回None
        """
        if pipeline_id not in self.pipelines:
            return None

        return self.pipelines[pipeline_id].get_pipeline_status()

    def list_pipelines(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all pipelines with optional status filter
        列出所有管道，可选状态过滤

        Args:
            status_filter: Status to filter by (optional)
                          要过滤的状态（可选）

        Returns:
            list: List of pipeline summaries
                  管道摘要列表
        """
        pipelines = []

        for pipeline_id, pipeline in self.pipelines.items():
            status = pipeline.get_pipeline_status()

            if status_filter is None or status['status'] == status_filter:
                pipelines.append({
                    'pipeline_id': pipeline_id,
                    'name': pipeline.name,
                    'status': status['status'],
                    'created_at': pipeline.created_at.isoformat(),
                    'execution_time': pipeline.execution_time,
                    'progress_percentage': status['progress_percentage']
                })

        return pipelines

    def get_manager_stats(self) -> Dict[str, Any]:
        """
        Get pipeline manager statistics
        获取管道管理器统计信息

        Returns:
            dict: Manager statistics
                  管理器统计信息
        """
        return {
            'manager_name': self.manager_name,
            'total_pipelines': len(self.pipelines),
            'active_pipelines': len(self.active_pipelines),
            'max_concurrent_pipelines': self.max_concurrent_pipelines,
            'stats': self.stats
        }

    def _update_execution_stats(self, success: bool, execution_time: float) -> None:
        """
        Update execution statistics
        更新执行统计信息

        Args:
            success: Whether execution was successful
                    执行是否成功
            execution_time: Execution time
                           执行时间
        """
        if success:
            self.stats['completed_pipelines'] += 1
        else:
            self.stats['failed_pipelines'] += 1

        # Update average execution time
        total_completed = self.stats['completed_pipelines'] + self.stats['failed_pipelines']
        current_avg = self.stats['average_execution_time']
        self.stats['average_execution_time'] = (
            (current_avg * (total_completed - 1)) + execution_time
        ) / total_completed


# Global data pipeline manager instance
# 全局数据管道管理器实例
data_pipeline_manager = DataPipelineManager()

__all__ = [
    'PipelineStatus',
    'PipelineStep',
    'PipelineMetrics',
    'DataPipelineStep',
    'DataPipeline',
    'DataPipelineManager',
    'data_pipeline_manager'
]
