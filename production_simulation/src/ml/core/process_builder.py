#!/usr/bin/env python3
"""精简版流程构建器，满足单元测试覆盖需求。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from .process_orchestrator import (
    MLProcess,
    MLProcessType,
    ProcessPriority,
    ProcessStep,
    get_ml_process_orchestrator,
)


def register_ml_step_executors(*args, **kwargs):
    """兼容旧接口，测试环境会对其进行 monkeypatch。"""
    return None


class MLProcessBuilder:
    def __init__(self):
        self.orchestrator = get_ml_process_orchestrator()
        self.steps: Dict[str, ProcessStep] = {}
        self.process_config: Dict[str, Any] = {}
        self.templates = self._default_templates()

    # ------------------------------------------------------------------ #
    # 模板定义
    # ------------------------------------------------------------------ #
    def _default_templates(self) -> Dict[str, Dict[str, Any]]:
        return {
            "basic_training": {
                "name": "基础模型训练流程",
                "type": MLProcessType.MODEL_TRAINING,
                "steps": {
                    "load_data": {
                        "step_name": "数据加载",
                        "step_type": "data_loading",
                        "config": {"data_source": "file"},
                    },
                    "feature_engineering": {
                        "step_name": "特征工程",
                        "step_type": "feature_engineering",
                        "dependencies": ["load_data"],
                        "config": {"scaling_method": "standard"},
                    },
                    "train_model": {
                        "step_name": "模型训练",
                        "step_type": "model_training",
                        "dependencies": ["feature_engineering"],
                        "config": {"model_type": "random_forest"},
                    },
                    "evaluate_model": {
                        "step_name": "模型评估",
                        "step_type": "model_evaluation",
                        "dependencies": ["train_model"],
                        "config": {"metrics": ["accuracy"]},
                    },
                },
            },
            "batch_prediction": {
                "name": "批量预测流程",
                "type": MLProcessType.BATCH_PROCESSING,
                "steps": {
                    "load_prediction_data": {
                        "step_name": "预测数据加载",
                        "step_type": "data_loading",
                        "config": {"data_source": "file"},
                    },
                    "feature_engineering": {
                        "step_name": "特征工程",
                        "step_type": "feature_engineering",
                        "dependencies": ["load_prediction_data"],
                        "config": {},
                    },
                    "predict": {
                        "step_name": "模型预测",
                        "step_type": "model_prediction",
                        "dependencies": ["feature_engineering"],
                        "config": {"output_format": "dataframe"},
                    },
                },
            },
        }

    # ------------------------------------------------------------------ #
    # 构建流程
    # ------------------------------------------------------------------ #
    def from_template(self, template_name: str, config: Optional[Dict[str, Any]] = None) -> "MLProcessBuilder":
        if template_name not in self.templates:
            raise ValueError(f"未找到流程模板: {template_name}")

        template = self.templates[template_name]
        self.steps = {}
        self.process_config = {
            "name": template["name"],
            "type": template["type"],
            "template": template_name,
            "callbacks": {},
        }

        for step_id, step_config in template["steps"].items():
            self.add_step(
                step_id=step_id,
                step_name=step_config["step_name"],
                step_type=step_config["step_type"],
                dependencies=step_config.get("dependencies", []),
                config=step_config.get("config", {}),
                timeout=step_config.get("timeout"),
            )

        if config:
            for key, value in config.items():
                if key == "config":
                    self.process_config.setdefault("config", {}).update(value)
                elif key == "metadata":
                    self.process_config.setdefault("metadata", {}).update(value)
                else:
                    self.process_config[key] = value

        return self

    def add_step(
        self,
        step_id: str,
        step_name: str,
        step_type: str,
        dependencies: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> "MLProcessBuilder":
        step = ProcessStep(
            step_id=step_id,
            step_name=step_name,
            step_type=step_type,
            dependencies=dependencies or [],
            config=config or {},
            timeout=timeout,
        )
        self.steps[step_id] = step
        return self

    def configure_step(self, step_id: str, config: Dict[str, Any]) -> "MLProcessBuilder":
        if step_id not in self.steps:
            raise ValueError(f"步骤不存在: {step_id}")
        self.steps[step_id].config.update(config)
        return self

    def configure_process(self, config: Dict[str, Any]) -> "MLProcessBuilder":
        self.process_config.setdefault("config", {}).update(config)
        return self

    def set_priority(self, priority: ProcessPriority) -> "MLProcessBuilder":
        self.process_config["priority"] = priority
        return self

    def set_timeout(self, timeout: int) -> "MLProcessBuilder":
        self.process_config["timeout"] = timeout
        return self

    def add_callback(self, event: str, callback):
        callbacks = self.process_config.setdefault("callbacks", {})
        callbacks.setdefault(event, []).append(callback)

    def build(self) -> MLProcess:
        if not self.steps:
            raise ValueError("流程至少需要一个步骤")

        process = MLProcess(
            process_id="",
            process_type=self.process_config.get("type", MLProcessType.MODEL_TRAINING),
            process_name=self.process_config.get("name", f"ML Process {datetime.now().isoformat()}"),
            priority=self.process_config.get("priority", ProcessPriority.NORMAL),
            steps=self.steps,
            config=self.process_config.get("config", {}),
            timeout=self.process_config.get("timeout"),
            metadata=self.process_config.get("metadata", {}),
        )

        return process

    def build_and_submit(self) -> str:
        process = self.build()
        callbacks = self.process_config.get("callbacks", {})
        for event, callback_list in callbacks.items():
            for callback in callback_list:
                self.orchestrator.add_process_callback(process.process_id, event, callback)

        process_id = self.orchestrator.submit_process(process)
        return process_id

    def reset(self):
        self.steps.clear()
        self.process_config.clear()


__all__ = ["MLProcessBuilder", "ProcessPriority"]

