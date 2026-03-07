import importlib
import json
import sys
from types import SimpleNamespace

import pytest

# 为不规范的绝对导入建立别名
sys.modules.setdefault(
    "cloud_native_configs",
    importlib.import_module("src.infrastructure.config.environment.cloud_native_configs"),
)
sys.modules.setdefault(
    "cloud_configs",
    importlib.import_module("src.infrastructure.config.environment.cloud_configs"),
)

from src.infrastructure.config.environment import cloud_native_enhanced as module
from src.infrastructure.config.environment.cloud_native_configs import (
    AutoScalingConfig,
    CloudNativeMonitoringConfig,
    CloudProvider,
    MultiCloudConfig,
    ScalingPolicy,
    ServiceMeshConfig,
    ServiceMeshType,
)


@pytest.fixture
def patched_managers(monkeypatch):
    created = {}

    class StubServiceMeshManager:
        def __init__(self, mesh_type, namespace):
            created["service_mesh"] = (mesh_type, namespace)
            self.mesh_type = mesh_type
            self.namespace = namespace
            self.install_called = 0
            self.should_succeed = True

        def install_service_mesh(self):
            self.install_called += 1
            return self.should_succeed

    class StubMultiCloudManager:
        def __init__(self, cfg):
            created["multi_cloud"] = cfg
            self.config = cfg

    class StubAutoScalingManager:
        def __init__(self, cfg):
            created["auto_scaling"] = cfg
            self.config = cfg
            self.desired_replicas = None
            self.check_calls = []
            self.execute_calls = []

        def check_scaling_needs(self, service_name, metrics):
            self.check_calls.append((service_name, metrics))
            return self.desired_replicas

        def execute_scaling(self, service_name, target):
            self.execute_calls.append((service_name, target))

    class StubMonitoringManager:
        def __init__(self, cfg):
            created["monitoring"] = cfg
            self.config = cfg
            self.collected = []

        def collect_metrics(self, service_name, metrics):
            self.collected.append((service_name, metrics))

        def generate_monitoring_report(self):
            return {"status": "ok"}

    monkeypatch.setattr(module, "ServiceMeshManager", StubServiceMeshManager)
    monkeypatch.setattr(module, "MultiCloudManager", StubMultiCloudManager)
    monkeypatch.setattr(module, "AutoScalingManager", StubAutoScalingManager)
    monkeypatch.setattr(module, "EnhancedMonitoringManager", StubMonitoringManager)

    return created, StubServiceMeshManager, StubAutoScalingManager, StubMonitoringManager


def test_initialize_managers_with_configs(patched_managers):
    created, service_mesh_cls, auto_scaling_cls, _ = patched_managers

    mesh_cfg = ServiceMeshConfig(mesh_type=ServiceMeshType.LINKERD, namespace="svc-mesh")
    multi_cfg = MultiCloudConfig(primary_provider=CloudProvider.AZURE, secondary_providers=[CloudProvider.GCP])
    # 兼容 cloud_native_enhanced 对 backup_providers 的引用
    setattr(multi_cfg, "backup_providers", multi_cfg.secondary_providers)
    scaling_cfg = AutoScalingConfig(min_replicas=2, max_replicas=5)
    monitor_cfg = CloudNativeMonitoringConfig(prometheus_enabled=False)

    enhanced = module.CloudNativeEnhanced(mesh_cfg, multi_cfg, scaling_cfg, monitor_cfg)

    assert isinstance(enhanced.service_mesh_manager, service_mesh_cls)
    assert enhanced.service_mesh_manager.mesh_type == ServiceMeshType.LINKERD
    assert isinstance(enhanced.auto_scaling_manager, auto_scaling_cls)
    assert created["service_mesh"][0] == ServiceMeshType.LINKERD
    assert created["auto_scaling"] is scaling_cfg
    assert created["monitoring"] is monitor_cfg


def test_deploy_with_service_mesh_success(patched_managers):
    _, service_mesh_cls, _, _ = patched_managers
    mesh_cfg = ServiceMeshConfig(mesh_type=ServiceMeshType.ISTIO, namespace="istio-system")
    enhanced = module.CloudNativeEnhanced(mesh_cfg)

    enhanced.service_mesh_manager.should_succeed = True
    result = enhanced.deploy_with_service_mesh("orders", {"image": "demo"})

    assert result is True
    assert enhanced.service_mesh_manager.install_called == 1


def test_deploy_with_service_mesh_requires_name(patched_managers):
    module.CloudNativeEnhanced(ServiceMeshConfig())
    with pytest.raises(ValueError):
        module.CloudNativeEnhanced(ServiceMeshConfig()).deploy_with_service_mesh("", {})


def test_deploy_with_service_mesh_without_manager_returns_false(patched_managers):
    enhanced = module.CloudNativeEnhanced()
    result = enhanced.deploy_with_service_mesh("svc", {})
    assert result is False


def test_enable_auto_scaling_invokes_hpa(monkeypatch, patched_managers):
    _, _, _, _ = patched_managers
    scaling_cfg = AutoScalingConfig(min_replicas=1, max_replicas=3)
    enhanced = module.CloudNativeEnhanced(auto_scaling_config=scaling_cfg)

    hpa_calls = {}

    def fake_apply(cmd, input=None, text=None, capture_output=None):
        hpa_calls["cmd"] = cmd
        hpa_calls["input"] = input
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_apply)

    assert enhanced.enable_auto_scaling("billing") is True

    payload = json.loads(hpa_calls["input"])
    assert payload["metadata"]["name"] == "billing-hpa"
    assert payload["spec"]["minReplicas"] == 1
    assert payload["spec"]["maxReplicas"] == 3


def test_create_kubernetes_hpa_handles_error(monkeypatch, patched_managers):
    scaling_cfg = AutoScalingConfig(min_replicas=1, max_replicas=2)
    enhanced = module.CloudNativeEnhanced(auto_scaling_config=scaling_cfg)

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="boom"),
    )

    assert enhanced._create_kubernetes_hpa("svc") is False


def test_collect_monitoring_data_triggers_scaling(monkeypatch, patched_managers):
    created, _, auto_scaling_cls, monitoring_cls = patched_managers
    scaling_cfg = AutoScalingConfig()
    monitor_cfg = CloudNativeMonitoringConfig()
    enhanced = module.CloudNativeEnhanced(
        service_mesh_config=None,
        multi_cloud_config=None,
        auto_scaling_config=scaling_cfg,
        monitoring_config=monitor_cfg,
    )

    auto_manager = enhanced.auto_scaling_manager
    auto_manager.desired_replicas = 4

    metrics = {"cpu_percent": 95}
    enhanced.collect_monitoring_data("orders", metrics)

    assert auto_manager.check_calls == [("orders", metrics)]
    assert auto_manager.execute_calls == [("orders", 4)]
    monitoring_manager = enhanced.monitoring_manager
    assert monitoring_manager.collected == [("orders", metrics)]


def test_get_monitoring_report_when_manager_missing():
    enhanced = module.CloudNativeEnhanced()
    assert enhanced.get_monitoring_report() == {"error": "监控管理器未初始化"}


def test_get_platform_status(monkeypatch, patched_managers):
    _, _, _, _ = patched_managers
    mesh_cfg = ServiceMeshConfig(mesh_type=ServiceMeshType.ISTIO, namespace="mesh-ns")
    multi_cfg = MultiCloudConfig(
        primary_provider=CloudProvider.AWS, secondary_providers=[CloudProvider.GCP]
    )
    setattr(multi_cfg, "backup_providers", multi_cfg.secondary_providers)
    scaling_cfg = AutoScalingConfig(scaling_policy=ScalingPolicy.MEMORY_UTILIZATION)
    monitor_cfg = CloudNativeMonitoringConfig(grafana_enabled=False)

    enhanced = module.CloudNativeEnhanced(mesh_cfg, multi_cfg, scaling_cfg, monitor_cfg)

    fake_now = SimpleNamespace(isoformat=lambda: "2025-01-01T00:00:00")

    class DummyDateTime:
        @staticmethod
        def now():
            return fake_now

    monkeypatch.setattr(module, "datetime", DummyDateTime)

    status = enhanced.get_platform_status()

    assert status["timestamp"] == "2025-01-01T00:00:00"
    assert status["service_mesh"]["type"] == ServiceMeshType.ISTIO.value
    assert status["multi_cloud"]["primary_provider"] == CloudProvider.AWS.value
    assert status["auto_scaling"]["scaling_policy"] == ScalingPolicy.MEMORY_UTILIZATION.value
    assert status["monitoring"]["grafana_enabled"] is False

