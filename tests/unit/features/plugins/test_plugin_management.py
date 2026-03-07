from __future__ import annotations

from typing import Dict, List, Optional

from src.features.plugins.base_plugin import BaseFeaturePlugin, PluginMetadata, PluginType
from src.features.plugins.plugin_manager import FeaturePluginManager
from src.features.plugins.plugin_registry import PluginRegistry
from src.features.plugins.plugin_validator import PluginValidator


def build_metadata(
    name: str = "alpha_plugin",
    *,
    plugin_type: PluginType = PluginType.PROCESSOR,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    config_schema: Optional[Dict] = None,
) -> PluginMetadata:
    return PluginMetadata(
        name=name,
        version=version,
        description=f"{name} description",
        author="feature-team",
        plugin_type=plugin_type,
        tags=tags or [],
        dependencies=[],
        config_schema=config_schema,
    )


class SamplePlugin(BaseFeaturePlugin):
    def __init__(self, metadata: PluginMetadata):
        self._metadata = metadata
        self.initialized = 0
        self.cleaned = 0
        super().__init__()

    def _get_metadata(self) -> PluginMetadata:
        return self._metadata

    def process(self, data, **kwargs):  # type: ignore[override]
        return data

    def _initialize_plugin(self):
        self.initialized += 1

    def _cleanup_plugin(self):
        self.cleaned += 1


def make_plugin(**metadata_kwargs) -> SamplePlugin:
    return SamplePlugin(build_metadata(**metadata_kwargs))


class DummyLoader:
    def __init__(self, plugin_dirs=None):
        self.plugin_dirs = list(plugin_dirs or [])
        self.discovered: List[str] = []
        self.from_file: Dict[str, SamplePlugin] = {}
        self.from_module: Dict[str, SamplePlugin] = {}
        self.reloaded: Dict[str, SamplePlugin] = {}
        self.unloaded: List[str] = []

    def discover_plugins(self):
        return list(self.discovered)

    def load_plugin_from_file(self, path: str):
        return self.from_file.get(path)

    def load_plugin_from_module(self, module_name: str, plugin_class_name=None):
        return self.from_module.get(module_name)

    def reload_plugin(self, plugin_name: str):
        return self.reloaded.get(plugin_name)

    def unload_plugin(self, plugin_name: str):
        self.unloaded.append(plugin_name)
        return True

    def add_plugin_dir(self, plugin_dir: str):
        if plugin_dir not in self.plugin_dirs:
            self.plugin_dirs.append(plugin_dir)

    def remove_plugin_dir(self, plugin_dir: str):
        if plugin_dir in self.plugin_dirs:
            self.plugin_dirs.remove(plugin_dir)


def test_plugin_registry_tracks_type_tag_and_stats():
    registry = PluginRegistry()
    plugin = make_plugin(name="alpha_plugin", tags=["trend", "beta"])

    assert registry.register_plugin(plugin) is True
    assert registry.register_plugin(plugin) is False  # duplicate
    assert registry.get_plugin("alpha_plugin") is plugin
    assert registry.get_plugins_by_type(PluginType.PROCESSOR) == [plugin]
    assert registry.get_plugins_by_tag("trend") == [plugin]

    stats = registry.get_plugin_stats()
    assert stats["total_plugins"] == 1
    assert stats["by_type"]["processor"] == 1
    assert stats["by_status"]["inactive"] == 1


def test_plugin_registry_unregister_removes_references():
    registry = PluginRegistry()
    plugin = make_plugin(name="beta_plugin", tags=["alpha"])
    assert registry.register_plugin(plugin)

    assert registry.unregister_plugin("beta_plugin") is True
    assert registry.get_plugin("beta_plugin") is None
    assert registry.get_plugins_by_type(PluginType.PROCESSOR) == []
    assert registry.get_plugins_by_tag("alpha") == []
    assert registry.list_plugins() == []


def test_plugin_validator_config_and_api_checks():
    validator = PluginValidator()
    plugin = make_plugin(
        name="config_plugin",
        plugin_type=PluginType.SELECTOR,
        config_schema={"threshold": {"type": float}},
    )

    assert validator.validate_plugin_instance(plugin) is True
    assert validator.validate_config(plugin, {"threshold": 0.5}) is True
    assert validator.validate_config(plugin, {"threshold": "bad"}) is False
    plugin.metadata.min_api_version = "1.0.0"
    plugin.metadata.max_api_version = "1.5.0"
    assert validator.validate_api_compatibility(plugin, min_version="0.9.0", max_version="1.6.0") is True
    assert validator.validate_api_compatibility(plugin, min_version="2.0.0", max_version="3.0.0") is False


def test_feature_plugin_manager_full_flow(monkeypatch):
    loader = DummyLoader()
    initial_plugin = make_plugin(name="alpha_plugin", tags=["trend"])
    reloaded_plugin = make_plugin(name="alpha_plugin", version="1.1.0")
    loader.discovered = ["alpha_path"]
    loader.from_file["alpha_path"] = initial_plugin
    loader.reloaded["alpha_plugin"] = reloaded_plugin

    monkeypatch.setattr("src.features.plugins.plugin_manager.PluginLoader", lambda plugin_dirs=None: loader)
    manager = FeaturePluginManager(plugin_dirs=["/tmp/plugins"])

    loaded_plugins = manager.discover_and_load_plugins()
    assert loaded_plugins == [initial_plugin]
    assert manager.get_plugin("alpha_plugin") is initial_plugin
    assert manager.list_plugins(PluginType.PROCESSOR) == ["alpha_plugin"]

    manager.enable_auto_discovery(False)
    manager.enable_auto_load(False)
    assert manager._auto_discovery is False
    assert manager._auto_load is False

    manager.add_plugin_dir("new_dir")
    assert "new_dir" in loader.plugin_dirs
    manager.remove_plugin_dir("new_dir")
    assert "new_dir" not in loader.plugin_dirs

    plugin_info = manager.get_plugin_info("alpha_plugin")
    assert plugin_info is not None
    assert plugin_info["metadata"]["name"] == "alpha_plugin"

    reloaded = manager.reload_plugin("alpha_plugin")
    assert reloaded is reloaded_plugin
    assert manager.get_plugin("alpha_plugin") is reloaded_plugin

    validation = manager.validate_all_plugins()
    assert validation == {"alpha_plugin": True}

    assert manager.unload_plugin("alpha_plugin") is True
    assert loader.unloaded == ["alpha_plugin"]
    assert manager.list_plugins() == []

