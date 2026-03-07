import configparser
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    from infrastructure.utils.exception_utils import DataLoaderError  # type: ignore
except Exception:  # pragma: no cover - 尝试 src 命名空间
    try:
        from src.infrastructure.utils.exception_utils import DataLoaderError  # type: ignore
    except Exception:  # pragma: no cover - 兼容降级环境
        class DataLoaderError(Exception):
            """数据加载异常基类（降级实现）"""

try:
    from src.data.quality.validator import DataValidator  # type: ignore
except Exception:  # pragma: no cover - 兼容缺失依赖
    class DataValidator:  # type: ignore
        """降级数据校验器"""

        def validate(self, data: Any) -> bool:
            return True


try:
    from src.data.monitoring.quality_monitor import DataQualityMonitor  # type: ignore
except Exception:  # pragma: no cover - 兼容缺失依赖
    class DataQualityMonitor:  # type: ignore
        """降级数据质量监控器"""

        def track_metrics(self, *args, **kwargs) -> None:
            return None

        def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
            return {}


try:
    from src.data.cache import CacheConfig, CacheManager, DiskCache  # type: ignore
except Exception:  # pragma: no cover - 兼容缺失依赖
    @dataclass
    class CacheConfig:  # type: ignore
        disk_cache_dir: str = "data/cache"

    class DiskCache:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.storage: Dict[str, Any] = {}

        def get(self, key: str) -> Any:
            return self.storage.get(key)

        def set(self, key: str, value: Any) -> None:
            self.storage[key] = value

    class CacheManager:  # type: ignore
        def __init__(self, config: CacheConfig):
            self.config = config
            self._disk_cache = DiskCache(config.disk_cache_dir)

        def get_cached_data(self, key: str) -> Any:
            return self._disk_cache.get(key)

        def save_to_cache(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
            self._disk_cache.set(key, value)

        def get(self, key: str, cache_type: Optional[str] = None) -> Any:
            return self._disk_cache.get(key)

        def set(self, key: str, value: Any, cache_type: Optional[str] = None) -> None:
            self._disk_cache.set(key, value)


class DataRegistry:
    """通用数据加载器注册表"""

    def __init__(self) -> None:
        self._loaders: Dict[str, Any] = {}

    def register(self, name: str, loader: Any) -> None:
        if name in self._loaders:
            raise ValueError(f"Loader '{name}' already registered")
        self._loaders[name] = loader

    def register_class(self, name: str, loader_cls: Any) -> None:
        self.register(name, loader_cls)

    def get_loader(self, name: str) -> Any:
        if name not in self._loaders:
            raise KeyError(name)
        return self._loaders[name]

    def is_registered(self, name: str) -> bool:
        return name in self._loaders

    def list_registered_loaders(self) -> Iterable[str]:
        return list(self._loaders.keys())

    def clear_loaders(self) -> None:
        self._loaders.clear()


@dataclass
class DataModel:
    """统一数据模型轻量实现"""

    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        return self.data is not None

    def to_dict(self) -> Dict[str, Any]:
        return {"data": self.data, "metadata": self.metadata}


class DataManager:
    """数据管理器轻量实现，用于协调数据加载、验证与缓存"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        registry: Optional[DataRegistry] = None,
        validator: Optional[DataValidator] = None,
        quality_monitor: Optional[DataQualityMonitor] = None,
        cache_manager: Optional[CacheManager] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.debug("初始化 DataManager: config_path=%s", config_path)

        self.config = self._load_config(config_path=config_path, config_dict=config_dict)
        self.config_path = config_path
        self.registry = registry or DataRegistry()
        self.validator = validator or DataValidator()
        self.quality_monitor = quality_monitor or DataQualityMonitor()

        cache_config = CacheConfig()
        cache_dir = Path(getattr(cache_config, "disk_cache_dir", "data/cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_manager = cache_manager or CacheManager(cache_config)

        max_workers = self._resolve_max_workers()
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

        self.data_lineage: Dict[str, list] = {}

    # --------------------------------------------------------------------- #
    # 配置管理
    # --------------------------------------------------------------------- #
    def _load_config(
        self,
        *,
        config_path: Optional[str],
        config_dict: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if config_dict is not None:
            return config_dict

        if config_path:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            parser = configparser.ConfigParser()
            parser.read(path, encoding="utf-8")
            return {section: dict(parser[section]) for section in parser.sections()}

        return {}

    def _resolve_max_workers(self) -> int:
        general = self.config.get("General", {})
        if hasattr(general, "getint"):
            return general.getint("max_concurrent_workers", fallback=4)
        if isinstance(general, dict):
            value = general.get("max_concurrent_workers")
            try:
                return int(value) if value is not None else 4
            except (TypeError, ValueError):
                self.logger.warning("max_concurrent_workers 配置无效: %s，使用默认值4", value)
        return 4

    # --------------------------------------------------------------------- #
    # 注册中心
    # --------------------------------------------------------------------- #
    def register_loader(self, name: str, loader: Any) -> None:
        self.registry.register(name, loader)

    # --------------------------------------------------------------------- #
    # 数据加载主流程
    # --------------------------------------------------------------------- #
    def load_data(self, data_type: str, start_date: str, end_date: str, **kwargs) -> Any:
        cache_key = self._build_cache_key(data_type, start_date, end_date, kwargs)

        try:
            cached = self.cache_manager.get_cached_data(cache_key)
        except AttributeError:
            cached = None

        if cached is not None:
            self.logger.debug("缓存命中: %s", cache_key)
            return cached

        if not self.registry.is_registered(data_type):
            raise DataLoaderError(f"数据类型未注册: {data_type}")

        loader = self.registry.get_loader(data_type)

        try:
            result = loader.load(start_date=start_date, end_date=end_date, **kwargs)
        except Exception as exc:
            raise DataLoaderError(f"加载数据失败: {exc}") from exc

        # 数据验证
        try:
            if hasattr(result, "validate"):
                if not result.validate():
                    return result
            else:
                if not self.validator.validate(result):
                    raise DataLoaderError("数据校验失败")
        except Exception as exc:
            raise DataLoaderError(f"数据校验失败: {exc}") from exc

        # 数据质量监控与缓存写入
        try:
            metadata = getattr(loader, "metadata", {})
            self.quality_monitor.track_metrics(data_type, metadata, result)
        except Exception as exc:
            self.logger.warning("质量监控记录失败: %s", exc)

        try:
            self.cache_manager.save_to_cache(cache_key, result, metadata=getattr(loader, "metadata", {}))
        except Exception as exc:
            raise DataLoaderError(f"缓存写入失败: {exc}") from exc

        self._record_data_lineage(data_type, loader, start_date, end_date, **kwargs)

        return result

    def _build_cache_key(self, data_type: str, start_date: str, end_date: str, extra: Dict[str, Any]) -> str:
        ordered = "_".join(f"{k}={extra[k]}" for k in sorted(extra))
        return f"{data_type}:{start_date}:{end_date}:{ordered}"

    # --------------------------------------------------------------------- #
    # 多源数据加载
    # --------------------------------------------------------------------- #
    def load_multi_source(
        self,
        market_symbols: Iterable[Any],
        index_symbols: Iterable[Any],
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> Dict[str, Any]:
        results = {
            "market": self._load_stock_data(market_symbols, start_date, end_date, **kwargs),
            "index": self._load_index_data(index_symbols, start_date, end_date, **kwargs),
            "news": self._load_news_data(start_date, end_date, **kwargs),
            "fundamental": self._load_financial_data(start_date, end_date, **kwargs),
        }
        self._record_data_version(results, start_date, end_date)
        return results

    def _load_stock_data(self, symbols: Iterable[Any], start_date: str, end_date: str, **kwargs) -> DataModel:
        return DataModel(data=list(symbols), metadata={"type": "stock", "start": start_date, "end": end_date})

    def _load_index_data(self, symbols: Iterable[Any], start_date: str, end_date: str, **kwargs) -> DataModel:
        return DataModel(data=list(symbols), metadata={"type": "index", "start": start_date, "end": end_date})

    def _load_news_data(self, start_date: str, end_date: str, **kwargs) -> DataModel:
        return DataModel(data=[], metadata={"type": "news", "start": start_date, "end": end_date})

    def _load_financial_data(self, start_date: str, end_date: str, **kwargs) -> DataModel:
        return DataModel(data=[], metadata={"type": "fundamental", "start": start_date, "end": end_date})

    # --------------------------------------------------------------------- #
    # 数据血缘与版本管理
    # --------------------------------------------------------------------- #
    def _record_data_lineage(
        self,
        data_type: str,
        loader: Any,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> None:
        lineage = {
            "loader": loader.__class__.__name__,
            "start_date": start_date,
            "end_date": end_date,
            "params": kwargs,
        }
        self.data_lineage.setdefault(data_type, []).append(lineage)

    def _record_data_version(self, sources: Dict[str, Any], start_date: str, end_date: str) -> None:
        version_info = {"start_date": start_date, "end_date": end_date, "sources": list(sources.keys())}
        self.data_lineage.setdefault("_versions", []).append(version_info)

    def track_data_lineage(self, data_type: str) -> Dict[str, Any]:
        entries = self.data_lineage.get(data_type, [])
        return {"data_type": data_type, "history": entries}


__all__ = [
    "DataManager",
    "DataModel",
    "DataRegistry",
    "DataLoaderError",
]

