import importlib


def test_ml_core_import_success():
    import src.ml.core.ml_core as ml_core

    importlib.reload(ml_core)

    assert hasattr(ml_core, "MLCore")


def test_ml_core_import_fallback(monkeypatch):
    monkeypatch.setenv("ML_CORE_FORCE_FALLBACK", "1")

    import src.ml.core.ml_core as ml_core

    importlib.reload(ml_core)

    core = ml_core.MLCore()
    assert core.cache_manager is None
    assert core.config["model_cache_enabled"] is True

    monkeypatch.delenv("ML_CORE_FORCE_FALLBACK", raising=False)
    importlib.reload(ml_core)

