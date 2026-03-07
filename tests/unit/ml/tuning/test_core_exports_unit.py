from importlib import reload


def test_tuning_core_exports_symbols():
    import src.ml.tuning.core as tuning_core

    reload(tuning_core)  # ensure fresh import for coverage
    expected = {
        'BaseTuner',
        'OptunaTuner',
        'MultiObjectiveTuner',
        'EarlyStopping',
        'TuningVisualizer',
        'TuningResult',
        'SearchMethod',
        'ObjectiveDirection',
    }
    assert set(tuning_core.__all__) == expected
    for name in expected:
        assert hasattr(tuning_core, name)

