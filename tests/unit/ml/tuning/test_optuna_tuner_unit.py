from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import optuna

from src.ml.tuning.optimizers.base import ObjectiveDirection, SearchMethod, TuningResult
from src.ml.tuning.optimizers.optuna_tuner import MultiObjectiveTuner, OptunaTuner
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class DummyTrial:
    def __init__(self):
        self.params = {}

    def suggest_float(self, name, low, high, log=False):
        value = (low + high) / 2
        self.params[name] = value
        return value

    def suggest_int(self, name, low, high, log=False):
        value = low
        self.params[name] = value
        return value

    def suggest_categorical(self, name, choices):
        value = choices[0]
        self.params[name] = value
        return value


class DummyStudy:
    def __init__(self, multi_objective=False):
        self.multi_objective = multi_objective
        self.optimize_called = False
        self.best_params = {}
        self.best_value = None
        self.best_trials = []
        self._trials = []

    def optimize(self, objective, n_trials, show_progress_bar=True):
        self.optimize_called = True
        trial = DummyTrial()
        value = objective(trial)
        if self.multi_objective:
            self.best_trials = [Mock(values=value, params=trial.params)]
            self.best_value = value
        else:
            self.best_trials = [Mock(values=[value], params=trial.params)]
            self.best_params = trial.params
            self.best_value = value
        self._trials.append(Mock(params=trial.params, values=value))

    def trials_dataframe(self):
        return pd.DataFrame([{"params": t.params, "values": t.values} for t in self._trials])


@pytest.fixture
def patched_optuna(monkeypatch):
    dummy_study = DummyStudy()
    monkeypatch.setattr("optuna.create_study", lambda **kwargs: dummy_study)
    samplers = Mock(
        TPESampler=lambda seed=None: "tpe",
        RandomSampler=lambda seed=None: "random",
        CmaEsSampler=lambda seed=None: "cmaes",
        NSGAIISampler=lambda seed=None: "nsgaii",
    )
    monkeypatch.setattr("optuna.samplers", samplers)
    importance_mock = Mock(return_value={"lr": 0.6})
    importance_module = SimpleNamespace(get_param_importances=importance_mock)
    monkeypatch.setattr("optuna.importance", importance_module, raising=False)
    dummy_study.importance_mock = importance_mock
    return dummy_study


def test_optuna_tuner_requires_param_space():
    tuner = OptunaTuner()
    with pytest.raises(ValueError):
        tuner.tune(lambda params: 1.0, {})


def test_optuna_tuner_returns_tuning_result(patched_optuna):
    tuner = OptunaTuner(method=SearchMethod.RANDOM, seed=123)
    param_space = {
        "lr": {"type": "float", "low": 0.01, "high": 0.03},
        "depth": {"type": "int", "low": 3, "high": 5},
        "feature": {"type": "categorical", "choices": ["a", "b"]},
    }

    result = tuner.tune(lambda params: 0.85, param_space, n_trials=5, direction=ObjectiveDirection.MAXIMIZE)

    assert isinstance(result, TuningResult)
    assert result.best_params == patched_optuna.best_params
    assert result.best_value == patched_optuna.best_value
    assert patched_optuna.importance_mock.called
    assert result.importance == {"lr": 0.6}
    assert not result.trials.empty


def test_optuna_tuner_trials_dataframe_failure(monkeypatch):
    class FailingStudy(DummyStudy):
        def trials_dataframe(self):
            raise RuntimeError("boom")

    dummy_study = FailingStudy()
    monkeypatch.setattr("optuna.create_study", lambda **kwargs: dummy_study)
    samplers = Mock(
        TPESampler=lambda seed=None: "tpe",
        RandomSampler=lambda seed=None: "random",
        CmaEsSampler=lambda seed=None: "cmaes",
        NSGAIISampler=lambda seed=None: "nsgaii",
    )
    monkeypatch.setattr("optuna.samplers", samplers)
    monkeypatch.setattr(
        "optuna.importance",
        SimpleNamespace(get_param_importances=lambda study: {}),
        raising=False,
    )

    tuner = OptunaTuner()
    param_space = {"lr": {"type": "float", "low": 0.01, "high": 0.02}}

    result = tuner.tune(lambda params: 0.1, param_space, n_trials=1)
    assert result.trials.empty


def test_optuna_tuner_handles_unknown_param_type(patched_optuna):
    tuner = OptunaTuner()
    with pytest.raises(ValueError):
        tuner.tune(lambda params: 0.0, {"bad": {"type": "unsupported"}}, n_trials=1)


def test_optuna_tuner_importance_failure(monkeypatch, patched_optuna):
    monkeypatch.setattr("optuna.importance.get_param_importances", Mock(side_effect=RuntimeError("boom")))

    tuner = OptunaTuner(method=SearchMethod.TPE)
    param_space = {"lr": {"type": "float", "low": 0.01, "high": 0.02}}

    result = tuner.tune(lambda params: 0.5, param_space, n_trials=1)

    assert isinstance(result, TuningResult)
    assert result.importance is None


def test_optuna_tuner_sampler_selection():
    tuner = OptunaTuner(method=SearchMethod.CMAES)
    sampler = tuner._get_sampler()
    assert type(sampler).__name__ == "CmaEsSampler"

    tuner_other = OptunaTuner(method=SearchMethod.BAYESIAN)
    assert type(tuner_other._get_sampler()).__name__ == "TPESampler"


def test_multi_objective_tuner(monkeypatch):
    dummy_study = DummyStudy(multi_objective=True)
    monkeypatch.setattr("optuna.create_study", lambda **kwargs: dummy_study)
    monkeypatch.setattr("optuna.samplers.NSGAIISampler", lambda seed=None: "nsgaii")

    def objective_a(params):
        return params.get("lr", 0.1)

    def objective_b(params):
        return 1 - params.get("lr", 0.1)

    tuner = MultiObjectiveTuner(
        [
            (objective_a, ObjectiveDirection.MAXIMIZE),
            (objective_b, ObjectiveDirection.MINIMIZE),
        ]
    )

    param_space = {
        "lr": {"type": "float", "low": 0.01, "high": 0.02},
        "depth": {"type": "int", "low": 2, "high": 4},
        "optimizer": {"type": "categorical", "choices": ["adam", "sgd"]},
    }
    results = tuner.tune(param_space, n_trials=2)

    assert set(results.keys()) == {"objective_a", "objective_b"}
    for res in results.values():
        assert isinstance(res, TuningResult)
        assert not res.trials.empty


def test_multi_objective_returns_empty_when_no_trials(monkeypatch):
    class EmptyStudy(DummyStudy):
        def __init__(self):
            super().__init__(multi_objective=True)

        def optimize(self, objective, n_trials, show_progress_bar=True):
            self.optimize_called = True
            self.best_trials = []

    dummy_study = EmptyStudy()
    monkeypatch.setattr("optuna.create_study", lambda **kwargs: dummy_study)
    monkeypatch.setattr("optuna.samplers.NSGAIISampler", lambda seed=None: "nsgaii")

    tuner = MultiObjectiveTuner([(lambda params: 0.0, ObjectiveDirection.MAXIMIZE)])
    results = tuner.tune({"x": {"type": "float", "low": 0.0, "high": 1.0}}, n_trials=1)
    assert results == {}


def test_multi_objective_trials_dataframe_failure(monkeypatch):
    class FailingStudy(DummyStudy):
        def __init__(self):
            super().__init__(multi_objective=True)

        def trials_dataframe(self):
            raise RuntimeError("boom")

    dummy_study = FailingStudy()
    monkeypatch.setattr("optuna.create_study", lambda **kwargs: dummy_study)
    monkeypatch.setattr("optuna.samplers.NSGAIISampler", lambda seed=None: "nsgaii")

    def objective(params):
        return params.get("lr", 0.1)

    tuner = MultiObjectiveTuner([(objective, ObjectiveDirection.MAXIMIZE)])
    param_space = {"lr": {"type": "float", "low": 0.01, "high": 0.02}}
    results = tuner.tune(param_space, n_trials=1)
    result = results["objective"]
    assert result.trials.empty


def test_multi_objective_requires_objectives():
    with pytest.raises(ValueError):
        MultiObjectiveTuner([])


def test_multi_objective_unsupported_param_type(monkeypatch):
    dummy_study = DummyStudy(multi_objective=True)
    monkeypatch.setattr("optuna.create_study", lambda **kwargs: dummy_study)
    monkeypatch.setattr("optuna.samplers.NSGAIISampler", lambda seed=None: "nsgaii")

    tuner = MultiObjectiveTuner([(lambda params: 0.0, ObjectiveDirection.MAXIMIZE)])
    with pytest.raises(ValueError):
        tuner.tune({"bad": {"type": "unknown"}}, n_trials=1)


def test_multi_objective_skips_trials_without_values(monkeypatch):
    class IncompleteStudy(DummyStudy):
        def __init__(self):
            super().__init__(multi_objective=True)

        def optimize(self, objective, n_trials, show_progress_bar=True):
            self.best_trials = [Mock(values=None, params={"lr": 0.1})]

    dummy_study = IncompleteStudy()
    monkeypatch.setattr("optuna.create_study", lambda **kwargs: dummy_study)
    monkeypatch.setattr("optuna.samplers.NSGAIISampler", lambda seed=None: "nsgaii")

    tuner = MultiObjectiveTuner([(lambda params: 0.0, ObjectiveDirection.MAXIMIZE)])
    results = tuner.tune({"lr": {"type": "float", "low": 0.0, "high": 1.0}}, n_trials=1)
    assert results == {}

def test_optuna_tuner_init_with_seed():
    """测试OptunaTuner初始化时设置seed"""
    tuner = OptunaTuner(method=SearchMethod.TPE, seed=42)
    assert tuner.method == SearchMethod.TPE
    assert tuner.seed == 42


def test_optuna_tuner_init_defaults():
    """测试OptunaTuner默认初始化参数"""
    tuner = OptunaTuner()
    assert tuner.method == SearchMethod.TPE
    assert tuner.seed is None


def test_optuna_tuner_sampler_tpe():
    """测试TPE采样器获取"""
    tuner = OptunaTuner(method=SearchMethod.TPE, seed=42)
    sampler = tuner._get_sampler()
    # 验证是TPESampler实例
    assert sampler.__class__.__name__ == 'TPESampler'


def test_optuna_tuner_sampler_random():
    """测试随机采样器获取"""
    tuner = OptunaTuner(method=SearchMethod.RANDOM, seed=123)
    sampler = tuner._get_sampler()
    # 验证是RandomSampler实例
    assert sampler.__class__.__name__ == 'RandomSampler'


def test_optuna_tuner_sampler_cmaes():
    """测试CMAES采样器获取"""
    tuner = OptunaTuner(method=SearchMethod.CMAES, seed=456)
    sampler = tuner._get_sampler()
    # 验证是CmaEsSampler实例
    assert sampler.__class__.__name__ == 'CmaEsSampler'


def test_optuna_tuner_sampler_bayesian():
    """测试BAYESIAN采样器获取（映射到TPE）"""
    tuner = OptunaTuner(method=SearchMethod.BAYESIAN, seed=456)
    sampler = tuner._get_sampler()
    assert isinstance(sampler, optuna.samplers.TPESampler)


def test_optuna_tuner_sampler_nsgaii():
    """测试NSGAII采样器获取（默认分支）"""
    tuner = OptunaTuner(method="unknown_method", seed=789)  # 未知方法会使用默认NSGAII
    sampler = tuner._get_sampler()
    assert isinstance(sampler, optuna.samplers.NSGAIISampler)


def test_multi_objective_init_empty_objectives():
    """测试多目标调参器空目标列表"""
    with pytest.raises(ValueError, match="Objectives cannot be empty"):
        MultiObjectiveTuner([])


def test_optuna_tuner_tune_minimize_direction():
    """测试tune方法的最小化方向"""
    tuner = OptunaTuner(method=SearchMethod.TPE, seed=42)

    def objective_func(params):
        # 返回一个需要最小化的值
        return params["lr"] ** 2

    param_space = {
        "lr": {"type": "float", "low": 0.1, "high": 1.0, "log": False}
    }

    # 使用很少的trial来快速完成测试
    result = tuner.tune(objective_func, param_space, n_trials=2, direction=ObjectiveDirection.MINIMIZE)

    # 验证结果结构
    assert isinstance(result, TuningResult)
    assert isinstance(result.best_params, dict)
    assert "lr" in result.best_params
    assert isinstance(result.best_value, (int, float))
    assert isinstance(result.trials, pd.DataFrame)
    # importance可能为None
    assert result.importance is None or isinstance(result.importance, dict)


def test_optuna_tuner_tune_int_param():
    """测试tune方法的整数参数"""
    tuner = OptunaTuner(method=SearchMethod.TPE, seed=42)

    def objective_func(params):
        return params["batch_size"]

    param_space = {
        "batch_size": {"type": "int", "low": 16, "high": 128}
    }

    result = tuner.tune(objective_func, param_space, n_trials=2)

    assert isinstance(result, TuningResult)
    assert isinstance(result.best_params, dict)
    assert "batch_size" in result.best_params
    assert isinstance(result.best_params["batch_size"], int)


def test_optuna_tuner_tune_categorical_param():
    """测试tune方法的分类参数"""
    tuner = OptunaTuner(method=SearchMethod.TPE, seed=42)

    def objective_func(params):
        return {"adam": 0.1, "sgd": 0.2, "rmsprop": 0.15}[params["optimizer"]]

    param_space = {
        "optimizer": {"type": "categorical", "choices": ["adam", "sgd", "rmsprop"]}
    }

    result = tuner.tune(objective_func, param_space, n_trials=2)

    assert isinstance(result, TuningResult)
    assert isinstance(result.best_params, dict)
    assert "optimizer" in result.best_params
    assert result.best_params["optimizer"] in ["adam", "sgd", "rmsprop"]


def test_multi_objective_tune_with_trials(monkeypatch):
    """测试多目标调参器的完整tune方法执行"""
    # 创建mock trials
    mock_trial1 = Mock()
    mock_trial1.params = {"lr": 0.01, "batch_size": 32}
    mock_trial1.values = [0.9, 0.1]  # 两个目标的值

    mock_trial2 = Mock()
    mock_trial2.params = {"lr": 0.02, "batch_size": 64}
    mock_trial2.values = [0.85, 0.15]

    mock_study = Mock()
    mock_study.best_trials = [mock_trial1, mock_trial2]
    mock_study.trials_dataframe.return_value = pd.DataFrame()
    mock_study.optimize = Mock()

    mock_create_study = Mock(return_value=mock_study)

    monkeypatch.setattr("optuna.create_study", mock_create_study)
    monkeypatch.setattr("optuna.samplers.NSGAIISampler", Mock(return_value="nsgaii"))

    def objective_a(params):
        return params["lr"] * 2  # MAXIMIZE

    def objective_b(params):
        return 1 - params["lr"]  # MINIMIZE

    tuner = MultiObjectiveTuner([
        (objective_a, ObjectiveDirection.MAXIMIZE),
        (objective_b, ObjectiveDirection.MINIMIZE)
    ])

    param_space = {
        "lr": {"type": "float", "low": 0.001, "high": 0.1},
        "batch_size": {"type": "categorical", "choices": [32, 64, 128]}
    }

    results = tuner.tune(param_space, n_trials=5)

    # 验证返回了两个目标的结果
    assert len(results) == 2
    assert "objective_a" in results
    assert "objective_b" in results

    # 验证第一个目标（MAXIMIZE）选择了最好的trial
    # 在mock_trial1和mock_trial2中，第一个目标值分别是0.9和0.85，0.9更大
    assert results["objective_a"].best_params["lr"] == 0.01
    assert results["objective_a"].best_value == 0.9

    # 验证第二个目标（MINIMIZE）选择了最好的trial
    # 在mock_trial1和mock_trial2中，第二个目标值分别是0.1和0.15，0.1更小
    assert results["objective_b"].best_params["lr"] == 0.01
    assert results["objective_b"].best_value == 0.1


def test_optuna_tuner_tune_with_real_calls(monkeypatch):
    """测试tune方法的实际optuna调用路径"""
    # Mock optuna组件
    mock_study = Mock()
    mock_study.best_params = {"lr": 0.01}
    mock_study.best_value = 0.95
    mock_study.trials_dataframe.return_value = pd.DataFrame()
    mock_study.optimize = Mock()

    mock_create_study = Mock(return_value=mock_study)
    mock_importance = Mock(return_value={"lr": 0.8})

    monkeypatch.setattr("optuna.create_study", mock_create_study)
    monkeypatch.setattr("optuna.importance.get_param_importances", mock_importance)
    monkeypatch.setattr("optuna.samplers.TPESampler", Mock(return_value="tpe_sampler"))

    tuner = OptunaTuner(method=SearchMethod.TPE, seed=42)

    def objective_func(params):
        return params["lr"] * 2

    param_space = {
        "lr": {"type": "float", "low": 0.001, "high": 0.1, "log": True}
    }

    result = tuner.tune(objective_func, param_space, n_trials=10)

    # 验证结果
    assert result.best_params == {"lr": 0.01}
    assert result.best_value == 0.95
    assert isinstance(result.trials, pd.DataFrame)
    # importance可能为None（如果optuna.importance失败），这是正常情况
    assert result.importance is None or isinstance(result.importance, dict)

    # 验证optuna.create_study被正确调用
    mock_create_study.assert_called_once()
    call_args = mock_create_study.call_args
    assert call_args[1]["direction"] == "maximize"
    assert "sampler" in call_args[1]


def test_multi_objective_requires_param_space(monkeypatch):
    monkeypatch.setattr("optuna.create_study", lambda **kwargs: DummyStudy(multi_objective=True))
    monkeypatch.setattr("optuna.samplers.NSGAIISampler", lambda seed=None: "nsgaii")

    tuner = MultiObjectiveTuner([(lambda params: 0.0, ObjectiveDirection.MAXIMIZE)])
    with pytest.raises(ValueError):
        tuner.tune({}, n_trials=1)

