import optuna
import pandas as pd
from typing import Callable, Dict, List, Tuple

from .base import BaseTuner, ObjectiveDirection, SearchMethod, TuningResult


class OptunaTuner(BaseTuner):
    """基于 Optuna 的单目标调参器实现。"""

    def __init__(self, method=SearchMethod.TPE, seed=None):
        self.method = method
        self.seed = seed

    def tune(
        self,
        objective_func: Callable[[Dict], float],
        param_space: Dict[str, Dict],
        n_trials: int = 100,
        direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE,
    ) -> TuningResult:
        if not param_space:
            raise ValueError("Parameter space cannot be empty")

        study = optuna.create_study(
            direction="maximize" if direction == ObjectiveDirection.MAXIMIZE else "minimize",
            sampler=self._get_sampler(),
        )

        def wrapped_objective(trial):
            params = {}
            for name, space in param_space.items():
                param_type = space.get("type")
                if param_type == "float":
                    params[name] = trial.suggest_float(
                        name,
                        space["low"],
                        space["high"],
                        log=space.get("log", False),
                    )
                elif param_type == "int":
                    params[name] = trial.suggest_int(
                        name,
                        space["low"],
                        space["high"],
                        log=space.get("log", False),
                    )
                elif param_type == "categorical":
                    params[name] = trial.suggest_categorical(name, space["choices"])
                else:
                    raise ValueError(f"Unsupported parameter type '{param_type}' for '{name}'")
            return objective_func(params)

        study.optimize(wrapped_objective, n_trials=n_trials, show_progress_bar=True)

        try:
            trials_df = study.trials_dataframe()
        except Exception:
            trials_df = pd.DataFrame()

        try:
            importance = optuna.importance.get_param_importances(study)
        except Exception:
            importance = None

        return TuningResult(
            best_params=study.best_params,
            best_value=study.best_value,
            trials=trials_df,
            importance=importance,
        )

    def _get_sampler(self):
        """获取搜索采样器"""
        if self.method == SearchMethod.TPE:
            return optuna.samplers.TPESampler(seed=self.seed)
        if self.method == SearchMethod.RANDOM:
            return optuna.samplers.RandomSampler(seed=self.seed)
        if self.method == SearchMethod.CMAES:
            return optuna.samplers.CmaEsSampler(seed=self.seed)
        if self.method == SearchMethod.BAYESIAN:
            return optuna.samplers.TPESampler(seed=self.seed)  # Bayesian optimization using TPE
        return optuna.samplers.NSGAIISampler(seed=self.seed)


class MultiObjectiveTuner(BaseTuner):
    """基于 Optuna 的多目标调参封装。"""

    def __init__(self, objectives: List[Tuple[Callable[[Dict], float], ObjectiveDirection]]):
        if not objectives:
            raise ValueError("Objectives cannot be empty")
        self.objectives = objectives

    def tune(self, param_space: Dict[str, Dict], n_trials: int = 100) -> Dict[str, TuningResult]:
        if not param_space:
            raise ValueError("Parameter space cannot be empty")

        directions = [
            "maximize" if direction == ObjectiveDirection.MAXIMIZE else "minimize"
            for _, direction in self.objectives
        ]

        study = optuna.create_study(
            directions=directions,
            sampler=optuna.samplers.NSGAIISampler(),
        )

        def wrapped_objective(trial):
            params = {}
            for name, space in param_space.items():
                param_type = space.get("type")
                if param_type == "float":
                    params[name] = trial.suggest_float(
                        name,
                        space["low"],
                        space["high"],
                        log=space.get("log", False),
                    )
                elif param_type == "int":
                    params[name] = trial.suggest_int(
                        name,
                        space["low"],
                        space["high"],
                        log=space.get("log", False),
                    )
                elif param_type == "categorical":
                    params[name] = trial.suggest_categorical(name, space["choices"])
                else:
                    raise ValueError(f"Unsupported parameter type '{param_type}' for '{name}'")
            return [obj(params) for obj, _ in self.objectives]

        study.optimize(wrapped_objective, n_trials=n_trials, show_progress_bar=True)

        try:
            trials_df = study.trials_dataframe()
        except Exception:
            trials_df = pd.DataFrame()

        results: Dict[str, TuningResult] = {}
        if not study.best_trials:
            return results

        for idx, (obj_func, direction) in enumerate(self.objectives):
            completed_trials = [
                trial for trial in study.best_trials if getattr(trial, "values", None) is not None
            ]
            if not completed_trials:
                continue

            if direction == ObjectiveDirection.MAXIMIZE:
                best_trial = max(completed_trials, key=lambda t: t.values[idx])
            else:
                best_trial = min(completed_trials, key=lambda t: t.values[idx])

            results[obj_func.__name__] = TuningResult(
                best_params=best_trial.params,
                best_value=best_trial.values[idx],
                trials=trials_df,
            )

        return results
