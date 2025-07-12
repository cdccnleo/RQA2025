import optuna
import pandas as pd
from typing import Dict, List, Tuple, Callable
from functools import partial
from tqdm import tqdm
from .base import BaseTuner, TuningResult, SearchMethod, ObjectiveDirection

class OptunaTuner(BaseTuner):
    """基于Optuna的调参器"""

    def __init__(self, method=SearchMethod.TPE, seed=None):
        self.method = method
        self.seed = seed

    def tune(self, objective_func, param_space,
            n_trials=100, direction=ObjectiveDirection.MAXIMIZE) -> TuningResult:
        # 创建study
        study = optuna.create_study(
            direction='maximize' if direction == ObjectiveDirection.MAXIMIZE else 'minimize',
            sampler=self._get_sampler()
        )

        # 定义优化目标函数
        def wrapped_objective(trial):
            params = {}
            for name, space in param_space.items():
                if space['type'] == 'float':
                    params[name] = trial.suggest_float(
                        name, space['low'], space['high'], log=space.get('log', False))
                elif space['type'] == 'int':
                    params[name] = trial.suggest_int(
                        name, space['low'], space['high'], log=space.get('log', False))
                elif space['type'] == 'categorical':
                    params[name] = trial.suggest_categorical(name, space['choices'])
            return objective_func(params)

        # 执行优化
        study.optimize(wrapped_objective, n_trials=n_trials, show_progress_bar=True)

        # 获取结果
        trials = study.trials_dataframe()
        importance = optuna.importance.get_param_importances(study)

        return TuningResult(
            best_params=study.best_params,
            best_value=study.best_value,
            trials=trials,
            importance=importance
        )

    def _get_sampler(self):
        """获取搜索采样器"""
        if self.method == SearchMethod.TPE:
            return optuna.samplers.TPESampler(seed=self.seed)
        elif self.method == SearchMethod.RANDOM:
            return optuna.samplers.RandomSampler(seed=self.seed)
        elif self.method == SearchMethod.CMAES:
            return optuna.samplers.CmaEsSampler(seed=self.seed)
        else:
            return optuna.samplers.NSGAIISampler(seed=self.seed)

class MultiObjectiveTuner(BaseTuner):
    """多目标调参器"""

    def __init__(self, objectives: List[Tuple[Callable, ObjectiveDirection]]):
        self.objectives = objectives

    def tune(self, param_space, n_trials=100) -> Dict[str, TuningResult]:
        # 创建多目标study
        study = optuna.create_study(
            directions=[dir.value for _, dir in self.objectives],
            sampler=optuna.samplers.NSGAIISampler()
        )

        # 定义优化目标函数
        def wrapped_objective(trial):
            params = {}
            for name, space in param_space.items():
                if space['type'] == 'float':
                    params[name] = trial.suggest_float(
                        name, space['low'], space['high'], log=space.get('log', False))
                elif space['type'] == 'int':
                    params[name] = trial.suggest_int(
                        name, space['low'], space['high'], log=space.get('log', False))
                elif space['type'] == 'categorical':
                    params[name] = trial.suggest_categorical(name, space['choices'])

            # 计算所有目标
            return [obj(params) for obj, _ in self.objectives]

        # 执行优化
        study.optimize(wrapped_objective, n_trials=n_trials, show_progress_bar=True)

        # 处理结果
        results = {}
        for i, (obj_func, direction) in enumerate(self.objectives):
            best_trial = max(study.best_trials,
                           key=lambda t: t.values[i] if direction == ObjectiveDirection.MAXIMIZE else -t.values[i])

            results[obj_func.__name__] = TuningResult(
                best_params=best_trial.params,
                best_value=best_trial.values[i],
                trials=study.trials_dataframe()
            )

        return results
