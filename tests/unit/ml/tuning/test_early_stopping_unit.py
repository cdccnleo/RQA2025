import pytest

from src.ml.tuning.evaluators.early_stopping import EarlyStopping
from src.ml.tuning.optimizers.base import ObjectiveDirection
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def test_early_stopping_maximize_improves_resets_counter():
    stopper = EarlyStopping(patience=2, min_delta=0.01, direction=ObjectiveDirection.MAXIMIZE)
    stopper(0.5)  # initialize best
    stopper(0.52)  # improvement resets counter
    assert stopper.counter == 0
    assert stopper.best_score == pytest.approx(0.52)


def test_early_stopping_minimize_triggers_after_patience():
    stopper = EarlyStopping(patience=2, min_delta=0.0, direction=ObjectiveDirection.MINIMIZE)
    stopper(0.2)
    stopper(0.25)  # worse
    assert stopper.early_stop is False
    should_stop = stopper(0.3)  # patience exhausted
    assert should_stop is True
    assert stopper.early_stop is True


def test_early_stopping_reset_restores_state():
    stopper = EarlyStopping(patience=1)
    stopper(0.4)
    stopper(0.3)
    assert stopper.early_stop is True
    stopper.reset()
    assert stopper.early_stop is False
    assert stopper.best_score is None
    assert stopper.counter == 0


def test_early_stopping_maximize_with_min_delta():
    """测试最大化方向下的最小改进量"""
    stopper = EarlyStopping(patience=2, min_delta=0.1, direction=ObjectiveDirection.MAXIMIZE)

    # 初始化
    stopper(0.5)

    # 小改进，不足以重置计数器
    stopper(0.55)  # 0.55 - 0.5 = 0.05 < 0.1
    assert stopper.counter == 1
    assert stopper.best_score == 0.5

    # 足够大的改进
    stopper(0.65)  # 0.65 - 0.5 = 0.15 > 0.1
    assert stopper.counter == 0
    assert stopper.best_score == 0.65


def test_early_stopping_minimize_with_min_delta():
    """测试最小化方向下的最小改进量"""
    stopper = EarlyStopping(patience=2, min_delta=0.05, direction=ObjectiveDirection.MINIMIZE)

    # 初始化
    stopper(0.5)

    # 小改进，不足以重置计数器（对于最小化，小改进意味着更大的值）
    stopper(0.52)  # 0.52 - 0.5 = 0.02 < 0.05
    assert stopper.counter == 1
    assert stopper.best_score == 0.5

    # 足够大的改进（更小的值）
    stopper(0.4)  # 0.5 - 0.4 = 0.1 > 0.05
    assert stopper.counter == 0
    assert stopper.best_score == 0.4


def test_early_stopping_no_early_stop_before_patience():
    """测试在达到耐心值之前不会早停"""
    stopper = EarlyStopping(patience=3, direction=ObjectiveDirection.MAXIMIZE)

    stopper(0.5)
    assert not stopper.early_stop

    stopper(0.4)  # worse
    assert not stopper.early_stop
    assert stopper.counter == 1

    stopper(0.3)  # worse
    assert not stopper.early_stop
    assert stopper.counter == 2

    # 第三次变差，达到耐心值
    should_stop = stopper(0.2)
    assert should_stop is True
    assert stopper.early_stop is True


def test_early_stopping_initialization_with_different_params():
    """测试不同初始化参数"""
    # 测试最小化方向
    stopper_min = EarlyStopping(patience=10, min_delta=0.001, direction=ObjectiveDirection.MINIMIZE)
    assert stopper_min.patience == 10
    assert stopper_min.min_delta == 0.001
    assert stopper_min.direction == ObjectiveDirection.MINIMIZE
    assert stopper_min.counter == 0
    assert stopper_min.best_score is None
    assert stopper_min.early_stop is False

    # 测试默认参数
    stopper_default = EarlyStopping()
    assert stopper_default.patience == 5
    assert stopper_default.min_delta == 0.0
    assert stopper_default.direction == ObjectiveDirection.MAXIMIZE


def test_early_stopping_min_delta_edge_cases():
    """测试min_delta的边界情况"""
    # 测试min_delta为0的情况
    stopper = EarlyStopping(patience=2, min_delta=0.0, direction=ObjectiveDirection.MAXIMIZE)
    stopper(0.5)

    # 任何改进都应该重置计数器（因为min_delta=0）
    stopper(0.500001)  # 极小改进
    assert stopper.counter == 0
    assert stopper.best_score == 0.500001

    # 测试精确相等的情况（不应该重置计数器）
    stopper2 = EarlyStopping(patience=2, min_delta=0.0, direction=ObjectiveDirection.MAXIMIZE)
    stopper2(0.5)
    stopper2(0.5)  # 完全相等
    assert stopper2.counter == 1  # 应该增加计数器
    assert stopper2.best_score == 0.5


def test_early_stopping_patience_one():
    """测试patience=1的情况"""
    stopper = EarlyStopping(patience=1, direction=ObjectiveDirection.MAXIMIZE)
    stopper(0.5)

    # 第一次变差就应该触发早停
    should_stop = stopper(0.4)
    assert should_stop is True
    assert stopper.early_stop is True
    assert stopper.counter == 1


def test_early_stopping_multiple_calls_after_early_stop():
    """测试早停后继续调用"""
    stopper = EarlyStopping(patience=1, direction=ObjectiveDirection.MAXIMIZE)
    stopper(0.5)
    stopper(0.4)  # 触发早停

    # 继续调用应该返回True
    assert stopper(0.6) is True  # 即使有改进，也已经早停了
    assert stopper(0.3) is True
    assert stopper.early_stop is True
