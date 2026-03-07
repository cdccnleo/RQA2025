import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.core.ml_core import MLCore
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def _train_dummy_model(core: MLCore) -> str:
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.5, 1.5, 2.5]})
    y = pd.Series([1.1, 2.6, 3.9])
    return core.train_model(X, y, model_type="linear")


def test_save_and_load_model(tmp_path):
    core = MLCore()
    model_id = _train_dummy_model(core)

    filepath = tmp_path / "model.joblib"
    assert core.save_model(model_id, str(filepath))

    new_core = MLCore()
    loaded_id = new_core.load_model(str(filepath))
    assert loaded_id is not None
    predictions = new_core.predict(loaded_id, pd.DataFrame({"x": [4.0], "y": [3.5]}))
    assert predictions.shape == (1,)

