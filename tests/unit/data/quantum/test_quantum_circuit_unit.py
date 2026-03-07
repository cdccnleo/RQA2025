import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest

from src.data.quantum.quantum_circuit import (
    QuantumCircuit,
    QuantumGate,
)


def test_validate_gate_and_set_get_qubit_state():
    qc = QuantumCircuit(qubits=3, depth=0)
    gate = QuantumGate("H", [0])
    assert qc.validate_gate_placement(gate) is True
    # 索引越界校验
    bad_gate = QuantumGate("X", [5])
    assert qc.validate_gate_placement(bad_gate) is False
    # 设置/读取量子比特状态（不触发随机执行路径）
    qc.set_qubit_state(1, 1)
    assert qc.get_qubit_state(1) == 1
    # 重置
    qc.reset()
    assert qc.get_qubit_state(1) == 0


def test_add_gate_and_measure_all_no_execute():
    qc = QuantumCircuit(qubits=2, depth=0)
    qc.add_gate("H", [0])
    qc.add_gate("X", [1])
    # 不调用 execute，避免进入随机与噪声路径
    results = qc.measure_all()
    assert isinstance(results, list)
    assert len(results) == 2


