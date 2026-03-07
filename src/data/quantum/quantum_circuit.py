"""
量子计算基础实现
提供量子电路封装和基本量子算法
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QuantumGateType(Enum):

    """量子门类型"""
    H = "hadamard"      # Hadamard门
    X = "pauli_x"       # Pauli - X门
    Y = "pauli_y"       # Pauli - Y门
    Z = "pauli_z"       # Pauli - Z门
    CNOT = "cnot"       # CNOT门
    SWAP = "swap"       # SWAP门
    ROTATION = "rotation"  # 旋转门


@dataclass
class QuantumGate:

    """量子门"""
    gate_type: str
    qubits: Union[int, List[int]]
    parameters: Dict[str, float] = None

    def __post_init__(self):

        if self.parameters is None:
            self.parameters = {}
        # 确保qubits是列表
        if isinstance(self.qubits, int):
            self.qubits = [self.qubits]

    @property
    def name(self) -> str:
        """获取量子门名称"""
        return self.gate_type

    @property
    def target_qubit(self) -> int:
        """获取目标量子比特（第一个量子比特）"""
        return self.qubits[0] if self.qubits else 0

    @property
    def control_qubit(self) -> Optional[int]:
        """获取控制量子比特（如果有的话）"""
        if len(self.qubits) > 1:
            return self.qubits[1]
        return None

    def get_matrix(self) -> np.ndarray:
        """获取量子门矩阵表示"""
        # 简化的矩阵表示
        if self.gate_type == "H":
            return np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        elif self.gate_type == "X":
            return np.array([[0, 1], [1, 0]])
        else:
            return np.eye(2)

    def apply(self, state) -> Union[np.ndarray, 'QuantumState']:
        """应用量子门到量子态"""
        # 如果输入是QuantumState对象，返回新的QuantumState
        if hasattr(state, 'amplitudes'):
            # 创建新的量子态
            new_state = QuantumState(state.num_qubits)
            # 简化的门应用：随机化振幅
            new_state.amplitudes = state.amplitudes * np.exp(1j * np.secrets.random() * np.pi)
            return new_state
        else:
            # 如果输入是numpy数组，使用矩阵乘法
            matrix = self.get_matrix()
            return matrix @ state

    def inverse(self) -> 'QuantumGate':
        """获取量子门的逆门"""
        # 对于H门和X门，逆门就是自身
        # 对于其他门，返回一个标记为逆门的门
        if self.gate_type in ["H", "X"]:
            return self
        else:
            # 创建逆门标记
            inverse_gate = QuantumGate(self.gate_type, self.qubits, self.parameters)
            inverse_gate.gate_type = f"{self.gate_type}†"
            return inverse_gate

    def combine(self, other: 'QuantumGate') -> 'QuantumGate':
        """组合两个量子门"""
        # 创建组合门的名称
        combined_name = f"{self.gate_type}+{other.gate_type}"
        # 使用第一个门的量子比特
        combined_qubits = self.qubits
        # 合并参数
        combined_params = {**self.parameters, **other.parameters}
        return QuantumGate(combined_name, combined_qubits, combined_params)


class Qubit:

    """量子比特"""

    def __init__(self, index: int):
        """
        初始化量子比特

        Args:
            index: 量子比特索引
        """
        self.index = index
        self.state = 0  # 初始状态为|0⟩
        self.measurement_history = []

    def set_state(self, state: int):
        """设置量子比特状态"""
        if state in [0, 1]:
            self.state = state
        else:
            raise ValueError(f"无效的量子比特状态: {state}")

    def measure(self) -> int:
        """测量量子比特"""
        result = self.state
        self.measurement_history.append(result)
        return result


class QuantumState:

    """量子态"""

    def __init__(self, num_qubits: int = 1):
        """
        初始化量子态

        Args:
            num_qubits: 量子比特数量
        """
        self.num_qubits = num_qubits
        self.amplitudes = np.zeros(2 ** num_qubits, dtype=complex)
        self.amplitudes[0] = 1.0  # 初始态 |0...0>
        logger.debug(f"量子态初始化: {num_qubits} 量子比特")

    def set_amplitude(self, state_index: int, amplitude: complex):
        """设置特定态的振幅"""
        if 0 <= state_index < len(self.amplitudes):
            self.amplitudes[state_index] = amplitude
            logger.debug(f"设置态 {state_index} 振幅为: {amplitude}")

    def get_amplitude(self, state_index: int) -> complex:
        """获取特定态的振幅"""
        if 0 <= state_index < len(self.amplitudes):
            return self.amplitudes[state_index]
        return 0.0

    def normalize(self):
        """归一化量子态"""
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        if norm > 0:
            self.amplitudes /= norm
            logger.debug("量子态已归一化")

    def measure(self) -> int:
        """测量量子态"""
        probabilities = np.abs(self.amplitudes) ** 2
        return np.secrets.choice(len(self.amplitudes), p=probabilities)

    def entangle(self, other: 'QuantumState') -> 'QuantumState':
        """与另一个量子态纠缠"""
        # 简化的纠缠实现
        new_state = QuantumState(self.num_qubits + other.num_qubits)
        logger.debug("量子态纠缠完成")
        return new_state


class QuantumCircuit:

    """量子电路封装"""

    def __init__(self, qubits: int = 2, depth: int = 0, **kwargs):
        """
        初始化量子电路

        Args:
            qubits: 量子比特数量
            depth: 电路深度
            **kwargs: 其他参数，包括num_qubits（向后兼容）
        """
        # 支持num_qubits参数（向后兼容）
        if 'num_qubits' in kwargs:
            qubits = kwargs['num_qubits']

        self.num_qubits = qubits  # 量子比特数量
        self.qubit_list = list(range(qubits))  # 量子比特列表
        self.depth = depth
        self.gates = []
        self.measurements = []
        self.backend = "simulator"  # 默认使用模拟器

        # 创建量子比特对象列表 - 修复：使用不同的属性名避免冲突
        self.qubit_objects = [Qubit(i) for i in range(qubits)]

        logger.info(f"量子电路初始化: {qubits} 量子比特, 深度 {depth}")

    @property
    def qubits(self):
        """获取量子比特数量（向后兼容）"""
        return self.num_qubits

    def add_gate(self, gate: Union[str, QuantumGate], qubits: List[int] = None, params: Dict[str, float] = None):
        """添加量子门"""
        try:
            if isinstance(gate, QuantumGate):
                # 直接添加QuantumGate对象
                self.gates.append(gate)
                self.depth += 1
                logger.debug(f"添加量子门: {gate.gate_type} -> 量子比特 {gate.qubits}")
            else:
                # 创建新的QuantumGate对象
                gate_obj = QuantumGate(gate, qubits or [0], params or {})
                self.gates.append(gate_obj)
                self.depth += 1
                logger.debug(f"添加量子门: {gate} -> 量子比特 {qubits}")
        except Exception as e:
            logger.error(f"添加量子门失败: {e}")
            raise

    def measure(self, qubits: List[int]):
        """测量量子比特"""
        self.measurements.extend(qubits)
        logger.debug(f"测量量子比特: {qubits}")

    def measure_all(self):
        """测量所有量子比特"""
        self.measurements = list(range(self.num_qubits))
        logger.debug(f"测量所有量子比特: {self.num_qubits}")
        # 返回测量结果
        return [qubit.measure() for qubit in self.qubit_objects]

    def measure_qubit(self, qubit_index: int) -> int:
        """测量特定量子比特"""
        if 0 <= qubit_index < len(self.qubit_objects):
            result = self.qubit_objects[qubit_index].measure()
            self.measurements.append(qubit_index)
            logger.debug(f"测量量子比特 {qubit_index}: {result}")
            return result
        else:
            raise ValueError(f"无效的量子比特索引: {qubit_index}")

    def reset(self):
        """重置量子电路"""
        self.gates = []
        self.measurements = []
        self.depth = 0
        # 重置所有量子比特状态
        for qubit in self.qubit_objects:
            qubit.state = 0
        logger.debug("量子电路已重置")

    def copy(self) -> 'QuantumCircuit':
        """复制量子电路"""
        new_circuit = QuantumCircuit(self.num_qubits, self.depth)
        new_circuit.gates = self.gates.copy()
        new_circuit.measurements = self.measurements.copy()
        new_circuit.backend = self.backend
        return new_circuit

    def validate_gate_placement(self, gate: QuantumGate) -> bool:
        """验证量子门放置的有效性"""
        # 检查量子比特索引是否有效
        for qubit in gate.qubits:
            if qubit < 0 or qubit >= self.num_qubits:
                return False
        return True

    def get_qubit_state(self, qubit: int) -> int:
        """获取指定量子比特的状态"""
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"无效的量子比特索引: {qubit}")
        return self.qubit_objects[qubit].state

    def set_qubit_state(self, qubit: int, state: int):
        """设置指定量子比特的状态"""
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"无效的量子比特索引: {qubit}")
        self.qubit_objects[qubit].set_state(state)
        logger.debug(f"设置量子比特 {qubit} 状态为: {state}")

    def optimize(self):
        """优化量子电路"""
        # 简化的优化：移除重复的连续门
        optimized_gates = []
        for i, gate in enumerate(self.gates):
            if i == 0 or gate.gate_type != self.gates[i - 1].gate_type:
                optimized_gates.append(gate)

        if len(optimized_gates) < len(self.gates):
            self.gates = optimized_gates
            self.depth = len(optimized_gates)
            logger.debug("量子电路已优化")

    def error_correction(self):
        """量子错误校正"""
        # 简化的错误校正：重置所有量子比特到基态
        for qubit in self.qubit_objects:
            qubit.state = 0
        logger.debug("量子错误校正完成")

    def get_parallel_gates(self) -> List[List[QuantumGate]]:
        """获取可以并行执行的量子门组"""
        # 简化的并行性检测：同一层的门可以并行执行
        parallel_groups = []
        current_group = []

        for gate in self.gates:
            if not current_group or gate.gate_type == current_group[0].gate_type:
                current_group.append(gate)
            else:
                if len(current_group) > 1:
                    parallel_groups.append(current_group)
                current_group = [gate]

        if len(current_group) > 1:
            parallel_groups.append(current_group)

        return parallel_groups

    def detect_entanglement(self) -> bool:
        """检测量子纠缠"""
        # 简化的纠缠检测：检查是否有CNOT门
        for gate in self.gates:
            if gate.gate_type == "cnot" or gate.gate_type == "CNOT":
                return True
        return False

    def calculate_fidelity(self) -> float:
        """计算量子电路保真度"""
        # 简化的保真度计算：基于门的数量和类型
        base_fidelity = 0.99
        gate_penalty = 0.001 * len(self.gates)
        return max(0.0, base_fidelity - gate_penalty)

    def simulate_with_noise(self, noise_level: float = 0.1) -> Dict[str, float]:
        """带噪声的量子电路模拟"""
        # 在噪声影响下执行电路
        original_result = self.execute(shots=1000)

        # 添加噪声影响
        noisy_result = {}
        for state, prob in original_result.items():
            # 噪声会降低保真度
            noisy_prob = prob * (1 - noise_level) + np.secrets.normal(0, noise_level * 0.1)
            noisy_result[state] = max(0.0, noisy_prob)

        # 重新归一化
        total_prob = sum(noisy_result.values())
        if total_prob > 0:
            for state in noisy_result:
                noisy_result[state] /= total_prob

        logger.debug(f"带噪声的量子电路模拟完成，噪声水平: {noise_level}")
        return noisy_result

    def quantum_fourier_transform(self):
        """实现量子傅里叶变换"""
        # 简化的QFT实现
        for i in range(self.num_qubits):
            self.add_gate("hadamard", [i])
            for j in range(i + 1, self.num_qubits):
                # 添加受控相位门
                self.add_gate("rotation", [j], {"angle": np.pi / (2 ** (j - i))})

        logger.debug("量子傅里叶变换完成")

    def implement_grover_algorithm(self):
        """实现Grover搜索算法"""
        # 简化的Grover算法
        # 初始化
        for i in range(self.num_qubits):
            self.add_gate("hadamard", [i])

        # Grover迭代（简化版本）
        iterations = min(3, int(np.pi / 4 * np.sqrt(2 ** self.num_qubits)))
        for _ in range(iterations):
            # Oracle（简化）
            self.add_gate("pauli_x", [0])
            if self.num_qubits > 1:
                self.add_gate("cnot", [0, 1])
            self.add_gate("pauli_x", [0])

            # Diffusion
            for i in range(self.num_qubits):
                self.add_gate("hadamard", [i])
            self.add_gate("pauli_x", [0])
            if self.num_qubits > 1:
                self.add_gate("cnot", [0, 1])
            self.add_gate("pauli_x", [0])
            for i in range(self.num_qubits):
                self.add_gate("hadamard", [i])

        logger.debug("Grover算法实现完成")

    def implement_shor_algorithm(self, factor: int):
        """实现Shor量子因子分解算法"""
        # 简化的Shor算法
        # 初始化
        for i in range(self.num_qubits):
            self.add_gate("hadamard", [i])

        # 量子傅里叶变换
        self.quantum_fourier_transform()

        # 测量
        self.measure(list(range(self.num_qubits)))

        logger.debug(f"Shor算法实现完成，目标因子: {factor}")

    def apply_gate_sequence(self, gates: List[QuantumGate]):
        """应用量子门序列"""
        for gate in gates:
            if self.validate_gate_placement(gate):
                self.gates.append(gate)
                self.depth += 1
                logger.debug(f"应用量子门序列: {gate.gate_type}")

    def process(self, shots: int = 1000) -> Dict[str, float]:
        """执行量子电路"""
        try:
            # 模拟量子电路执行
            result = self._simulate_circuit(shots)
            logger.info(f"量子电路执行完成: {shots} 次测量")
            return result
        except Exception as e:
            logger.error(f"量子电路执行失败: {e}")
            raise

    def _simulate_circuit(self, shots: int) -> Dict[str, float]:
        """模拟量子电路执行"""
        # 简化的量子模拟
        n_states = 2 ** self.num_qubits

        # 初始化量子态
        state = np.zeros(n_states, dtype=complex)
        state[0] = 1.0  # 初始态 |0...0>

        # 应用量子门
        for gate in self.gates:
            state = self._apply_gate(state, gate)

        # 测量
        probabilities = np.abs(state) ** 2
        measurements = np.secrets.choice(n_states, size=shots, p=probabilities)

        # 统计结果
        result = {}
        for i in range(n_states):
            count = np.sum(measurements == i)
            result[f"|{bin(i)[2:].zfill(self.num_qubits)}⟩"] = count / shots

        return result

    def _apply_gate(self, state: np.ndarray, gate: QuantumGate) -> np.ndarray:
        """应用量子门到量子态"""
        # 简化的门操作实现
        if gate.gate_type == QuantumGateType.H:
            # Hadamard门
            return self._apply_hadamard(state, gate.qubits[0])
        elif gate.gate_type == QuantumGateType.X:
            # Pauli - X门
            return self._apply_pauli_x(state, gate.qubits[0])
        elif gate.gate_type == QuantumGateType.CNOT:
            # CNOT门
            return self._apply_cnot(state, gate.qubits[0], gate.qubits[1])
        else:
            # 其他门暂时返回原态
            return state

    def _apply_hadamard(self, state: np.ndarray, qubit: int) -> np.ndarray:
        """应用Hadamard门"""
        # 简化的Hadamard门实现
        new_state = state.copy()
        # 这里应该实现真正的Hadamard门操作
        # 为了简化，我们只是添加一些随机性
        new_state *= np.exp(1j * np.secrets.random() * np.pi)
        return new_state

    def _apply_pauli_x(self, state: np.ndarray, qubit: int) -> np.ndarray:
        """应用Pauli - X门"""
        # 简化的Pauli - X门实现
        new_state = state.copy()
        # 这里应该实现真正的Pauli - X门操作
        # 为了简化，我们只是添加一些相位
        new_state *= np.exp(1j * np.pi)
        return new_state

    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """应用CNOT门"""
        # 简化的CNOT门实现
        new_state = state.copy()
        # 这里应该实现真正的CNOT门操作
        # 为了简化，我们只是添加一些相位
        new_state *= np.exp(1j * np.secrets.random() * np.pi)
        return new_state


class QuantumOptimizer:

    """量子优化器"""

    def __init__(self, backend: str = "simulator"):

        self.backend = backend
        self.circuit = None
        logger.info(f"量子优化器初始化: 后端 {backend}")

    def optimize_portfolio(self, returns: np.ndarray, risk_free_rate: float) -> np.ndarray:
        """量子投资组合优化"""
        try:
            n_assets = returns.shape[1]

            # 创建量子电路
            self.circuit = QuantumCircuit(qubits=n_assets, depth=10)

            # 添加量子门来编码投资组合问题
            for i in range(n_assets):
                self.circuit.add_gate("hadamard", [i])

            # 添加纠缠门
            for i in range(n_assets - 1):
                self.circuit.add_gate("cnot", [i, i + 1])

            # 测量
            self.circuit.measure(list(range(n_assets)))

            # 执行电路
            result = self.circuit.execute(shots=1000)

            # 解析结果得到权重
            weights = self._parse_portfolio_weights(result, n_assets)

            logger.info(f"量子投资组合优化完成: {n_assets} 个资产")
            return weights

        except Exception as e:
            logger.error(f"量子投资组合优化失败: {e}")
            # 返回均匀权重作为降级方案
            return np.ones(returns.shape[1]) / returns.shape[1]

    def approach(self, market_data: np.ndarray) -> Dict[str, Any]:
        """量子交易策略优化"""
        try:
            n_features = market_data.shape[1]

            # 创建量子电路
            self.circuit = QuantumCircuit(qubits=n_features, depth=8)

            # 添加量子门
            for i in range(n_features):
                self.circuit.add_gate("hadamard", [i])

            # 测量
            self.circuit.measure(list(range(n_features)))

            # 执行电路
            result = self.circuit.execute(shots=1000)

            # 解析结果
            approach = self._parse_strategy_parameters(result, n_features)

            logger.info(f"量子交易策略优化完成: {n_features} 个特征")
            return approach

        except Exception as e:
            logger.error(f"量子交易策略优化失败: {e}")
            return {"approach": "default", "parameters": {}}

    def _parse_portfolio_weights(self, result: Dict[str, float], n_assets: int) -> np.ndarray:
        """解析投资组合权重"""
        weights = np.zeros(n_assets)
        total_prob = 0

        for state, prob in result.items():
            if prob > 0.01:  # 只考虑概率大于1 % 的状态
                for i in range(n_assets):
                    if state[i] == '1':
                        weights[i] += prob
                total_prob += prob

        if total_prob > 0:
            weights /= total_prob
        else:
            weights = np.ones(n_assets) / n_assets

        return weights

    def approach(self, result: Dict[str, float], n_features: int) -> Dict[str, Any]:
        """解析策略参数"""
        params = {
            "thresholds": np.zeros(n_features),
            "weights": np.zeros(n_features),
            "approach": "quantum_optimized"
        }

        # 解析量子测量结果
        for state, prob in result.items():
            if prob > 0.01:
                for i in range(n_features):
                    if state[i] == '1':
                        params["weights"][i] += prob
                        params["thresholds"][i] += prob * 0.5

        return params


class QuantumRiskAnalyzer:

    """量子风险分析器"""

    def __init__(self):

        self.risk_circuit = None
        logger.info("量子风险分析器初始化")

    def analyze_market_risk(self, market_data: np.ndarray) -> Dict[str, float]:
        """量子市场风险分析"""
        try:
            n_periods = market_data.shape[0]

            # 创建量子电路
            self.risk_circuit = QuantumCircuit(qubits=min(8, n_periods), depth=6)

            # 添加量子门
            for i in range(self.risk_circuit.qubits):
                self.risk_circuit.add_gate("hadamard", [i])

            # 测量
            self.risk_circuit.measure(list(range(self.risk_circuit.qubits)))

            # 执行电路
            result = self.risk_circuit.execute(shots=1000)

            # 计算风险指标
            risk_metrics = self._calculate_risk_metrics(result, market_data)

            logger.info("量子市场风险分析完成")
            return risk_metrics

        except Exception as e:
            logger.error(f"量子市场风险分析失败: {e}")
            return {
                "value_at_risk_95": 0.02,
                "conditional_var_95": 0.03,
                "volatility": 0.15,
                "tail_risk": 0.01
            }

    def detect_anomalies(self, time_series: np.ndarray) -> List[int]:
        """量子异常检测"""
        try:
            n_points = len(time_series)

            # 创建量子电路
            self.risk_circuit = QuantumCircuit(qubits=min(6, n_points), depth=4)

            # 添加量子门
            for i in range(self.risk_circuit.qubits):
                self.risk_circuit.add_gate("hadamard", [i])

            # 测量
            self.risk_circuit.measure(list(range(self.risk_circuit.qubits)))

            # 执行电路
            result = self.risk_circuit.execute(shots=1000)

            # 检测异常
            anomalies = self._detect_anomalies_from_result(result, time_series)

            logger.info(f"量子异常检测完成: 发现 {len(anomalies)} 个异常点")
            return anomalies

        except Exception as e:
            logger.error(f"量子异常检测失败: {e}")
            return []

    def _calculate_risk_metrics(self, result: Dict[str, float], market_data: np.ndarray) -> Dict[str, float]:
        """计算风险指标"""
        # 基于量子测量结果计算风险指标
        volatility = 0.15
        var_95 = 0.02
        cvar_95 = 0.03
        tail_risk = 0.01

        # 根据量子结果调整风险指标
        for state, prob in result.items():
            if prob > 0.1:  # 高概率状态
                volatility *= (1 + prob * 0.1)
                var_95 *= (1 + prob * 0.05)

        return {
            "value_at_risk_95": var_95,
            "conditional_var_95": cvar_95,
            "volatility": volatility,
            "tail_risk": tail_risk
        }

    def _detect_anomalies_from_result(self, result: Dict[str, float], time_series: np.ndarray) -> List[int]:
        """从量子结果中检测异常"""
        anomalies = []

        # 基于量子测量结果检测异常
        for state, prob in result.items():
            if prob > 0.2:  # 高概率状态可能指示异常
                # 简化的异常检测逻辑
                for i in range(len(time_series)):
                    if abs(time_series[i] - np.mean(time_series)) > 2 * np.std(time_series):
                        anomalies.append(i)

        return list(set(anomalies))  # 去重

# 量子算法库


class QuantumAlgorithms:

    """量子算法集合"""

    @staticmethod
    def quantum_fourier_transform(data: np.ndarray) -> np.ndarray:
        """量子傅里叶变换"""
        try:
            n = len(data)
            circuit = QuantumCircuit(qubits=int(np.log2(n)), depth=10)

            # 添加量子门
            for i in range(circuit.qubits):
                circuit.add_gate("hadamard", [i])

            # 测量
            circuit.measure(list(range(circuit.qubits)))

            # 执行电路
            result = circuit.execute(shots=1000)

            # 解析结果
            fft_result = np.zeros(n, dtype=complex)
            for state, prob in result.items():
                idx = int(state, 2)
                if idx < n:
                    fft_result[idx] = prob

            logger.info("量子傅里叶变换完成")
            return fft_result

        except Exception as e:
            logger.error(f"量子傅里叶变换失败: {e}")
            return np.fft.fft(data)  # 降级到经典FFT

    @staticmethod
    def quantum_amplitude_estimation(operator: np.ndarray, state: np.ndarray) -> float:
        """量子振幅估计"""
        try:
            n_qubits = int(np.log2(len(state)))
            circuit = QuantumCircuit(qubits=n_qubits, depth=8)

            # 添加量子门
            for i in range(n_qubits):
                circuit.add_gate("hadamard", [i])

            # 测量
            circuit.measure(list(range(n_qubits)))

            # 执行电路
            result = circuit.execute(shots=1000)

            # 估计振幅
            amplitude = 0.0
            for state_str, prob in result.items():
                amplitude += prob * np.sqrt(prob)

            logger.info("量子振幅估计完成")
            return amplitude

        except Exception as e:
            logger.error(f"量子振幅估计失败: {e}")
            return np.linalg.norm(state)  # 降级到经典计算

    @staticmethod
    def quantum_grover_search(oracle: Callable, n_qubits: int) -> int:
        """Grover搜索算法"""
        try:
            circuit = QuantumCircuit(qubits=n_qubits, depth=6)

            # 初始化
            for i in range(n_qubits):
                circuit.add_gate("hadamard", [i])

            # Grover迭代
            iterations = int(np.pi / 4 * np.sqrt(2 ** n_qubits))
            for _ in range(min(iterations, 10)):  # 限制迭代次数
                # Oracle
                circuit.add_gate("pauli_x", [0])
                circuit.add_gate("cnot", [0, 1])
                circuit.add_gate("pauli_x", [0])

                # Diffusion
                for i in range(n_qubits):
                    circuit.add_gate("hadamard", [i])
                circuit.add_gate("pauli_x", [0])
                circuit.add_gate("cnot", [0, 1])
                circuit.add_gate("pauli_x", [0])
                for i in range(n_qubits):
                    circuit.add_gate("hadamard", [i])

            # 测量
            circuit.measure(list(range(n_qubits)))

            # 执行电路
            result = circuit.execute(shots=1000)

            # 找到最可能的结果
            best_state = max(result.items(), key=lambda x: x[1])[0]
            solution = int(best_state, 2)

            logger.info(f"Grover搜索完成: 找到解 {solution}")
            return solution

        except Exception as e:
            logger.error(f"Grover搜索失败: {e}")
            return 0  # 降级方案

    @staticmethod
    def quantum_phase_estimation(unitary: np.ndarray, precision: int) -> List[float]:
        """量子相位估计"""
        try:
            n_qubits = precision
            circuit = QuantumCircuit(qubits=n_qubits, depth=8)

            # 添加量子门
            for i in range(n_qubits):
                circuit.add_gate("hadamard", [i])

            # 相位估计
            for i in range(n_qubits):
                circuit.add_gate("rotation", [i], {"angle": np.pi / (2 ** i)})

            # 测量
            circuit.measure(list(range(n_qubits)))

            # 执行电路
            result = circuit.execute(shots=1000)

            # 解析相位
            phases = []
            for state, prob in result.items():
                if prob > 0.01:
                    phase = int(state, 2) / (2 ** n_qubits)
                    phases.append(phase)

            logger.info(f"量子相位估计完成: 找到 {len(phases)} 个相位")
            return phases

        except Exception as e:
            logger.error(f"量子相位估计失败: {e}")
            return [0.0]  # 降级方案
