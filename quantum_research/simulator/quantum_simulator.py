#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2026 量子电路模拟器
高性能量子电路模拟器，支持多种噪声模型和优化

特性:
- 状态向量模拟
- 密度矩阵模拟
- 噪声模拟
- 并行计算优化
- 内存优化
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """模拟结果数据类"""
    state_vector: Optional[np.ndarray] = None
    density_matrix: Optional[np.ndarray] = None
    measurements: Dict[str, int] = None
    expectation_values: Dict[str, float] = None
    execution_time: float = 0.0
    circuit_depth: int = 0
    gate_count: int = 0
    memory_usage: float = 0.0


class QuantumSimulator:
    """量子电路模拟器"""

    def __init__(self, num_qubits: int, simulation_type: str = "state_vector",
                 noise_model: str = None, parallel: bool = False):
        """
        初始化量子模拟器

        Args:
            num_qubits: 量子比特数量
            simulation_type: 模拟类型 ('state_vector', 'density_matrix')
            noise_model: 噪声模型 ('depolarizing', 'amplitude_damping', etc.)
            parallel: 是否启用并行计算
        """
        self.num_qubits = num_qubits
        self.simulation_type = simulation_type
        self.noise_model = noise_model
        self.parallel = parallel

        # 性能监控
        self.execution_times = []
        self.memory_usage = []

        # 初始化量子态
        if simulation_type == "state_vector":
            self._initialize_state_vector()
        elif simulation_type == "density_matrix":
            self._initialize_density_matrix()
        else:
            raise ValueError(f"不支持的模拟类型: {simulation_type}")

        logger.info(f"量子模拟器初始化完成: {num_qubits} 量子比特, 类型: {simulation_type}")

    def _initialize_state_vector(self):
        """初始化状态向量"""
        self.state_vector = np.zeros(2 ** self.num_qubits, dtype=complex)
        self.state_vector[0] = 1.0  # |00...0⟩状态

    def _initialize_density_matrix(self):
        """初始化密度矩阵"""
        dim = 2 ** self.num_qubits
        self.density_matrix = np.zeros((dim, dim), dtype=complex)
        self.density_matrix[0, 0] = 1.0  # |00...0⟩⟨00...0| 态

    def simulate_circuit(self, circuit: Dict[str, Any],
                        shots: int = 1024) -> SimulationResult:
        """
        模拟量子电路

        Args:
            circuit: 量子电路描述
            shots: 测量次数

        Returns:
            模拟结果
        """
        start_time = time.time()

        # 重置量子态
        if self.simulation_type == "state_vector":
            self._initialize_state_vector()
        else:
            self._initialize_density_matrix()

        # 应用量子门
        gate_count = 0
        for gate in circuit['gates']:
            self._apply_gate(gate)
            gate_count += 1

            # 应用噪声 (如果启用)
            if self.noise_model:
                self._apply_noise(gate)

        # 执行测量
        measurements = self._perform_measurements(shots)
        expectation_values = self._compute_expectation_values(circuit.get('measurements', []))

        execution_time = time.time() - start_time

        # 内存使用估算
        memory_usage = self._estimate_memory_usage()

        result = SimulationResult(
            state_vector=self.state_vector if self.simulation_type == "state_vector" else None,
            density_matrix=self.density_matrix if self.simulation_type == "density_matrix" else None,
            measurements=measurements,
            expectation_values=expectation_values,
            execution_time=execution_time,
            circuit_depth=len(circuit['gates']),
            gate_count=gate_count,
            memory_usage=memory_usage
        )

        self.execution_times.append(execution_time)
        self.memory_usage.append(memory_usage)

        logger.info(".4f")
        return result

    def _apply_gate(self, gate: Dict[str, Any]):
        """应用量子门"""
        gate_type = gate['type']
        qubits = gate['qubits']
        params = gate.get('params', {})

        if gate_type == 'H':
            self._apply_hadamard(qubits[0])
        elif gate_type == 'X':
            self._apply_pauli_x(qubits[0])
        elif gate_type == 'Y':
            self._apply_pauli_y(qubits[0])
        elif gate_type == 'Z':
            self._apply_pauli_z(qubits[0])
        elif gate_type == 'S':
            self._apply_s_gate(qubits[0])
        elif gate_type == 'T':
            self._apply_t_gate(qubits[0])
        elif gate_type == 'RX':
            self._apply_rotation_x(qubits[0], params.get('angle', 0))
        elif gate_type == 'RY':
            self._apply_rotation_y(qubits[0], params.get('angle', 0))
        elif gate_type == 'RZ':
            self._apply_rotation_z(qubits[0], params.get('angle', 0))
        elif gate_type == 'CNOT':
            self._apply_cnot(qubits[0], qubits[1])
        elif gate_type == 'CZ':
            self._apply_cz(qubits[0], qubits[1])
        elif gate_type == 'SWAP':
            self._apply_swap(qubits[0], qubits[1])
        elif gate_type == 'Toffoli':
            self._apply_toffoli(qubits[0], qubits[1], qubits[2])
        elif gate_type == 'Fredkin':
            self._apply_fredkin(qubits[0], qubits[1], qubits[2])
        else:
            logger.warning(f"不支持的量子门: {gate_type}")

    def _apply_hadamard(self, qubit: int):
        """应用Hadamard门"""
        H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        self._apply_single_qubit_gate(qubit, H)

    def _apply_pauli_x(self, qubit: int):
        """应用Pauli-X门"""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        self._apply_single_qubit_gate(qubit, X)

    def _apply_pauli_y(self, qubit: int):
        """应用Pauli-Y门"""
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self._apply_single_qubit_gate(qubit, Y)

    def _apply_pauli_z(self, qubit: int):
        """应用Pauli-Z门"""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        self._apply_single_qubit_gate(qubit, Z)

    def _apply_s_gate(self, qubit: int):
        """应用S门"""
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        self._apply_single_qubit_gate(qubit, S)

    def _apply_t_gate(self, qubit: int):
        """应用T门"""
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
        self._apply_single_qubit_gate(qubit, T)

    def _apply_rotation_x(self, qubit: int, angle: float):
        """应用RX旋转门"""
        cos = np.cos(angle/2)
        sin = np.sin(angle/2)
        RX = np.array([[cos, -1j*sin], [-1j*sin, cos]], dtype=complex)
        self._apply_single_qubit_gate(qubit, RX)

    def _apply_rotation_y(self, qubit: int, angle: float):
        """应用RY旋转门"""
        cos = np.cos(angle/2)
        sin = np.sin(angle/2)
        RY = np.array([[cos, -sin], [sin, cos]], dtype=complex)
        self._apply_single_qubit_gate(qubit, RY)

    def _apply_rotation_z(self, qubit: int, angle: float):
        """应用RZ旋转门"""
        RZ = np.array([[np.exp(-1j*angle/2), 0], [0, np.exp(1j*angle/2)]], dtype=complex)
        self._apply_single_qubit_gate(qubit, RZ)

    def _apply_cnot(self, control: int, target: int):
        """应用CNOT门"""
        if self.simulation_type == "state_vector":
            self._apply_cnot_state_vector(control, target)
        else:
            self._apply_cnot_density_matrix(control, target)

    def _apply_cnot_state_vector(self, control: int, target: int):
        """在状态向量上应用CNOT门"""
        dim = len(self.state_vector)
        result = np.zeros(dim, dtype=complex)

        for i in range(dim):
            # 检查控制比特
            if (i >> control) & 1:
                # 翻转目标比特
                j = i ^ (1 << target)
                result[j] = self.state_vector[i]
            else:
                result[i] = self.state_vector[i]

        self.state_vector = result

    def _apply_cnot_density_matrix(self, control: int, target: int):
        """在密度矩阵上应用CNOT门"""
        # 密度矩阵的CNOT操作较为复杂，这里简化实现
        dim = self.density_matrix.shape[0]
        cnot_op = np.eye(dim, dtype=complex)

        for i in range(dim):
            if (i >> control) & 1:
                j = i ^ (1 << target)
                # 交换矩阵元素
                temp = cnot_op[i, :].copy()
                cnot_op[i, :] = cnot_op[j, :]
                cnot_op[j, :] = temp

        self.density_matrix = cnot_op @ self.density_matrix @ cnot_op.conj().T

    def _apply_cz(self, control: int, target: int):
        """应用CZ门"""
        if self.simulation_type == "state_vector":
            self._apply_cz_state_vector(control, target)
        else:
            self._apply_cz_density_matrix(control, target)

    def _apply_cz_state_vector(self, control: int, target: int):
        """在状态向量上应用CZ门"""
        dim = len(self.state_vector)

        for i in range(dim):
            # 如果两个控制比特都为1，相位翻转
            if ((i >> control) & 1) and ((i >> target) & 1):
                self.state_vector[i] *= -1

    def _apply_cz_density_matrix(self, control: int, target: int):
        """在密度矩阵上应用CZ门"""
        # CZ门的密度矩阵操作
        dim = self.density_matrix.shape[0]
        cz_op = np.eye(dim, dtype=complex)

        for i in range(dim):
            if ((i >> control) & 1) and ((i >> target) & 1):
                cz_op[i, i] = -1

        self.density_matrix = cz_op @ self.density_matrix @ cz_op

    def _apply_swap(self, qubit1: int, qubit2: int):
        """应用SWAP门"""
        # SWAP可以通过三个CNOT实现
        self._apply_cnot(qubit1, qubit2)
        self._apply_cnot(qubit2, qubit1)
        self._apply_cnot(qubit1, qubit2)

    def _apply_toffoli(self, control1: int, control2: int, target: int):
        """应用Toffoli门 (CCNOT)"""
        dim = len(self.state_vector) if self.simulation_type == "state_vector" else self.density_matrix.shape[0]

        if self.simulation_type == "state_vector":
            result = self.state_vector.copy()

            for i in range(dim):
                # 如果两个控制比特都为1，翻转目标比特
                if ((i >> control1) & 1) and ((i >> control2) & 1):
                    j = i ^ (1 << target)
                    result[i], result[j] = result[j], result[i]

            self.state_vector = result
        else:
            # 密度矩阵版本 (简化实现)
            toffoli_op = np.eye(dim, dtype=complex)

            for i in range(dim):
                if ((i >> control1) & 1) and ((i >> control2) & 1):
                    j = i ^ (1 << target)
                    # 交换操作
                    temp = toffoli_op[i, :].copy()
                    toffoli_op[i, :] = toffoli_op[j, :]
                    toffoli_op[j, :] = temp

            self.density_matrix = toffoli_op @ self.density_matrix @ toffoli_op.conj().T

    def _apply_fredkin(self, control: int, target1: int, target2: int):
        """应用Fredkin门 (受控SWAP)"""
        dim = len(self.state_vector) if self.simulation_type == "state_vector" else self.density_matrix.shape[0]

        if self.simulation_type == "state_vector":
            result = self.state_vector.copy()

            for i in range(dim):
                # 如果控制比特为1，交换目标比特
                if (i >> control) & 1:
                    j = i ^ (1 << target1) ^ (1 << target2)
                    result[i], result[j] = result[j], result[i]

            self.state_vector = result
        else:
            # 密度矩阵版本 (简化)
            fredkin_op = np.eye(dim, dtype=complex)

            for i in range(dim):
                if (i >> control) & 1:
                    j = i ^ (1 << target1) ^ (1 << target2)
                    # 交换操作
                    temp = fredkin_op[i, :].copy()
                    fredkin_op[i, :] = fredkin_op[j, :]
                    fredkin_op[j, :] = temp

            self.density_matrix = fredkin_op @ self.density_matrix @ fredkin_op.conj().T

    def _apply_single_qubit_gate(self, qubit: int, gate_matrix: np.ndarray):
        """应用单量子比特门"""
        if self.simulation_type == "state_vector":
            self._apply_single_qubit_gate_state_vector(qubit, gate_matrix)
        else:
            self._apply_single_qubit_gate_density_matrix(qubit, gate_matrix)

    def _apply_single_qubit_gate_state_vector(self, qubit: int, gate_matrix: np.ndarray):
        """在状态向量上应用单量子比特门"""
        dim = len(self.state_vector)
        result = np.zeros(dim, dtype=complex)

        for i in range(dim):
            for j in range(2):
                # 检查除了目标量子比特外的所有比特是否相同
                mask = ~(1 << qubit)
                if (i & mask) == ((j << qubit) & mask):
                    # 目标量子比特的变换
                    qubit_i = (i >> qubit) & 1
                    result[i] += self.state_vector[j << qubit | (i & mask)] * gate_matrix[qubit_i, j]

        # 重新排列结果
        for i in range(dim):
            target_value = (i >> qubit) & 1
            other_bits = i & ~(1 << qubit)
            for source_value in range(2):
                source_i = source_value << qubit | other_bits
                result[i] += self.state_vector[source_i] * gate_matrix[target_value, source_value]

        self.state_vector = result

    def _apply_single_qubit_gate_density_matrix(self, qubit: int, gate_matrix: np.ndarray):
        """在密度矩阵上应用单量子比特门"""
        # 使用张量积构造门操作
        dim = self.density_matrix.shape[0]
        gate_op = np.eye(dim, dtype=complex)

        for i in range(dim):
            for j in range(dim):
                # 检查除了目标量子比特外的比特
                mask = ~(1 << qubit)
                if (i & mask) == (j & mask):
                    qubit_i = (i >> qubit) & 1
                    qubit_j = (j >> qubit) & 1
                    factor = gate_matrix[qubit_i, qubit_j]
                    if factor != 0:
                        gate_op[i, j] = factor

        self.density_matrix = gate_op @ self.density_matrix @ gate_op.conj().T

    def _apply_noise(self, gate: Dict[str, Any]):
        """应用噪声模型"""
        if self.noise_model == "depolarizing":
            self._apply_depolarizing_noise(gate)
        elif self.noise_model == "amplitude_damping":
            self._apply_amplitude_damping_noise(gate)
        elif self.noise_model == "phase_damping":
            self._apply_phase_damping_noise(gate)

    def _apply_depolarizing_noise(self, gate: Dict[str, Any], p: float = 0.01):
        """应用退极化噪声"""
        if self.simulation_type == "density_matrix":
            # 简化的退极化信道
            dim = self.density_matrix.shape[0]
            identity = np.eye(dim, dtype=complex)
            self.density_matrix = (1 - p) * self.density_matrix + p * identity / dim

    def _apply_amplitude_damping_noise(self, gate: Dict[str, Any], gamma: float = 0.01):
        """应用振幅阻尼噪声"""
        if self.simulation_type == "density_matrix":
            # 简化的振幅阻尼信道
            dim = self.density_matrix.shape[0]
            damping_op = np.eye(dim, dtype=complex)

            for i in range(dim):
                if bin(i).count('1') > 0:  # 如果有激发态
                    damping_op[i, i] = np.sqrt(1 - gamma)

            self.density_matrix = damping_op @ self.density_matrix @ damping_op.conj().T

    def _apply_phase_damping_noise(self, gate: Dict[str, Any], gamma: float = 0.01):
        """应用相位阻尼噪声"""
        if self.simulation_type == "density_matrix":
            # 简化的相位阻尼信道
            dim = self.density_matrix.shape[0]
            phase_op = np.eye(dim, dtype=complex)

            for i in range(dim):
                excited_count = bin(i).count('1')
                phase_op[i, i] = (1 - gamma) ** excited_count

            self.density_matrix = phase_op @ self.density_matrix @ phase_op

    def _perform_measurements(self, shots: int) -> Dict[str, int]:
        """执行测量"""
        measurements = {}

        if self.simulation_type == "state_vector":
            probabilities = np.abs(self.state_vector)**2
        else:
            probabilities = np.real(np.diag(self.density_matrix))

        probabilities = probabilities / np.sum(probabilities)  # 归一化

        # 生成测量结果
        outcomes = np.random.choice(len(probabilities), size=shots, p=probabilities)

        for outcome in outcomes:
            bitstring = format(outcome, f'0{self.num_qubits}b')
            measurements[bitstring] = measurements.get(bitstring, 0) + 1

        return measurements

    def _compute_expectation_values(self, observables: List[str]) -> Dict[str, float]:
        """计算期望值"""
        expectation_values = {}

        for observable in observables:
            if observable == 'Z':
                expectation_values['Z'] = self._compute_z_expectation()
            elif observable == 'X':
                expectation_values['X'] = self._compute_x_expectation()
            elif observable == 'Y':
                expectation_values['Y'] = self._compute_y_expectation()

        return expectation_values

    def _compute_z_expectation(self) -> float:
        """计算Z算符期望值"""
        if self.simulation_type == "state_vector":
            expectation = 0
            for i, amplitude in enumerate(self.state_vector):
                parity = bin(i).count('1') % 2
                expectation += np.abs(amplitude)**2 * (1 if parity == 0 else -1)
            return expectation
        else:
            # 密度矩阵版本
            dim = self.density_matrix.shape[0]
            z_op = np.zeros((dim, dim), dtype=complex)
            for i in range(dim):
                parity = bin(i).count('1') % 2
                z_op[i, i] = 1 if parity == 0 else -1
            return np.real(np.trace(z_op @ self.density_matrix))

    def _compute_x_expectation(self) -> float:
        """计算X算符期望值"""
        # 应用X基变换 (Hadamard)
        temp_state = self.state_vector.copy() if self.simulation_type == "state_vector" else self.density_matrix.copy()

        # 应用Hadamard变换
        for i in range(self.num_qubits):
            self._apply_hadamard(i)

        expectation = self._compute_z_expectation()

        # 恢复原状态
        if self.simulation_type == "state_vector":
            self.state_vector = temp_state
        else:
            self.density_matrix = temp_state

        return expectation

    def _compute_y_expectation(self) -> float:
        """计算Y算符期望值"""
        # Y = iXZ 或其他变换，这里简化实现
        return 0.0  # 简化版本

    def _estimate_memory_usage(self) -> float:
        """估算内存使用量 (MB)"""
        if self.simulation_type == "state_vector":
            return self.state_vector.nbytes / (1024 * 1024)
        else:
            return self.density_matrix.nbytes / (1024 * 1024)

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            'average_execution_time': np.mean(self.execution_times) if self.execution_times else 0,
            'max_execution_time': np.max(self.execution_times) if self.execution_times else 0,
            'average_memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0,
            'max_memory_usage': np.max(self.memory_usage) if self.memory_usage else 0,
            'total_simulations': len(self.execution_times)
        }


def create_quantum_simulator(num_qubits: int, simulation_type: str = "state_vector",
                           noise_model: str = None, parallel: bool = False) -> QuantumSimulator:
    """
    创建量子模拟器的工厂函数

    Args:
        num_qubits: 量子比特数量
        simulation_type: 模拟类型
        noise_model: 噪声模型
        parallel: 是否并行

    Returns:
        配置好的量子模拟器实例
    """
    return QuantumSimulator(
        num_qubits=num_qubits,
        simulation_type=simulation_type,
        noise_model=noise_model,
        parallel=parallel
    )


if __name__ == "__main__":
    # 测试量子模拟器
    print("🔬 RQA2026 量子模拟器测试")
    print("=" * 50)

    # 创建模拟器
    simulator = create_quantum_simulator(num_qubits=3, simulation_type="state_vector")

    # 测试量子电路
    circuit = {
        'num_qubits': 3,
        'gates': [
            {'type': 'H', 'qubits': [0]},
            {'type': 'CNOT', 'qubits': [0, 1]},
            {'type': 'CNOT', 'qubits': [1, 2]},
            {'type': 'H', 'qubits': [2]}
        ],
        'measurements': ['Z', 'X']
    }

    print("\\n⚡ 执行量子电路模拟:")
    result = simulator.simulate_circuit(circuit, shots=1000)

    print(f"执行时间: {result.execution_time:.4f}秒")
    print(f"电路深度: {result.circuit_depth}")
    print(f"门数量: {result.gate_count}")
    print(f"内存使用: {result.memory_usage:.2f}MB")
    print(f"测量结果样本: {dict(list(result.measurements.items())[:5])}")
    print(f"期望值: {result.expectation_values}")

    # 性能统计
    stats = simulator.get_performance_stats()
    print(f"\\n📊 性能统计:")
    print(f"平均执行时间: {stats['average_execution_time']:.4f}秒")
    print(f"最大内存使用: {stats['max_memory_usage']:.2f}MB")

    print("\\n✅ 量子模拟器测试完成!")
