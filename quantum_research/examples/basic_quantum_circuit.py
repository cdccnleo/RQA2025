# -*- coding: utf-8 -*-
"""
基础量子电路示例
演示量子计算的基本概念和操作
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer


def create_bell_state():
    """创建Bell状态 (量子纠缠示例)"""
    # 创建2量子比特电路
    qc = QuantumCircuit(2, 2)

    # 应用Hadamard门到第一个量子比特
    qc.h(0)

    # 应用CNOT门
    qc.cx(0, 1)

    # 测量所有量子比特
    qc.measure_all()

    return qc


def run_quantum_circuit(qc, shots=1000):
    """运行量子电路并返回结果"""
    # 使用Aer模拟器
    simulator = AerSimulator()

    # 编译电路
    compiled_circuit = transpile(qc, simulator)

    # 运行模拟
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()

    # 获取计数
    counts = result.get_counts()

    return counts


def demonstrate_superposition():
    """演示量子叠加原理"""
    print("🔬 量子叠加演示")

    # 创建单量子比特电路
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # Hadamard门创建叠加状态
    qc.measure_all()

    # 运行电路
    counts = run_quantum_circuit(qc, shots=10000)

    print("测量结果 (期望约50% |0⟩ 和 50% |1⟩):")
    for outcome, count in counts.items():
        probability = count / 10000 * 100
        print(f"  |{outcome}⟩: {probability:.1f}%")

    return counts


def demonstrate_entanglement():
    """演示量子纠缠"""
    print("\n🔗 量子纠缠演示 (Bell状态)")

    # 创建Bell状态电路
    qc = create_bell_state()

    # 运行电路
    counts = run_quantum_circuit(qc, shots=10000)

    print("Bell状态测量结果 (期望约50% |00⟩ 和 50% |11⟩):")
    for outcome, count in counts.items():
        probability = count / 10000 * 100
        print(f"  |{outcome}⟩: {probability:.1f}%")

    return counts


def visualize_circuit():
    """可视化量子电路"""
    print("\n🎨 生成电路图...")

    qc = create_bell_state()

    # 保存电路图
    circuit_drawer(qc, output='mpl', filename='bell_state_circuit.png')
    print("电路图已保存为: bell_state_circuit.png")


if __name__ == "__main__":
    print("🚀 RQA2026量子计算基础示例")
    print("=" * 50)

    try:
        # 演示量子叠加
        demonstrate_superposition()

        # 演示量子纠缠
        demonstrate_entanglement()

        # 可视化电路
        visualize_circuit()

        print("\n🎉 所有示例运行完成!")

    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("请确保已正确安装Qiskit和相关依赖包")
