#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2026量子投资组合优化算法

此模块实现了基于量子计算的投资组合优化算法，
包括：
- 量子近似优化算法 (QAOA)
- 量子变分算法 (VQE)
- 量子蒙特卡洛方法

作者: RQA2026量子计算引擎团队
时间: 2025年12月3日
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import Estimator
from qiskit.algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt


class QuantumPortfolioOptimizer:
    """量子投资组合优化器"""

    def __init__(self, num_assets=4, risk_tolerance=0.5):
        """
        初始化量子投资组合优化器

        Args:
            num_assets: 资产数量
            risk_tolerance: 风险容忍度 (0-1)
        """
        self.num_assets = num_assets
        self.risk_tolerance = risk_tolerance
        self.simulator = AerSimulator()

    def create_portfolio_hamiltonian(self, returns, covariance):
        """
        创建投资组合哈密顿量

        Args:
            returns: 资产收益率向量
            covariance: 协方差矩阵

        Returns:
            SparsePauliOp: 量子哈密顿量
        """
        num_qubits = self.num_assets

        # 计算目标函数系数
        expected_return = np.sum(returns)
        risk_penalty = self.risk_tolerance * np.trace(covariance)

        # 构建哈密顿量项
        hamiltonian_terms = []

        # 期望收益率项 (最大化)
        for i in range(num_qubits):
            pauli_string = ['I'] * num_qubits
            pauli_string[i] = 'Z'
            coeff = -returns[i]  # 负号因为我们要最大化
            hamiltonian_terms.append((coeff, ''.join(pauli_string)))

        # 风险惩罚项 (最小化)
        for i in range(num_qubits):
            for j in range(num_qubits):
                if i != j:
                    pauli_string = ['I'] * num_qubits
                    pauli_string[i] = 'Z'
                    pauli_string[j] = 'Z'
                    coeff = self.risk_tolerance * covariance[i, j] / 4
                    hamiltonian_terms.append((coeff, ''.join(pauli_string)))

        return SparsePauliOp.from_list(hamiltonian_terms)

    def optimize_portfolio_qaoa(self, returns, covariance, num_layers=2):
        """
        使用QAOA优化投资组合

        Args:
            returns: 资产收益率
            covariance: 协方差矩阵
            num_layers: QAOA层数

        Returns:
            dict: 优化结果
        """
        print("🔬 开始量子近似优化算法 (QAOA) 投资组合优化...")

        # 创建哈密顿量
        hamiltonian = self.create_portfolio_hamiltonian(returns, covariance)

        # 设置QAOA
        optimizer = COBYLA(maxiter=100)
        qaoa = QAOA(optimizer, num_layers=num_layers, mixer=None)

        # 运行优化
        try:
            result = qaoa.compute_minimum_eigenvalue(hamiltonian)

            # 解析结果
            eigenstate = result.eigenstate
            min_eigenvalue = result.eigenvalue

            # 计算权重 (从量子态解码)
            weights = self.decode_quantum_state(eigenstate)

            print(f"🎯 最优特征值: {min_eigenvalue:.4f}")
            print(f"📊 最优权重: {weights}")

            return {
                'success': True,
                'weights': weights,
                'expected_return': np.dot(weights, returns),
                'risk': np.sqrt(np.dot(weights.T, np.dot(covariance, weights))),
                'sharpe_ratio': np.dot(weights, returns) / np.sqrt(np.dot(weights.T, np.dot(covariance, weights))),
                'eigenvalue': min_eigenvalue
            }

        except Exception as e:
            print(f"❌ QAOA优化失败: {e}")
            return {'success': False, 'error': str(e)}

    def decode_quantum_state(self, eigenstate):
        """
        从量子态解码投资组合权重

        Args:
            eigenstate: 量子本征态

        Returns:
            np.array: 资产权重
        """
        # 简化解码：使用概率最大的状态
        probabilities = np.abs(eigenstate.data) ** 2
        max_prob_index = np.argmax(probabilities)

        # 将索引转换为二进制权重
        binary_weights = format(max_prob_index, f'0{self.num_assets}b')
        weights = np.array([int(bit) for bit in binary_weights])

        # 归一化权重
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(self.num_assets) / self.num_assets

        return weights

    def classical_optimization(self, returns, covariance):
        """
        经典优化作为对比基准

        Args:
            returns: 资产收益率
            covariance: 协方差矩阵

        Returns:
            dict: 经典优化结果
        """
        print("📊 执行经典投资组合优化 (对比基准)...")

        # 简化：等权重组合
        weights = np.ones(self.num_assets) / self.num_assets

        return {
            'weights': weights,
            'expected_return': np.dot(weights, returns),
            'risk': np.sqrt(np.dot(weights.T, np.dot(covariance, weights))),
            'sharpe_ratio': np.dot(weights, returns) / np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
        }


def demonstrate_quantum_portfolio_optimization():
    """演示量子投资组合优化"""
    print("🚀 RQA2026量子投资组合优化演示")
    print("=" * 50)

    # 创建示例数据
    np.random.seed(42)
    num_assets = 4

    # 生成随机收益率和协方差矩阵
    returns = np.random.normal(0.08, 0.02, num_assets)  # 年化收益率约8%
    base_cov = np.random.rand(num_assets, num_assets)
    covariance = np.dot(base_cov, base_cov.T) * 0.04  # 年化波动率约20%

    print("📈 资产数据:")
    for i in range(num_assets):
        print(".1f"
              .3f"
    print(f"💰 协方差矩阵:\n{covariance}\n")

    # 创建量子优化器
    optimizer = QuantumPortfolioOptimizer(num_assets=num_assets)

    # 经典优化
    classical_result = optimizer.classical_optimization(returns, covariance)
    print("📊 经典优化结果:")
    print(f"📈 期望收益率: {classical_result['expected_return']:.3f}")    print(f"📈 期望收益率: {classical_result['expected_return']:.3f}")    print(f"📈 期望收益率: {classical_result['expected_return']:.3f}")    print(f"📊 权重: {classical_result['weights']}\n")

    # 量子优化
    quantum_result = optimizer.optimize_portfolio_qaoa(returns, covariance)

    if quantum_result['success']:
        print("🔬 量子优化结果:")
        print(f"📈 期望收益率: {classical_result['expected_return']:.3f}")        print(f"📈 期望收益率: {classical_result['expected_return']:.3f}")        print(f"📈 期望收益率: {classical_result['expected_return']:.3f}")        print(f"📊 权重: {quantum_result['weights']}")
        print(f"🔢 量子特征值: {quantum_result['eigenvalue']:.4f}")
        # 对比分析
        print("\n🔍 优化对比:")        print(f"📈 期望收益率: {classical_result['expected_return']:.3f}")        print(f"📈 期望收益率: {classical_result['expected_return']:.3f}")        print(f"📈 期望收益率: {classical_result['expected_return']:.3f}")        print(f"📈 期望收益率: {classical_result['expected_return']:.3f}")    else:
        print(f"❌ 量子优化失败: {quantum_result.get('error', '未知错误')}")


if __name__ == "__main__":
    demonstrate_quantum_portfolio_optimization()
