#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2026 QAOA投资组合优化算法

此模块实现了使用量子近似优化算法(QAOA)进行投资组合优化的完整解决方案。

核心功能：
- QAOA算法实现
- 投资组合优化哈密顿量构造
- 量子电路优化
- 经典对比分析

作者: RQA2026量子计算引擎团队
时间: 2025年12月3日
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import Estimator
from qiskit.algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import time


class QAOAPortfolioOptimizer:
    """基于QAOA的量子投资组合优化器"""

    def __init__(self,
                 num_assets: int = 4,
                 risk_tolerance: float = 0.3,
                 max_budget: float = 1.0):
        """
        初始化QAOA投资组合优化器

        Args:
            num_assets: 资产数量
            risk_tolerance: 风险容忍度 (0-1)
            max_budget: 最大预算比例
        """
        self.num_assets = num_assets
        self.risk_tolerance = risk_tolerance
        self.max_budget = max_budget
        self.simulator = AerSimulator()

        # 记录优化历史
        self.optimization_history = []

    def generate_realistic_portfolio_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成现实的投资组合数据

        Returns:
            Tuple[np.ndarray, np.ndarray]: (预期收益率, 协方差矩阵)
        """
        np.random.seed(42)  # 保证结果可重现

        # 生成预期收益率 (年化)
        # 假设资产年化收益率在5%-15%之间
        returns = np.random.uniform(0.05, 0.15, self.num_assets)

        # 生成协方差矩阵
        # 使用合理的波动率假设
        base_cov = np.random.randn(self.num_assets, self.num_assets)
        cov_matrix = np.dot(base_cov, base_cov.T) * 0.04  # 年化波动率约20%

        # 确保协方差矩阵正定
        cov_matrix = cov_matrix + np.eye(self.num_assets) * 0.01

        return returns, cov_matrix

    def create_portfolio_hamiltonian(self,
                                   returns: np.ndarray,
                                   cov_matrix: np.ndarray) -> SparsePauliOp:
        """
        创建投资组合优化问题的哈密顿量

        目标函数: max(returns · weights) - risk_penalty · (weights · cov_matrix · weights)
        约束条件: sum(weights) = budget, 0 ≤ weights_i ≤ 1

        Args:
            returns: 资产预期收益率向量
            cov_matrix: 协方差矩阵

        Returns:
            SparsePauliOp: 量子哈密顿量
        """
        num_qubits = self.num_assets
        hamiltonian_terms = []

        # 1. 预期收益率项 (最大化 -> 负系数最小化)
        for i in range(num_qubits):
            pauli_string = ['I'] * num_qubits
            pauli_string[i] = 'Z'
            coeff = -returns[i]  # 负号将最大化转为最小化
            hamiltonian_terms.append((coeff, ''.join(pauli_string)))

        # 2. 风险惩罚项 (最小化方差)
        risk_penalty = self.risk_tolerance
        for i in range(num_qubits):
            for j in range(num_qubits):
                if i <= j:  # 只计算上三角部分
                    pauli_string = ['I'] * num_qubits
                    if i == j:
                        # 对角项: w_i^2
                        pauli_string[i] = 'Z'
                        coeff = risk_penalty * cov_matrix[i, j] / 4
                    else:
                        # 交叉项: 2*w_i*w_j
                        pauli_string[i] = 'Z'
                        pauli_string[j] = 'Z'
                        coeff = risk_penalty * cov_matrix[i, j] / 2
                    hamiltonian_terms.append((coeff, ''.join(pauli_string)))

        # 3. 预算约束项 (sum(weights) ≈ budget)
        # 使用惩罚函数: penalty * (sum(weights) - budget)^2
        budget_penalty = 2.0  # 约束惩罚强度
        for i in range(num_qubits):
            for j in range(num_qubits):
                pauli_string = ['I'] * num_qubits
                if i == j:
                    pauli_string[i] = 'Z'
                    coeff = budget_penalty * (1 - 2 * self.max_budget) / 4
                else:
                    pauli_string[i] = 'Z'
                    pauli_string[j] = 'Z'
                    coeff = budget_penalty * (1 - 2 * self.max_budget) / 2
                hamiltonian_terms.append((coeff, ''.join(pauli_string)))

        return SparsePauliOp.from_list(hamiltonian_terms)

    def decode_quantum_solution(self, eigenstate) -> np.ndarray:
        """
        从量子本征态解码投资组合权重

        Args:
            eigenstate: 量子本征态

        Returns:
            np.ndarray: 资产权重向量
        """
        # 获取概率最大的状态
        probabilities = np.abs(eigenstate.data) ** 2
        max_prob_index = np.argmax(probabilities)

        # 将索引转换为二进制权重
        binary_string = format(max_prob_index, f'0{self.num_assets}b')

        # 将二进制转换为权重 (0 -> 0, 1 -> 预算份额)
        weights = np.array([int(bit) for bit in binary_string])

        # 归一化权重
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights * (self.max_budget / weight_sum)
        else:
            # 如果全为0，使用等权重
            weights = np.ones(self.num_assets) * (self.max_budget / self.num_assets)

        return weights

    def optimize_portfolio_qaoa(self,
                              returns: np.ndarray,
                              cov_matrix: np.ndarray,
                              num_layers: int = 2,
                              optimizer_name: str = 'COBYLA') -> Dict[str, Any]:
        """
        使用QAOA优化投资组合

        Args:
            returns: 资产预期收益率
            cov_matrix: 协方差矩阵
            num_layers: QAOA层数
            optimizer_name: 经典优化器名称

        Returns:
            Dict[str, Any]: 优化结果
        """
        print("🔬 开始QAOA投资组合优化...")

        start_time = time.time()

        # 创建哈密顿量
        hamiltonian = self.create_portfolio_hamiltonian(returns, cov_matrix)
        print(f"📐 哈密顿量创建完成，包含{len(hamiltonian)}个项")

        # 选择经典优化器
        if optimizer_name == 'COBYLA':
            optimizer = COBYLA(maxiter=100)
        elif optimizer_name == 'SPSA':
            optimizer = SPSA(maxiter=100)
        elif optimizer_name == 'ADAM':
            optimizer = ADAM(maxiter=100)
        else:
            optimizer = COBYLA(maxiter=100)

        # 设置QAOA
        qaoa = QAOA(optimizer, num_layers=num_layers, mixer=None)

        try:
            # 运行优化
            result = qaoa.compute_minimum_eigenvalue(hamiltonian)

            # 解码结果
            optimal_weights = self.decode_quantum_solution(result.eigenstate)

            # 计算投资组合指标
            expected_return = np.dot(optimal_weights, returns)
            risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            sharpe_ratio = expected_return / risk if risk > 0 else 0

            optimization_time = time.time() - start_time

            result_dict = {
                'success': True,
                'weights': optimal_weights,
                'expected_return': expected_return,
                'risk': risk,
                'sharpe_ratio': sharpe_ratio,
                'eigenvalue': result.eigenvalue,
                'optimization_time': optimization_time,
                'num_layers': num_layers,
                'optimizer': optimizer_name,
                'convergence': len(result.cost_function_evals) if hasattr(result, 'cost_function_evals') else None
            }

            self.optimization_history.append(result_dict)

            print(f"🎯 最优特征值: {result.eigenvalue:.4f}")
            print(f"📈 预期收益率: {expected_return:.3f}")
            print(f"⚠️ 投资风险: {risk:.3f}")
            print(f"📊 Sharpe比率: {sharpe_ratio:.3f}")
            print(f"⏱️ 优化时间: {optimization_time:.2f}s")
            return result_dict

        except Exception as e:
            error_time = time.time() - start_time
            print(f"❌ QAOA优化失败 ({error_time:.2f}s): {e}")
            return {
                'success': False,
                'error': str(e),
                'optimization_time': error_time
            }

    def compare_with_classical(self,
                              returns: np.ndarray,
                              cov_matrix: np.ndarray) -> Dict[str, Any]:
        """
        与经典优化方法比较

        Args:
            returns: 资产预期收益率
            cov_matrix: 协方差矩阵

        Returns:
            Dict[str, Any]: 比较结果
        """
        print("\n📊 执行经典投资组合优化 (对比基准)...")

        # 简单等权重组合
        equal_weights = np.ones(self.num_assets) * (self.max_budget / self.num_assets)

        # 风险平价组合 (简化版)
        volatilities = np.sqrt(np.diag(cov_matrix))
        risk_parity_weights = 1.0 / volatilities
        risk_parity_weights = risk_parity_weights * (self.max_budget / np.sum(risk_parity_weights))

        # 计算指标
        def calculate_metrics(weights):
            expected_return = np.dot(weights, returns)
            risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = expected_return / risk if risk > 0 else 0
            return {
                'weights': weights,
                'expected_return': expected_return,
                'risk': risk,
                'sharpe_ratio': sharpe_ratio
            }

        classical_results = {
            'equal_weight': calculate_metrics(equal_weights),
            'risk_parity': calculate_metrics(risk_parity_weights)
        }

        return classical_results

    def visualize_optimization_results(self, quantum_result: Dict, classical_results: Dict):
        """
        可视化优化结果对比

        Args:
            quantum_result: 量子优化结果
            classical_results: 经典优化结果
        """
        if not quantum_result['success']:
            print("⚠️ 量子优化失败，跳过可视化")
            return

        # 准备数据
        methods = ['Quantum QAOA'] + list(classical_results.keys())
        returns = [quantum_result['expected_return']] + [r['expected_return'] for r in classical_results.values()]
        risks = [quantum_result['risk']] + [r['risk'] for r in classical_results.values()]
        sharpes = [quantum_result['sharpe_ratio']] + [r['sharpe_ratio'] for r in classical_results.values()]

        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 收益率对比
        bars1 = axes[0].bar(methods, returns, color=['blue', 'red', 'green', 'orange'])
        axes[0].set_title('Expected Returns Comparison')
        axes[0].set_ylabel('Expected Return')
        axes[0].tick_params(axis='x', rotation=45)

        # 风险对比
        bars2 = axes[1].bar(methods, risks, color=['blue', 'red', 'green', 'orange'])
        axes[1].set_title('Risk Comparison')
        axes[1].set_ylabel('Risk (Volatility)')
        axes[1].tick_params(axis='x', rotation=45)

        # Sharpe比率对比
        bars3 = axes[2].bar(methods, sharpes, color=['blue', 'red', 'green', 'orange'])
        axes[2].set_title('Sharpe Ratio Comparison')
        axes[2].set_ylabel('Sharpe Ratio')
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('quantum_portfolio_optimization_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("📊 优化结果对比图已保存为: quantum_portfolio_optimization_comparison.png")

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        运行完整的投资组合优化分析

        Returns:
            Dict[str, Any]: 完整分析结果
        """
        print("🚀 RQA2026量子投资组合优化综合分析")
        print("=" * 60)

        # 生成数据
        returns, cov_matrix = self.generate_realistic_portfolio_data()

        print("📈 资产数据:")
        asset_names = [f"Asset_{i+1}" for i in range(self.num_assets)]
        for i, (name, ret, vol) in enumerate(zip(asset_names, returns, np.sqrt(np.diag(cov_matrix)))):
            print(f"  {name}: 收益率={ret:.1f}, 波动率={vol:.1f}")
        print(f"\n💰 协方差矩阵:\n{cov_matrix}")

        # 量子优化
        quantum_result = self.optimize_portfolio_qaoa(returns, cov_matrix, num_layers=3)

        # 经典优化对比
        classical_results = self.compare_with_classical(returns, cov_matrix)

        # 输出经典方法结果
        print("\n📊 经典优化结果:")
        for method_name, result in classical_results.items():
            print(f"\n{method_name.upper()}:")
            print(f"    预期收益率: {result['expected_return']:.3f}")
            print(f"    投资风险: {result['risk']:.3f}")
            print(f"    Sharpe比率: {result['sharpe_ratio']:.3f}")
            print(f"  权重: {result['weights']}")

        # 性能分析
        if quantum_result['success']:
            print("\n🎯 性能分析:")
            quantum_sharpe = quantum_result['sharpe_ratio']
            best_classical_sharpe = max(r['sharpe_ratio'] for r in classical_results.values())
            improvement = (quantum_sharpe - best_classical_sharpe) / best_classical_sharpe * 100

            print(f"  量子Sharpe比率: {quantum_sharpe:.3f}")
            print(f"  最佳经典Sharpe比率: {best_classical_sharpe:.3f}")
            if improvement > 0:
                print(f"  📈 相对于经典方法提升: +{improvement:.1f}%")
            else:
                print(f"  📉 相对于经典方法下降: {improvement:.1f}%")
            # 可视化
            try:
                self.visualize_optimization_results(quantum_result, classical_results)
            except Exception as e:
                print(f"⚠️ 可视化失败: {e}")

        return {
            'quantum_result': quantum_result,
            'classical_results': classical_results,
            'asset_data': {
                'names': asset_names,
                'returns': returns,
                'cov_matrix': cov_matrix
            }
        }


def demonstrate_quantum_portfolio_optimization():
    """演示量子投资组合优化"""
    optimizer = QAOAPortfolioOptimizer(num_assets=4, risk_tolerance=0.3)
    results = optimizer.run_comprehensive_analysis()

    return results


if __name__ == "__main__":
    results = demonstrate_quantum_portfolio_optimization()
