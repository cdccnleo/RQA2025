# -*- coding: utf-8 -*-
"""
量子计算在量化金融中的应用示例
演示投资组合优化问题的量子求解
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator


def create_portfolio_optimization_problem():
    """创建投资组合优化问题"""
    print("📊 创建投资组合优化问题...")

    # 示例数据：3个资产
    n_assets = 3

    # 预期收益率 (期望值)
    expected_returns = np.array([0.08, 0.12, 0.10])  # 8%, 12%, 10%

    # 协方差矩阵 (风险)
    covariance_matrix = np.array([
        [0.04, 0.006, 0.002],
        [0.006, 0.09, 0.008],
        [0.002, 0.008, 0.06]
    ])

    # 创建二次规划问题
    qp = QuadraticProgram()

    # 添加二进制变量 (是否选择该资产)
    for i in range(n_assets):
        qp.binary_var(f"x_{i}")

    # 目标函数：最小化风险 (方差)
    # 0.5 * x^T * Σ * x (忽略常数项)
    quadratic_terms = {}
    for i in range(n_assets):
        for j in range(n_assets):
            if i <= j:  # 对称矩阵，只添加上三角
                coeff = covariance_matrix[i, j]
                if i == j:
                    coeff *= 0.5  # 对角线元素
                quadratic_terms[(f"x_{i}", f"x_{j}")] = coeff

    # 添加二次项到目标函数
    for vars_tuple, coeff in quadratic_terms.items():
        qp.minimize_quadratic_term(coeff, vars_tuple[0], vars_tuple[1])

    # 添加约束：至少选择一个资产
    qp.add_linear_constraint(
        linear_expression={f"x_{i}": 1 for i in range(n_assets)},
        sense=">=",
        rhs=1,
        name="min_assets"
    )

    print(f"✅ 创建了{n_assets}资产的投资组合优化问题")
    return qp


def solve_classical_optimization(qp):
    """使用经典方法求解 (穷举法，演示用)"""
    print("\n🔍 使用经典方法求解...")

    n_assets = 3
    best_solution = None
    best_value = float('inf')

    # 穷举所有可能的投资组合 (2^3 = 8种)
    for i in range(2**n_assets):
        # 转换为二进制选择
        selection = [(i >> j) & 1 for j in range(n_assets)]

        # 计算风险 (方差)
        x = np.array(selection)
        risk = 0.5 * x.T @ np.array([
            [0.04, 0.006, 0.002],
            [0.006, 0.09, 0.008],
            [0.002, 0.008, 0.06]
        ]) @ x

        # 检查约束
        if sum(selection) >= 1 and risk < best_value:
            best_value = risk
            best_solution = selection

    print(f"最优解: {best_solution}, 风险值: {best_value:.6f}")
    return best_solution, best_value


def solve_quantum_optimization(qp):
    """使用量子算法求解"""
    print("\n🔬 使用量子算法求解...")

    try:
        # 转换为QUBO问题
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)

        # 使用VQE算法
        estimator = Estimator()
        optimizer = COBYLA(maxiter=100)

        vqe = VQE(estimator=estimator, optimizer=optimizer)

        # 创建最小特征值优化器
        vqe_optimizer = MinimumEigenOptimizer(vqe)

        # 求解
        result = vqe_optimizer.solve(qubo)

        print(f"量子算法最优解: {[int(x) for x in result.x]}")
        print(f"最优值: {result.fval:.6f}")

        return result.x, result.fval

    except Exception as e:
        print(f"量子算法求解失败: {e}")
        print("这可能是由于量子模拟器限制或参数设置问题")
        return None, None


def compare_solutions():
    """比较经典和量子解法"""
    print("🚀 投资组合优化：经典vs量子对比")
    print("=" * 50)

    # 创建优化问题
    qp = create_portfolio_optimization_problem()

    # 经典方法
    classical_solution, classical_value = solve_classical_optimization(qp)

    # 量子方法
    quantum_solution, quantum_value = solve_quantum_optimization(qp)

    # 对比结果
    print("\n📊 结果对比:")
    print(f"经典方法 - 解: {classical_solution}, 值: {classical_value:.6f}")
    if quantum_solution is not None:
        print(f"量子方法 - 解: {[int(x) for x in quantum_solution]}, 值: {quantum_value:.6f}")
    else:
        print("量子方法 - 计算失败")

    print("\n💡 说明:")
    print("- 经典方法使用穷举搜索 (适用于小规模问题)")
    print("- 量子方法使用VQE算法 (适用于大规模优化问题)")
    print("- 在实际应用中，量子算法在大规模问题上具有理论优势")


if __name__ == "__main__":
    try:
        compare_solutions()
        print("\n🎉 量化金融量子优化示例完成!")

    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("请确保已安装qiskit-optimization和其他相关依赖")
