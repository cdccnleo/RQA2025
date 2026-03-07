#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2026量子计算开发环境搭建脚本

此脚本用于自动搭建量子计算开发环境，包括：
- 核心量子计算库安装
- 开发工具配置
- 示例代码和文档
- 环境验证测试

作者: RQA2026创新项目组
时间: 2025年12月1日
"""

import sys
import subprocess
import os
import platform
from pathlib import Path


class QuantumEnvironmentSetup:
    """量子计算开发环境搭建类"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.quantum_dir = self.project_root / "quantum_research"
        self.requirements_file = self.quantum_dir / "requirements.txt"
        self.env_file = self.quantum_dir / "quantum_env.py"

    def run_command(self, command, description=""):
        """执行系统命令"""
        print(f"🔧 {description}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            print(f"✅ {description} - 成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} - 失败")
            print(f"错误信息: {e.stderr}")
            return False

    def create_directories(self):
        """创建必要的目录结构"""
        print("📁 创建量子计算研究目录结构...")

        directories = [
            self.quantum_dir,
            self.quantum_dir / "algorithms",
            self.quantum_dir / "finance",
            self.quantum_dir / "optimization",
            self.quantum_dir / "examples",
            self.quantum_dir / "tests",
            self.quantum_dir / "docs",
            self.quantum_dir / "notebooks"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ 创建目录: {directory}")

        return True

    def create_requirements(self):
        """创建依赖包列表"""
        print("📦 生成量子计算依赖包列表...")

        requirements = """# RQA2026量子计算开发环境依赖包
# 核心量子计算框架
qiskit>=0.40.0
qiskit-aer>=0.11.0
qiskit-optimization>=0.5.0
qiskit-machine-learning>=0.6.0
qiskit-finance>=0.3.0

# 可视化和分析
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
jupyter>=1.0.0
ipykernel>=6.0.0

# 数值计算和优化
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
scikit-learn>=1.0.0
cvxopt>=1.3.0

# 开发工具
pytest>=7.0.0
pytest-cov>=4.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.950

# 云服务集成
requests>=2.25.0
boto3>=1.20.0  # AWS
azure-identity>=1.7.0  # Azure
google-cloud-storage>=2.0.0  # GCP

# 性能监控
psutil>=5.8.0
memory-profiler>=0.60.0

# 文档生成
sphinx>=4.0.0
sphinx-rtd-theme>=1.0.0
"""

        with open(self.requirements_file, 'w', encoding='utf-8') as f:
            f.write(requirements)

        print(f"✅ 依赖包列表已保存到: {self.requirements_file}")
        return True

    def install_packages(self):
        """安装Python包"""
        if not self.requirements_file.exists():
            print("❌ 依赖包列表文件不存在，请先运行 create_requirements()")
            return False

        print("📦 安装量子计算相关Python包...")
        return self.run_command(
            f"{sys.executable} -m pip install -r {self.requirements_file}",
            "安装量子计算依赖包"
        )

    def create_environment_file(self):
        """创建量子计算环境配置文件"""
        print("⚙️ 创建量子计算环境配置...")

        env_config = '''# -*- coding: utf-8 -*-
"""
RQA2026量子计算开发环境配置

此文件包含量子计算开发环境的基本配置和工具函数。
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional


class QuantumConfig:
    """量子计算环境配置类"""

    def __init__(self):
        # 项目路径配置
        self.project_root = Path(__file__).parent.parent
        self.quantum_dir = self.project_root / "quantum_research"

        # 量子计算提供商配置
        self.providers = {
            "ibm": {
                "token": os.getenv("IBM_QUANTUM_TOKEN", ""),
                "hub": os.getenv("IBM_QUANTUM_HUB", "ibm-q"),
                "group": os.getenv("IBM_QUANTUM_GROUP", "open"),
                "project": os.getenv("IBM_QUANTUM_PROJECT", "main")
            },
            "aws": {
                "region": os.getenv("AWS_REGION", "us-east-1"),
                "bucket": os.getenv("AWS_S3_BUCKET", "")
            },
            "azure": {
                "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID", ""),
                "resource_group": os.getenv("AZURE_RESOURCE_GROUP", "")
            }
        }

        # 计算资源配置
        self.compute_resources = {
            "max_shots": int(os.getenv("QUANTUM_MAX_SHOTS", "1000")),
            "optimization_level": int(os.getenv("QUANTUM_OPT_LEVEL", "1")),
            "timeout": int(os.getenv("QUANTUM_TIMEOUT", "300"))
        }

        # 开发环境配置
        self.development = {
            "debug_mode": os.getenv("QUANTUM_DEBUG", "false").lower() == "true",
            "log_level": os.getenv("QUANTUM_LOG_LEVEL", "INFO"),
            "cache_dir": self.quantum_dir / "cache",
            "results_dir": self.quantum_dir / "results"
        }

        # 创建必要的目录
        self.development["cache_dir"].mkdir(exist_ok=True)
        self.development["results_dir"].mkdir(exist_ok=True)

    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """获取指定提供商的配置"""
        return self.providers.get(provider_name, {})

    def is_provider_configured(self, provider_name: str) -> bool:
        """检查提供商是否已配置"""
        config = self.get_provider_config(provider_name)
        if provider_name == "ibm":
            return bool(config.get("token"))
        elif provider_name == "aws":
            return bool(config.get("region"))
        elif provider_name == "azure":
            return bool(config.get("subscription_id"))
        return False

    def get_compute_config(self) -> Dict[str, Any]:
        """获取计算资源配置"""
        return self.compute_resources

    def get_development_config(self) -> Dict[str, Any]:
        """获取开发环境配置"""
        return self.development


# 全局配置实例
quantum_config = QuantumConfig()


def setup_quantum_logging():
    """设置量子计算相关的日志配置"""
    import logging

    # 配置根日志器
    logging.basicConfig(
        level=getattr(logging, quantum_config.development["log_level"]),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(quantum_config.development["cache_dir"] / "quantum.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # 创建量子计算专用日志器
    quantum_logger = logging.getLogger("quantum")
    quantum_logger.setLevel(logging.DEBUG if quantum_config.development["debug_mode"] else logging.INFO)

    return quantum_logger


def check_quantum_environment():
    """检查量子计算环境是否正确配置"""
    print("🔍 检查量子计算环境配置...")

    issues = []

    # 检查Qiskit
    try:
        import qiskit
        print(f"✅ Qiskit版本: {qiskit.__version__}")
    except ImportError:
        issues.append("❌ Qiskit未安装")

    # 检查提供商配置
    configured_providers = []
    for provider in ["ibm", "aws", "azure"]:
        if quantum_config.is_provider_configured(provider):
            configured_providers.append(provider)

    if configured_providers:
        print(f"✅ 已配置提供商: {', '.join(configured_providers)}")
    else:
        print("⚠️  警告: 未配置任何量子计算提供商")

    # 检查目录权限
    dirs_to_check = [
        quantum_config.development["cache_dir"],
        quantum_config.development["results_dir"]
    ]

    for directory in dirs_to_check:
        if directory.exists() and os.access(directory, os.W_OK):
            print(f"✅ 目录权限正常: {directory}")
        else:
            issues.append(f"❌ 目录权限问题: {directory}")

    if issues:
        print("\n⚠️  发现以下问题:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("\n🎉 量子计算环境配置检查通过!")
        return True


if __name__ == "__main__":
    # 初始化日志
    logger = setup_quantum_logging()
    logger.info("量子计算环境配置模块加载完成")

    # 运行环境检查
    check_quantum_environment()
'''

        with open(self.env_file, 'w', encoding='utf-8') as f:
            f.write(env_config)

        print(f"✅ 环境配置文件已保存到: {self.env_file}")
        return True

    def create_example_code(self):
        """创建示例代码"""
        print("💡 创建量子计算示例代码...")

        # 基础量子电路示例
        basic_circuit = '''# -*- coding: utf-8 -*-
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
'''

        example_file = self.quantum_dir / "examples" / "basic_quantum_circuit.py"
        with open(example_file, 'w', encoding='utf-8') as f:
            f.write(basic_circuit)

        # 量化金融示例
        finance_example = '''# -*- coding: utf-8 -*-
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
    print("\\n🔍 使用经典方法求解...")

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
    print("\\n🔬 使用量子算法求解...")

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
    print("\\n📊 结果对比:")
    print(f"经典方法 - 解: {classical_solution}, 值: {classical_value:.6f}")
    if quantum_solution is not None:
        print(f"量子方法 - 解: {[int(x) for x in quantum_solution]}, 值: {quantum_value:.6f}")
    else:
        print("量子方法 - 计算失败")

    print("\\n💡 说明:")
    print("- 经典方法使用穷举搜索 (适用于小规模问题)")
    print("- 量子方法使用VQE算法 (适用于大规模优化问题)")
    print("- 在实际应用中，量子算法在大规模问题上具有理论优势")


if __name__ == "__main__":
    try:
        compare_solutions()
        print("\\n🎉 量化金融量子优化示例完成!")

    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("请确保已安装qiskit-optimization和其他相关依赖")
'''

        finance_file = self.quantum_dir / "examples" / "quantum_finance_example.py"
        with open(finance_file, 'w', encoding='utf-8') as f:
            f.write(finance_example)

        print("✅ 示例代码已创建")
        return True

    def create_readme(self):
        """创建README文档"""
        print("📖 创建README文档...")

        readme_content = '''# 🔬 RQA2026量子计算研究环境

## 🎯 项目概述

这是RQA2026创新项目中量子计算创新引擎的开发环境，为量化交易中的量子计算应用提供完整的开发和研究平台。

## 📁 目录结构

```
quantum_research/
├── algorithms/          # 量子算法实现
├── finance/            # 量化金融应用
├── optimization/       # 优化问题求解
├── examples/           # 示例代码
├── tests/             # 测试代码
├── docs/              # 文档
├── notebooks/         # Jupyter笔记本
├── requirements.txt   # 依赖包列表
└── quantum_env.py     # 环境配置
```

## 🚀 快速开始

### 1. 环境搭建

```bash
# 安装依赖包
pip install -r requirements.txt

# 验证环境配置
python quantum_env.py
```

### 2. 运行示例

```bash
# 基础量子电路示例
python examples/basic_quantum_circuit.py

# 量化金融应用示例
python examples/quantum_finance_example.py
```

### 3. 开发环境配置

```python
from quantum_env import quantum_config, setup_quantum_logging

# 初始化配置
config = quantum_config

# 设置日志
logger = setup_quantum_logging()
```

## 🔧 核心组件

### 量子计算框架
- **Qiskit**: IBM开源量子计算框架
- **Qiskit Optimization**: 量子优化算法
- **Qiskit Finance**: 量子金融应用
- **Qiskit Machine Learning**: 量子机器学习

### 开发工具
- **Jupyter**: 交互式开发环境
- **pytest**: 测试框架
- **Sphinx**: 文档生成

## 📊 应用场景

### 1. 投资组合优化
使用量子近似优化算法(QAOA)解决投资组合选择问题。

### 2. 风险建模
量子蒙特卡洛方法加速VaR计算和风险评估。

### 3. 期权定价
量子算法解决多资产期权定价的复杂计算。

### 4. 机器学习
量子支持向量机和量子神经网络优化。

## 🔐 访问配置

### IBM Quantum
```bash
export IBM_QUANTUM_TOKEN="your_token_here"
```

### AWS Braket
```bash
export AWS_REGION="us-east-1"
export AWS_S3_BUCKET="your_bucket"
```

### Azure Quantum
```bash
export AZURE_SUBSCRIPTION_ID="your_subscription"
export AZURE_RESOURCE_GROUP="your_resource_group"
```

## 📚 学习资源

### 官方文档
- [Qiskit文档](https://qiskit.org/documentation/)
- [IBM Quantum](https://quantum-computing.ibm.com/)
- [Qiskit Finance](https://qiskit.org/ecosystem/finance/)

### 教程和示例
- [Qiskit教程](https://qiskit.org/learn/)
- [量子计算入门](https://quantum.country/)
- [量子算法](https://quantumalgorithms.org/)

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](../LICENSE) 文件了解详情。

## 📞 联系方式

- 项目维护者: RQA2026创新项目组
- 项目邮箱: quantum@rqatech.com
- 项目主页: [https://github.com/rqa2026/quantum-research](https://github.com/rqa2026/quantum-research)

## 🙏 致谢

感谢IBM Quantum、Google Quantum AI等量子计算平台的开源贡献，以及学术界的量子计算研究成果。

---

*RQA2026量子计算创新引擎*
*让量子计算赋能量化交易*
'''

        readme_file = self.quantum_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"✅ README文档已创建: {readme_file}")
        return True

    def setup_environment(self):
        """完整的环境搭建流程"""
        print("🚀 开始搭建RQA2026量子计算开发环境")
        print("=" * 60)

        steps = [
            ("创建目录结构", self.create_directories),
            ("生成依赖包列表", self.create_requirements),
            ("创建环境配置", self.create_environment_file),
            ("创建示例代码", self.create_example_code),
            ("创建README文档", self.create_readme),
            ("安装Python包", self.install_packages),
        ]

        for step_name, step_func in steps:
            print(f"\n📋 执行步骤: {step_name}")
            if not step_func():
                print(f"❌ 步骤 '{step_name}' 失败，环境搭建终止")
                return False

        print("\n" + "=" * 60)
        print("🎉 RQA2026量子计算开发环境搭建完成!")
        print("\n📚 接下来你可以:")
        print("   1. 运行示例: python examples/basic_quantum_circuit.py")
        print("   2. 查看文档: README.md")
        print("   3. 开始开发: 创建你的第一个量子算法")
        print("\n🔗 相关资源:")
        print("   • Qiskit文档: https://qiskit.org/documentation/")
        print("   • IBM Quantum: https://quantum-computing.ibm.com/")
        print("   • 量子计算入门: https://quantum.country/")

        return True


def main():
    """主函数"""
    print("🔬 RQA2026量子计算开发环境自动搭建工具")
    print("作者: RQA2026创新项目组")
    print("时间: 2025年12月1日")
    print("-" * 50)

    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        sys.exit(1)

    print(f"✅ Python版本: {sys.version}")

    # 检查操作系统
    print(f"✅ 操作系统: {platform.system()} {platform.release()}")

    # 创建搭建器并执行
    setup = QuantumEnvironmentSetup()

    if setup.setup_environment():
        print("\n🎊 恭喜！量子计算开发环境已成功搭建！")
        print("🌟 现在开始你的量子计算探索之旅吧！")
    else:
        print("\n❌ 环境搭建失败，请检查错误信息并重试")
        sys.exit(1)


if __name__ == "__main__":
    main()
