#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU环境配置脚本

帮助用户安装和配置GPU加速支持
"""

import os
import subprocess
import platform
from typing import Dict, Optional


def check_system_info() -> Dict[str, str]:
    """检查系统信息"""
    info = {
        'os': platform.system(),
        'os_version': platform.version(),
        'python_version': platform.python_version(),
        'architecture': platform.architecture()[0]
    }

    print("系统信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    return info


def check_cuda_installation() -> Dict[str, bool]:
    """检查CUDA安装状态"""
    cuda_info = {
        'nvidia_driver': False,
        'cuda_toolkit': False,
        'cuda_runtime': False
    }

    print("\n检查CUDA安装状态:")

    # 检查NVIDIA驱动
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            cuda_info['nvidia_driver'] = True
            print("  ✅ NVIDIA驱动已安装")
        else:
            print("  ❌ NVIDIA驱动未安装或不可用")
    except FileNotFoundError:
        print("  ❌ nvidia-smi命令不可用")

    # 检查CUDA工具包
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            cuda_info['cuda_toolkit'] = True
            print("  ✅ CUDA工具包已安装")
        else:
            print("  ❌ CUDA工具包未安装")
    except FileNotFoundError:
        print("  ❌ nvcc命令不可用")

    # 检查CUDA运行时
    try:
        import torch
        if torch.cuda.is_available():
            cuda_info['cuda_runtime'] = True
            print(f"  ✅ CUDA运行时可用 (PyTorch检测到 {torch.cuda.device_count()} 个GPU)")
        else:
            print("  ❌ CUDA运行时不可用")
    except ImportError:
        print("  ❌ PyTorch未安装")

    return cuda_info


def check_python_packages() -> Dict[str, bool]:
    """检查Python包安装状态"""
    packages = {
        'torch': False,
        'cupy': False,
        'numpy': False,
        'pandas': False
    }

    print("\n检查Python包:")

    for package in packages.keys():
        try:
            __import__(package)
            packages[package] = True
            print(f"  ✅ {package} 已安装")
        except ImportError:
            print(f"  ❌ {package} 未安装")

    return packages


def get_cuda_version() -> Optional[str]:
    """获取CUDA版本"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            # 解析版本号
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    version = line.split('release')[1].strip().split(',')[0]
                    return version
    except FileNotFoundError:
        pass

    # 尝试从环境变量获取
    cuda_version = os.environ.get('CUDA_VERSION')
    if cuda_version:
        return cuda_version

    return None


def install_cupy(cuda_version: Optional[str] = None) -> bool:
    """安装CuPy"""
    print(f"\n安装CuPy...")

    if not cuda_version:
        print("  ⚠️  无法检测CUDA版本，尝试安装通用版本")
        package = "cupy-cuda11x"  # 默认版本
    else:
        # 根据CUDA版本选择包
        if cuda_version.startswith('11'):
            package = "cupy-cuda11x"
        elif cuda_version.startswith('12'):
            package = "cupy-cuda12x"
        else:
            package = "cupy-cuda11x"  # 默认版本

    print(f"  选择包: {package}")

    try:
        # 尝试conda安装
        result = subprocess.run([
            'conda', 'install', '-c', 'conda-forge', package, '-y'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("  ✅ CuPy安装成功 (conda)")
            return True
        else:
            print("  ⚠️  conda安装失败，尝试pip安装")

            # 尝试pip安装
            result = subprocess.run([
                'pip', 'install', package
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("  ✅ CuPy安装成功 (pip)")
                return True
            else:
                print(f"  ❌ CuPy安装失败: {result.stderr}")
                return False

    except Exception as e:
        print(f"  ❌ 安装过程出错: {e}")
        return False


def install_pytorch_gpu() -> bool:
    """安装GPU版本的PyTorch"""
    print(f"\n安装GPU版本PyTorch...")

    try:
        # 使用conda安装
        result = subprocess.run([
            'conda', 'install', 'pytorch', 'pytorch-cuda=11.8', '-c', 'pytorch', '-c', 'nvidia', '-y'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("  ✅ PyTorch GPU版本安装成功")
            return True
        else:
            print("  ⚠️  conda安装失败，尝试pip安装")

            # 尝试pip安装
            result = subprocess.run([
                'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("  ✅ PyTorch GPU版本安装成功 (pip)")
                return True
            else:
                print(f"  ❌ PyTorch GPU版本安装失败: {result.stderr}")
                return False

    except Exception as e:
        print(f"  ❌ 安装过程出错: {e}")
        return False


def test_gpu_functionality() -> bool:
    """测试GPU功能"""
    print(f"\n测试GPU功能...")

    try:
        # 测试CuPy
        import cupy as cp
        if cp.cuda.is_available():
            print("  ✅ CuPy GPU功能正常")
            device_count = cp.cuda.runtime.getDeviceCount()
            print(f"  检测到 {device_count} 个GPU设备")

            # 简单计算测试
            x = cp.array([1, 2, 3, 4, 5])
            y = cp.array([2, 3, 4, 5, 6])
            z = x + y
            print(f"  GPU计算测试: {cp.asnumpy(z)}")

            return True
        else:
            print("  ❌ CuPy GPU功能不可用")
            return False

    except ImportError:
        print("  ❌ CuPy未安装")
        return False
    except Exception as e:
        print(f"  ❌ GPU测试失败: {e}")
        return False


def create_gpu_config() -> None:
    """创建GPU配置文件"""
    config_content = """# GPU配置
gpu:
  enabled: true
  memory_limit: 0.8  # GPU内存使用限制
  batch_size: 1000   # 批处理大小
  fallback_to_cpu: true  # 失败时回退到CPU

# 技术指标GPU加速配置
technical_indicators:
  use_gpu: true
  indicators:
    - sma
    - ema
    - rsi
    - macd
    - bollinger
    - atr
"""

    config_file = "config/gpu_config.yaml"
    os.makedirs("config", exist_ok=True)

    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)

    print(f"\n✅ GPU配置文件已创建: {config_file}")


def main():
    """主函数"""
    print("GPU环境配置脚本")
    print("=" * 50)

    # 检查系统信息
    system_info = check_system_info()

    # 检查CUDA安装
    cuda_info = check_cuda_installation()

    # 检查Python包
    packages = check_python_packages()

    # 获取CUDA版本
    cuda_version = get_cuda_version()
    if cuda_version:
        print(f"\n检测到CUDA版本: {cuda_version}")

    # 安装建议
    print("\n" + "=" * 50)
    print("安装建议:")

    if not cuda_info['nvidia_driver']:
        print("1. 请先安装NVIDIA驱动")
        print("   下载地址: https://www.nvidia.com/Download/index.aspx")
        return

    if not cuda_info['cuda_toolkit']:
        print("2. 请安装CUDA工具包")
        print("   下载地址: https://developer.nvidia.com/cuda-downloads")
        return

    # 安装CuPy
    if not packages['cupy']:
        if not install_cupy(cuda_version):
            print("❌ CuPy安装失败，请手动安装")
            return

    # 安装PyTorch GPU版本
    if not packages['torch'] or not cuda_info['cuda_runtime']:
        if not install_pytorch_gpu():
            print("❌ PyTorch GPU版本安装失败，请手动安装")
            return

    # 测试GPU功能
    if not test_gpu_functionality():
        print("❌ GPU功能测试失败")
        return

    # 创建配置文件
    create_gpu_config()

    print("\n" + "=" * 50)
    print("✅ GPU环境配置完成!")
    print("\n下一步:")
    print("1. 运行GPU加速演示: python scripts/features/demo_gpu_acceleration.py")
    print("2. 运行GPU测试: python scripts/testing/run_tests.py tests/unit/features/processors/gpu/")
    print("3. 查看GPU状态报告: GPU_ACCELERATION_STATUS_REPORT.md")


if __name__ == "__main__":
    main()
