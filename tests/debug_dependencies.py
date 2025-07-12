import sys
import pkg_resources
from pathlib import Path

# 检查核心依赖是否安装
REQUIRED_PACKAGES = [
    'numpy',
    'pandas',
    'scikit-learn',
    'tensorflow',
    'pytest',
    'configparser'
]

def check_environment():
    print("Python路径:", sys.executable)
    print("\n已安装包版本:")
    for pkg in REQUIRED_PACKAGES:
        try:
            version = pkg_resources.get_distribution(pkg).version
            print(f"{pkg:20s}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"{pkg:20s}: 未安装")

if __name__ == "__main__":
    check_environment()
