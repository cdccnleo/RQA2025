#!/usr/bin/env python3
"""
导入路径优化脚本
将复杂的绝对导入路径优化为相对导入或简化的绝对导入
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict


class ImportOptimizer:
    """导入路径优化器"""

    def __init__(self, src_dir: str = "src"):
        self.src_dir = Path(src_dir)
        self.import_patterns = [
            # 匹配 from src.xxx import yyy
            (r'from src\.([a-zA-Z_][a-zA-Z0-9_.]*) import ([a-zA-Z_][a-zA-Z0-9_, \n]*)',
             self._optimize_src_import),
            # 匹配 import src.xxx
            (r'import src\.([a-zA-Z_][a-zA-Z0-9_.]*)',
             self._optimize_src_import_simple)
        ]

    def _get_relative_path(self, file_path: Path, target_module: str) -> str:
        """计算相对导入路径"""
        file_dir = file_path.parent
        target_path = self.src_dir / target_module.replace('.', '/')

        # 计算相对路径
        try:
            relative_path = os.path.relpath(target_path, file_dir)
            return relative_path.replace('\\', '/').replace('/', '.')
        except ValueError:
            # 如果无法计算相对路径，返回绝对路径
            return f"src.{target_module}"

    def _optimize_src_import(self, match) -> str:
        """优化 from src.xxx import yyy 格式的导入"""
        module_path = match.group(1)
        imports = match.group(2)

        # 对于跨层级的导入，保持绝对导入但简化
        if module_path.startswith(('infrastructure.', 'data.', 'features.', 'models.', 'trading.')):
            return f'from src.{module_path} import {imports}'

        # 对于同层级的导入，尝试使用相对导入
        return f'from .{module_path} import {imports}'

    def _optimize_src_import_simple(self, match) -> str:
        """优化 import src.xxx 格式的导入"""
        module_path = match.group(1)

        # 对于跨层级的导入，保持绝对导入
        if module_path.startswith(('infrastructure.', 'data.', 'features.', 'models.', 'trading.')):
            return f'import src.{module_path}'

        # 对于同层级的导入，尝试使用相对导入
        return f'import .{module_path}'

    def optimize_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """优化单个文件的导入路径"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            changes = []

            for pattern, replacement_func in self.import_patterns:
                def replace_func(match):
                    new_import = replacement_func(match)
                    if new_import != match.group(0):
                        changes.append(f"  {match.group(0)} -> {new_import}")
                    return new_import

                content = re.sub(pattern, replace_func, content)

            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, changes

            return False, []

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            return False, []

    def optimize_directory(self, directory: str = None) -> Dict[str, List[str]]:
        """优化目录下所有Python文件的导入路径"""
        if directory is None:
            directory = self.src_dir

        dir_path = Path(directory)
        python_files = list(dir_path.rglob("*.py"))

        results = {}
        total_changes = 0

        print(f"🔍 开始优化 {len(python_files)} 个Python文件的导入路径...")

        for file_path in python_files:
            if file_path.name == "__init__.py":
                continue  # 跳过__init__.py文件

            changed, changes = self.optimize_file(file_path)
            if changed:
                results[str(file_path)] = changes
                total_changes += len(changes)
                print(f"✅ 优化了 {file_path}: {len(changes)} 个导入")

        print(f"📊 总共优化了 {len(results)} 个文件，{total_changes} 个导入路径")
        return results


def create_import_aliases():
    """创建导入别名文件"""
    aliases_content = '''
"""
导入别名定义
简化复杂的导入路径
"""

# 加速层别名
from src.acceleration.fpga import FpgaManager as FPGA
from src.acceleration.gpu import GPUManager as GPU, CUDAComputeEngine as CUDA

# 基础设施层别名
from src.infrastructure.config.unified_config import get_config, set_config, ConfigScope
from src.infrastructure.cache.thread_safe_cache import ThreadSafeCache, CachePolicy

# 数据层别名
from src.data.loader.stock_loader import StockDataLoader as StockLoader
from src.data.loader.news_loader import FinancialNewsLoader as NewsLoader

# 特征层别名
from src.features.signal_generator import SignalGenerator as SignalGen
from src.features.config import feature_config_manager as FeatureConfig

# 模型层别名
from src.models.model_manager import ModelManager as ModelMgr
from src.models.base_model import BaseModel as Model

# 交易层别名
from src.trading.trading_engine import TradingEngine as Trading
from src.trading.risk.risk_controller import RiskController as Risk

# 导出所有别名
__all__ = [
    'FPGA', 'GPU', 'CUDA',
    'get_config', 'set_config', 'ConfigScope',
    'ThreadSafeCache', 'CachePolicy',
    'StockLoader', 'NewsLoader',
    'SignalGen', 'FeatureConfig',
    'ModelMgr', 'Model',
    'Trading', 'Risk'
]
'''

    aliases_file = Path("src/aliases.py")
    with open(aliases_file, 'w', encoding='utf-8') as f:
        f.write(aliases_content)

    print(f"✅ 创建了导入别名文件: {aliases_file}")


def main():
    """主函数"""
    print("🚀 开始导入路径优化...")

    # 创建导入别名
    create_import_aliases()

    # 优化导入路径
    optimizer = ImportOptimizer()
    results = optimizer.optimize_directory()

    # 生成优化报告
    report_file = Path("reports/import_optimization_report.md")
    report_file.parent.mkdir(exist_ok=True)

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 导入路径优化报告\n\n")
        f.write(f"**优化时间**: 2025-07-19\n")
        f.write(f"**优化文件数**: {len(results)}\n\n")

        f.write("## 优化详情\n\n")
        for file_path, changes in results.items():
            f.write(f"### {file_path}\n")
            for change in changes:
                f.write(f"- {change}\n")
            f.write("\n")

    print(f"📋 优化报告已保存到: {report_file}")


if __name__ == "__main__":
    main()
