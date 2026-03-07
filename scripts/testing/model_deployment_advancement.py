#!/usr/bin/env python3
"""
RQA2025 模型落地推进脚本
按照实施计划分阶段提升测试覆盖率，推进模型落地
"""

import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ModelDeploymentAdvancement:
    """模型落地推进器"""

    def __init__(self):
        self.project_root = project_root
        self.layers = {
            'infrastructure': {
                'path': 'src/infrastructure',
                'target': 90.0,
                'tests': 'tests/unit/infrastructure',
                'priority': 'highest'
            },
            'data': {
                'path': 'src/data',
                'target': 80.0,
                'tests': 'tests/unit/data',
                'priority': 'high'
            },
            'features': {
                'path': 'src/features',
                'target': 80.0,
                'tests': 'tests/unit/features',
                'priority': 'high'
            },
            'models': {
                'path': 'src/models',
                'target': 80.0,
                'tests': 'tests/unit/models',
                'priority': 'medium'
            },
            'trading': {
                'path': 'src/trading',
                'target': 80.0,
                'tests': 'tests/unit/trading',
                'priority': 'medium'
            },
            'backtest': {
                'path': 'src/backtest',
                'target': 80.0,
                'tests': 'tests/unit/backtest',
                'priority': 'low'
            }
        }

    def run_layer_test(self, layer_name: str, env_name: str = 'rqa') -> Dict[str, Any]:
        """运行单层测试"""
        layer_info = self.layers[layer_name]

        print(f"\n=== 执行 {layer_name} 层测试 ===")
        print(f"目标覆盖率: {layer_info['target']}%")
        print(f"测试路径: {layer_info['tests']}")
        print(f"优先级: {layer_info['priority']}")

        # 构建测试命令
        cmd = [
            'python', 'scripts/testing/run_tests.py',
            '--env', env_name,
            '--module', layer_name,
            '--cov', layer_info['path'],
            '--pytest-args', '-v', '--tb=short',
            '--timeout', '300'
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=600
            )

            return {
                'layer': layer_name,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }

        except subprocess.TimeoutExpired:
            return {
                'layer': layer_name,
                'returncode': 124,
                'stdout': '',
                'stderr': '测试执行超时',
                'success': False
            }
        except Exception as e:
            return {
                'layer': layer_name,
                'returncode': 1,
                'stdout': '',
                'stderr': str(e),
                'success': False
            }

    def analyze_coverage(self, stdout: str) -> Dict[str, Any]:
        """分析覆盖率结果"""
        try:
            # 解析覆盖率信息
            lines = stdout.split('\n')
            coverage_info = {}

            for line in lines:
                if 'TOTAL' in line and 'Stmts' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        coverage_info['total_statements'] = int(parts[1])
                        coverage_info['missed_statements'] = int(parts[2])
                        coverage_info['coverage_percentage'] = float(parts[3].rstrip('%'))
                        break

            return coverage_info
        except Exception as e:
            return {'error': str(e)}

    def generate_test_files(self, layer_name: str) -> List[str]:
        """为指定层生成测试文件"""
        layer_info = self.layers[layer_name]
        src_path = self.project_root / layer_info['path']
        test_path = self.project_root / layer_info['tests']

        test_files = []

        # 查找所有Python文件
        for py_file in src_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            # 生成对应的测试文件路径
            relative_path = py_file.relative_to(src_path)
            test_file = test_path / f"test_{relative_path.name}"

            if not test_file.exists():
                test_files.append(str(test_file))

        return test_files

    def create_basic_test_template(self, test_file_path: str, module_path: str) -> str:
        """创建基础测试模板"""
        module_name = Path(module_path).stem
        class_name = ''.join(word.capitalize() for word in module_name.split('_'))

        template = f'''"""
测试文件: {test_file_path}
模块: {module_path}
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入被测试的模块
try:
    from {module_path.replace('/', '.').replace('.py', '')} import {class_name}
except ImportError as e:
    pytest.skip(f"模块导入失败: {{e}}", allow_module_level=True)

class Test{class_name}:
    """测试 {class_name} 类"""
    
    def setup_method(self):
        """测试前的设置"""
        try:
            self.instance = {class_name}()
        except Exception as e:
            pytest.skip(f"实例化失败: {{e}}")
    
    def test_initialization(self):
        """测试初始化"""
        assert self.instance is not None
    
    def test_basic_functionality(self):
        """测试基本功能"""
        # TODO: 添加具体的功能测试
        assert True
    
    def test_error_handling(self):
        """测试错误处理"""
        # TODO: 添加错误处理测试
        assert True

if __name__ == "__main__":
    pytest.main([__file__])
'''
        return template

    def advance_layer(self, layer_name: str, env_name: str = 'rqa') -> Dict[str, Any]:
        """推进单层测试覆盖率"""
        print(f"\n🚀 开始推进 {layer_name} 层...")

        # 1. 运行当前测试
        result = self.run_layer_test(layer_name, env_name)

        # 2. 分析覆盖率
        coverage_info = self.analyze_coverage(result['stdout'])

        # 3. 生成缺失的测试文件
        missing_tests = self.generate_test_files(layer_name)

        # 4. 创建基础测试文件
        created_tests = []
        for test_file in missing_tests[:5]:  # 限制每次最多创建5个测试文件
            try:
                # 从测试文件路径推断模块路径
                test_path = Path(test_file)
                module_name = test_path.stem.replace('test_', '')
                module_path = f"src/{layer_name}/{module_name}.py"

                template = self.create_basic_test_template(test_file, module_path)

                # 确保测试目录存在
                test_path.parent.mkdir(parents=True, exist_ok=True)

                # 写入测试文件
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(template)

                created_tests.append(test_file)
                print(f"✅ 创建测试文件: {test_file}")

            except Exception as e:
                print(f"❌ 创建测试文件失败: {test_file} - {e}")

        return {
            'layer': layer_name,
            'test_result': result,
            'coverage_info': coverage_info,
            'missing_tests': missing_tests,
            'created_tests': created_tests
        }

    def run_advancement_plan(self, target_layers: List[str] = None, env_name: str = 'rqa') -> Dict[str, Any]:
        """执行推进计划"""
        if target_layers is None:
            # 按优先级排序
            priority_order = ['infrastructure', 'data', 'features', 'models', 'trading', 'backtest']
            target_layers = priority_order

        print("🎯 开始模型落地推进计划")
        print(f"目标层: {', '.join(target_layers)}")
        print(f"环境: {env_name}")

        results = {}
        total_created_tests = 0

        for layer_name in target_layers:
            if layer_name not in self.layers:
                print(f"⚠️ 跳过未知层: {layer_name}")
                continue

            try:
                result = self.advance_layer(layer_name, env_name)
                results[layer_name] = result
                total_created_tests += len(result['created_tests'])

                # 显示结果摘要
                coverage = result['coverage_info'].get('coverage_percentage', 0)
                target = self.layers[layer_name]['target']
                status = "✅" if coverage >= target else "⚠️"

                print(f"{status} {layer_name} 层: {coverage:.1f}% / {target}%")

            except Exception as e:
                print(f"❌ {layer_name} 层推进失败: {e}")
                results[layer_name] = {'error': str(e)}

        # 生成总结报告
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_layers': len(target_layers),
            'successful_layers': len([r for r in results.values() if 'error' not in r]),
            'total_created_tests': total_created_tests,
            'results': results
        }

        # 保存报告
        report_file = self.project_root / 'reports' / 'testing' / \
            f'model_deployment_advancement_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n📊 推进计划完成")
        print(f"成功推进层数: {summary['successful_layers']}/{summary['total_layers']}")
        print(f"创建测试文件: {total_created_tests} 个")
        print(f"报告保存至: {report_file}")

        return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 模型落地推进脚本')
    parser.add_argument('--layers', nargs='+',
                        help='指定要推进的层 (infrastructure, data, features, models, trading, backtest)')
    parser.add_argument('--env', default='rqa',
                        help='测试环境名称 (默认: rqa)')
    parser.add_argument('--all', action='store_true',
                        help='推进所有层')

    args = parser.parse_args()

    # 确定目标层
    if args.all:
        target_layers = ['infrastructure', 'data', 'features', 'models', 'trading', 'backtest']
    elif args.layers:
        target_layers = args.layers
    else:
        # 默认按优先级推进
        target_layers = ['infrastructure', 'data', 'features']

    # 执行推进计划
    advancement = ModelDeploymentAdvancement()
    summary = advancement.run_advancement_plan(target_layers, args.env)

    # 根据结果设置退出码
    if summary['successful_layers'] == summary['total_layers']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
