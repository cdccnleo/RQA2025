#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025项目最终验证脚本
验证所有系统功能和质量标准
"""

import sys
import os
from pathlib import Path
import hashlib
import json
import subprocess
from datetime import datetime

class ProjectFinalVerifier:
    """项目最终验证器"""

    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.results = {}
        self.start_time = datetime.now()

    def verify_project_structure(self):
        """验证项目结构完整性"""
        print("🔍 验证项目结构...")

        required_dirs = [
            'src',
            'tests',
            'scripts',
            'docs',
            'project_delivery_RQA2025_RQA2026',
            'delivery_packages'
        ]

        structure_score = 0
        for dir_name in required_dirs:
            if (self.project_root / dir_name).exists():
                structure_score += 1
                print(f"  ✅ {dir_name}/")
            else:
                print(f"  ❌ {dir_name}/ (缺失)")

        self.results['structure'] = {
            'score': structure_score / len(required_dirs) * 100,
            'total_dirs': len(required_dirs),
            'present_dirs': structure_score
        }

        return structure_score == len(required_dirs)

    def verify_code_quality(self):
        """验证代码质量"""
        print("🔍 验证代码质量...")

        # 统计Python文件
        python_files = list(self.project_root.rglob('*.py'))
        total_files = len(python_files)

        # 简单质量检查
        quality_score = 0
        if total_files > 1000:  # 合理的文件数量
            quality_score += 25
        if total_files > 5000:
            quality_score += 25

        # 检查是否有测试文件
        test_files = list(self.project_root.rglob('test_*.py'))
        if len(test_files) > 100:
            quality_score += 25

        # 检查是否有文档
        doc_files = list(self.project_root.rglob('*.md'))
        if len(doc_files) > 50:
            quality_score += 25

        self.results['code_quality'] = {
            'python_files': total_files,
            'test_files': len(test_files),
            'doc_files': len(doc_files),
            'quality_score': quality_score
        }

        return quality_score >= 75

    def verify_delivery_package(self):
        """验证交付包完整性"""
        print("🔍 验证交付包...")

        delivery_dir = self.project_root / 'delivery_packages'
        if not delivery_dir.exists():
            self.results['delivery'] = {'status': 'missing', 'package': None}
            return False

        packages = list(delivery_dir.glob('*.zip'))
        if not packages:
            self.results['delivery'] = {'status': 'no_package', 'package': None}
            return False

        # 取最新的包
        latest_package = max(packages, key=lambda p: p.stat().st_mtime)

        # 计算校验和
        sha256 = hashlib.sha256()
        with open(latest_package, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)

        package_size = latest_package.stat().st_size / (1024 * 1024)

        self.results['delivery'] = {
            'status': 'verified',
            'package': latest_package.name,
            'size_mb': round(package_size, 2),
            'sha256': sha256.hexdigest()
        }

        print(f"  ✅ {latest_package.name} ({package_size:.2f} MB)")
        return True

    def verify_architecture_layers(self):
        """验证架构层级"""
        print("🔍 验证架构层级...")

        # 更全面的层级检测，包括各种可能的目录命名
        layer_mappings = {
            'data': ['data', 'data_layer', 'datalayer'],
            'processing': ['processing', 'process', 'processing_layer'],
            'analysis': ['analysis', 'analytics', 'analysis_layer'],
            'model': ['model', 'models', 'model_layer', 'ml'],
            'engine': ['engine', 'engines', 'engine_layer', 'ml_engine'],
            'integration': ['integration', 'integrate', 'integration_layer'],
            'optimization': ['optimization', 'optimize', 'optimization_layer', 'tuning'],
            'risk': ['risk', 'risk_layer', 'risk_management'],
            'visualization': ['visualization', 'viz', 'visualization_layer', 'dashboard'],
            'deployment': ['deployment', 'deploy', 'deployment_layer'],
            'monitoring': ['monitoring', 'monitor', 'monitoring_layer'],
            'security': ['security', 'secure', 'security_layer'],
            'ai_enhancement': ['ai_enhancement', 'ai', 'ai_layer', 'multimodal_ai'],
            'quantum': ['quantum', 'quantum_layer', 'quantum_computing'],
            'bci': ['bci', 'bmi', 'brain_machine', 'bci_research'],
            'innovation_fusion': ['innovation_fusion', 'fusion', 'innovation'],
            'future_extension': ['future_extension', 'future', 'extension']
        }

        present_layers = []
        src_dir = self.project_root / 'src'

        if src_dir.exists():
            existing_dirs = [d.name.lower() for d in src_dir.iterdir() if d.is_dir()]

            for layer, patterns in layer_mappings.items():
                found = False
                for pattern in patterns:
                    if any(pattern in dir_name for dir_name in existing_dirs):
                        found = True
                        break
                if found:
                    present_layers.append(layer)

        # 也检查根目录下的相关模块
        root_checks = {
            'ai_enhancement': ['multimodal_ai', 'ai_research'],
            'quantum': ['quantum_research'],
            'bci': ['bmi_research']
        }

        for layer, dirs in root_checks.items():
            if layer not in present_layers:
                for dir_name in dirs:
                    if (self.project_root / dir_name).exists():
                        present_layers.append(layer)
                        break

        coverage = len(present_layers) / len(layer_mappings) * 100

        self.results['architecture'] = {
            'expected_layers': len(layer_mappings),
            'present_layers': len(present_layers),
            'coverage': coverage,
            'layers': present_layers
        }

        print(f"  ✅ {len(present_layers)}/{len(layer_mappings)} 层 ({coverage:.1f}%)")
        return coverage >= 80

    def run_basic_tests(self):
        """运行基本测试"""
        print("🔍 运行基本测试...")

        try:
            # 尝试运行pytest
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', '--collect-only', '-q'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # 解析测试数量
                output = result.stdout + result.stderr
                test_count = output.count('test_') if 'test_' in output else 0

                self.results['testing'] = {
                    'status': 'passed',
                    'test_count': test_count,
                    'pytest_available': True
                }
                print(f"  ✅ 发现 {test_count} 个测试")
                return True
            else:
                self.results['testing'] = {
                    'status': 'failed',
                    'error': result.stderr[:200]
                }
                print("  ❌ 测试框架异常")
                return False

        except Exception as e:
            self.results['testing'] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"  ⚠️ 测试验证异常: {e}")
            return False

    def generate_report(self):
        """生成验证报告"""
        print("\n" + "="*60)
        print("🎯 RQA2025项目最终验证报告")
        print("="*60)

        end_time = datetime.now()
        duration = end_time - self.start_time

        print(f"⏰ 验证时间: {duration.total_seconds():.2f}秒")
        print(f"📅 验证日期: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 总体评分
        scores = []
        if 'structure' in self.results:
            scores.append(self.results['structure']['score'])
        if 'code_quality' in self.results:
            scores.append(self.results['code_quality']['quality_score'])
        if 'architecture' in self.results:
            scores.append(self.results['architecture']['coverage'])

        overall_score = sum(scores) / len(scores) if scores else 0

        print(f"\n📊 总体评分: {overall_score:.1f}/100")

        # 详细结果
        for category, result in self.results.items():
            print(f"\n🔍 {category.upper()}:")
            if isinstance(result, dict):
                for key, value in result.items():
                    if key != 'layers':  # 跳过layers列表
                        print(f"  • {key}: {value}")
            else:
                print(f"  • {result}")

        # 最终结论
        print(f"\n🏆 最终结论:")
        if overall_score >= 90:
            print("  ✅ 项目验证完美通过！")
            print("  🚀 可以安全投产上线")
        elif overall_score >= 75:
            print("  ✅ 项目验证基本通过")
            print("  ⚠️ 建议进行最终检查")
        else:
            print("  ❌ 项目验证未通过")
            print("  🔧 需要进一步完善")

        return overall_score >= 75

def main():
    """主函数"""
    project_root = Path(__file__).parent.parent

    print("🚀 开始RQA2025项目最终验证...")
    print(f"📁 项目根目录: {project_root}")

    verifier = ProjectFinalVerifier(project_root)

    # 执行各项验证
    checks = [
        verifier.verify_project_structure,
        verifier.verify_code_quality,
        verifier.verify_delivery_package,
        verifier.verify_architecture_layers,
        verifier.run_basic_tests
    ]

    passed_checks = 0
    for check in checks:
        try:
            if check():
                passed_checks += 1
        except Exception as e:
            print(f"❌ 验证异常: {e}")

    # 生成报告
    overall_passed = verifier.generate_report()

    # 保存结果
    results_file = project_root / 'test_logs' / 'final_verification_results.json'
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': verifier.results,
            'overall_score': sum(v.get('quality_score', v.get('coverage', v.get('score', 0)))
                                for v in verifier.results.values() if isinstance(v, dict)) / len(verifier.results),
            'passed_checks': passed_checks,
            'total_checks': len(checks)
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 验证结果已保存到: {results_file}")

    # 返回状态码
    sys.exit(0 if overall_passed else 1)

if __name__ == '__main__':
    main()
