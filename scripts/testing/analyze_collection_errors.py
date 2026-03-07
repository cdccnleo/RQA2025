#!/usr/bin/env python3
"""
分析pytest收集错误的工具

用于分析第一阶段收集错误，分类错误类型，制定修复策略
"""

import subprocess
import sys
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple


class CollectionErrorAnalyzer:
    """收集错误分析器"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.errors = []
        self.error_by_type = defaultdict(list)
        self.error_by_file = defaultdict(list)
        self.error_patterns = defaultdict(int)
        
    def collect_errors(self) -> List[str]:
        """收集所有pytest收集错误"""
        print("🔍 正在收集pytest错误信息...")
        
        cmd = [sys.executable, '-m', 'pytest', 'tests/', '--collect-only', '--tb=short']
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                cwd=str(self.project_root),
                timeout=300  # 5分钟超时
            )
            
            # 提取ERROR行
            error_lines = []
            in_error_block = False
            current_error = []
            
            for line in result.stderr.split('\n') + result.stdout.split('\n'):
                if line.strip().startswith('ERROR'):
                    in_error_block = True
                    current_error = [line]
                elif in_error_block:
                    if line.strip() and not line.strip().startswith('='):
                        current_error.append(line)
                    elif line.strip().startswith('='):
                        if current_error:
                            error_lines.append('\n'.join(current_error))
                            current_error = []
                        in_error_block = False
                elif 'ModuleNotFoundError' in line or 'ImportError' in line or 'SyntaxError' in line:
                    if current_error:
                        current_error.append(line)
                    
            if current_error:
                error_lines.append('\n'.join(current_error))
                
            self.errors = error_lines
            return error_lines
            
        except subprocess.TimeoutExpired:
            print("❌ 收集超时")
            return []
        except Exception as e:
            print(f"❌ 收集失败: {e}")
            return []
    
    def categorize_errors(self):
        """分类错误"""
        print("📊 正在分析错误类型...")
        
        for error_text in self.errors:
            # 提取文件路径
            file_match = re.search(r'tests[\\/][^\s]+\.py', error_text)
            if file_match:
                file_path = file_match.group(0)
                self.error_by_file[file_path].append(error_text)
            
            # 分类错误类型
            error_lower = error_text.lower()
            
            if 'modulenotfounderror' in error_lower or 'no module named' in error_lower:
                module_match = re.search(r"no module named ['\"]([^'\"]+)['\"]", error_lower)
                if module_match:
                    module_name = module_match.group(1)
                    self.error_by_type['ModuleNotFoundError'].append({
                        'module': module_name,
                        'file': file_match.group(0) if file_match else 'unknown',
                        'error': error_text
                    })
                    self.error_patterns[f'ModuleNotFoundError: {module_name}'] += 1
                else:
                    self.error_by_type['ModuleNotFoundError'].append({'error': error_text})
                    self.error_patterns['ModuleNotFoundError (unknown)'] += 1
                    
            elif 'importerror' in error_lower:
                self.error_by_type['ImportError'].append({'error': error_text})
                self.error_patterns['ImportError'] += 1
                
            elif 'syntaxerror' in error_lower:
                self.error_by_type['SyntaxError'].append({'error': error_text})
                self.error_patterns['SyntaxError'] += 1
                
            elif 'indentationerror' in error_lower:
                self.error_by_type['IndentationError'].append({'error': error_text})
                self.error_patterns['IndentationError'] += 1
                
            elif 'attributeerror' in error_lower:
                self.error_by_type['AttributeError'].append({'error': error_text})
                self.error_patterns['AttributeError'] += 1
                
            else:
                self.error_by_type['Other'].append({'error': error_text})
                self.error_patterns['Other'] += 1
    
    def generate_report(self) -> str:
        """生成分析报告"""
        report = []
        report.append("# 📊 pytest收集错误分析报告\n")
        report.append(f"**总错误数**: {len(self.errors)}\n\n")
        
        # 错误类型统计
        report.append("## 📈 错误类型分布\n\n")
        report.append("| 错误类型 | 数量 | 占比 |\n")
        report.append("|---------|------|------|\n")
        
        total = len(self.errors)
        for error_type, count in sorted(self.error_by_type.items(), key=lambda x: len(x[1]), reverse=True):
            percentage = (len(count) / total * 100) if total > 0 else 0
            report.append(f"| {error_type} | {len(count)} | {percentage:.1f}% |\n")
        
        # 高频错误模式
        report.append("\n## 🔍 高频错误模式\n\n")
        report.append("| 错误模式 | 出现次数 |\n")
        report.append("|---------|----------|\n")
        
        for pattern, count in sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:20]:
            report.append(f"| {pattern} | {count} |\n")
        
        # 文件错误统计
        report.append("\n## 📁 文件错误统计\n\n")
        report.append("| 文件 | 错误数 |\n")
        report.append("|------|--------|\n")
        
        for file_path, errors in sorted(self.error_by_file.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
            report.append(f"| {file_path} | {len(errors)} |\n")
        
        # ModuleNotFoundError详细分析
        if self.error_by_type['ModuleNotFoundError']:
            report.append("\n## 🔴 ModuleNotFoundError详细分析\n\n")
            
            # 统计缺失的模块
            missing_modules = Counter()
            for error_info in self.error_by_type['ModuleNotFoundError']:
                if 'module' in error_info:
                    missing_modules[error_info['module']] += 1
            
            report.append("### 缺失模块统计（Top 10）\n\n")
            report.append("| 模块名 | 出现次数 |\n")
            report.append("|--------|----------|\n")
            
            for module, count in missing_modules.most_common(10):
                report.append(f"| {module} | {count} |\n")
        
        # 修复建议
        report.append("\n## 💡 修复建议\n\n")
        
        if self.error_by_type['ModuleNotFoundError']:
            report.append("### ModuleNotFoundError修复策略\n")
            report.append("1. 检查缺失模块是否存在\n")
            report.append("2. 修复导入路径（如：`.exception_utils` → `.core.exceptions`）\n")
            report.append("3. 创建缺失的模块文件（如我们刚创建的`exception_utils.py`）\n\n")
        
        if self.error_by_type['ImportError']:
            report.append("### ImportError修复策略\n")
            report.append("1. 检查循环导入问题\n")
            report.append("2. 修复相对导入路径\n")
            report.append("3. 确保所有依赖已安装\n\n")
        
        if self.error_by_type['SyntaxError']:
            report.append("### SyntaxError修复策略\n")
            report.append("1. 使用linter检查语法错误\n")
            report.append("2. 修复缩进问题\n")
            report.append("3. 修复括号匹配问题\n\n")
        
        return ''.join(report)
    
    def run_analysis(self) -> Dict:
        """运行完整分析"""
        print("🚀 开始分析pytest收集错误...\n")
        
        # 收集错误
        errors = self.collect_errors()
        print(f"✅ 收集到 {len(errors)} 个错误\n")
        
        # 分类错误
        self.categorize_errors()
        print(f"✅ 错误分类完成\n")
        
        # 生成报告
        report = self.generate_report()
        
        # 保存报告
        report_path = self.project_root / 'test_logs' / 'collection_errors_analysis.md'
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 报告已保存到: {report_path}\n")
        print(report)
        
        return {
            'total_errors': len(self.errors),
            'errors_by_type': {k: len(v) for k, v in self.error_by_type.items()},
            'error_patterns': dict(self.error_patterns),
            'errors_by_file': {k: len(v) for k, v in self.error_by_file.items()},
            'report_path': str(report_path)
        }


def main():
    """主函数"""
    analyzer = CollectionErrorAnalyzer()
    results = analyzer.run_analysis()
    
    print("\n" + "="*60)
    print("📊 分析完成！")
    print(f"总错误数: {results['total_errors']}")
    print(f"报告路径: {results['report_path']}")
    print("="*60)


if __name__ == '__main__':
    main()

