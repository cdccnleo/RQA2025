#!/usr/bin/env python3
"""
RQA2025 基础设施层安全漏洞扫描脚本

扫描基础设施层代码中的安全漏洞
"""

import os
import re


class SecurityScanner:
    def __init__(self):
        self.vulnerabilities = []
        self.security_patterns = {
            'hardcoded_passwords': re.compile(r'password\s*=\s*[\'\"](?!\{\{).+?[\'\"]'),
            'hardcoded_secrets': re.compile(r'(secret|token|key)\s*=\s*[\'\"](?!\{\{).+?[\'\"]'),
            'sql_injection': re.compile(r'execute\(.+\+\s*[\'\"]|[\'\"]\s*\+.+\)'),
            'eval_usage': re.compile(r'\beval\s*\('),
            'exec_usage': re.compile(r'\bexec\s*\('),
            'pickle_usage': re.compile(r'\bpickle\.(loads?|dumps?)'),
            'shell_injection': re.compile(r'subprocess\.(run|call|Popen|check_output)\(.+shell\s*=\s*True'),
            'weak_crypto': re.compile(r'(md5|sha1)\('),
            'insecure_random': re.compile(r'random\.(randint|random|choice|sample)'),
            'path_traversal': re.compile(r'\.\./|\.\.\\'),
            'command_injection': re.compile(r'os\.system|os\.popen'),
        }

    def scan_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            for line_num, line in enumerate(lines, 1):
                for vuln_type, pattern in self.security_patterns.items():
                    if pattern.search(line):
                        # 排除注释和测试文件
                        if not line.strip().startswith('#') and not 'test' in filepath.lower():
                            self.vulnerabilities.append({
                                'file': filepath,
                                'line': line_num,
                                'type': vuln_type,
                                'code': line.strip(),
                                'severity': self._get_severity(vuln_type)
                            })

        except Exception as e:
            pass  # 跳过无法读取的文件

    def _get_severity(self, vuln_type):
        severity_map = {
            'hardcoded_passwords': 'critical',
            'hardcoded_secrets': 'critical',
            'sql_injection': 'high',
            'shell_injection': 'high',
            'eval_usage': 'high',
            'exec_usage': 'high',
            'weak_crypto': 'medium',
            'insecure_random': 'medium',
            'path_traversal': 'medium',
            'command_injection': 'medium',
            'pickle_usage': 'low'
        }
        return severity_map.get(vuln_type, 'low')

    def scan_directory(self, directory):
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    self.scan_file(filepath)

    def print_report(self):
        if not self.vulnerabilities:
            print('✅ 未发现安全漏洞')
            return

        print(f'❌ 发现 {len(self.vulnerabilities)} 个安全问题:')
        print()

        # 按严重程度分组
        by_severity = {}
        for vuln in self.vulnerabilities:
            severity = vuln['severity']
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(vuln)

        for severity in ['critical', 'high', 'medium', 'low']:
            if severity in by_severity:
                vulns = by_severity[severity]
                print(f'🔴 {severity.upper()} ({len(vulns)}个):')
                for vuln in vulns[:5]:  # 只显示前5个
                    print(f'   📁 {os.path.basename(vuln["file"])}:{vuln["line"]}')
                    print(f'      {vuln["type"]}: {vuln["code"][:60]}...')
                if len(vulns) > 5:
                    print(f'      ... 还有 {len(vulns) - 5} 个类似问题')
                print()

    def get_vulnerability_stats(self):
        """获取漏洞统计信息"""
        stats = {
            'total': len(self.vulnerabilities),
            'by_type': {},
            'by_severity': {}
        }

        for vuln in self.vulnerabilities:
            # 按类型统计
            vuln_type = vuln['type']
            stats['by_type'][vuln_type] = stats['by_type'].get(vuln_type, 0) + 1

            # 按严重程度统计
            severity = vuln['severity']
            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1

        return stats


def main():
    print('🔒 RQA2025 基础设施层安全漏洞扫描')
    print('=' * 50)

    scanner = SecurityScanner()
    scanner.scan_directory('src/infrastructure')
    scanner.print_report()

    stats = scanner.get_vulnerability_stats()
    print(f'📊 扫描完成统计:')
    print(f'   总问题数: {stats["total"]}')
    print(f'   按严重程度: {stats["by_severity"]}')
    print(
        f'   主要问题类型: {dict(sorted(stats["by_type"].items(), key=lambda x: x[1], reverse=True)[:5])}')


if __name__ == "__main__":
    main()
