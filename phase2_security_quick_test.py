#!/usr/bin/env python3
"""
Phase 2 安全性测试验证（简化版）
"""

import os
from pathlib import Path
import json
from datetime import datetime

def main():
    print('🔒 Phase 2 安全性测试开始')
    print('=' * 60)

    # 安全检查结果收集
    security_findings = {
        'vulnerabilities': [],
        'configuration_issues': [],
        'access_control_issues': [],
        'data_protection_issues': [],
        'compliance_issues': []
    }

    # 查找Python文件
    def find_python_files():
        python_files = []
        src_dir = Path('src')

        for file_path in src_dir.rglob('*.py'):
            if not any(skip in str(file_path) for skip in ['__pycache__', '.git', 'test']):
                python_files.append(file_path)

        return python_files[:10]  # 限制文件数量

    print('🔍 扫描源代码安全问题...')

    # 扫描文件
    python_files = find_python_files()
    print(f'扫描 {len(python_files)} 个Python文件')

    total_vulnerabilities = 0

    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            file_vulnerabilities = 0

            # 检查不安全导入
            insecure_imports = ['pickle', 'eval', 'exec', 'input', 'os.system']
            for insecure_import in insecure_imports:
                if insecure_import in content:
                    security_findings['vulnerabilities'].append({
                        'type': 'insecure_import',
                        'file': str(file_path),
                        'severity': 'high',
                        'description': f'发现不安全的导入: {insecure_import}'
                    })
                    file_vulnerabilities += 1

            # 检查硬编码密码（简单检查）
            if 'password=' in content.lower() or 'api_key=' in content.lower():
                security_findings['vulnerabilities'].append({
                    'type': 'potential_hardcoded_secret',
                    'file': str(file_path),
                    'severity': 'medium',
                    'description': '发现可能的硬编码敏感信息'
                })
                file_vulnerabilities += 1

            if file_vulnerabilities > 0:
                print(f'⚠️ {file_path.name}: 发现 {file_vulnerabilities} 个安全问题')

            total_vulnerabilities += file_vulnerabilities

        except Exception as e:
            print(f'❌ 处理文件 {file_path.name} 时出错')

    # 生成安全报告
    total_findings = sum(len(findings) for findings in security_findings.values())

    # 评估风险等级
    severity_weights = {'critical': 10, 'high': 7, 'medium': 4, 'low': 2, 'info': 1}
    total_risk_score = 0

    for category, findings in security_findings.items():
        for finding in findings:
            severity = finding.get('severity', 'info')
            total_risk_score += severity_weights.get(severity, 1)

    if total_risk_score >= 30:
        risk_level = 'high'
    elif total_risk_score >= 10:
        risk_level = 'medium'
    elif total_risk_score >= 3:
        risk_level = 'low'
    else:
        risk_level = 'minimal'

    print('=' * 60)
    print(f'📊 安全审计结果:')
    print(f'扫描文件数: {len(python_files)}')
    print(f'总发现数: {total_findings}')
    print(f'漏洞数: {len(security_findings["vulnerabilities"])}')
    print(f'风险等级: {risk_level}')
    print(f'风险评分: {total_risk_score}')

    if total_findings == 0:
        print('🎉 安全审计通过！未发现严重安全问题')
    elif risk_level in ['minimal', 'low']:
        print('✅ 安全状况良好，发现的问题风险较低')
    else:
        print('⚠️ 发现较高风险问题，建议及时修复')

    # 保存报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 2 - Security Quick Audit',
        'summary': {
            'files_scanned': len(python_files),
            'total_findings': total_findings,
            'risk_level': risk_level,
            'risk_score': total_risk_score,
            'status': 'passed' if risk_level in ['minimal', 'low'] else 'needs_attention'
        },
        'findings': security_findings
    }

    report_file = f'phase2_security_quick_audit_{int(datetime.now().timestamp())}.json'
    os.makedirs('test_logs', exist_ok=True)
    with open(f'test_logs/{report_file}', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f'📄 报告已保存: test_logs/{report_file}')
    print('=' * 60)

    return risk_level, total_findings

if __name__ == "__main__":
    main()
