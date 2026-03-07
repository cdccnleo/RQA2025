import os
import subprocess
import sys

layers = [
    ('automation', '自动化层'),
    ('resilience', '弹性层'),
    ('testing', '测试层'),
    ('tools', '工具层'),
    ('distributed', '分布式协调器'),
    ('async_processor', '异步处理器'),
    ('mobile', '移动端层'),
    ('boundary', '业务边界层')
]

results = {}

for layer_dir, layer_name in layers:
    print(f'\n验证{layer_name} ({layer_dir})...')
    try:
        # 运行pytest
        result = subprocess.run([
            sys.executable, '-m', 'pytest',
            f'tests/unit/{layer_dir}/',
            '--tb=short', '-x', '--disable-warnings', '-q'
        ], capture_output=True, text=True, timeout=300)

        # 解析结果
        output = result.stdout + result.stderr
        passed = 0
        failed = 0
        skipped = 0

        for line in output.split('\n'):
            if 'PASSED' in line and 'tests/unit/' in line:
                passed += 1
            elif 'FAILED' in line and 'tests/unit/' in line:
                failed += 1
            elif 'SKIPPED' in line and 'tests/unit/' in line:
                skipped += 1

        # 查找最终统计
        final_stats = f'{passed} passed, {failed} failed, {skipped} skipped'
        for line in reversed(output.split('\n')):
            if 'passed' in line.lower() and 'failed' in line.lower():
                final_stats = line.strip()
                break

        results[layer_name] = {
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'stats': final_stats,
            'success': result.returncode == 0
        }

        print(f'  结果: {final_stats}')

    except Exception as e:
        results[layer_name] = {
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'stats': f'错误: {str(e)}',
            'success': False
        }
        print(f'  错误: {str(e)}')

print('\n' + '='*80)
print('各层级测试汇总:')
print('='*80)

total_passed = 0
total_failed = 0
total_skipped = 0

for layer_name, stats in results.items():
    print(f'{layer_name:15} | {stats["stats"]}')
    total_passed += stats['passed']
    total_failed += stats['failed']
    total_skipped += stats['skipped']

print('='*80)
print(f'总计: {total_passed} 通过, {total_failed} 失败, {total_skipped} 跳过')
