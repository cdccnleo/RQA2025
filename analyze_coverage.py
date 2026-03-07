import json

with open('test_logs/infrastructure_coverage.json', 'r') as f:
    data = json.load(f)

print('基础设施层覆盖率统计:')
print(f'总文件数: {data["meta"]["num_files"]}')
print(f'总行数: {data["meta"]["num_statements"]}')
print(f'覆盖行数: {data["meta"]["num_statements"] - data["meta"]["num_missing"]}')
print(f'覆盖率: {data["meta"]["percent_covered"]:.1f}%')

print('\n覆盖率最低的10个文件:')
files = [(k, v['percent_covered']) for k, v in data['files'].items() if v['num_statements'] > 0]
files.sort(key=lambda x: x[1])
for file_path, coverage in files[:10]:
    if coverage < 50:
        num_statements = data['files'][file_path]['num_statements']
        print(f'{file_path}: {coverage:.1f}% ({num_statements}行)')

print('\n覆盖率最高的10个文件:')
files.sort(key=lambda x: x[1], reverse=True)
for file_path, coverage in files[:10]:
    if coverage >= 80:
        num_statements = data['files'][file_path]['num_statements']
        print(f'{file_path}: {coverage:.1f}% ({num_statements}行)')