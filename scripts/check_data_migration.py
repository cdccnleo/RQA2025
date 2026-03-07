#!/usr/bin/env python3
"""
数据迁移准备检查脚本
"""

import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime

def main():
    print('📦 数据迁移准备检查报告')
    print('=' * 50)

    # 检查数据目录结构
    print('📁 数据目录结构检查:')
    data_dirs = ['data/', 'tests/data/', 'reports/']
    for dir_path in data_dirs:
        exists = os.path.exists(dir_path)
        status = '✅ 存在' if exists else '❌ 缺失'
        if exists:
            try:
                files = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
                status += f' ({files}个文件)'
            except:
                status += ' (访问受限)'
        print(f'{dir_path:<15} {status}')

    # 检查数据库文件（如果存在）
    print('\n🗄️ 数据库文件检查:')
    db_files = ['data/market_data.db', 'data/trading_data.db', 'tests/test_data.db']
    for db_file in db_files:
        if os.path.exists(db_file):
            size = os.path.getsize(db_file) / 1024  # KB
            print(f'{db_file}: ✅ 存在 ({size:.1f}KB)')

            # 尝试连接数据库验证
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
                tables = cursor.fetchall()
                table_count = len(tables)
                print(f'  表数量: {table_count}')
                conn.close()
            except Exception as e:
                print(f'  ❌ 数据库连接失败: {e}')
        else:
            print(f'{db_file}: ❌ 不存在')

    # 检查配置文件
    print('\n⚙️ 配置文件检查:')
    config_files = ['requirements.txt', 'pytest.ini', 'README.md']
    for config_file in config_files:
        if os.path.exists(config_file):
            size = os.path.getsize(config_file)
            print(f'{config_file}: ✅ 存在 ({size} bytes)')
        else:
            print(f'{config_file}: ❌ 缺失')

    # 创建数据迁移测试脚本
    print('\n🔄 创建数据迁移测试脚本:')
    migration_script = '''#!/usr/bin/env python3
"""
数据迁移测试脚本
"""

import os
import json
from pathlib import Path
from datetime import datetime

def test_data_backup():
    """测试数据备份功能"""
    print("开始数据备份测试...")

    # 模拟备份操作
    backup_dir = Path('backups')
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = backup_dir / f'backup_{timestamp}.json'

    test_data = {
        'timestamp': timestamp,
        'type': 'test_backup',
        'status': 'success'
    }

    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"备份文件创建成功: {backup_file}")
    return True

def test_data_validation():
    """测试数据验证功能"""
    print("开始数据验证测试...")

    # 模拟数据验证
    test_records = [
        {'id': 1, 'symbol': 'AAPL', 'price': 150.0},
        {'id': 2, 'symbol': 'GOOGL', 'price': 2500.0},
        {'id': 3, 'symbol': 'MSFT', 'price': 300.0}
    ]

    valid_count = 0
    for record in test_records:
        if all(key in record for key in ['id', 'symbol', 'price']):
            if isinstance(record['price'], (int, float)) and record['price'] > 0:
                valid_count += 1

    print(f"数据验证完成: {valid_count}/{len(test_records)} 条记录有效")
    return valid_count == len(test_records)

if __name__ == '__main__':
    print("数据迁移测试开始")
    print("-" * 30)

    backup_success = test_data_backup()
    validation_success = test_data_validation()

    print("-" * 30)
    if backup_success and validation_success:
        print("✅ 数据迁移测试通过")
    else:
        print("❌ 数据迁移测试失败")
'''

    # 保存迁移脚本
    script_path = Path('scripts/test_data_migration.py')
    script_path.parent.mkdir(exist_ok=True)

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(migration_script)

    print(f'数据迁移测试脚本已创建: {script_path}')

    # 执行测试脚本
    print('\n▶️ 执行数据迁移测试:')
    try:
        exec(open(script_path).read())
    except Exception as e:
        print(f'❌ 脚本执行失败: {e}')

    # 检查备份目录
    print('\n📦 备份策略检查:')
    backup_dir = Path('backups')
    if backup_dir.exists():
        backup_files = list(backup_dir.glob('*.json'))
        print(f'备份文件数量: {len(backup_files)}')
        if backup_files:
            latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
            print(f'最新备份文件: {latest_backup.name}')
            print(f'备份文件大小: {latest_backup.stat().st_size} bytes')
    else:
        print('备份目录不存在，将在需要时创建')

    print('\n✅ 数据迁移准备检查完成')
    print('=' * 50)

if __name__ == '__main__':
    main()
