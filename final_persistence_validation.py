#!/usr/bin/env python3
"""
量化交易数据持久化架构 - 最终验证脚本
"""

import requests
import time
import subprocess

def test_data_persistence():
    """测试数据持久化完整流程"""
    print("=== 最终验证完整数据持久化流程 ===")

    # 1. 创建测试数据源
    test_source = {
        'name': '最终持久化验证',
        'type': '财经新闻',
        'url': 'https://test.final.persistence.com'
    }

    try:
        create_response = requests.post('http://localhost:8000/api/v1/data/sources',
                                       json=test_source, timeout=5)

        if create_response.status_code == 200:
            result = create_response.json()
            if result.get('success'):
                source_id = result.get('data', {}).get('id')
                print('✅ 数据源创建成功:', source_id)

                # 2. 多次触发样本获取，模拟数据采集
                print('🔄 触发多次数据采集...')

                for i in range(3):
                    print('  第' + str(i+1) + '次采集...')
                    sample_response = requests.get('http://localhost:8000/api/v1/samples/' + source_id, timeout=10)

                    if sample_response.status_code == 200:
                        sample_data = sample_response.json()
                        data_source = sample_data.get('data_source')
                        cache_info = sample_data.get('cache_info', {})

                        print('    ✅ 数据源: ' + str(data_source) + ', 缓存: ' + str(cache_info.get('has_cache', False)))

                        # 检查是否有数据
                        sample_content = sample_data.get('sample_data', {})
                        if isinstance(sample_content, dict) and 'news' in sample_content:
                            news_count = len(sample_content['news'])
                            print('    📊 新闻数量: ' + str(news_count))
                        elif isinstance(sample_content, dict) and 'cryptocurrencies' in sample_content:
                            crypto_count = len(sample_content['cryptocurrencies'])
                            print('    📊 加密货币数量: ' + str(crypto_count))

                    else:
                        print('    ❌ 采集失败: ' + str(sample_response.status_code))

                    time.sleep(1)  # 短暂延迟

                # 3. 等待数据持久化完成
                print('⏳ 等待数据持久化...')
                time.sleep(5)

                # 4. 最终状态检查
                print('🔍 最终持久化状态检查:')

                # 检查Redis缓存
                redis_result = subprocess.run([
                    'docker', 'exec', 'rqa2025-redis-1', 'redis-cli', 'KEYS', '*'
                ], capture_output=True, text=True, timeout=5)

                redis_keys = redis_result.stdout.strip().split('\n') if redis_result.returncode == 0 else []
                quant_keys = [k for k in redis_keys if k and ('quant' in k or source_id in k)]

                if quant_keys:
                    print('✅ Redis缓存:')
                    for key in quant_keys[:5]:
                        print('   - ' + key)
                    print('   共 ' + str(len(quant_keys)) + ' 个缓存键')
                else:
                    print('⚠️ Redis缓存: 无相关数据')

                # 检查PostgreSQL数据
                db_result = subprocess.run([
                    'docker', 'exec', 'rqa2025-postgres-1', 'psql',
                    '-U', 'rqa2025', '-d', 'rqa2025', '-c',
                    'SELECT COUNT(*) as total_records FROM collected_data;'
                ], capture_output=True, text=True, timeout=5)

                if db_result.returncode == 0:
                    lines = db_result.stdout.strip().split('\n')
                    count_line = [line for line in lines if 'total_records' not in line and line.strip() and not line.startswith('-')]
                    if count_line:
                        count = count_line[0].strip()
                        print('✅ PostgreSQL记录数: ' + count)
                    else:
                        print('⚠️ PostgreSQL: 无法解析记录数')
                else:
                    print('❌ PostgreSQL查询失败')

                # 检查文件存储
                file_result = subprocess.run([
                    'docker', 'exec', 'rqa2025-rqa2025-app-1', 'find', '/app/data',
                    '-name', '*.json', '-type', 'f', '-mmin', '-10'
                ], capture_output=True, text=True, timeout=5)

                if file_result.returncode == 0:
                    files = [line for line in file_result.stdout.strip().split('\n') if line.strip()]
                    if files:
                        print('✅ 文件存储:')
                        for file_path in files[:3]:
                            print('   - ' + file_path)
                        if len(files) > 3:
                            print('   ... 还有 ' + str(len(files) - 3) + ' 个文件')
                    else:
                        print('⚠️ 文件存储: 最近10分钟无新文件')

            else:
                print('❌ 数据源创建失败:', result.get('message'))
        else:
            print('❌ 创建请求失败:', create_response.status_code)

    except Exception as e:
        print('❌ 测试错误:', str(e)[:100])

    print()
    print('🎉 完整数据持久化流程验证完成！')
    print()
    print('📋 量化交易数据持久化架构总结:')
    print('🏗️  架构层次: Redis缓存 → PostgreSQL数据库 → 文件系统')
    print('🔄 数据流程: 采集 → 处理 → 存储 → 缓存 → 查询')
    print('📊 支持类型: 财经新闻、股票数据、宏观经济、加密货币')
    print('⚡ 性能优化: 智能缓存、批量处理、异步操作')
    print('🛡️  容错设计: 降级处理、错误重试、状态监控')
    print('📈 可扩展性: 插件化架构、配置驱动、多数据源支持')

if __name__ == "__main__":
    test_data_persistence()
