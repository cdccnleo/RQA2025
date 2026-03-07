#!/usr/bin/env python3
"""
RQA2025 数据迁移验证脚本

验证预投产环境的数据迁移完整性和一致性
"""

import sys
import time
import json
from datetime import datetime
import psycopg2
from influxdb_client import InfluxDBClient
import redis
import requests


class DataMigrationValidator:
    """数据迁移验证器"""

    def __init__(self):
        # 数据库连接配置
        self.pg_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'rqa2025',
            'user': 'rqa2025_user',
            'password': 'rqa2025_secure_pass'
        }

        self.influx_config = {
            'url': 'http://localhost:8086',
            'token': 'rqa2025_token_preprod_12345',
            'org': 'rqa2025_org',
            'bucket': 'rqa2025_metrics'
        }

        self.redis_config = {
            'host': 'localhost',
            'port': 6379,
            'password': 'rqa2025_redis_pass_preprod',
            'db': 0
        }

    def validate_postgres_connection(self):
        """验证PostgreSQL连接"""
        print("🔍 验证PostgreSQL连接...")

        try:
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor()

            # 测试查询
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"   ✅ PostgreSQL连接成功: {version[:50]}...")

            # 检查数据库结构
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = cursor.fetchall()
            table_names = [table[0] for table in tables]

            expected_tables = [
                'health_checks', 'monitoring_metrics', 'alerts',
                'configurations', 'user_sessions', 'api_access_logs',
                'performance_metrics'
            ]

            missing_tables = [t for t in expected_tables if t not in table_names]
            if missing_tables:
                print(f"   ⚠️  缺少表: {missing_tables}")
            else:
                print(f"   ✅ 所有必需表存在: {len(table_names)} 个表")

            # 检查数据量
            for table in expected_tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table};")
                count = cursor.fetchone()[0]
                print(f"   📊 {table}: {count} 条记录")

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            print(f"   ❌ PostgreSQL连接失败: {e}")
            return False

    def validate_influxdb_connection(self):
        """验证InfluxDB连接"""
        print("🔍 验证InfluxDB连接...")

        try:
            client = InfluxDBClient(**self.influx_config)
            health = client.health()

            if health.status == 'pass':
                print("   ✅ InfluxDB连接成功")

                # 检查bucket
                buckets_api = client.buckets_api()
                buckets = buckets_api.find_buckets().buckets

                rqa2025_bucket = None
                for bucket in buckets:
                    if bucket.name == 'rqa2025_metrics':
                        rqa2025_bucket = bucket
                        break

                if rqa2025_bucket:
                    print("   ✅ rqa2025_metrics bucket存在")
                else:
                    print("   ⚠️  rqa2025_metrics bucket不存在")

                # 查询一些数据
                query_api = client.query_api()
                query = '''
                from(bucket: "rqa2025_metrics")
                  |> range(start: -1h)
                  |> limit(n: 5)
                '''

                try:
                    result = query_api.query(query)
                    record_count = sum(len(table.records) for table in result)
                    print(f"   📊 最近1小时数据点: {record_count}")
                except Exception as e:
                    print(f"   ℹ️  数据查询: {e}")

                client.close()
                return True
            else:
                print(f"   ❌ InfluxDB健康检查失败: {health.status}")
                return False

        except Exception as e:
            print(f"   ❌ InfluxDB连接失败: {e}")
            return False

    def validate_redis_connection(self):
        """验证Redis连接"""
        print("🔍 验证Redis连接...")

        try:
            r = redis.Redis(**self.redis_config)
            pong = r.ping()

            if pong:
                print("   ✅ Redis连接成功")

                # 检查内存使用
                info = r.info('memory')
                used_memory = info['used_memory_human']
                max_memory = info.get('maxmemory_human', 'unlimited')
                print(f"   📊 内存使用: {used_memory} / {max_memory}")

                # 检查连接数
                clients = r.info('clients')
                connected_clients = clients['connected_clients']
                print(f"   👥 连接客户端: {connected_clients}")

                # 检查键数量
                db_info = r.info('keyspace')
                if 'db0' in db_info:
                    keys = db_info['db0']['keys']
                    print(f"   🔑 数据库键数量: {keys}")
                else:
                    print("   🔑 数据库键数量: 0")

                r.close()
                return True
            else:
                print("   ❌ Redis ping失败")
                return False

        except Exception as e:
            print(f"   ❌ Redis连接失败: {e}")
            return False

    def validate_application_health(self):
        """验证应用健康状态"""
        print("🔍 验证应用健康状态...")

        try:
            response = requests.get('http://localhost:8000/health', timeout=10)

            if response.status_code == 200:
                health_data = response.json()
                print("   ✅ 应用健康检查通过")

                # 检查各个组件状态
                components = health_data.get('components', {})
                for component, status in components.items():
                    if status.get('healthy', False):
                        print(f"   ✅ {component}: 健康")
                    else:
                        print(f"   ⚠️  {component}: {status.get('error', '未知错误')}")

                # 检查响应时间
                response_time = health_data.get('response_time', 0)
                print(".2f" return True
            else:
                print(f"   ❌ 应用健康检查失败: HTTP {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"   ❌ 应用连接失败: {e}")
            return False

    def validate_monitoring_stack(self):
        """验证监控栈状态"""
        print("🔍 验证监控栈状态...")

        services=[
            ('Prometheus', 'http://localhost:9090/-/healthy'),
            ('Grafana', 'http://localhost:3000/api/health'),
            ('Elasticsearch', 'http://localhost:9200/_cluster/health'),
            ('Kibana', 'http://localhost:5601/api/status')
        ]

        healthy_count=0

        for service_name, url in services:
            try:
                response=requests.get(url, timeout=10)

                if response.status_code == 200:
                    print(f"   ✅ {service_name}: 正常")
                    healthy_count += 1
                else:
                    print(f"   ⚠️  {service_name}: HTTP {response.status_code}")

            except requests.exceptions.RequestException as e:
                print(f"   ❌ {service_name}: 连接失败 - {e}")

        success_rate=healthy_count / len(services) * 100
        print(".1f"
        return success_rate >= 75  # 至少75%的服务正常

    def validate_data_consistency(self):
        """验证数据一致性"""
        print("🔍 验证数据一致性...")

        try:
            # 连接数据库
            conn=psycopg2.connect(**self.pg_config)
            cursor=conn.cursor()

            # 检查健康检查数据一致性
            cursor.execute("""
                SELECT
                    service_name,
                    COUNT(*) as total_checks,
                    COUNT(CASE WHEN status = 'healthy' THEN 1 END) as healthy_checks,
                    ROUND(
                        COUNT(CASE WHEN status = 'healthy' THEN 1 END)::numeric /
                        COUNT(*)::numeric * 100, 2
                    ) as health_rate
                FROM health_checks
                WHERE checked_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
                GROUP BY service_name
                ORDER BY service_name;
            """)

            results=cursor.fetchall()

            if results:
                print("   📊 健康检查数据一致性:")
                for row in results:
                    service, total, healthy, rate=row
                    print(".1f" else:
                print("   ℹ️  最近1小时无健康检查数据")

            # 检查配置数据
            cursor.execute("SELECT COUNT(*) FROM configurations;")
            config_count=cursor.fetchone()[0]
            print(f"   📋 配置项数量: {config_count}")

            # 检查告警数据
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM alerts
                GROUP BY status
                ORDER BY status;
            """)
            alert_stats=cursor.fetchall()

            if alert_stats:
                print("   🚨 告警统计:")
                for status, count in alert_stats:
                    print(f"     {status}: {count}")

            cursor.close()
            conn.close()

            print("   ✅ 数据一致性验证完成")
            return True

        except Exception as e:
            print(f"   ❌ 数据一致性验证失败: {e}")
            return False

    def generate_validation_report(self):
        """生成验证报告"""
        print("\n" + "="*60)
        print("🎯 RQA2025 数据迁移验证报告")
        print("="*60)
        print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        validations=[
            ("PostgreSQL连接", self.validate_postgres_connection()),
            ("InfluxDB连接", self.validate_influxdb_connection()),
            ("Redis连接", self.validate_redis_connection()),
            ("应用健康状态", self.validate_application_health()),
            ("监控栈状态", self.validate_monitoring_stack()),
            ("数据一致性", self.validate_data_consistency())
        ]

        passed=0
        total=len(validations)

        print("📋 验证结果:")
        for name, result in validations:
            status="✅ 通过" if result else "❌ 失败"
            print(f"  {status} {name}")
            if result:
                passed += 1

        print()
        print("📊 总体结果:")
        success_rate=passed / total * 100
        print(".1f"
        if success_rate >= 90:
            print("🎉 数据迁移验证完全通过！")
            return True
        elif success_rate >= 75:
            print("✅ 数据迁移验证基本通过")
            return True
        else:
            print("❌ 数据迁移验证失败，需要修复")
            return False

    def run_full_validation(self):
        """运行完整验证"""
        print("🚀 开始RQA2025预投产环境数据迁移验证")
        print("="*50)

        # 等待服务就绪
        print("⏳ 等待服务启动...")
        time.sleep(10)

        # 执行验证
        success=self.generate_validation_report()

        print("\n" + "="*50)
        if success:
            print("🎯 数据迁移验证成功！预投产环境准备就绪。")
            print("\n📋 后续步骤:")
            print("1. 执行业务验收测试")
            print("2. 进行性能基准测试")
            print("3. 执行安全评估")
            print("4. 准备灰度发布计划")
        else:
            print("❌ 数据迁移验证失败！请检查服务状态和配置。")
            sys.exit(1)


def main():
    """主函数"""
    validator=DataMigrationValidator()
    validator.run_full_validation()


if __name__ == "__main__":
    main()
