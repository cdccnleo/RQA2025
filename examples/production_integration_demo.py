"""
生产环境集成演示脚本

展示如何在实际生产环境中使用企业级数据治理和多市场同步功能
"""
from src.data import (
    initialize_data_layer,
    get_governance_manager,
    get_sync_manager
)
import sys
import os
import time
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demo_enterprise_governance():
    """演示企业级数据治理功能"""
    print("\n" + "="*60)
    print("企业级数据治理功能演示")
    print("="*60)

    try:
        # 获取治理管理器
        governance_manager = get_governance_manager()

        # 初始化治理框架
        print("1. 初始化治理框架...")
        init_result = governance_manager.initialize_governance_framework()
        print(f"   初始化结果: {init_result}")

        # 创建数据策略
        print("\n2. 创建数据策略...")
        from src.data.governance.enterprise_governance import PolicyType, EnforcementLevel

        privacy_policy = governance_manager.policy_manager.create_policy(
            policy_name="数据隐私保护策略",
            policy_type=PolicyType.PRIVACY,
            description="保护用户隐私数据，符合GDPR要求",
            rules=[
                {"rule_type": "data_encryption", "method": "AES-256"},
                {"rule_type": "access_control", "method": "role_based"},
                {"rule_type": "data_retention", "method": "time_limited"}
            ],
            enforcement_level=EnforcementLevel.CRITICAL
        )
        print(f"   创建隐私策略: {privacy_policy.policy_name}")

        # 添加合规要求
        print("\n3. 添加合规要求...")
        from src.data.governance.enterprise_governance import RegulationType

        ccpa_requirement = governance_manager.compliance_manager.add_requirement(
            regulation_name=RegulationType.CCPA,
            requirement_type="consumer_privacy",
            description="加州消费者隐私法案合规要求",
            mandatory=True
        )
        print(f"   添加CCPA合规要求: {ccpa_requirement.requirement_id}")

        # 实施合规要求
        print("\n4. 实施合规要求...")
        governance_manager.compliance_manager.implement_requirement(
            ccpa_requirement.requirement_id,
            {
                "implementation_method": "automated_compliance_check",
                "data_processing_consent": True,
                "consumer_rights_management": True
            }
        )
        print("   CCPA合规要求已实施")

        # 生成治理报告
        print("\n5. 生成治理报告...")
        governance_report = governance_manager.generate_governance_report()

        print(f"   活跃策略数量: {governance_report['active_policies_count']}")
        print(f"   治理评分: {governance_report['overall_governance_score']:.2f}")
        print(f"   高风险发现: {governance_report['high_risk_findings_count']}")
        print(f"   建议: {governance_report['recommendations']}")

        return governance_report

    except Exception as e:
        print(f"   错误: {e}")
        return None


def demo_multi_market_sync():
    """演示多市场数据同步功能"""
    print("\n" + "="*60)
    print("多市场数据同步功能演示")
    print("="*60)

    try:
        # 获取同步管理器
        sync_manager = get_sync_manager()

        # 初始化市场
        print("1. 初始化市场...")
        init_result = sync_manager.initialize_markets()
        print(f"   注册市场数量: {init_result['markets_registered']}")
        print(f"   时区映射数量: {init_result['timezone_mappings']}")
        print(f"   汇率设置数量: {init_result['exchange_rates_set']}")

        # 添加模拟市场数据
        print("\n2. 添加模拟市场数据...")
        from src.data.sync.multi_market_sync import MarketData, DataType

        # 添加上海市场数据
        shanghai_data = MarketData(
            market_id="SHANGHAI",
            symbol="000001.SZ",
            price=15.80,
            volume=1000000,
            timestamp=datetime.now(),
            timezone="Asia/Shanghai",
            currency="CNY",
            data_type=DataType.TICK,
            source="wind"
        )
        sync_manager.global_manager.add_market_data("SHANGHAI", shanghai_data)
        print("   添加上海市场数据")

        # 添加纽约市场数据
        nyse_data = MarketData(
            market_id="NYSE",
            symbol="AAPL",
            price=150.25,
            volume=500000,
            timestamp=datetime.now(),
            timezone="America/New_York",
            currency="USD",
            data_type=DataType.TICK,
            source="yahoo"
        )
        sync_manager.global_manager.add_market_data("NYSE", nyse_data)
        print("   添加纽约市场数据")

        # 启动同步任务
        print("\n3. 启动同步任务...")
        from src.data.sync.multi_market_sync import SyncType

        shanghai_task = sync_manager.start_sync_task("SHANGHAI", SyncType.REAL_TIME)
        nyse_task = sync_manager.start_sync_task("NYSE", SyncType.BATCH)
        print(f"   上海市场同步任务: {shanghai_task}")
        print(f"   纽约市场同步任务: {nyse_task}")

        # 模拟任务完成
        time.sleep(1)
        sync_manager.complete_sync_task(shanghai_task, records_synced=1000, error_count=0)
        sync_manager.complete_sync_task(nyse_task, records_synced=500, error_count=5)
        print("   同步任务已完成")

        # 时区转换演示
        print("\n4. 时区转换演示...")
        shanghai_time = datetime.now()
        ny_time = sync_manager.timezone_synchronizer.convert_timezone(
            shanghai_time, "Asia/Shanghai", "America/New_York"
        )
        print(f"   上海时间: {shanghai_time}")
        print(f"   纽约时间: {ny_time}")

        # 货币转换演示
        print("\n5. 货币转换演示...")
        usd_amount = sync_manager.currency_processor.convert_currency(
            1000, "CNY", "USD"
        )
        print(f"   1000 CNY = {usd_amount:.2f} USD")

        # 获取同步报告
        print("\n6. 获取同步报告...")
        sync_report = sync_manager.get_sync_report()

        print(f"   活跃任务数量: {sync_report['active_tasks_count']}")
        print(f"   完成任务数量: {sync_report['completed_tasks_count']}")
        print(f"   失败任务数量: {sync_report['failed_tasks_count']}")
        print(f"   总同步记录: {sync_report['total_records_synced']}")
        print(f"   总体成功率: {sync_report['overall_success_rate']:.2%}")

        # 市场统计信息
        print("\n7. 市场统计信息...")
        for market_id, stats in sync_report['market_statistics'].items():
            print(f"   {market_id}: {stats['data_count']} 条数据")
            if 'price_stats' in stats:
                print(
                    f"     价格范围: {stats['price_stats']['min']:.2f} - {stats['price_stats']['max']:.2f}")

        return sync_report

    except Exception as e:
        print(f"   错误: {e}")
        return None


def demo_integrated_usage():
    """演示集成使用场景"""
    print("\n" + "="*60)
    print("集成使用场景演示")
    print("="*60)

    try:
        # 初始化数据层
        print("1. 初始化数据层...")
        init_result = initialize_data_layer()
        print(f"   初始化状态: {init_result['status']}")

        if init_result['status'] == 'initialized':
            # 获取增强的数据管理器
            data_manager = init_result['data_manager']
            governance_manager = init_result['governance_manager']
            sync_manager = init_result['sync_manager']

            # 演示带治理检查的数据加载
            print("\n2. 带治理检查的数据加载...")
            try:
                # 检查治理状态
                governance_report = governance_manager.generate_governance_report()
                governance_score = governance_report['overall_governance_score']

                if governance_score >= 80:
                    print(f"   治理评分: {governance_score:.2f} (通过)")
                    print("   允许数据加载操作")

                    # 模拟数据加载
                    print("   执行数据加载...")
                    time.sleep(0.5)
                    print("   数据加载完成")

                else:
                    print(f"   治理评分: {governance_score:.2f} (不通过)")
                    print("   拒绝数据加载操作")

            except Exception as e:
                print(f"   数据加载错误: {e}")

            # 演示市场数据同步
            print("\n3. 市场数据同步...")
            try:
                # 启动同步任务
                from src.data.sync.multi_market_sync import SyncType
                task_id = sync_manager.start_sync_task("SHANGHAI", SyncType.REAL_TIME)
                print(f"   启动同步任务: {task_id}")

                # 模拟同步过程
                time.sleep(1)
                sync_manager.complete_sync_task(task_id, records_synced=2000, error_count=10)

                # 获取同步结果
                sync_report = sync_manager.get_sync_report()
                success_rate = sync_report['overall_success_rate']
                print(f"   同步成功率: {success_rate:.2%}")

            except Exception as e:
                print(f"   同步错误: {e}")

            # 演示性能监控
            print("\n4. 性能监控...")
            try:
                # 数据管理器性能
                cache_stats = data_manager.get_cache_stats()
                print(f"   缓存统计: {cache_stats}")

                # 治理性能
                governance_report = governance_manager.generate_governance_report()
                print(f"   治理评分: {governance_report['overall_governance_score']:.2f}")

                # 同步性能
                sync_report = sync_manager.get_sync_report()
                print(f"   同步成功率: {sync_report['overall_success_rate']:.2%}")

            except Exception as e:
                print(f"   监控错误: {e}")

        return init_result

    except Exception as e:
        print(f"   错误: {e}")
        return None


def main():
    """主函数"""
    print("RQA2025 生产环境集成演示")
    print("="*60)
    print("本演示展示企业级数据治理和多市场数据同步功能")
    print("在生产环境中的实际应用")

    # 演示企业级数据治理
    governance_result = demo_enterprise_governance()

    # 演示多市场数据同步
    sync_result = demo_multi_market_sync()

    # 演示集成使用
    integration_result = demo_integrated_usage()

    # 总结
    print("\n" + "="*60)
    print("演示总结")
    print("="*60)

    if governance_result:
        print("✅ 企业级数据治理功能正常")
    else:
        print("❌ 企业级数据治理功能异常")

    if sync_result:
        print("✅ 多市场数据同步功能正常")
    else:
        print("❌ 多市场数据同步功能异常")

    if integration_result and integration_result.get('status') == 'initialized':
        print("✅ 集成功能正常")
    else:
        print("❌ 集成功能异常")

    print("\n脚本与生产代码的关系:")
    print("- 脚本: 功能验证、原型实现、测试驱动")
    print("- 生产代码: 模块化设计、接口规范、性能优化、监控集成")
    print("- 演进过程: 从原型到产品的完整实现")


if __name__ == "__main__":
    main()
