"""
RQA2025 基础设施层 - 日志分析平台标准格式示例

演示如何使用各种日志分析平台的标准格式输出日志。
"""

from infrastructure.logging.standards.base_standard import StandardFormatType
from infrastructure.logging.standards.standard_manager import StandardFormatManager, StandardOutputConfig
from infrastructure.logging.standards.standard_formatter import StandardFormatter
from infrastructure.logging.core.interfaces import LogLevel, LogCategory
import sys
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, 'src')


def create_sample_entries():
    """创建示例日志条目"""
    base_time = datetime.now()

    entries = [
        StandardFormatter.create_standard_entry(
            timestamp=base_time,
            level=LogLevel.INFO,
            message="用户登录成功",
            category=LogCategory.SECURITY,
            service="auth-service",
            host="web-01",
            environment="production",
            user_id="user_123",
            session_id="session_abc",
            request_id="req_001",
            trace_id="trace_123456",
            correlation_id="corr_789",
            metadata={"login_method": "oauth", "ip": "192.168.1.100"},
            tags=["auth", "login", "success"]
        ),
        StandardFormatter.create_standard_entry(
            timestamp=base_time,
            level=LogLevel.ERROR,
            message="数据库连接失败",
            category=LogCategory.DATABASE,
            service="order-service",
            host="app-02",
            environment="production",
            request_id="req_002",
            trace_id="trace_123457",
            correlation_id="corr_790",
            metadata={"db_host": "db-01", "error_code": "ECONNREFUSED"},
            tags=["database", "connection", "error"]
        ),
        StandardFormatter.create_standard_entry(
            timestamp=base_time,
            level=LogLevel.WARNING,
            message="API响应时间过长",
            category=LogCategory.PERFORMANCE,
            service="api-gateway",
            host="gw-01",
            environment="staging",
            request_id="req_003",
            trace_id="trace_123458",
            correlation_id="corr_791",
            metadata={"response_time": 2500, "endpoint": "/api/orders"},
            tags=["performance", "latency", "warning"]
        ),
        StandardFormatter.create_standard_entry(
            timestamp=base_time,
            level=LogLevel.DEBUG,
            message="交易处理完成",
            category=LogCategory.TRADING,
            service="trading-engine",
            host="trade-01",
            environment="production",
            user_id="trader_456",
            request_id="req_004",
            trace_id="trace_123459",
            correlation_id="corr_792",
            metadata={"symbol": "AAPL", "quantity": 100, "price": 150.25},
            tags=["trading", "transaction", "completed"]
        )
    ]

    return entries


def demonstrate_individual_formats():
    """演示单个格式转换"""
    print("🔄 单个格式转换演示")
    print("=" * 50)

    formatter = StandardFormatter()
    entries = create_sample_entries()

    formats_to_demo = [
        (StandardFormatType.ELK, "ELK Stack (Elasticsearch)"),
        (StandardFormatType.SPLUNK, "Splunk"),
        (StandardFormatType.DATADOG, "Datadog"),
        (StandardFormatType.LOKI, "Loki (Prometheus)"),
        (StandardFormatType.GRAYLOG, "Graylog GELF"),
    ]

    for entry in entries[:2]:  # 只演示前两个条目
        print(f"\n📝 原始日志: {entry.message}")
        print(f"🔖 级别: {entry.level.value}, 类别: {entry.category.value}")

        for format_type, format_name in formats_to_demo:
            try:
                formatted = formatter.format_log_entry(entry, format_type)
                content_type = formatter.get_content_type(format_type)

                print(f"\n  🎯 {format_name} 格式:")
                print(f"    Content-Type: {content_type}")

                if isinstance(formatted, str):
                    # 对于字符串格式，显示前200个字符
                    preview = formatted[:200] + "..." if len(formatted) > 200 else formatted
                    print(f"    输出: {preview}")
                else:
                    # 对于字典格式，显示键
                    print(f"    字段: {list(formatted.keys())}")

            except Exception as e:
                print(f"    ❌ 错误: {e}")


def demonstrate_batch_formats():
    """演示批量格式转换"""
    print("\n\n📦 批量格式转换演示")
    print("=" * 50)

    formatter = StandardFormatter()
    entries = create_sample_entries()

    batch_formats = [
        (StandardFormatType.ELK, "ELK Stack"),
        (StandardFormatType.SPLUNK, "Splunk"),
        (StandardFormatType.DATADOG, "Datadog"),
    ]

    for format_type, format_name in batch_formats:
        try:
            batch_result = formatter.format_batch(entries, format_type)
            supports_batch = formatter.supports_batch(format_type)

            print(f"\n🎯 {format_name} 批量格式:")
            print(f"   支持批量: {supports_batch}")

            if isinstance(batch_result, str):
                size_kb = len(batch_result) / 1024
                print(f"   数据大小: {size_kb:.2f} KB")
            elif isinstance(batch_result, list):
                print(f"   条目数量: {len(batch_result)}")
                if batch_result:
                    print(
                        f"   首条字段: {list(batch_result[0].keys()) if isinstance(batch_result[0], dict) else 'N/A'}")

        except Exception as e:
            print(f"   ❌ 错误: {e}")


def demonstrate_manager_usage():
    """演示标准格式管理器使用"""
    print("\n\n⚙️ 标准格式管理器演示")
    print("=" * 50)

    manager = StandardFormatManager()

    # 注册示例配置
    sample_configs = manager.create_sample_configs()
    for name, config in sample_configs.items():
        manager.register_config(name, config)
        print(f"✅ 注册配置: {name} -> {config.format_type.value}")

    print(f"\n📋 支持的目标: {manager.get_supported_targets()}")

    # 演示目标信息
    for target in ["elk-dev", "datadog-staging"]:
        try:
            info = manager.get_target_info(target)
            print(f"\n🎯 目标 {target} 信息:")
            for key, value in info.items():
                print(f"   {key}: {value}")
        except Exception as e:
            print(f"   ❌ 获取 {target} 信息失败: {e}")

    # 演示格式化
    entries = create_sample_entries()[:1]  # 只用一个条目
    for target in ["elk-dev", "datadog-staging"]:
        try:
            formatted = manager.format_for_target(entries[0], target)
            print(f"\n📤 {target} 格式化结果类型: {type(formatted).__name__}")
        except Exception as e:
            print(f"   ❌ {target} 格式化失败: {e}")


def demonstrate_fluentd_format():
    """演示Fluentd特殊格式"""
    print("\n\n🔥 Fluentd 特殊格式演示")
    print("=" * 50)

    from infrastructure.logging.standards.fluentd_standard import FluentdStandardFormat

    fluentd_formatter = FluentdStandardFormat()
    entries = create_sample_entries()[:2]

    print("JSON格式 (用于HTTP输入):")
    json_payload = fluentd_formatter.create_json_payload(entries)
    print(f"负载大小: {len(json_payload)} 字符")
    print(f"预览: {json_payload[:300]}...")

    print("\nMessagePack格式 (用于Forward协议):")
    try:
        msgpack_payload = fluentd_formatter.create_forward_payload(entries)
        print(f"负载大小: {len(msgpack_payload)} 字节")
        print("MessagePack编码成功")
    except ImportError:
        print("MessagePack未安装，跳过二进制格式演示")


async def demonstrate_async_sending():
    """演示异步发送"""
    print("\n\n🚀 异步发送演示")
    print("=" * 50)

    manager = StandardFormatManager()

    # 注册一个模拟配置
    config = StandardOutputConfig(
        format_type=StandardFormatType.ELK,
        endpoint="http://mock-endpoint:9200/_bulk",
        batch_size=10,
        async_mode=True
    )
    manager.register_config("mock-elk", config)

    entries = create_sample_entries()

    try:
        result = await manager.send_to_target(entries, "mock-elk")
        print("异步发送结果:")
        for key, value in result.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ 异步发送失败: {e}")


def main():
    """主函数"""
    print("🌟 RQA2025 日志分析平台标准格式演示")
    print("=" * 60)

    try:
        # 演示单个格式转换
        demonstrate_individual_formats()

        # 演示批量格式转换
        demonstrate_batch_formats()

        # 演示管理器使用
        demonstrate_manager_usage()

        # 演示Fluentd特殊格式
        demonstrate_fluentd_format()

        # 演示异步发送
        import asyncio
        asyncio.run(demonstrate_async_sending())

        print("\n✅ 所有演示完成！")

    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
