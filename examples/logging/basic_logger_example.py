#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 基础设施层日志系统 - 基础Logger使用示例

演示BaseLogger的基本使用方法，包括单例模式、基本日志记录等。
"""

from infrastructure.logging import BaseLogger, LogLevel, LogFormat, LogCategory
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def basic_logger_example():
    """基础Logger使用示例"""
    print("=== 基础Logger使用示例 ===\n")

    # 1. 创建基础Logger实例
    print("1. 创建基础Logger实例:")
    logger = BaseLogger(
        name="example.basic",
        level=LogLevel.INFO,
        category=LogCategory.GENERAL,
        format_type=LogFormat.STRUCTURED,
        log_dir="logs/examples"
    )
    logger.info("基础Logger创建成功")
    print()

    # 2. 基本日志记录
    print("2. 基本日志记录:")
    logger.debug("调试信息 - 通常在生产环境关闭")
    logger.info("普通信息 - 重要业务事件")
    logger.warning("警告信息 - 潜在问题")
    logger.error("错误信息 - 可恢复错误")
    logger.critical("严重错误 - 系统级问题")
    print()

    # 3. 结构化日志记录
    print("3. 结构化日志记录:")
    logger.info("用户登录",
                user_id="user123",
                login_method="password",
                ip_address="192.168.1.100",
                user_agent="Chrome/91.0")

    logger.warning("缓存命中率低",
                   hit_rate=0.65,
                   threshold=0.8,
                   cache_size=1000,
                   eviction_count=50)
    print()


def singleton_logger_example():
    """单例模式Logger使用示例"""
    print("=== 单例模式Logger使用示例 ===\n")

    # 1. 使用单例模式创建Logger
    print("1. 使用单例模式创建Logger:")
    logger1 = BaseLogger.get_instance("example.singleton", level=LogLevel.INFO)
    logger2 = BaseLogger.get_instance("example.singleton", level=LogLevel.INFO)

    print(f"两个Logger实例是否相同: {logger1 is logger2}")
    logger1.info("通过logger1记录")
    logger2.info("通过logger2记录")
    print()

    # 2. 单例模式的好处
    print("2. 单例模式的好处:")
    import time

    start_time = time.time()
    loggers = []
    for i in range(100):
        logger = BaseLogger.get_instance(f"example.perf.{i}", level=LogLevel.DEBUG)
        loggers.append(logger)
    end_time = time.time()

    print(f"{end_time - start_time:.4f}")
    print(f"创建的Logger实例数量: {len(set(loggers))} (复用实例)")
    print()


def configuration_example():
    """Logger配置示例"""
    print("=== Logger配置示例 ===\n")

    # 1. 不同日志级别
    print("1. 不同日志级别配置:")
    debug_logger = BaseLogger.get_instance("example.debug",
                                           level=LogLevel.DEBUG,
                                           category=LogCategory.SYSTEM)
    info_logger = BaseLogger.get_instance("example.info",
                                          level=LogLevel.INFO,
                                          category=LogCategory.BUSINESS)

    debug_logger.debug("调试级别日志")
    debug_logger.info("信息级别日志")
    info_logger.debug("这个不会显示 - 级别不够")
    info_logger.info("这个会显示")
    print()

    # 2. 不同日志格式
    print("2. 不同日志格式:")
    text_logger = BaseLogger.get_instance("example.text",
                                          format_type=LogFormat.TEXT)
    json_logger = BaseLogger.get_instance("example.json",
                                          format_type=LogFormat.JSON)
    structured_logger = BaseLogger.get_instance("example.structured",
                                                format_type=LogFormat.STRUCTURED)

    text_logger.info("纯文本格式日志")
    json_logger.info("JSON格式日志", key="value")
    structured_logger.info("结构化格式日志", component="Example", operation="demo")
    print()


def performance_example():
    """性能优化示例"""
    print("=== 性能优化示例 ===\n")

    import time

    # 1. 普通创建 vs 单例模式性能对比
    print("1. 性能对比测试:")

    # 普通创建性能
    start_time = time.time()
    normal_loggers = []
    for i in range(100):
        logger = BaseLogger(f"perf.normal.{i}", level=LogLevel.INFO)
        normal_loggers.append(logger)
    normal_time = time.time() - start_time

    # 单例模式性能
    start_time = time.time()
    singleton_loggers = []
    for i in range(100):
        logger = BaseLogger.get_instance(f"perf.singleton.{i}", level=LogLevel.INFO)
        singleton_loggers.append(logger)
    singleton_time = time.time() - start_time

    print(f"普通创建时间: {normal_time:.4f}")
    print(f"单例创建时间: {singleton_time:.4f}")
    print(f"性能提升: {(normal_time - singleton_time) / normal_time * 100:.1f}%")
    print()


def cleanup_example():
    """清理示例"""
    print("=== 清理示例 ===\n")

    # 清理单例实例 (主要用于测试)
    print("清理Logger单例实例...")
    BaseLogger.clear_instances()
    print("清理完成")
    print()


def main():
    """主函数"""
    print("RQA2025 基础设施层日志系统 - 基础Logger使用示例")
    print("=" * 60)
    print()

    try:
        basic_logger_example()
        singleton_logger_example()
        configuration_example()
        performance_example()
        cleanup_example()

        print("🎉 所有示例执行完成！")
        print("\n提示:")
        print("- 查看 logs/examples/ 目录中的日志文件")
        print("- 调整LogLevel查看不同详细程度的日志")
        print("- 使用单例模式提升性能")

    except Exception as e:
        print(f"❌ 示例执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
