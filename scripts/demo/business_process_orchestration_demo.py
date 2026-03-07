#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
业务流程编排演示脚本
展示完整的业务流程驱动架构集成
"""

from src.core.business_process_demo import BusinessProcessDemo, DemoConfig
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('business_process_demo.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ 加载配置文件失败: {e}")
        return {}


def create_demo_config() -> DemoConfig:
    """创建演示配置"""
    return DemoConfig(
        symbols=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
        strategy_name="demo_strategy",
        max_processes=5,
        process_timeout=300,
        enable_monitoring=True
    )


def run_basic_demo():
    """运行基础演示"""
    logger.info("🚀 启动业务流程编排基础演示...")

    # 创建演示配置
    config = create_demo_config()

    # 创建演示实例
    demo = BusinessProcessDemo(config)

    try:
        # 初始化演示环境
        logger.info("📋 初始化演示环境...")
        if not demo.initialize():
            logger.error("❌ 演示环境初始化失败")
            return False

        # 启动演示
        logger.info("▶️ 启动演示...")
        if not demo.start_demo():
            logger.error("❌ 演示启动失败")
            return False

        # 监控演示状态
        logger.info("📊 开始监控演示状态...")
        monitor_duration = 60  # 监控60秒
        start_time = time.time()

        while demo.is_running and (time.time() - start_time) < monitor_duration:
            try:
                status = demo.get_demo_status()
                metrics = demo.get_demo_metrics()

                logger.info(f"📊 演示状态: "
                            f"活跃进程 {status['active_processes']}, "
                            f"完成 {metrics['completed_processes']}, "
                            f"失败 {metrics['failed_processes']}")

                # 检查是否所有进程都已完成
                if status['active_processes'] == 0:
                    logger.info("✅ 所有进程已完成")
                    break

                time.sleep(10)  # 每10秒检查一次状态

            except KeyboardInterrupt:
                logger.info("⏹️ 收到中断信号，正在停止演示...")
                break
            except Exception as e:
                logger.error(f"❌ 监控过程中发生错误: {e}")
                break

        # 停止演示
        logger.info("⏹️ 停止演示...")
        demo.stop_demo()

        # 输出最终结果
        final_metrics = demo.get_demo_metrics()
        logger.info("🎉 演示完成！")
        logger.info(f"📈 总进程数: {final_metrics['total_processes']}")
        logger.info(f"✅ 成功完成: {final_metrics['completed_processes']}")
        logger.info(f"❌ 失败数量: {final_metrics['failed_processes']}")
        if 'success_rate' in final_metrics:
            logger.info(f"📊 成功率: {final_metrics['success_rate']:.1f}%")

        return True

    except Exception as e:
        logger.error(f"❌ 演示运行失败: {e}")
        return False


def run_interactive_demo():
    """运行交互式演示"""
    logger.info("🎮 启动业务流程编排交互式演示...")

    # 创建演示配置
    config = create_demo_config()

    # 创建演示实例
    demo = BusinessProcessDemo(config)

    try:
        # 初始化演示环境
        logger.info("📋 初始化演示环境...")
        if not demo.initialize():
            logger.error("❌ 演示环境初始化失败")
            return False

        # 启动演示
        logger.info("▶️ 启动演示...")
        if not demo.start_demo():
            logger.error("❌ 演示启动失败")
            return False

        logger.info("🎮 交互式演示已启动！")
        logger.info("可用命令:")
        logger.info("  status - 查看演示状态")
        logger.info("  metrics - 查看演示指标")
        logger.info("  stop - 停止演示")
        logger.info("  quit - 退出演示")

        while demo.is_running:
            try:
                command = input("\n请输入命令: ").strip().lower()

                if command == "status":
                    status = demo.get_demo_status()
                    logger.info(f"📊 演示状态: {json.dumps(status, indent=2, ensure_ascii=False)}")

                elif command == "metrics":
                    metrics = demo.get_demo_metrics()
                    logger.info(f"📈 演示指标: {json.dumps(metrics, indent=2, ensure_ascii=False)}")

                elif command == "stop":
                    logger.info("⏹️ 停止演示...")
                    demo.stop_demo()
                    break

                elif command == "quit":
                    logger.info("👋 退出演示...")
                    demo.stop_demo()
                    break

                elif command == "help":
                    logger.info("可用命令:")
                    logger.info("  status - 查看演示状态")
                    logger.info("  metrics - 查看演示指标")
                    logger.info("  stop - 停止演示")
                    logger.info("  quit - 退出演示")
                    logger.info("  help - 显示帮助")

                else:
                    logger.warning(f"❓ 未知命令: {command}")
                    logger.info("输入 'help' 查看可用命令")

            except KeyboardInterrupt:
                logger.info("\n⏹️ 收到中断信号，正在停止演示...")
                demo.stop_demo()
                break
            except Exception as e:
                logger.error(f"❌ 命令执行失败: {e}")

        return True

    except Exception as e:
        logger.error(f"❌ 交互式演示运行失败: {e}")
        return False


def run_stress_test():
    """运行压力测试"""
    logger.info("🔥 启动业务流程编排压力测试...")

    # 创建压力测试配置
    config = DemoConfig(
        symbols=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX", "AMD", "INTC"],
        strategy_name="stress_test_strategy",
        max_processes=10,
        process_timeout=180,
        enable_monitoring=True
    )

    # 创建演示实例
    demo = BusinessProcessDemo(config)

    try:
        # 初始化演示环境
        logger.info("📋 初始化演示环境...")
        if not demo.initialize():
            logger.error("❌ 演示环境初始化失败")
            return False

        # 启动演示
        logger.info("▶️ 启动压力测试...")
        if not demo.start_demo():
            logger.error("❌ 压力测试启动失败")
            return False

        # 监控压力测试状态
        logger.info("📊 开始监控压力测试状态...")
        test_duration = 120  # 测试120秒
        start_time = time.time()

        while demo.is_running and (time.time() - start_time) < test_duration:
            try:
                status = demo.get_demo_status()
                metrics = demo.get_demo_metrics()

                logger.info(f"🔥 压力测试状态: "
                            f"活跃进程 {status['active_processes']}, "
                            f"完成 {metrics['completed_processes']}, "
                            f"失败 {metrics['failed_processes']}")

                # 检查是否所有进程都已完成
                if status['active_processes'] == 0:
                    logger.info("✅ 所有进程已完成")
                    break

                time.sleep(15)  # 每15秒检查一次状态

            except KeyboardInterrupt:
                logger.info("⏹️ 收到中断信号，正在停止压力测试...")
                break
            except Exception as e:
                logger.error(f"❌ 压力测试监控过程中发生错误: {e}")
                break

        # 停止演示
        logger.info("⏹️ 停止压力测试...")
        demo.stop_demo()

        # 输出最终结果
        final_metrics = demo.get_demo_metrics()
        logger.info("🎉 压力测试完成！")
        logger.info(f"📈 总进程数: {final_metrics['total_processes']}")
        logger.info(f"✅ 成功完成: {final_metrics['completed_processes']}")
        logger.info(f"❌ 失败数量: {final_metrics['failed_processes']}")
        if 'success_rate' in final_metrics:
            logger.info(f"📊 成功率: {final_metrics['success_rate']:.1f}%")

        return True

    except Exception as e:
        logger.error(f"❌ 压力测试运行失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("🎯 业务流程编排演示系统")
    logger.info("=" * 50)

    if len(sys.argv) > 1:
        demo_type = sys.argv[1].lower()
    else:
        print("\n请选择演示类型:")
        print("1. basic - 基础演示")
        print("2. interactive - 交互式演示")
        print("3. stress - 压力测试")
        print("4. all - 运行所有演示")

        choice = input("\n请输入选择 (1-4): ").strip()

        if choice == "1":
            demo_type = "basic"
        elif choice == "2":
            demo_type = "interactive"
        elif choice == "3":
            demo_type = "stress"
        elif choice == "4":
            demo_type = "all"
        else:
            logger.error("❌ 无效选择")
            return

    success_count = 0
    total_count = 0

    try:
        if demo_type == "basic" or demo_type == "all":
            total_count += 1
            logger.info("\n" + "="*50)
            if run_basic_demo():
                success_count += 1
                logger.info("✅ 基础演示成功")
            else:
                logger.error("❌ 基础演示失败")

        if demo_type == "interactive" or demo_type == "all":
            total_count += 1
            logger.info("\n" + "="*50)
            if run_interactive_demo():
                success_count += 1
                logger.info("✅ 交互式演示成功")
            else:
                logger.error("❌ 交互式演示失败")

        if demo_type == "stress" or demo_type == "all":
            total_count += 1
            logger.info("\n" + "="*50)
            if run_stress_test():
                success_count += 1
                logger.info("✅ 压力测试成功")
            else:
                logger.error("❌ 压力测试失败")

        # 输出总结
        logger.info("\n" + "="*50)
        logger.info("🎯 演示总结")
        logger.info(f"📊 总演示数: {total_count}")
        logger.info(f"✅ 成功数: {success_count}")
        logger.info(f"❌ 失败数: {total_count - success_count}")
        logger.info(f"📈 成功率: {(success_count/total_count)*100:.1f}%")

        if success_count == total_count:
            logger.info("🎉 所有演示都成功完成！")
        else:
            logger.warning("⚠️ 部分演示失败，请检查日志")

    except KeyboardInterrupt:
        logger.info("\n⏹️ 演示被用户中断")
    except Exception as e:
        logger.error(f"❌ 演示系统发生错误: {e}")


if __name__ == "__main__":
    main()
