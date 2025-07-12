"""压力测试主执行脚本"""
import logging
from stress_test_executor import StressTestExecutor
from stress_test_monitor import StressTestMonitor, StressTestAnalyzer

def configure_logging():
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('stress_test.log'),
            logging.StreamHandler()
        ]
    )

def main():
    # 1. 初始化
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("=== 开始执行压力测试 ===")

    # 2. 初始化监控系统
    monitor = StressTestMonitor()
    monitor.start_monitoring()

    # 3. 初始化执行器
    executor = StressTestExecutor(max_workers=5)

    try:
        # 4. 执行测试场景
        logger.info("执行基础场景测试...")
        base_results = executor._execute_scenarios([
            "2015股灾重现",
            "千股跌停",
            "Level2数据风暴"
        ])

        logger.info("执行极端场景测试...")
        extreme_results = executor._execute_scenarios([
            "流动性危机",
            "政策突变",
            "熔断压力测试"
        ])

        # 5. 生成最终报告
        all_results = base_results + extreme_results
        report = monitor.generate_final_report()
        analyzer = StressTestAnalyzer()

        # 6. 分析结果
        throughput_stats = analyzer.analyze_throughput(report)
        latency_stats = analyzer.analyze_latency(report)
        resource_usage = analyzer.analyze_resource_usage(report)

        # 7. 输出报告
        logger.info("\n=== 压力测试报告 ===")
        logger.info(f"总耗时: {report['summary']['total_duration']}")
        logger.info(f"场景通过率: {report['summary']['success_rate']}")

        logger.info("\n性能指标:")
        logger.info(f"最大吞吐量: {throughput_stats['max']:.2f}/s")
        logger.info(f"平均延迟: {latency_stats['mean']:.2f}ms")
        logger.info(f"峰值CPU使用率: {resource_usage['max_cpu']:.1f}%")
        logger.info(f"峰值内存使用率: {resource_usage['max_memory']:.1f}%")

        # 8. 生成可视化图表
        monitor.visualize_metrics()
        logger.info("指标图表已生成: stress_test_metrics.png")

    except Exception as e:
        logger.error(f"压力测试执行失败: {str(e)}", exc_info=True)
        raise

    finally:
        # 9. 清理资源
        executor._cleanup()
        logger.info("=== 压力测试执行完成 ===")

if __name__ == "__main__":
    main()
