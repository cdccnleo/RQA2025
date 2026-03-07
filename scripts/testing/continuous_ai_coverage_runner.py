#!/usr/bin/env python3
"""
RQA2025 持续AI覆盖率自动化运行器
定期执行AI增强的测试覆盖率提升任务
"""

from scripts.testing.ai_enhanced_coverage_automation import DeepseekAIConnector, AICoverageAutomation
import os
import sys
import json
import schedule
import argparse
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入AI自动化模块

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/continuous_ai_coverage.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ContinuousAICoverageRunner:
    """持续AI覆盖率运行器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_root = project_root
        self.ai_connector = None
        self.automation = None
        self.running = False
        self.execution_history = []

        # 创建必要的目录
        self._create_directories()

    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            'logs',
            'cache/ai_test_generation',
            'reports/testing',
            'data/coverage_history'
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    async def initialize_ai_connector(self):
        """初始化AI连接器"""
        try:
            self.ai_connector = DeepseekAIConnector(
                api_base=self.config.get('api_base', 'http://localhost:11434'),
                model=self.config.get('model', 'deepseek-coder')
            )
            await self.ai_connector.__aenter__()

            self.automation = AICoverageAutomation(self.ai_connector)

            # 更新目标覆盖率
            target_coverage = self.config.get('target_coverage', 85.0)
            for layer in self.automation.target_coverage:
                self.automation.target_coverage[layer] = target_coverage

            logger.info("✅ AI连接器初始化成功")
            return True

        except Exception as e:
            logger.error(f"❌ AI连接器初始化失败: {e}")
            return False

    async def cleanup_ai_connector(self):
        """清理AI连接器"""
        if self.ai_connector:
            await self.ai_connector.__aexit__(None, None, None)
            logger.info("🧹 AI连接器已清理")

    async def execute_coverage_automation(self) -> Dict[str, Any]:
        """执行覆盖率自动化"""
        try:
            logger.info("🚀 开始执行AI覆盖率自动化...")

            # 执行自动化流程
            results = await self.automation.execute_ai_automation()

            # 记录执行历史
            execution_record = {
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'results': {
                    'generated_files': len(results['generated_files']),
                    'test_passed': results['test_results']['passed'],
                    'test_failed': results['test_results']['failed'],
                    'current_coverage': results['current_coverage']
                }
            }

            self.execution_history.append(execution_record)

            # 保存执行历史
            self._save_execution_history()

            logger.info("✅ AI覆盖率自动化执行完成")
            return results

        except Exception as e:
            logger.error(f"❌ AI覆盖率自动化执行失败: {e}")

            # 记录失败历史
            execution_record = {
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            }

            self.execution_history.append(execution_record)
            self._save_execution_history()

            return {'error': str(e)}

    def _save_execution_history(self):
        """保存执行历史"""
        history_file = 'data/coverage_history/execution_history.json'

        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.execution_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存执行历史失败: {e}")

    def _load_execution_history(self):
        """加载执行历史"""
        history_file = 'data/coverage_history/execution_history.json'

        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.execution_history = json.load(f)
            except Exception as e:
                logger.error(f"加载执行历史失败: {e}")
                self.execution_history = []
        else:
            self.execution_history = []

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 检查AI连接
            if not self.ai_connector:
                logger.warning("AI连接器未初始化")
                return False

            # 检查项目结构
            required_dirs = ['src', 'tests', 'logs']
            for dir_name in required_dirs:
                if not os.path.exists(dir_name):
                    logger.error(f"缺少必要目录: {dir_name}")
                    return False

            # 检查conda环境
            if 'rqa' not in sys.prefix and 'rqa' not in sys.executable:
                logger.error("未在conda rqa环境中运行")
                return False

            # 检查Python依赖
            required_packages = ['aiohttp', 'pytest', 'pytest-cov', 'schedule']
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    logger.error(f"缺少Python包: {package}")
                    return False

            # 检查AI服务可用性
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.config.get('api_base', 'http://localhost:11434')}/v1/models",
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status != 200:
                            logger.error(f"AI服务响应异常: {response.status}")
                            return False
            except Exception as e:
                logger.error(f"AI服务连接失败: {e}")
                return False

            # 检查磁盘空间
            try:
                import shutil
                total, used, free = shutil.disk_usage('.')
                free_gb = free / (1024**3)
                if free_gb < 1.0:  # 少于1GB
                    logger.warning(f"磁盘空间不足: {free_gb:.2f}GB")
            except Exception as e:
                logger.warning(f"无法检查磁盘空间: {e}")

            logger.info("✅ 健康检查通过")
            return True

        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False

    async def run_scheduled_task(self):
        """运行定时任务"""
        if not self.running:
            logger.warning("运行器已停止，跳过定时任务")
            return

        logger.info("⏰ 执行定时AI覆盖率自动化任务...")

        # 健康检查
        if not await self.health_check():
            logger.error("健康检查失败，跳过本次执行")
            return

        # 执行自动化
        results = await self.execute_coverage_automation()

        if 'error' not in results:
            logger.info("✅ 定时任务执行成功")
        else:
            logger.error(f"❌ 定时任务执行失败: {results['error']}")

    async def start_scheduler(self):
        """启动调度器"""
        logger.info("🔄 启动持续AI覆盖率自动化调度器...")

        # 加载执行历史
        self._load_execution_history()

        # 设置定时任务
        schedule_time = self.config.get('schedule_time', '02:00')
        schedule.every().day.at(schedule_time).do(
            lambda: asyncio.create_task(self.run_scheduled_task())
        )

        # 设置立即执行一次（如果配置了）
        if self.config.get('run_immediately', False):
            asyncio.create_task(self.run_scheduled_task())

        logger.info(f"📅 调度器已启动，每日 {schedule_time} 执行")
        logger.info("按 Ctrl+C 停止调度器")

        self.running = True

        try:
            while self.running:
                schedule.run_pending()
                await asyncio.sleep(60)  # 每分钟检查一次

        except KeyboardInterrupt:
            logger.info("🛑 收到停止信号，正在关闭...")
            self.running = False

    async def run_once(self):
        """运行一次自动化任务"""
        logger.info("🔄 执行单次AI覆盖率自动化...")

        # 初始化AI连接器
        if not await self.initialize_ai_connector():
            logger.error("AI连接器初始化失败")
            return

        try:
            # 执行自动化
            results = await self.execute_coverage_automation()

            if 'error' not in results:
                logger.info("✅ 单次执行成功")
                logger.info(f"📊 生成测试文件: {len(results['generated_files'])} 个")
                logger.info(f"🧪 测试结果: {results['test_results']['passed']} 通过")
            else:
                logger.error(f"❌ 单次执行失败: {results['error']}")

        finally:
            # 清理AI连接器
            await self.cleanup_ai_connector()

    def generate_status_report(self) -> str:
        """生成状态报告"""
        report_file = "reports/testing/continuous_ai_coverage_status.md"

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 统计执行历史
        total_executions = len(self.execution_history)
        successful_executions = len(
            [r for r in self.execution_history if r.get('status') == 'success'])
        failed_executions = total_executions - successful_executions

        # 计算平均覆盖率
        avg_coverage = 0.0
        if successful_executions > 0:
            coverage_sum = 0
            count = 0
            for record in self.execution_history:
                if record.get('status') == 'success' and 'results' in record:
                    current_cov = record['results'].get('current_coverage', {})
                    if current_cov:
                        avg = sum(current_cov.values()) / len(current_cov)
                        coverage_sum += avg
                        count += 1

            if count > 0:
                avg_coverage = coverage_sum / count

        report_content = f"""# RQA2025 持续AI覆盖率自动化状态报告

## 📊 运行状态

**最后更新时间**: {current_time}
**运行状态**: {'🟢 运行中' if self.running else '🔴 已停止'}
**总执行次数**: {total_executions}
**成功执行**: {successful_executions}
**失败执行**: {failed_executions}
**成功率**: {successful_executions / max(total_executions, 1) * 100:.1f}%
**平均覆盖率**: {avg_coverage:.2f}%

## ⚙️ 配置信息

- **API端点**: {self.config.get('api_base', 'http://localhost:11434')}
- **AI模型**: {self.config.get('model', 'deepseek-coder')}
- **目标覆盖率**: {self.config.get('target_coverage', 85.0)}%
- **执行时间**: {self.config.get('schedule_time', '02:00')}
- **立即执行**: {self.config.get('run_immediately', False)}

## 📈 最近执行记录

"""

        # 显示最近10次执行记录
        recent_executions = self.execution_history[-10:]
        for i, record in enumerate(reversed(recent_executions), 1):
            timestamp = record.get('timestamp', 'Unknown')
            status = record.get('status', 'Unknown')
            status_emoji = "✅" if status == 'success' else "❌"

            report_content += f"### {i}. {timestamp}\n"
            report_content += f"**状态**: {status_emoji} {status}\n"

            if status == 'success' and 'results' in record:
                results = record['results']
                report_content += f"- 生成文件: {results.get('generated_files', 0)} 个\n"
                report_content += f"- 测试通过: {results.get('test_passed', 0)} 个\n"
                report_content += f"- 测试失败: {results.get('test_failed', 0)} 个\n"

                current_cov = results.get('current_coverage', {})
                if current_cov:
                    avg_cov = sum(current_cov.values()) / len(current_cov)
                    report_content += f"- 平均覆盖率: {avg_cov:.2f}%\n"
            elif status == 'failed':
                error = record.get('error', 'Unknown error')
                report_content += f"- 错误: {error}\n"

            report_content += "\n"

        report_content += f"""
## 🚀 下一步行动

1. **监控执行状态**: 定期检查执行日志
2. **优化AI提示**: 根据失败情况调整AI提示词
3. **扩展测试范围**: 增加更多模块的测试覆盖
4. **性能优化**: 优化AI生成和测试执行性能

## 📊 性能指标

- [ ] 执行成功率 ≥ 95%
- [ ] 平均覆盖率 ≥ 85%
- [ ] 测试通过率 ≥ 90%
- [ ] 自动化程度 ≥ 100%

---
**报告版本**: v1.0
**最后更新**: {current_time}
"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"📄 状态报告已生成: {report_file}")
        return report_file


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 持续AI覆盖率自动化运行器')
    parser.add_argument('--mode', choices=['once', 'continuous'], default='once',
                        help='运行模式: once(单次) 或 continuous(持续)')
    parser.add_argument('--api-base', default='http://localhost:11434',
                        help='Deepseek API基础URL')
    parser.add_argument('--model', default='deepseek-coder',
                        help='使用的AI模型')
    parser.add_argument('--target', type=float, default=85.0,
                        help='目标覆盖率')
    parser.add_argument('--schedule-time', default='02:00',
                        help='定时执行时间 (HH:MM)')
    parser.add_argument('--run-immediately', action='store_true',
                        help='立即执行一次')
    parser.add_argument('--status', action='store_true',
                        help='生成状态报告')

    args = parser.parse_args()

    # 检查conda环境
    if 'rqa' not in sys.prefix and 'rqa' not in sys.executable:
        logger.error('❌ 请先激活conda rqa环境后再运行本脚本！')
        sys.exit(1)

    # 创建配置
    config = {
        'api_base': args.api_base,
        'model': args.model,
        'target_coverage': args.target,
        'schedule_time': args.schedule_time,
        'run_immediately': args.run_immediately
    }

    # 创建运行器
    runner = ContinuousAICoverageRunner(config)

    if args.status:
        # 生成状态报告
        runner._load_execution_history()
        report_file = runner.generate_status_report()
        logger.info(f"📄 状态报告已生成: {report_file}")
        return

    if args.mode == 'once':
        # 单次执行
        await runner.run_once()
    else:
        # 持续运行
        await runner.start_scheduler()


if __name__ == "__main__":
    asyncio.run(main())
