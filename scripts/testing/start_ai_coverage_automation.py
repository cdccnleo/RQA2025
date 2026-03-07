#!/usr/bin/env python3
"""
RQA2025 AI覆盖率自动化启动脚本
提供便捷的启动和管理功能
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 日志配置，确保控制台和文件都能输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/start_ai_coverage_automation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    logger.info("【日志测试】check_environment已进入")

    # 检查conda环境
    if 'rqa' not in sys.prefix and 'rqa' not in sys.executable:
        print("❌ 请先激活conda rqa环境后再运行本脚本！")
        logger.error("【日志测试】conda rqa环境未激活")
        return False

    # 检查必要目录
    required_dirs = ['src', 'tests', 'logs']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"❌ 缺少必要目录: {dir_name}")
            logger.error(f"【日志测试】缺少必要目录: {dir_name}")
            return False

    # 检查配置文件
    config_file = "scripts/testing/ai_coverage_config.json"
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        logger.error(f"【日志测试】配置文件不存在: {config_file}")
        return False

    print("✅ 环境检查通过")
    logger.info("【日志测试】环境检查通过")
    return True


def load_config():
    """加载配置文件"""
    config_file = "scripts/testing/ai_coverage_config.json"
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        logger.error(f"【日志测试】加载配置文件失败: {e}")
        return None


def check_ai_service():
    """检查AI服务是否可用"""
    print("🤖 检查AI服务...")
    logger.info("【日志测试】check_ai_service已进入")

    config = load_config()
    if not config:
        logger.error("【日志测试】加载配置文件失败，AI服务不可用")
        return False

    api_base = config['ai_config']['api_base']

    try:
        import aiohttp
        import asyncio

        async def test_connection():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{api_base}/v1/models",
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            # 检查响应内容
                            data = await response.json()
                            if 'data' in data and len(data['data']) > 0:
                                return True
                            else:
                                print("⚠️ AI服务响应格式异常")
                                logger.warning("【日志测试】AI服务响应格式异常")
                                return False
                        else:
                            print(f"⚠️ AI服务响应状态码: {response.status}")
                            logger.warning(f"【日志测试】AI服务响应状态码: {response.status}")
                            return False
            except asyncio.TimeoutError:
                print("❌ AI服务连接超时")
                logger.error("【日志测试】AI服务连接超时")
                return False
            except Exception as e:
                print(f"❌ AI服务连接异常: {e}")
                logger.error(f"【日志测试】AI服务连接异常: {e}")
                return False

        # 运行连接测试
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(test_connection())
        loop.close()

        if result:
            print("✅ AI服务连接正常")
            logger.info("【日志测试】AI服务连接正常")
            return True
        else:
            print("❌ AI服务连接失败")
            logger.error("【日志测试】AI服务连接失败")
            return False

    except Exception as e:
        print(f"❌ AI服务检查失败: {e}")
        logger.error(f"【日志测试】AI服务检查失败: {e}")
        return False


def run_single_execution():
    """运行单次执行"""
    print("🚀 启动单次AI覆盖率自动化...")
    logger.info("【日志测试】run_single_execution已进入")
    cmd = [
        "python", "scripts/testing/ai_enhanced_coverage_automation.py",
        "--target", "85.0",
        "--layers", "infrastructure", "data", "features", "trading"
    ]
    logger.info(f"【日志测试】即将调用子进程: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=600, encoding='utf-8', errors='replace')
        logger.info(f"【日志测试】子进程返回码: {result.returncode}")

        if result.returncode == 0:
            print("✅ 单次执行成功")
            print(result.stdout)
            logger.info("【日志测试】单次执行成功")
        else:
            print("❌ 单次执行失败")
            print(result.stderr)
            logger.error(f"【日志测试】单次执行失败: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("❌ 执行超时")
        logger.error("【日志测试】执行超时")
    except Exception as e:
        print(f"❌ 执行异常: {e}")
        logger.error(f"【日志测试】执行异常: {e}")


def run_continuous_execution():
    """运行持续执行"""
    print("🔄 启动持续AI覆盖率自动化...")

    cmd = [
        "python", "scripts/testing/continuous_ai_coverage_runner.py",
        "--mode", "continuous",
        "--run-immediately"
    ]

    try:
        # 在后台运行
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ 持续执行已启动 (PID: {process.pid})")
        print("📝 日志文件: logs/continuous_ai_coverage.log")

        # 保存PID到文件
        with open("logs/ai_coverage_automation.pid", "w") as f:
            f.write(str(process.pid))

        return process

    except Exception as e:
        print(f"❌ 启动持续执行失败: {e}")
        logger.error(f"【日志测试】启动持续执行失败: {e}")
        return None


def stop_continuous_execution():
    """停止持续执行"""
    print("🛑 停止持续AI覆盖率自动化...")
    logger.info("【日志测试】stop_continuous_execution已进入")

    pid_file = "logs/ai_coverage_automation.pid"
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())

            # 终止进程
            os.kill(pid, 15)  # SIGTERM
            print(f"✅ 已发送停止信号到进程 {pid}")
            logger.info(f"【日志测试】已发送停止信号到进程 {pid}")

            # 删除PID文件
            os.remove(pid_file)
            print("✅ PID文件已删除")
            logger.info("【日志测试】PID文件已删除")

        except Exception as e:
            print(f"❌ 停止进程失败: {e}")
            logger.error(f"【日志测试】停止进程失败: {e}")
    else:
        print("❌ 未找到PID文件")
        logger.warning("【日志测试】未找到PID文件")


def show_status():
    """显示状态"""
    print("📊 AI覆盖率自动化状态...")
    logger.info("【日志测试】show_status已进入")

    # 检查PID文件
    pid_file = "logs/ai_coverage_automation.pid"
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())

            # 检查进程是否运行
            try:
                os.kill(pid, 0)  # 发送信号0检查进程是否存在
                print(f"🟢 持续执行正在运行 (PID: {pid})")
                logger.info(f"【日志测试】持续执行正在运行 (PID: {pid})")
            except OSError:
                print("🔴 进程不存在，可能已停止")
                logger.warning(f"【日志测试】进程不存在，可能已停止 (PID: {pid})")
                os.remove(pid_file)
                print("✅ PID文件已删除")
                logger.info("【日志测试】PID文件已删除")
        except Exception as e:
            print(f"❌ 读取PID文件失败: {e}")
            logger.error(f"【日志测试】读取PID文件失败: {e}")
    else:
        print("🔴 持续执行未运行")
        logger.warning("【日志测试】持续执行未运行")

    # 显示最近日志
    log_file = "logs/continuous_ai_coverage.log"
    if os.path.exists(log_file):
        print(f"\n📝 最近日志 (最后10行):")
        try:
            with open(log_file, "r", encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(f"  {line.strip()}")
        except Exception as e:
            print(f"❌ 读取日志失败: {e}")
            logger.error(f"【日志测试】读取日志失败: {e}")


def generate_report():
    """生成报告"""
    print("📄 生成AI覆盖率报告...")
    logger.info("【日志测试】generate_report已进入")

    cmd = [
        "python", "scripts/testing/continuous_ai_coverage_runner.py",
        "--status"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ 报告生成成功")
            print("📄 报告位置: reports/testing/continuous_ai_coverage_status.md")
            logger.info("【日志测试】报告生成成功")
        else:
            print("❌ 报告生成失败")
            print(result.stderr)
            logger.error(f"【日志测试】报告生成失败: {result.stderr}")

    except Exception as e:
        print(f"❌ 生成报告异常: {e}")
        logger.error(f"【日志测试】生成报告异常: {e}")


def main():
    """主函数"""
    logger.info("【日志测试】main已进入")
    parser = argparse.ArgumentParser(description='RQA2025 AI覆盖率自动化启动脚本')
    parser.add_argument('action', choices=['start', 'stop', 'status', 'once', 'report', 'check'],
                        help='执行动作')
    parser.add_argument('--mode', choices=['single', 'continuous'], default='single',
                        help='运行模式')
    args = parser.parse_args()
    logger.info(f"【日志测试】解析参数: action={args.action}, mode={args.mode}")
    # 检查环境
    if not check_environment():
        logger.error("【日志测试】环境检查未通过，退出")
        sys.exit(1)
    logger.info("【日志测试】环境检查通过")
    if args.action == 'check':
        # 检查AI服务
        if check_ai_service():
            print("✅ 所有检查通过，可以开始使用AI覆盖率自动化")
            logger.info("【日志测试】AI服务检查通过")
        else:
            print("❌ AI服务不可用，请检查Deepseek服务是否启动")
            logger.error("【日志测试】AI服务不可用")
        return

    elif args.action == 'once':
        logger.info("【日志测试】进入once分支，准备单次执行")
        # 单次执行
        run_single_execution()

    elif args.action == 'start':
        logger.info("【日志测试】进入start分支")
        # 启动服务
        if args.mode == 'single':
            run_single_execution()
        else:
            run_continuous_execution()

    elif args.action == 'stop':
        logger.info("【日志测试】进入stop分支")
        # 停止服务
        stop_continuous_execution()

    elif args.action == 'status':
        logger.info("【日志测试】进入status分支")
        # 显示状态
        show_status()

    elif args.action == 'report':
        logger.info("【日志测试】进入report分支")
        # 生成报告
        generate_report()

    print("\n🎯 使用说明:")
    print("  python scripts/testing/start_ai_coverage_automation.py check    # 检查环境")
    print("  python scripts/testing/start_ai_coverage_automation.py once     # 单次执行")
    print("  python scripts/testing/start_ai_coverage_automation.py start    # 启动持续执行")
    print("  python scripts/testing/start_ai_coverage_automation.py stop     # 停止持续执行")
    print("  python scripts/testing/start_ai_coverage_automation.py status   # 查看状态")
    print("  python scripts/testing/start_ai_coverage_automation.py report   # 生成报告")


if __name__ == "__main__":
    main()
