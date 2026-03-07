#!/usr/bin/env python3
"""
带API服务器的压力测试脚本
先启动API服务器，然后运行压力测试
"""

import sys
import time
import subprocess
import requests
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def wait_for_api_server(url: str, max_wait: int = 30) -> bool:
    """等待API服务器启动"""
    logger.info(f"等待API服务器启动: {url}")

    for i in range(max_wait):
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                logger.info("API服务器已启动")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(1)
        if (i + 1) % 5 == 0:
            logger.info(f"等待API服务器启动... ({i + 1}/{max_wait})")

    logger.error("API服务器启动超时")
    return False


def run_api_server():
    """启动API服务器"""
    logger.info("启动API服务器...")

    # 启动API服务器进程
    server_process = subprocess.Popen([
        sys.executable, "scripts/simple_api_server.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return server_process


def run_stress_test():
    """运行压力测试"""
    logger.info("运行压力测试...")

    # 运行压力测试
    test_process = subprocess.run([
        sys.executable, "scripts/run_stress_test.py"
    ], capture_output=True, text=True)

    return test_process


def main():
    """主函数"""
    logger.info("开始带API服务器的压力测试...")

    api_server_process = None

    try:
        # 1. 启动API服务器
        api_server_process = run_api_server()

        # 2. 等待API服务器启动
        if not wait_for_api_server("http://localhost:5000"):
            logger.error("API服务器启动失败")
            return 1

        # 3. 运行压力测试
        test_result = run_stress_test()

        # 4. 输出测试结果
        if test_result.returncode == 0:
            logger.info("🎉 压力测试通过!")
        else:
            logger.error("❌ 压力测试失败")
            logger.error(f"测试输出: {test_result.stdout}")
            logger.error(f"测试错误: {test_result.stderr}")

        return test_result.returncode

    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
        return 1

    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        return 1

    finally:
        # 清理：停止API服务器
        if api_server_process:
            logger.info("停止API服务器...")
            try:
                api_server_process.terminate()
                api_server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                api_server_process.kill()
            logger.info("API服务器已停止")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
