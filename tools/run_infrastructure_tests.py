#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 基础设施层测试运行脚本
使用conda test环境运行测试，避免PowerShell环境问题
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_infrastructure_tests():
    """运行基础设施层测试"""

    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    # 设置测试环境变量
    test_env = os.environ.copy()
    test_env.update({
        'PYTEST_CURRENT_TEST': '1',
        'TESTING': '1',
        'DISABLE_BACKGROUND_TASKS': '1',
        'DISABLE_PERFORMANCE_MONITORING': '1',
        'PYTHONPATH': str(project_root),
        'DISABLE_THREAD_CLEANUP': '0',  # 启用线程清理
        'TEST_TIMEOUT': '60',  # 60秒超时
    })

    # 分模块运行测试，避免大批量测试超时
    test_modules = [
        {
            'name': '健康检查模块',
            'path': 'tests/unit/infrastructure/test_health/health_check.py',
            'timeout': 120
        },
        {
            'name': '存储模块',
            'path': 'tests/unit/infrastructure/storage/test_storage.py',
            'timeout': 90
        },
        {
            'name': '云原生模块',
            'path': 'tests/unit/infrastructure/cloud_native/',
            'timeout': 150
        },
        {
            'name': '分布式系统模块',
            'path': 'tests/unit/infrastructure/distributed/',
            'timeout': 120
        },
        {
            'name': '错误处理模块',
            'path': 'tests/unit/infrastructure/error/test_retry_handler_fixed.py',
            'timeout': 100
        }
    ]

    total_success = 0
    total_tests = len(test_modules)

    for module in test_modules:
        module_name = module['name']
        module_path = module['path']
        module_timeout = module['timeout']

        logger.info(f"开始测试模块: {module_name}")

        # 检查测试文件是否存在
        if not os.path.exists(module_path):
            logger.warning(f"测试路径不存在: {module_path}")
            continue

        # 构建测试命令
        test_command = [
            'python', '-m', 'pytest',
            module_path,
            '--tb=short',
            '--maxfail=3',
            '-x',
            '-v',
            '--disable-warnings',
            '--durations=5',
            f'--timeout={module_timeout}'
        ]

        # 为特定模块添加特殊配置
        if 'cloud_native' in module_path:
            test_command.extend(['--timeout=200', '--capture=no'])
        elif 'distributed' in module_path:
            test_command.extend(['--timeout=180', '--capture=no'])

        logger.info(f"测试命令: {' '.join(test_command)}")

        try:
            # 运行测试
            result = subprocess.run(
                test_command,
                env=test_env,
                capture_output=False,
                text=True,
                timeout=module_timeout + 30  # 额外30秒缓冲时间
            )

            if result.returncode == 0:
                logger.info(f"✅ {module_name}测试通过！")
                total_success += 1
            else:
                logger.error(f"❌ {module_name}测试失败，退出码: {result.returncode}")

        except subprocess.TimeoutExpired:
            logger.error(f"❌ {module_name}测试超时（{module_timeout}秒）")
        except Exception as e:
            logger.error(f"❌ 运行{module_name}测试时发生错误: {e}")

        # 在模块间添加短暂延迟，避免资源冲突
        time.sleep(2)

    return 0 if total_success == total_tests else 1


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("RQA2025 基础设施层测试")
    logger.info("=" * 60)

    # 检查是否在正确的环境中
    python_path = sys.executable
    logger.info(f"Python路径: {python_path}")

    # 运行测试
    exit_code = run_infrastructure_tests()

    logger.info("=" * 60)
    if exit_code == 0:
        logger.info("🎉 测试完成！")
    else:
        logger.error("💥 测试失败！")
    logger.info("=" * 60)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
