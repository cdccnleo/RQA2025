#!/usr/bin/env python3
"""
Docker容器化编排器集成测试脚本
测试DataCollectionOrchestrator在Docker环境中的集成情况
"""

import os
import sys
import time
import json
import asyncio
import subprocess
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_docker_environment():
    """检查Docker环境"""
    logger.info("检查Docker环境...")

    try:
        # 检查Docker是否安装
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Docker未安装或不可用")
            return False
        logger.info(f"Docker版本: {result.stdout.strip()}")

        # 检查Docker Compose是否安装
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Docker Compose未安装或不可用")
            return False
        logger.info(f"Docker Compose版本: {result.stdout.strip()}")

        return True

    except Exception as e:
        logger.error(f"Docker环境检查失败: {e}")
        return False

def check_project_structure():
    """检查项目结构"""
    logger.info("检查项目结构...")

    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    logger.info(f"项目根目录: {project_root}")

    required_files = [
        'docker-compose.yml',
        'Dockerfile',
        'requirements.txt',
        'scripts/start_production.py',
        'src/core/orchestration/business_process/data_collection_orchestrator.py',
        'src/gateway/web/data_source_config_manager.py'
    ]

    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)

    if missing_files:
        logger.error(f"缺少必需文件: {missing_files}")
        return False

    logger.info("项目结构检查通过")
    return True

def validate_docker_compose_config():
    """验证Docker Compose配置"""
    logger.info("验证Docker Compose配置...")

    try:
        project_root = Path(__file__).parent.parent
        result = subprocess.run(['docker-compose', 'config'], capture_output=True, text=True, cwd=str(project_root), encoding='utf-8')
        if result.returncode != 0:
            logger.error(f"Docker Compose配置无效: {result.stderr}")
            return False

        # 检查是否包含编排器服务
        if 'data-collection-orchestrator' not in result.stdout:
            logger.error("Docker Compose配置中缺少data-collection-orchestrator服务")
            return False

        logger.info("Docker Compose配置验证通过")
        return True

    except Exception as e:
        logger.error(f"Docker Compose配置验证失败: {e}")
        return False

def test_orchestrator_imports():
    """测试编排器相关模块的导入"""
    logger.info("测试编排器模块导入...")

    try:
        # 添加src路径
        project_root = Path(__file__).parent.parent
        src_path = project_root / 'src'
        sys.path.insert(0, str(src_path))
        sys.path.insert(0, str(project_root))

        # 测试核心编排器导入
        from src.core.orchestration.business_process.data_collection_orchestrator import DataCollectionWorkflow
        logger.info("✅ DataCollectionOrchestrator导入成功")

        # 测试状态机导入
        from src.core.orchestration.business_process.data_collection_state_machine import StateMachineManager
        logger.info("✅ DataCollectionStateMachine导入成功")

        # 测试服务治理导入
        from src.core.orchestration.business_process.service_governance import ServiceGovernanceManager
        logger.info("✅ ServiceGovernanceManager导入成功")

        # 测试监控告警导入
        from src.core.orchestration.business_process.monitoring_alerts import AlertManager, DataCollectionMonitor
        logger.info("✅ MonitoringAlerts导入成功")

        # 测试数据源配置管理器导入
        from src.gateway.web.data_source_config_manager import DataSourceConfigManager
        logger.info("✅ DataSourceConfigManager导入成功")

        return True

    except Exception as e:
        logger.error(f"编排器模块导入失败: {e}")
        return False

def test_data_source_config_manager():
    """测试数据源配置管理器功能"""
    logger.info("测试数据源配置管理器功能...")

    try:
        sys.path.insert(0, 'src')
        from src.gateway.web.data_source_config_manager import DataSourceConfigManager

        # 创建配置管理器实例
        manager = DataSourceConfigManager()

        # 测试获取数据源
        sources = manager.get_data_sources()
        logger.info(f"✅ 获取数据源成功: {len(sources)} 个数据源")

        # 测试配置验证
        validation = manager.validate_all_sources()
        logger.info(f"✅ 配置验证完成: {validation}")

        # 测试配置统计
        stats = manager.get_config_stats()
        logger.info(f"✅ 配置统计: {stats}")

        return True

    except Exception as e:
        logger.error(f"数据源配置管理器测试失败: {e}")
        return False

def test_orchestrator_initialization():
    """测试编排器初始化"""
    logger.info("测试编排器初始化...")

    async def test_orchestrator_init():
        try:
            sys.path.insert(0, 'src')
            from src.core.orchestration.business_process.data_collection_orchestrator import DataCollectionWorkflow

            # 创建编排器实例
            orchestrator = DataCollectionWorkflow()
            logger.info("✅ DataCollectionOrchestrator初始化成功")

            # 测试工作流统计查询
            stats = orchestrator.get_workflow_stats()
            logger.info(f"✅ 工作流统计查询成功: {stats}")

            return True
        except Exception as e:
            logger.error(f"编排器初始化测试失败: {e}")
            return False

    # 在异步上下文中运行测试
    result = asyncio.run(test_orchestrator_init())
    return result

def test_docker_build():
    """测试Docker镜像构建"""
    logger.info("测试Docker镜像构建...")

    try:
        # 构建Docker镜像
        logger.info("开始构建Docker镜像...")
        project_root = Path(__file__).parent.parent
        result = subprocess.run(
            ['docker-compose', 'build', '--no-cache'],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=600  # 10分钟超时
        )

        if result.returncode != 0:
            logger.error(f"Docker镜像构建失败: {result.stderr}")
            return False

        logger.info("✅ Docker镜像构建成功")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Docker镜像构建超时")
        return False
    except Exception as e:
        logger.error(f"Docker镜像构建异常: {e}")
        return False

def test_container_startup():
    """测试容器启动"""
    logger.info("测试容器启动...")

    try:
        # 启动容器（后台模式）
        logger.info("启动Docker容器...")
        project_root = Path(__file__).parent.parent
        result = subprocess.run(
            ['docker-compose', 'up', '-d'],
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )

        if result.returncode != 0:
            logger.error(f"容器启动失败: {result.stderr}")
            return False

        logger.info("✅ 容器启动成功")

        # 等待容器完全启动
        logger.info("等待容器完全启动...")
        time.sleep(30)

        # 检查容器状态
        project_root = Path(__file__).parent.parent
        result = subprocess.run(
            ['docker-compose', 'ps'],
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )

        if 'Up' not in result.stdout:
            logger.error("部分容器未正常启动")
            logger.info(f"容器状态: {result.stdout}")
            return False

        logger.info("✅ 容器状态检查通过")

        # 测试API端点
        logger.info("测试API端点...")
        time.sleep(10)  # 额外等待

        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=10)
            if response.status_code == 200:
                logger.info("✅ API端点测试通过")
            else:
                logger.warning(f"API端点返回状态码: {response.status_code}")
        except Exception as e:
            logger.warning(f"API端点测试失败: {e}")

        return True

    except Exception as e:
        logger.error(f"容器启动测试失败: {e}")
        return False

    finally:
        # 清理容器
        logger.info("清理测试容器...")
        project_root = Path(__file__).parent.parent
        subprocess.run(['docker-compose', 'down'], capture_output=True, cwd=str(project_root))

def run_integration_tests():
    """运行集成测试"""
    logger.info("=" * 60)
    logger.info("🚀 开始Docker容器化编排器集成测试")
    logger.info("=" * 60)

    tests = [
        ("Docker环境检查", check_docker_environment),
        ("项目结构检查", check_project_structure),
        ("Docker Compose配置验证", validate_docker_compose_config),
        ("编排器模块导入测试", test_orchestrator_imports),
        ("数据源配置管理器测试", test_data_source_config_manager),
        ("编排器初始化测试", test_orchestrator_initialization),
        ("Docker镜像构建测试", test_docker_build),
        ("容器启动测试", test_container_startup),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n执行测试: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ 通过" if result else "❌ 失败"
            logger.info(f"测试结果: {status}")
        except Exception as e:
            logger.error(f"测试异常: {e}")
            results.append((test_name, False))

    # 输出测试总结
    logger.info("\n" + "=" * 60)
    logger.info("📊 测试结果汇总")
    logger.info("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{status} {test_name}")
        if result:
            passed += 1

    logger.info(f"\n总测试数: {total}")
    logger.info(f"通过测试: {passed}")
    logger.info(f"失败测试: {total - passed}")
    logger.info(".1f")

    if passed == total:
        logger.info("\n🎉 所有Docker容器化编排器集成测试通过！")
        return True
    else:
        logger.error("\n❌ 部分集成测试失败，请检查相关配置和实现。")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
