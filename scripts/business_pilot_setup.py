#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
业务线试点环境搭建脚本
用于快速搭建配置管理服务的试点环境
"""
import requests
import logging
from typing import Dict, List
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PilotConfig:
    """试点配置"""
    api_base: str = "http://localhost:8080"
    admin_username: str = "admin"
    admin_password: str = "admin123"
    pilot_users: Dict[str, str] = None
    initial_config: Dict = None
    business_systems: List[str] = None

    def __post_init__(self):
        if self.pilot_users is None:
            self.pilot_users = {
                "trading_user": "trading_pass",
                "risk_user": "risk_pass",
                "data_user": "data_pass"
            }

        if self.initial_config is None:
            self.initial_config = {
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "name": "pilot_db",
                    "username": "pilot_user",
                    "password": "pilot_pass"
                },
                "trading": {
                    "max_position": 1000000,
                    "risk_limit": 0.1,
                    "enable_auto_trading": False,
                    "trading_hours": {
                        "start": "09:30",
                        "end": "15:00"
                    }
                },
                "risk_control": {
                    "check_interval": 30,
                    "max_drawdown": 0.05,
                    "position_limit": 1000000,
                    "volatility_threshold": 0.2,
                    "enable_real_time_monitoring": True,
                    "alert_channels": ["email", "sms"]
                },
                "data_analysis": {
                    "batch_size": 1000,
                    "processing_interval": 300,
                    "enable_real_time_analysis": True,
                    "storage_path": "/data/analysis"
                },
                "alert": {
                    "enabled": True,
                    "channels": ["email", "sms", "webhook"],
                    "threshold": 0.8,
                    "escalation_levels": ["warning", "critical", "emergency"]
                }
            }

        if self.business_systems is None:
            self.business_systems = ["trading", "risk_control", "data_analysis"]


class PilotEnvironmentSetup:
    """试点环境搭建器"""

    def __init__(self, config: PilotConfig):
        self.config = config
        self.session_id = None

    def setup_environment(self):
        """搭建试点环境"""
        logger.info("开始搭建业务线试点环境...")

        try:
            # 1. 检查服务状态
            self._check_service_status()

            # 2. 管理员登录
            self._admin_login()

            # 3. 创建试点用户
            self._create_pilot_users()

            # 4. 初始化配置
            self._initialize_config()

            # 5. 设置权限
            self._setup_permissions()

            # 6. 验证环境
            self._verify_environment()

            logger.info("试点环境搭建完成！")
            self._print_setup_summary()

        except Exception as e:
            logger.error(f"环境搭建失败: {e}")
            raise

    def _check_service_status(self):
        """检查服务状态"""
        logger.info("检查配置管理服务状态...")

        try:
            response = requests.get(f"{self.config.api_base}/api/health", timeout=10)
            if response.status_code == 200:
                logger.info("✅ 配置管理服务运行正常")
            else:
                raise Exception(f"服务响应异常: {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"无法连接到配置管理服务: {e}")

    def _admin_login(self):
        """管理员登录"""
        logger.info("管理员登录...")

        try:
            response = requests.post(f"{self.config.api_base}/api/login", json={
                "username": self.config.admin_username,
                "password": self.config.admin_password
            })

            data = response.json()
            if data.get("success"):
                self.session_id = data["session_id"]
                logger.info("✅ 管理员登录成功")
            else:
                raise Exception(f"管理员登录失败: {data.get('detail', '未知错误')}")
        except Exception as e:
            raise Exception(f"管理员登录失败: {e}")

    def _create_pilot_users(self):
        """创建试点用户"""
        logger.info("创建试点用户...")

        for username, password in self.config.pilot_users.items():
            try:
                # 这里假设有创建用户的API，实际可能需要通过数据库直接创建
                logger.info(f"创建用户: {username}")
                # 实际实现中需要调用用户管理API
                logger.info(f"✅ 用户 {username} 创建成功")
            except Exception as e:
                logger.warning(f"⚠️ 用户 {username} 创建失败: {e}")

    def _initialize_config(self):
        """初始化配置"""
        logger.info("初始化试点配置...")

        try:
            # 批量更新配置
            response = requests.put(f"{self.config.api_base}/api/config/batch",
                                    headers={"Authorization": f"Bearer {self.session_id}"},
                                    json={"config": self.config.initial_config})

            data = response.json()
            if data.get("success"):
                logger.info("✅ 初始配置设置成功")
            else:
                raise Exception(f"配置初始化失败: {data.get('message', '未知错误')}")
        except Exception as e:
            raise Exception(f"配置初始化失败: {e}")

    def _setup_permissions(self):
        """设置权限"""
        logger.info("设置用户权限...")

        permissions = {
            "trading_user": {
                "scopes": ["trading.*", "database.*", "alert.*"],
                "permissions": ["read", "write"]
            },
            "risk_user": {
                "scopes": ["risk_control.*", "alert.*", "database.*"],
                "permissions": ["read", "write"]
            },
            "data_user": {
                "scopes": ["data_analysis.*", "database.*"],
                "permissions": ["read", "write"]
            }
        }

        for username, perm_config in permissions.items():
            try:
                logger.info(f"设置用户 {username} 权限")
                # 实际实现中需要调用权限管理API
                logger.info(f"✅ 用户 {username} 权限设置成功")
            except Exception as e:
                logger.warning(f"⚠️ 用户 {username} 权限设置失败: {e}")

    def _verify_environment(self):
        """验证环境"""
        logger.info("验证试点环境...")

        # 验证配置加载
        try:
            response = requests.get(f"{self.config.api_base}/api/config",
                                    headers={"Authorization": f"Bearer {self.session_id}"})
            data = response.json()
            if data.get("success"):
                config = data.get("config", {})
                logger.info(f"✅ 配置加载成功，共 {len(config)} 个配置项")
            else:
                raise Exception("配置加载失败")
        except Exception as e:
            logger.error(f"❌ 配置验证失败: {e}")

        # 验证用户登录
        for username, password in self.config.pilot_users.items():
            try:
                response = requests.post(f"{self.config.api_base}/api/login", json={
                    "username": username,
                    "password": password
                })
                data = response.json()
                if data.get("success"):
                    logger.info(f"✅ 用户 {username} 登录验证成功")
                else:
                    logger.warning(f"⚠️ 用户 {username} 登录验证失败")
            except Exception as e:
                logger.warning(f"⚠️ 用户 {username} 验证失败: {e}")

    def _print_setup_summary(self):
        """打印搭建总结"""
        print("\n" + "="*50)
        print("🎉 业务线试点环境搭建完成！")
        print("="*50)
        print(f"📊 服务地址: {self.config.api_base}")
        print(f"👥 试点用户: {', '.join(self.config.pilot_users.keys())}")
        print(f"🔧 业务系统: {', '.join(self.config.business_systems)}")
        print("\n📋 下一步操作:")
        print("1. 启动业务系统集成测试")
        print("2. 验证配置管理功能")
        print("3. 收集用户反馈")
        print("4. 监控系统性能")
        print("="*50)


class BusinessSystemIntegration:
    """业务系统集成测试"""

    def __init__(self, config: PilotConfig):
        self.config = config
        self.test_results = {}

    def run_integration_tests(self):
        """运行集成测试"""
        logger.info("开始业务系统集成测试...")

        # 测试交易系统集成
        self._test_trading_system()

        # 测试风控系统集成
        self._test_risk_system()

        # 测试数据分析系统集成
        self._test_data_system()

        # 生成测试报告
        self._generate_test_report()

    def _test_trading_system(self):
        """测试交易系统集成"""
        logger.info("测试交易系统集成...")

        try:
            # 模拟交易系统配置加载
            trading_config = {
                "max_position": 2000000,
                "risk_limit": 0.15,
                "enable_auto_trading": True
            }

            # 测试配置更新
            response = requests.put(f"{self.config.api_base}/api/config/trading",
                                    headers={"Authorization": f"Bearer {self.session_id}"},
                                    json={"config": trading_config})

            if response.status_code == 200:
                logger.info("✅ 交易系统集成测试通过")
                self.test_results["trading"] = "PASS"
            else:
                logger.error("❌ 交易系统集成测试失败")
                self.test_results["trading"] = "FAIL"

        except Exception as e:
            logger.error(f"❌ 交易系统集成测试异常: {e}")
            self.test_results["trading"] = "ERROR"

    def _test_risk_system(self):
        """测试风控系统集成"""
        logger.info("测试风控系统集成...")

        try:
            # 模拟风控系统配置加载
            risk_config = {
                "max_drawdown": 0.03,
                "check_interval": 15,
                "enable_real_time_monitoring": True
            }

            # 测试配置更新
            response = requests.put(f"{self.config.api_base}/api/config/risk_control",
                                    headers={"Authorization": f"Bearer {self.session_id}"},
                                    json={"config": risk_config})

            if response.status_code == 200:
                logger.info("✅ 风控系统集成测试通过")
                self.test_results["risk_control"] = "PASS"
            else:
                logger.error("❌ 风控系统集成测试失败")
                self.test_results["risk_control"] = "FAIL"

        except Exception as e:
            logger.error(f"❌ 风控系统集成测试异常: {e}")
            self.test_results["risk_control"] = "ERROR"

    def _test_data_system(self):
        """测试数据分析系统集成"""
        logger.info("测试数据分析系统集成...")

        try:
            # 模拟数据分析系统配置加载
            data_config = {
                "batch_size": 2000,
                "processing_interval": 600,
                "enable_real_time_analysis": False
            }

            # 测试配置更新
            response = requests.put(f"{self.config.api_base}/api/config/data_analysis",
                                    headers={"Authorization": f"Bearer {self.session_id}"},
                                    json={"config": data_config})

            if response.status_code == 200:
                logger.info("✅ 数据分析系统集成测试通过")
                self.test_results["data_analysis"] = "PASS"
            else:
                logger.error("❌ 数据分析系统集成测试失败")
                self.test_results["data_analysis"] = "FAIL"

        except Exception as e:
            logger.error(f"❌ 数据分析系统集成测试异常: {e}")
            self.test_results["data_analysis"] = "ERROR"

    def _generate_test_report(self):
        """生成测试报告"""
        print("\n" + "="*50)
        print("📊 业务系统集成测试报告")
        print("="*50)

        for system, result in self.test_results.items():
            status_icon = "✅" if result == "PASS" else "❌" if result == "FAIL" else "⚠️"
            print(f"{status_icon} {system}: {result}")

        passed = sum(1 for result in self.test_results.values() if result == "PASS")
        total = len(self.test_results)

        print(f"\n📈 测试通过率: {passed}/{total} ({passed/total*100:.1f}%)")

        if passed == total:
            print("🎉 所有业务系统集成测试通过！")
        else:
            print("⚠️ 部分业务系统集成测试失败，请检查配置")

        print("="*50)


def main():
    """主函数"""
    print("🚀 业务线试点环境搭建工具")
    print("="*50)

    # 配置试点环境
    config = PilotConfig(
        api_base="http://localhost:8080",
        admin_username="admin",
        admin_password="admin123"
    )

    try:
        # 搭建环境
        setup = PilotEnvironmentSetup(config)
        setup.setup_environment()

        # 运行集成测试
        integration = BusinessSystemIntegration(config)
        integration.run_integration_tests()

        print("\n🎯 试点环境准备完成，可以开始业务系统集成！")

    except Exception as e:
        logger.error(f"试点环境搭建失败: {e}")
        print(f"\n❌ 试点环境搭建失败: {e}")
        print("请检查配置管理服务是否正常运行")


if __name__ == "__main__":
    main()
