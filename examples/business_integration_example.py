#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
业务线集成示例
展示如何在实际业务中集成配置管理Web服务
"""
import requests
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConfigServiceConfig:
    """配置服务配置"""
    api_base: str = "http://localhost:8080"
    username: str = "admin"
    password: str = "admin123"
    session_id: Optional[str] = None
    auto_reload: bool = True
    reload_interval: int = 30  # 秒


class BusinessConfigClient:
    """业务配置客户端"""

    def __init__(self, config: ConfigServiceConfig):
        self.config = config
        self._cached_config = {}
        self._last_reload_time = 0

        # 登录获取session
        self._login()

    def _login(self):
        """登录获取session"""
        try:
            url = f"{self.config.api_base}/api/login"
            resp = requests.post(url, json={
                "username": self.config.username,
                "password": self.config.password
            })
            data = resp.json()

            if data.get("success"):
                self.config.session_id = data["session_id"]
                logger.info("登录成功")
            else:
                raise Exception(f"登录失败: {data.get('detail', '未知错误')}")
        except Exception as e:
            logger.error(f"登录失败: {e}")
            raise

    def _get_auth_header(self):
        """获取认证头"""
        return {"Authorization": f"Bearer {self.config.session_id}"}

    def get_config(self, path: Optional[str] = None, force_reload: bool = False) -> Dict[str, Any]:
        """获取配置"""
        # 检查是否需要重新加载
        if (force_reload or
            self.config.auto_reload and
                time.time() - self._last_reload_time > self.config.reload_interval):
            self._reload_config()

        if path:
            return self._get_config_value(path)
        else:
            return self._cached_config

    def _reload_config(self):
        """重新加载配置"""
        try:
            url = f"{self.config.api_base}/api/config"
            resp = requests.get(url, headers=self._get_auth_header())
            data = resp.json()

            if data.get("success"):
                self._cached_config = data.get("config", {})
                self._last_reload_time = time.time()
                logger.info("配置重新加载成功")
            else:
                logger.warning(f"配置重新加载失败: {data.get('message', '未知错误')}")
        except Exception as e:
            logger.error(f"配置重新加载失败: {e}")

    def _get_config_value(self, path: str) -> Any:
        """获取指定路径的配置值"""
        try:
            url = f"{self.config.api_base}/api/config/{path}"
            resp = requests.get(url, headers=self._get_auth_header())
            data = resp.json()

            if data.get("success"):
                return data.get("value")
            else:
                logger.warning(f"获取配置失败: {data.get('message', '未知错误')}")
                return None
        except Exception as e:
            logger.error(f"获取配置失败: {e}")
            return None

    def update_config(self, path: str, value: Any) -> bool:
        """更新配置"""
        try:
            url = f"{self.config.api_base}/api/config/{path}"
            payload = {"path": path, "value": value}
            resp = requests.put(url, headers=self._get_auth_header(), json=payload)
            data = resp.json()

            if data.get("success"):
                logger.info(f"配置更新成功: {path} = {value}")
                # 重新加载配置缓存
                self._reload_config()
                return True
            else:
                logger.error(f"配置更新失败: {data.get('message', '未知错误')}")
                return False
        except Exception as e:
            logger.error(f"配置更新失败: {e}")
            return False

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置"""
        try:
            url = f"{self.config.api_base}/api/config/validate"
            resp = requests.post(url, headers=self._get_auth_header(), json={"config": config})
            data = resp.json()

            if data.get("valid"):
                logger.info("配置验证通过")
                return True
            else:
                logger.error("配置验证失败:")
                for error in data.get("errors", []):
                    logger.error(f"  - {error}")
                return False
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False

    def sync_config(self, target_nodes: Optional[list] = None) -> bool:
        """同步配置"""
        try:
            url = f"{self.config.api_base}/api/sync"
            payload = {"target_nodes": target_nodes} if target_nodes else {}
            resp = requests.post(url, headers=self._get_auth_header(), json=payload)
            data = resp.json()

            if data.get("success"):
                logger.info("配置同步成功")
                return True
            else:
                logger.error(f"配置同步失败: {data.get('message', '未知错误')}")
                return False
        except Exception as e:
            logger.error(f"配置同步失败: {e}")
            return False


class TradingSystem:
    """交易系统示例"""

    def __init__(self):
        # 初始化配置客户端
        config = ConfigServiceConfig(
            api_base="http://localhost:8080",
            username="trading_user",
            password="trading_pass",
            auto_reload=True,
            reload_interval=60  # 交易系统配置变更频率较低
        )
        self.config_client = BusinessConfigClient(config)

        # 加载初始配置
        self._load_config()

    def _load_config(self):
        """加载配置"""
        try:
            config = self.config_client.get_config()

            # 数据库配置
            self.db_host = config.get("database", {}).get("host", "localhost")
            self.db_port = config.get("database", {}).get("port", 5432)
            self.db_name = config.get("database", {}).get("name", "trading_db")

            # 交易配置
            self.max_position = config.get("trading", {}).get("max_position", 1000000)
            self.risk_limit = config.get("trading", {}).get("risk_limit", 0.1)
            self.enable_auto_trading = config.get("trading", {}).get("enable_auto_trading", False)

            # 风控配置
            self.risk_check_interval = config.get("risk_control", {}).get("check_interval", 30)
            self.max_drawdown = config.get("risk_control", {}).get("max_drawdown", 0.05)

            logger.info("交易系统配置加载完成")

        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            # 使用默认配置
            self._load_default_config()

    def _load_default_config(self):
        """加载默认配置"""
        self.db_host = "localhost"
        self.db_port = 5432
        self.db_name = "trading_db"
        self.max_position = 1000000
        self.risk_limit = 0.1
        self.enable_auto_trading = False
        self.risk_check_interval = 30
        self.max_drawdown = 0.05
        logger.warning("使用默认配置")

    def start_trading(self):
        """启动交易"""
        logger.info("启动交易系统...")
        logger.info(f"数据库连接: {self.db_host}:{self.db_port}/{self.db_name}")
        logger.info(f"最大持仓: {self.max_position}")
        logger.info(f"风险限制: {self.risk_limit}")
        logger.info(f"自动交易: {'启用' if self.enable_auto_trading else '禁用'}")

        # 模拟交易逻辑
        while True:
            try:
                # 检查配置是否需要重新加载
                if self.config_client.config.auto_reload:
                    self._load_config()

                # 执行交易逻辑
                self._execute_trading_logic()

                time.sleep(5)  # 模拟交易间隔

            except KeyboardInterrupt:
                logger.info("交易系统停止")
                break
            except Exception as e:
                logger.error(f"交易系统错误: {e}")
                time.sleep(10)  # 错误后等待

    def _execute_trading_logic(self):
        """执行交易逻辑"""
        # 模拟交易逻辑

    def update_trading_config(self, key: str, value: Any):
        """更新交易配置"""
        path = f"trading.{key}"
        if self.config_client.update_config(path, value):
            logger.info(f"交易配置更新成功: {key} = {value}")
            # 重新加载配置
            self._load_config()
        else:
            logger.error(f"交易配置更新失败: {key}")


class RiskControlSystem:
    """风控系统示例"""

    def __init__(self):
        # 初始化配置客户端
        config = ConfigServiceConfig(
            api_base="http://localhost:8080",
            username="risk_user",
            password="risk_pass",
            auto_reload=True,
            reload_interval=30  # 风控系统需要更频繁的配置更新
        )
        self.config_client = BusinessConfigClient(config)

        # 加载初始配置
        self._load_config()

    def _load_config(self):
        """加载配置"""
        try:
            config = self.config_client.get_config()

            # 风控配置
            risk_config = config.get("risk_control", {})
            self.check_interval = risk_config.get("check_interval", 30)
            self.max_drawdown = risk_config.get("max_drawdown", 0.05)
            self.position_limit = risk_config.get("position_limit", 1000000)
            self.volatility_threshold = risk_config.get("volatility_threshold", 0.2)
            self.enable_real_time_monitoring = risk_config.get("enable_real_time_monitoring", True)

            # 告警配置
            alert_config = config.get("alert", {})
            self.alert_enabled = alert_config.get("enabled", True)
            self.alert_channels = alert_config.get("channels", ["email", "sms"])
            self.alert_threshold = alert_config.get("threshold", 0.8)

            logger.info("风控系统配置加载完成")

        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            self._load_default_config()

    def _load_default_config(self):
        """加载默认配置"""
        self.check_interval = 30
        self.max_drawdown = 0.05
        self.position_limit = 1000000
        self.volatility_threshold = 0.2
        self.enable_real_time_monitoring = True
        self.alert_enabled = True
        self.alert_channels = ["email", "sms"]
        self.alert_threshold = 0.8
        logger.warning("使用默认配置")

    def start_monitoring(self):
        """启动风控监控"""
        logger.info("启动风控监控...")
        logger.info(f"检查间隔: {self.check_interval}秒")
        logger.info(f"最大回撤: {self.max_drawdown}")
        logger.info(f"持仓限制: {self.position_limit}")
        logger.info(f"实时监控: {'启用' if self.enable_real_time_monitoring else '禁用'}")

        # 模拟风控逻辑
        while True:
            try:
                # 检查配置是否需要重新加载
                if self.config_client.config.auto_reload:
                    self._load_config()

                # 执行风控检查
                self._execute_risk_check()

                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("风控监控停止")
                break
            except Exception as e:
                logger.error(f"风控监控错误: {e}")
                time.sleep(10)

    def _execute_risk_check(self):
        """执行风控检查"""
        # 模拟风控检查逻辑

    def update_risk_config(self, key: str, value: Any):
        """更新风控配置"""
        path = f"risk_control.{key}"
        if self.config_client.update_config(path, value):
            logger.info(f"风控配置更新成功: {key} = {value}")
            # 重新加载配置
            self._load_config()
        else:
            logger.error(f"风控配置更新失败: {key}")


def main():
    """主函数 - 演示业务系统集成"""
    print("=== 业务线集成示例 ===")

    # 示例1: 交易系统
    print("\n1. 交易系统集成示例")
    try:
        trading_system = TradingSystem()
        # 更新配置示例
        trading_system.update_trading_config("max_position", 2000000)
        trading_system.update_trading_config("enable_auto_trading", True)

        # 启动交易系统（模拟）
        print("交易系统配置:")
        print(f"  数据库: {trading_system.db_host}:{trading_system.db_port}")
        print(f"  最大持仓: {trading_system.max_position}")
        print(f"  自动交易: {trading_system.enable_auto_trading}")

    except Exception as e:
        print(f"交易系统集成失败: {e}")

    # 示例2: 风控系统
    print("\n2. 风控系统集成示例")
    try:
        risk_system = RiskControlSystem()
        # 更新配置示例
        risk_system.update_risk_config("max_drawdown", 0.03)
        risk_system.update_risk_config("check_interval", 15)

        # 启动风控系统（模拟）
        print("风控系统配置:")
        print(f"  检查间隔: {risk_system.check_interval}秒")
        print(f"  最大回撤: {risk_system.max_drawdown}")
        print(f"  实时监控: {risk_system.enable_real_time_monitoring}")

    except Exception as e:
        print(f"风控系统集成失败: {e}")

    print("\n=== 集成示例完成 ===")
    print("提示: 实际使用时需要:")
    print("1. 配置正确的API地址和认证信息")
    print("2. 根据业务需求调整配置重载间隔")
    print("3. 添加错误处理和重试机制")
    print("4. 集成到现有的监控和日志系统")


if __name__ == "__main__":
    main()
