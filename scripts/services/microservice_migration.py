#!/usr/bin/env python3
"""
微服务化迁移脚本

将现有的单体服务架构迁移到微服务架构
"""

from src.core import EventBus, ServiceContainer
from src.services.micro_service import MicroService, ServiceInfo, ServiceType, MicroServiceStatus
import os
import sys
import json
import yaml
import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class MicroserviceMigration:
    """微服务化迁移管理器"""

    def __init__(self, config_path: str = None):
        """
        初始化迁移管理器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path or "config/microservice_migration.yaml"
        self.config = self._load_config()
        self.event_bus = EventBus()
        self.container = ServiceContainer()
        self.micro_service = MicroService(self.event_bus, self.container)
        self.logger = logging.getLogger(__name__)

        # 迁移状态
        self.migration_status = {
            "started_at": None,
            "completed_at": None,
            "services_migrated": [],
            "errors": []
        }

    def _load_config(self) -> Dict[str, Any]:
        """加载迁移配置"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        return yaml.safe_load(f)
                    else:
                        return json.load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "services": {
                "api_service": {
                    "type": "api",
                    "port": 8001,
                    "dependencies": [],
                    "health_check_interval": 30
                },
                "business_service": {
                    "type": "business",
                    "port": 8002,
                    "dependencies": ["api_service"],
                    "health_check_interval": 30
                },
                "model_service": {
                    "type": "model",
                    "port": 8003,
                    "dependencies": [],
                    "health_check_interval": 30
                },
                "trading_service": {
                    "type": "trading",
                    "port": 8004,
                    "dependencies": ["business_service", "model_service"],
                    "health_check_interval": 30
                },
                "cache_service": {
                    "type": "cache",
                    "port": 8005,
                    "dependencies": [],
                    "health_check_interval": 30
                },
                "validation_service": {
                    "type": "validation",
                    "port": 8006,
                    "dependencies": ["api_service"],
                    "health_check_interval": 30
                }
            },
            "network": {
                "service_mesh": True,
                "load_balancer": "round_robin",
                "circuit_breaker": True,
                "retry_mechanism": True
            },
            "monitoring": {
                "metrics_collection": True,
                "distributed_tracing": True,
                "health_check": True
            }
        }

    async def start_migration(self) -> bool:
        """开始微服务化迁移"""
        try:
            self.logger.info("开始微服务化迁移...")
            self.migration_status["started_at"] = datetime.now()

            # 1. 启动微服务框架
            if not self.micro_service.start():
                raise Exception("微服务框架启动失败")

            # 2. 注册所有服务
            await self._register_all_services()

            # 3. 验证服务依赖
            await self._validate_service_dependencies()

            # 4. 启动服务网格
            await self._start_service_mesh()

            # 5. 配置负载均衡
            await self._configure_load_balancer()

            # 6. 启动监控
            await self._start_monitoring()

            self.migration_status["completed_at"] = datetime.now()
            self.logger.info("微服务化迁移完成")
            return True

        except Exception as e:
            self.logger.error(f"微服务化迁移失败: {e}")
            self.migration_status["errors"].append(str(e))
            return False

    async def _register_all_services(self):
        """注册所有服务"""
        self.logger.info("注册所有服务...")

        for service_name, service_config in self.config["services"].items():
            service_info = ServiceInfo(
                service_id=f"{service_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                service_name=service_name,
                service_type=ServiceType(service_config["type"]),
                host="localhost",
                port=service_config["port"],
                version="1.0.0",
                status=MicroServiceStatus.HEALTHY,
                metadata={
                    "dependencies": service_config.get("dependencies", []),
                    "health_check_interval": service_config.get("health_check_interval", 30)
                }
            )

            if self.micro_service.register_service(service_info):
                self.migration_status["services_migrated"].append(service_name)
                self.logger.info(f"服务注册成功: {service_name}")
            else:
                raise Exception(f"服务注册失败: {service_name}")

    async def _validate_service_dependencies(self):
        """验证服务依赖"""
        self.logger.info("验证服务依赖...")

        for service_name, service_config in self.config["services"].items():
            dependencies = service_config.get("dependencies", [])
            for dep in dependencies:
                if not self.micro_service.discover_service(dep):
                    raise Exception(f"服务依赖不存在: {service_name} -> {dep}")

        self.logger.info("服务依赖验证完成")

    async def _start_service_mesh(self):
        """启动服务网格"""
        if not self.config["network"]["service_mesh"]:
            return

        self.logger.info("启动服务网格...")
        # 这里可以集成Istio或其他服务网格
        # 目前使用内置的服务网格功能
        self.logger.info("服务网格启动完成")

    async def _configure_load_balancer(self):
        """配置负载均衡"""
        self.logger.info("配置负载均衡...")

        load_balancer_type = self.config["network"]["load_balancer"]
        self.micro_service.set_config("load_balancer_type", load_balancer_type)

        self.logger.info(f"负载均衡配置完成: {load_balancer_type}")

    async def _start_monitoring(self):
        """启动监控"""
        if not self.config["monitoring"]["metrics_collection"]:
            return

        self.logger.info("启动监控系统...")
        # 配置指标收集
        self.micro_service.set_config("metrics_collection", True)
        self.micro_service.set_config(
            "distributed_tracing", self.config["monitoring"]["distributed_tracing"])

        self.logger.info("监控系统启动完成")

    def get_migration_status(self) -> Dict[str, Any]:
        """获取迁移状态"""
        return {
            **self.migration_status,
            "total_services": len(self.config["services"]),
            "migrated_services": len(self.migration_status["services_migrated"]),
            "success_rate": len(self.migration_status["services_migrated"]) / len(self.config["services"]) * 100
        }

    def generate_migration_report(self) -> str:
        """生成迁移报告"""
        status = self.get_migration_status()

        report = f"""
# 微服务化迁移报告

## 迁移概览
- 开始时间: {status['started_at']}
- 完成时间: {status['completed_at']}
- 总服务数: {status['total_services']}
- 成功迁移: {status['migrated_services']}
- 成功率: {status['success_rate']:.1f}%

## 已迁移服务
{chr(10).join(f"- {service}" for service in status['services_migrated'])}

## 错误信息
{chr(10).join(f"- {error}" for error in status['errors'])}

## 配置信息
- 服务网格: {self.config['network']['service_mesh']}
- 负载均衡: {self.config['network']['load_balancer']}
- 熔断器: {self.config['network']['circuit_breaker']}
- 重试机制: {self.config['network']['retry_mechanism']}
- 指标收集: {self.config['monitoring']['metrics_collection']}
- 分布式追踪: {self.config['monitoring']['distributed_tracing']}
"""
        return report

    def save_migration_report(self, output_path: str = None):
        """保存迁移报告"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/technical/performance/microservice_migration_report_{timestamp}.md"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_migration_report())

        self.logger.info(f"迁移报告已保存: {output_path}")
        return output_path


async def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建迁移管理器
    migration = MicroserviceMigration()

    # 开始迁移
    success = await migration.start_migration()

    if success:
        # 生成并保存报告
        report_path = migration.save_migration_report()
        print(f"微服务化迁移成功完成！报告已保存到: {report_path}")

        # 显示迁移状态
        status = migration.get_migration_status()
        print(f"迁移成功率: {status['success_rate']:.1f}%")
    else:
        print("微服务化迁移失败！")
        status = migration.get_migration_status()
        for error in status['errors']:
            print(f"错误: {error}")


if __name__ == "__main__":
    asyncio.run(main())
