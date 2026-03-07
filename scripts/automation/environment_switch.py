#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境切换脚本
支持多环境配置管理、环境状态检查、配置同步等功能
"""
import os
import json
import requests
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EnvironmentInfo:
    """环境信息"""
    name: str
    api_base: str
    username: str
    password: str
    description: str = ""
    is_active: bool = False
    last_check: Optional[datetime] = None
    status: str = "unknown"


class EnvironmentSwitchManager:
    """环境切换管理器"""

    def __init__(self, config_file: str = "environment_config.json"):
        self.config_file = config_file
        self.environments = {}
        self.current_environment = None
        self.load_environments()

    def load_environments(self):
        """加载环境配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.environments = {
                        name: EnvironmentInfo(**env_config)
                        for name, env_config in config.get("environments", {}).items()
                    }
                    self.current_environment = config.get("current_environment")
            else:
                # 默认环境配置
                self.environments = {
                    "development": EnvironmentInfo(
                        name="development",
                        api_base="http://localhost:8080",
                        username="dev_user",
                        password="dev_pass",
                        description="开发环境"
                    ),
                    "staging": EnvironmentInfo(
                        name="staging",
                        api_base="http://staging-config.example.com:8080",
                        username="staging_user",
                        password="staging_pass",
                        description="预发布环境"
                    ),
                    "production": EnvironmentInfo(
                        name="production",
                        api_base="http://prod-config.example.com:8080",
                        username="prod_user",
                        password="prod_pass",
                        description="生产环境"
                    )
                }
                self.current_environment = "development"
                self.save_environments()

        except Exception as e:
            logger.error(f"加载环境配置失败: {e}")

    def save_environments(self):
        """保存环境配置"""
        try:
            config = {
                "environments": {
                    name: {
                        "name": env.name,
                        "api_base": env.api_base,
                        "username": env.username,
                        "password": env.password,
                        "description": env.description,
                        "is_active": env.is_active,
                        "last_check": env.last_check.isoformat() if env.last_check else None,
                        "status": env.status
                    }
                    for name, env in self.environments.items()
                },
                "current_environment": self.current_environment
            }

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 环境配置已保存: {self.config_file}")

        except Exception as e:
            logger.error(f"保存环境配置失败: {e}")

    def list_environments(self) -> List[EnvironmentInfo]:
        """列出所有环境"""
        return list(self.environments.values())

    def get_environment(self, name: str) -> Optional[EnvironmentInfo]:
        """获取指定环境"""
        return self.environments.get(name)

    def add_environment(self, name: str, api_base: str, username: str, password: str, description: str = ""):
        """添加新环境"""
        if name in self.environments:
            raise ValueError(f"环境 {name} 已存在")

        self.environments[name] = EnvironmentInfo(
            name=name,
            api_base=api_base,
            username=username,
            password=password,
            description=description
        )

        self.save_environments()
        logger.info(f"✅ 环境 {name} 已添加")

    def remove_environment(self, name: str):
        """删除环境"""
        if name not in self.environments:
            raise ValueError(f"环境 {name} 不存在")

        if self.current_environment == name:
            self.current_environment = None

        del self.environments[name]
        self.save_environments()
        logger.info(f"✅ 环境 {name} 已删除")

    def switch_environment(self, name: str) -> bool:
        """切换环境"""
        if name not in self.environments:
            logger.error(f"环境 {name} 不存在")
            return False

        # 检查环境状态
        if not self.check_environment_status(name):
            logger.error(f"环境 {name} 不可用")
            return False

        # 切换环境
        old_environment = self.current_environment
        self.current_environment = name

        # 更新环境状态
        self.environments[name].is_active = True
        self.environments[name].last_check = datetime.now()
        self.environments[name].status = "active"

        if old_environment and old_environment in self.environments:
            self.environments[old_environment].is_active = False

        self.save_environments()

        logger.info(f"✅ 已切换到环境: {name}")
        return True

    def check_environment_status(self, name: str) -> bool:
        """检查环境状态"""
        if name not in self.environments:
            return False

        env = self.environments[name]

        try:
            response = requests.get(f"{env.api_base}/api/health", timeout=10)
            is_healthy = response.status_code == 200

            env.last_check = datetime.now()
            env.status = "healthy" if is_healthy else "unhealthy"

            return is_healthy

        except Exception as e:
            env.last_check = datetime.now()
            env.status = "error"
            logger.error(f"检查环境 {name} 状态失败: {e}")
            return False

    def check_all_environments(self):
        """检查所有环境状态"""
        logger.info("检查所有环境状态...")

        for name, env in self.environments.items():
            is_healthy = self.check_environment_status(name)
            status_icon = "✅" if is_healthy else "❌"
            logger.info(f"{status_icon} {name}: {env.status}")

    def get_current_environment(self) -> Optional[EnvironmentInfo]:
        """获取当前环境"""
        if self.current_environment:
            return self.environments.get(self.current_environment)
        return None

    def sync_config_between_environments(self, source_env: str, target_env: str) -> bool:
        """在环境间同步配置"""
        try:
            source = self.get_environment(source_env)
            target = self.get_environment(target_env)

            if not source or not target:
                logger.error("源环境或目标环境不存在")
                return False

            # 检查环境状态
            if not self.check_environment_status(source_env):
                logger.error(f"源环境 {source_env} 不可用")
                return False

            if not self.check_environment_status(target_env):
                logger.error(f"目标环境 {target_env} 不可用")
                return False

            # 登录源环境
            source_session = self._login_environment(source)
            if not source_session:
                logger.error(f"登录源环境 {source_env} 失败")
                return False

            # 登录目标环境
            target_session = self._login_environment(target)
            if not target_session:
                logger.error(f"登录目标环境 {target_env} 失败")
                return False

            # 获取源环境配置
            source_config = self._get_environment_config(source, source_session)
            if not source_config:
                logger.error(f"获取源环境 {source_env} 配置失败")
                return False

            # 部署到目标环境
            success = self._deploy_config_to_environment(target, target_session, source_config)

            if success:
                logger.info(f"✅ 配置从 {source_env} 同步到 {target_env} 成功")
            else:
                logger.error(f"❌ 配置从 {source_env} 同步到 {target_env} 失败")

            return success

        except Exception as e:
            logger.error(f"环境间配置同步失败: {e}")
            return False

    def _login_environment(self, env: EnvironmentInfo) -> Optional[str]:
        """登录环境"""
        try:
            response = requests.post(f"{env.api_base}/api/login", json={
                "username": env.username,
                "password": env.password
            }, timeout=10)

            data = response.json()
            if data.get("success"):
                return data["session_id"]
            else:
                return None

        except Exception:
            return None

    def _get_environment_config(self, env: EnvironmentInfo, session_id: str) -> Optional[Dict]:
        """获取环境配置"""
        try:
            response = requests.get(f"{env.api_base}/api/config",
                                    headers={"Authorization": f"Bearer {session_id}"},
                                    timeout=10)

            if response.status_code == 200:
                data = response.json()
                return data.get("config", {})
            else:
                return None

        except Exception:
            return None

    def _deploy_config_to_environment(self, env: EnvironmentInfo, session_id: str, config: Dict) -> bool:
        """部署配置到环境"""
        try:
            response = requests.put(f"{env.api_base}/api/config/batch",
                                    headers={"Authorization": f"Bearer {session_id}"},
                                    json={"config": config},
                                    timeout=30)

            data = response.json()
            return data.get("success", False)

        except Exception:
            return False

    def export_environment_config(self, name: str, output_file: str):
        """导出环境配置"""
        env = self.get_environment(name)
        if not env:
            logger.error(f"环境 {name} 不存在")
            return

        try:
            session_id = self._login_environment(env)
            if not session_id:
                logger.error(f"登录环境 {name} 失败")
                return

            config = self._get_environment_config(env, session_id)
            if not config:
                logger.error(f"获取环境 {name} 配置失败")
                return

            # 保存配置到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 环境 {name} 配置已导出到: {output_file}")

        except Exception as e:
            logger.error(f"导出环境配置失败: {e}")

    def import_environment_config(self, name: str, config_file: str):
        """导入环境配置"""
        env = self.get_environment(name)
        if not env:
            logger.error(f"环境 {name} 不存在")
            return

        try:
            # 读取配置文件
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            session_id = self._login_environment(env)
            if not session_id:
                logger.error(f"登录环境 {name} 失败")
                return

            # 部署配置
            success = self._deploy_config_to_environment(env, session_id, config)

            if success:
                logger.info(f"✅ 配置已导入到环境 {name}")
            else:
                logger.error(f"❌ 导入配置到环境 {name} 失败")

        except Exception as e:
            logger.error(f"导入环境配置失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="环境切换管理工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 列出环境
    list_parser = subparsers.add_parser('list', help='列出所有环境')

    # 切换环境
    switch_parser = subparsers.add_parser('switch', help='切换环境')
    switch_parser.add_argument('environment', help='目标环境名称')

    # 检查环境状态
    check_parser = subparsers.add_parser('check', help='检查环境状态')
    check_parser.add_argument('--all', action='store_true', help='检查所有环境')
    check_parser.add_argument('environment', nargs='?', help='指定环境名称')

    # 添加环境
    add_parser = subparsers.add_parser('add', help='添加新环境')
    add_parser.add_argument('name', help='环境名称')
    add_parser.add_argument('--api-base', required=True, help='API服务地址')
    add_parser.add_argument('--username', required=True, help='用户名')
    add_parser.add_argument('--password', required=True, help='密码')
    add_parser.add_argument('--description', help='环境描述')

    # 删除环境
    remove_parser = subparsers.add_parser('remove', help='删除环境')
    remove_parser.add_argument('name', help='环境名称')

    # 同步配置
    sync_parser = subparsers.add_parser('sync', help='同步环境配置')
    sync_parser.add_argument('source', help='源环境')
    sync_parser.add_argument('target', help='目标环境')

    # 导出配置
    export_parser = subparsers.add_parser('export', help='导出环境配置')
    export_parser.add_argument('environment', help='环境名称')
    export_parser.add_argument('--output', required=True, help='输出文件路径')

    # 导入配置
    import_parser = subparsers.add_parser('import', help='导入环境配置')
    import_parser.add_argument('environment', help='环境名称')
    import_parser.add_argument('--config-file', required=True, help='配置文件路径')

    args = parser.parse_args()

    # 创建环境管理器
    manager = EnvironmentSwitchManager()

    try:
        if args.command == 'list':
            environments = manager.list_environments()
            current = manager.get_current_environment()

            print("\n📋 环境列表:")
            print("="*60)
            for env in environments:
                status_icon = "🟢" if env.is_active else "⚪"
                current_mark = " (当前)" if current and current.name == env.name else ""
                print(f"{status_icon} {env.name}{current_mark}")
                print(f"   地址: {env.api_base}")
                print(f"   描述: {env.description}")
                print(f"   状态: {env.status}")
                if env.last_check:
                    print(f"   最后检查: {env.last_check.strftime('%Y-%m-%d %H:%M:%S')}")
                print()

        elif args.command == 'switch':
            if manager.switch_environment(args.environment):
                print(f"\n✅ 已切换到环境: {args.environment}")
            else:
                print(f"\n❌ 切换到环境 {args.environment} 失败")

        elif args.command == 'check':
            if args.all:
                manager.check_all_environments()
            elif args.environment:
                if manager.check_environment_status(args.environment):
                    print(f"\n✅ 环境 {args.environment} 状态正常")
                else:
                    print(f"\n❌ 环境 {args.environment} 状态异常")
            else:
                print("请指定环境名称或使用 --all 检查所有环境")

        elif args.command == 'add':
            manager.add_environment(
                name=args.name,
                api_base=args.api_base,
                username=args.username,
                password=args.password,
                description=args.description or ""
            )
            print(f"\n✅ 环境 {args.name} 已添加")

        elif args.command == 'remove':
            manager.remove_environment(args.name)
            print(f"\n✅ 环境 {args.name} 已删除")

        elif args.command == 'sync':
            if manager.sync_config_between_environments(args.source, args.target):
                print(f"\n✅ 配置从 {args.source} 同步到 {args.target} 成功")
            else:
                print(f"\n❌ 配置同步失败")

        elif args.command == 'export':
            manager.export_environment_config(args.environment, args.output)

        elif args.command == 'import':
            manager.import_environment_config(args.environment, args.config_file)

        else:
            parser.print_help()

    except Exception as e:
        logger.error(f"操作失败: {e}")
        print(f"\n❌ 操作失败: {e}")


if __name__ == "__main__":
    main()
