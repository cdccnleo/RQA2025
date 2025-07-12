#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQA2025 测试环境管理工具
用于自动化部署和清理测试环境
"""

import os
import subprocess
import docker
import time
from typing import Dict, List

class TestEnvironmentManager:
    def __init__(self, config_path="config/test_env_config.json"):
        """
        初始化测试环境管理器
        :param config_path: 环境配置文件路径
        """
        self.config = self._load_config(config_path)
        self.docker_client = docker.from_env()
        self.containers = {}

    def setup(self):
        """部署测试环境"""
        print("🚀 开始部署测试环境...")

        # 1. 启动数据库
        self._start_database()

        # 2. 启动Redis
        self._start_redis()

        # 3. 启动FPGA模拟器
        if self.config.get("enable_fpga", False):
            self._start_fpga_emulator()

        # 4. 等待服务就绪
        self._wait_for_services()

        print("✅ 测试环境部署完成")

    def teardown(self):
        """清理测试环境"""
        print("🧹 开始清理测试环境...")

        # 停止所有容器
        for name, container in self.containers.items():
            print(f"  正在停止 {name}...")
            container.stop()
            container.remove()

        # 清理临时文件
        self._clean_temp_files()

        print("✅ 测试环境清理完成")

    def _start_database(self):
        """启动测试数据库"""
        db_config = self.config["database"]
        print(f"  启动 {db_config['type']} 数据库...")

        volumes = {}
        if db_config.get("data_volume"):
            volumes[db_config["data_volume"]] = {
                "bind": "/var/lib/postgresql/data",
                "mode": "rw"
            }

        container = self.docker_client.containers.run(
            image=db_config["image"],
            name="rqa2025_test_db",
            ports={f"{db_config['port']}/tcp": db_config["port"]},
            environment={
                "POSTGRES_USER": db_config["user"],
                "POSTGRES_PASSWORD": db_config["password"],
                "POSTGRES_DB": db_config["dbname"]
            },
            volumes=volumes,
            detach=True
        )

        self.containers["database"] = container

    def _start_redis(self):
        """启动Redis缓存"""
        redis_config = self.config["redis"]
        print("  启动 Redis...")

        container = self.docker_client.containers.run(
            image=redis_config["image"],
            name="rqa2025_test_redis",
            ports={f"{redis_config['port']}/tcp": redis_config["port"]},
            detach=True
        )

        self.containers["redis"] = container

    def _start_fpga_emulator(self):
        """启动FPGA模拟器"""
        fpga_config = self.config["fpga"]
        print("  启动 FPGA 模拟器...")

        # FPGA模拟器需要特殊权限
        container = self.docker_client.containers.run(
            image=fpga_config["image"],
            name="rqa2025_test_fpga",
            privileged=True,
            devices=["/dev/fpga0:/dev/fpga0"],
            detach=True
        )

        self.containers["fpga"] = container

    def _wait_for_services(self):
        """等待服务就绪"""
        print("⏳ 等待服务启动...")

        services = [
            ("database", self.config["database"]["port"], "PostgreSQL"),
            ("redis", self.config["redis"]["port"], "Redis")
        ]

        if self.config.get("enable_fpga", False):
            services.append(("fpga", self.config["fpga"]["port"], "FPGA Emulator"))

        for name, port, service_name in services:
            print(f"  等待 {service_name} 就绪...", end="", flush=True)

            start_time = time.time()
            while True:
                try:
                    # 尝试连接服务
                    if self._check_service_ready(name, port):
                        print(" ✅")
                        break

                    # 超时检查
                    if time.time() - start_time > 120:  # 2分钟超时
                        raise TimeoutError(f"{service_name} 启动超时")

                    time.sleep(2)
                    print(".", end="", flush=True)
                except Exception as e:
                    print(f"\n❌ {service_name} 启动失败: {str(e)}")
                    raise

    def _check_service_ready(self, service_name: str, port: int) -> bool:
        """检查服务是否就绪"""
        if service_name == "database":
            # 检查PostgreSQL是否就绪
            cmd = [
                "pg_isready",
                "-h", "localhost",
                "-p", str(port),
                "-U", self.config["database"]["user"]
            ]

            env = os.environ.copy()
            env["PGPASSWORD"] = self.config["database"]["password"]

            result = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            return result.returncode == 0

        elif service_name == "redis":
            # 检查Redis是否就绪
            import redis
            try:
                r = redis.Redis(
                    host="localhost",
                    port=port,
                    socket_timeout=1
                )
                return r.ping()
            except:
                return False

        elif service_name == "fpga":
            # 检查FPGA模拟器是否就绪
            # 这里需要根据实际FPGA模拟器实现
            return True

        return False

    def _clean_temp_files(self):
        """清理临时文件"""
        temp_dirs = [
            "/tmp/rqa2025_test",
            "tests/temp_data"
        ]

        for dir_path in temp_dirs:
            if os.path.exists(dir_path):
                print(f"  清理临时目录 {dir_path}...")
                subprocess.run(["rm", "-rf", dir_path])

    def _load_config(self, config_path: str) -> Dict:
        """加载环境配置"""
        import json
        with open(config_path, "r") as f:
            return json.load(f)

    def __enter__(self):
        """上下文管理器入口"""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.teardown()


if __name__ == "__main__":
    # 示例用法
    with TestEnvironmentManager() as env:
        print("测试环境已就绪，开始执行测试...")
        # 在这里执行测试
        input("按Enter键结束测试并清理环境...")
