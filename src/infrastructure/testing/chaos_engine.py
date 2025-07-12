#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQA2025 混沌工程测试框架
用于模拟生产环境故障，验证系统健壮性
"""

import random
import time
import subprocess
from enum import Enum, auto
from typing import Optional, Dict, List
import logging
from dataclasses import dataclass
import docker
import psutil

logger = logging.getLogger(__name__)

class ChaosError(Exception):
    """混沌工程专用异常"""
    pass

class FaultType(Enum):
    """故障类型枚举"""
    NETWORK_PARTITION = auto()
    FPGA_FAILURE = auto()
    HIGH_CPU = auto()
    MEMORY_LEAK = auto()
    DISK_FULL = auto()

@dataclass
class ChaosReport:
    """混沌测试报告"""
    fault_type: FaultType
    start_time: float
    end_time: float
    affected_components: List[str]
    recovery_time: float
    is_success: bool

class ChaosEngine:
    def __init__(self, enable_production: bool = False):
        """
        初始化混沌引擎
        :param enable_production: 是否允许在生产环境运行(默认为False)
        """
        self.enable_production = enable_production
        self.docker_client = docker.from_env()
        self.active_faults = set()

        # 安全保护机制
        self.safeguard = ChaosSafeguard()

    def simulate_network_partition(self, duration: int = 60, target_services: Optional[List[str]] = None) -> ChaosReport:
        """
        模拟网络分区故障
        :param duration: 故障持续时间(秒)
        :param target_services: 目标服务列表(为空则随机选择)
        :return: 测试报告
        """
        if not self.safeguard.check_environment():
            raise ChaosError("Safety check failed: not allowed in production")

        logger.warning("🚨 开始模拟网络分区故障...")
        report = ChaosReport(
            fault_type=FaultType.NETWORK_PARTITION,
            start_time=time.time(),
            end_time=0,
            affected_components=target_services or [],
            recovery_time=0,
            is_success=False
        )

        try:
            # 1. 选择目标容器
            containers = self._select_target_containers(target_services)
            if not containers:
                raise ChaosError("No containers available for network partition")

            # 2. 隔离网络
            for container in containers:
                self._isolate_container_network(container)
                self.active_faults.add((FaultType.NETWORK_PARTITION, container.id))

            # 3. 等待故障持续时间
            time.sleep(duration)

            # 4. 恢复网络
            for container in containers:
                self._restore_container_network(container)
                self.active_faults.discard((FaultType.NETWORK_PARTITION, container.id))

            # 生成报告
            report.end_time =time.time()
            report.recovery_time = report.end_time - report.start_time
            report.is_success = True
            logger.info(f"✅ 网络分区测试完成，隔离时长: {duration}秒")

        except Exception as e:
            logger.error(f"网络分区测试失败: {str(e)}")
            self._emergency_recovery()
            raise

        return report

    def simulate_fpga_failure(self, duration: int = 30, failure_mode: str = "complete") -> ChaosReport:
        """
        模拟FPGA设备故障
        :param duration: 故障持续时间(秒)
        :param failure_mode: 故障模式(complete/partial/noise)
        :return: 测试报告
        """
        if not self.safeguard.check_environment():
            raise ChaosError("Safety check failed: not allowed in production")

        logger.warning("🚨 开始模拟FPGA故障...")
        report = ChaosReport(
            fault_type=FaultType.FPGA_FAILURE,
            start_time=time.time(),
            end_time=0,
            affected_components=["fpga_accelerator"],
            recovery_time=0,
            is_success=False
        )

        try:
            # 1. 检查FPGA设备状态
            if not self._check_fpga_available():
                raise ChaosError("FPGA device not available")

            # 2. 根据故障模式模拟不同故障
            if failure_mode == "complete":
                self._simulate_fpga_crash()
            elif failure_mode == "partial":
                self._simulate_fpga_partial_failure()
            elif failure_mode == "noise":
                self._simulate_fpga_noise()
            else:
                raise ChaosError(f"Unknown failure mode: {failure_mode}")

            self.active_faults.add((FaultType.FPGA_FAILURE, "fpga"))

            # 3. 等待故障持续时间
            time.sleep(duration)

            # 4. 恢复FPGA
            self._restore_fpga()
            self.active_faults.discard((FaultType.FPGA_FAILURE, "fpga"))

            # 生成报告
            report.end_time = time.time()
            report.recovery_time = report.end_time - report.start_time
            report.is_success = True
            logger.info(f"✅ FPGA故障测试完成，故障模式: {failure_mode}, 时长: {duration}秒")

        except Exception as e:
            logger.error(f"FPGA故障测试失败: {str(e)}")
            self._emergency_recovery()
            raise

        return report

    def _select_target_containers(self, target_services: Optional[List[str]] = None) -> List:
        """选择目标容器"""
        containers = self.docker_client.containers.list()

        if target_services:
            # 过滤指定服务
            return [c for c in containers if any(s in c.name for s in target_services)]
        else:
            # 随机选择1-2个非关键容器
            non_critical = [c for c in containers if "redis" not in c.name and "db" not in c.name]
            return random.sample(non_critical, min(2, len(non_critical)))

    def _isolate_container_network(self, container) -> None:
        """隔离容器网络"""
        # 使用Linux网络命名空间隔离
        cmd = f"sudo iptables -A DOCKER -s {container.id[:12]} -j DROP"
        subprocess.run(cmd, shell=True, check=True)
        logger.debug(f"已隔离容器网络: {container.name}")

    def _restore_container_network(self, container) -> None:
        """恢复容器网络"""
        cmd = f"sudo iptables -D DOCKER -s {container.id[:12]} -j DROP"
        subprocess.run(cmd, shell=True, check=True)
        logger.debug(f"已恢复容器网络: {container.name}")

    def _check_fpga_available(self) -> bool:
        """检查FPGA设备是否可用"""
        try:
            # 检查FPGA设备文件是否存在
            return subprocess.run(["ls", "/dev/fpga0"], capture_output=True).returncode == 0
        except:
            return False

    def _simulate_fpga_crash(self) -> None:
        """模拟FPGA完全故障"""
        # 卸载FPGA驱动
        subprocess.run(["sudo", "rmmod", "fpga_driver"], check=True)
        logger.debug("已模拟FPGA完全故障")

    def _simulate_fpga_partial_failure(self) -> None:
        """模拟FPGA部分故障"""
        # 降低FPGA时钟频率
        subprocess.run(["sudo", "fpga-clk", "set", "50"], check=True)
        logger.debug("已模拟FPGA部分故障(降频)")

    def _simulate_fpga_noise(self) -> None:
        """模拟FPGA噪声干扰"""
        # 注入随机错误
        subprocess.run(["sudo", "fpga-error", "inject", "random"], check=True)
        logger.debug("已模拟FPGA噪声干扰")

    def _restore_fpga(self) -> None:
        """恢复FPGA设备"""
        # 重新加载驱动
        subprocess.run(["sudo", "modprobe", "fpga_driver"], check=True)
        # 恢复时钟频率
        subprocess.run(["sudo", "fpga-clk", "set", "100"], check=True)
        logger.debug("已恢复FPGA设备")

    def _emergency_recovery(self) -> None:
        """紧急恢复所有故障"""
        logger.warning("⚠️ 执行紧急恢复...")

        # 恢复网络隔离
        for fault_type, target in list(self.active_faults):
            if fault_type == FaultType.NETWORK_PARTITION:
                container = self.docker_client.containers.get(target)
                self._restore_container_network(container)
            elif fault_type == FaultType.FPGA_FAILURE:
                self._restore_fpga()

        self.active_faults.clear()

    def list_available_faults(self) -> Dict[str, str]:
        """获取可用的故障模拟类型"""
        return {
            "network_partition": "模拟网络分区",
            "fpga_failure": "模拟FPGA故障",
            "high_cpu": "模拟CPU过载(待实现)",
            "memory_leak": "模拟内存泄漏(待实现)",
            "disk_full": "模拟磁盘满(待实现)"
        }


class ChaosSafeguard:
    """混沌工程安全保护机制"""

    def check_environment(self) -> bool:
        """检查是否允许执行混沌测试"""
        # 1. 检查是否在生产环境
        if self._is_production():
            logger.critical("拒绝在生产环境执行混沌测试!")
            return False

        # 2. 检查系统负载
        if self._high_system_load():
            logger.warning("系统负载过高，暂不执行混沌测试")
            return False

        # 3. 检查关键服务状态
        if not self._critical_services_ok():
            logger.warning("关键服务异常，暂不执行混沌测试")
            return False

        return True

    def _is_production(self) -> bool:
        """检查是否生产环境"""
        try:
            with open("/etc/environment") as f:
                return "production" in f.read().lower()
        except:
            return False

    def _high_system_load(self) -> bool:
        """检查系统负载"""
        load = psutil.getloadavg()[0] / psutil.cpu_count()
        return load > 0.7

    def _critical_services_ok(self) -> bool:
        """检查关键服务状态"""
        try:
            client = docker.from_env()
            critical_services = ["redis", "postgres", "fpga_manager"]

            for service in critical_services:
                containers = client.containers.list(filters={"name": service})
                if not containers or containers[0].status != "running":
                    return False

            return True
        except:
            return False


if __name__ == "__main__":
    # 示例用法
    logging.basicConfig(level=logging.INFO)

    engine = ChaosEngine()
    print("可用的故障模拟类型:", engine.list_available_faults())

    # 模拟网络分区(随机选择2个非关键容器，持续30秒)
    report = engine.simulate_network_partition(duration=30)
    print("测试报告:", report)

    # 模拟FPGA完全故障，持续20秒
    report = engine.simulate_fpga_failure(duration=20, failure_mode="complete")
    print("测试报告:", report)
