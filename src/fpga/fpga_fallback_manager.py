#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FPGA降级管理器
实现FPGA与软件降级方案的自动切换
"""

import time
from typing import Optional, Dict, Callable
import logging
from .fpga_manager import FPGAManager

class FPGAFallbackManager:
    def __init__(self, fpga_manager: FPGAManager):
        self.fpga_manager = fpga_manager
        self.fallback_mode = False
        self.last_failure_time = 0
        self.health_check_interval = 30  # 健康检查间隔(秒)
        self.logger = logging.getLogger(__name__)

    def initialize(self) -> bool:
        """初始化降级管理器

        Returns:
            bool: 初始化是否成功
        """
        if not self.fpga_manager.initialize():
            self.logger.warning("FPGA管理器初始化失败，进入降级模式")
            self.fallback_mode = True
            return False

        self.fallback_mode = False
        return True

    def execute_with_fallback(self,
                           fpga_func: Callable,
                           software_func: Callable,
                           *args, **kwargs) -> any:
        """带降级功能的执行方法

        Args:
            fpga_func: FPGA实现函数
            software_func: 软件降级实现函数
            *args: 函数参数
            **kwargs: 函数关键字参数

        Returns:
            函数执行结果
        """
        if self.fallback_mode:
            return software_func(*args, **kwargs)

        try:
            # 检查FPGA健康状态
            if not self._check_fpga_health():
                self.logger.warning("FPGA设备异常，切换到降级模式")
                self.fallback_mode = True
                return software_func(*args, **kwargs)

            # 执行FPGA函数
            start_time = time.time()
            result = fpga_func(*args, **kwargs)
            elapsed = time.time() - start_time

            # 简单性能监控
            if elapsed > 0.1:  # 超过100ms认为性能下降
                self.logger.warning(f"FPGA性能下降，耗时: {elapsed:.3f}s")

            return result

        except Exception as e:
            self.logger.error(f"FPGA执行失败: {str(e)}")
            self.last_failure_time = time.time()
            self.fallback_mode = True
            return software_func(*args, **kwargs)

    def _check_fpga_health(self) -> bool:
        """检查FPGA健康状态

        Returns:
            bool: FPGA是否健康
        """
        # 检查失败时间是否在冷却期内
        if time.time() - self.last_failure_time < 60:
            return False

        # 检查设备状态
        status = self.fpga_manager.get_device_status()
        if not status or status['status'] != 'ready':
            return False

        # 检查心跳
        if time.time() - status['last_heartbeat'] > self.health_check_interval:
            return False

        return True

    def auto_recovery(self) -> bool:
        """尝试自动恢复FPGA模式

        Returns:
            bool: 是否成功恢复
        """
        if not self.fallback_mode:
            return True

        if self._check_fpga_health():
            self.logger.info("FPGA设备已恢复，退出降级模式")
            self.fallback_mode = False
            return True

        return False
