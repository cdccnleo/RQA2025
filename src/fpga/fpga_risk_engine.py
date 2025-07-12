#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FPGA加速风控引擎
实现硬件加速的风控检查功能
"""

import numpy as np
from typing import Dict, List, Optional
from src.fpga.fpga_manager import FPGAManager
from src.utils.logger import get_logger
from src.trading.risk.china.circuit_breaker import CircuitBreakerChecker
from src.trading.risk.china.price_limit import PriceLimitChecker
from src.trading.risk.china.t1_restriction import T1RestrictionChecker

logger = get_logger(__name__)

class FPGARiskEngine:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化FPGA风控引擎
        :param config: 配置参数
        """
        self.config = config
        self.fpga_manager = FPGAManager(config.get("fpga", {}))
        self.fallback_mode = False
        self.initialized = False

        # 软件降级方案
        self.software_checkers = {
            "circuit_breaker": CircuitBreakerChecker(config),
            "price_limit": PriceLimitChecker(config),
            "t1_restriction": T1RestrictionChecker(config)
        }

    def initialize(self) -> bool:
        """
        初始化FPGA设备
        :return: 是否初始化成功
        """
        try:
            if not self.fpga_manager.connect():
                logger.warning("FPGA连接失败，启用软件降级模式")
                self.fallback_mode = True
                return False

            # 加载FPGA程序
            if not self.fpga_manager.load_program("risk_engine"):
                logger.error("FPGA程序加载失败")
                self.fallback_mode = True
                return False

            # 初始化FPGA内存
            self._init_fpga_memory()

            self.initialized = True
            logger.info("FPGA风控引擎初始化成功")
            return True

        except Exception as e:
            logger.error(f"FPGA初始化异常: {str(e)}")
            self.fallback_mode = True
            return False

    def _init_fpga_memory(self):
        """初始化FPGA内存区域"""
        # 配置寄存器
        self.fpga_manager.write_register(0x00, 0x01)  # 启用所有检查

        # 设置风控参数
        self.set_circuit_breaker_thresholds([0.05, 0.07, 0.10])
        self.set_price_limit_thresholds({
            "main_board": 0.10,
            "star_market": 0.20
        })

    def check_order(self,
                   order: Dict[str, Any],
                   position: Dict[str, Any],
                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查订单风险(FPGA加速)
        :param order: 订单信息
        :param position: 持仓信息
        :param market_data: 市场数据
        :return: 检查结果
        """
        if not self.initialized or self.fallback_mode:
            return self._software_check(order, position, market_data)

        try:
            # 准备输入数据
            input_data = self._prepare_input_data(order, position, market_data)

            # 写入FPGA输入缓冲区
            self.fpga_manager.write_buffer(0, input_data)

            # 启动计算
            self.fpga_manager.start_calculation()

            # 读取结果
            result_data = self.fpga_manager.read_buffer(1)

            # 解析结果
            return self._parse_result(result_data)

        except Exception as e:
            logger.error(f"FPGA检查异常: {str(e)}")
            return self._software_check(order, position, market_data)

    def _prepare_input_data(self, order, position, market_data) -> np.ndarray:
        """
        准备FPGA输入数据
        :return: 打包后的numpy数组
        """
        # 将各种数据打包为FPGA可处理的格式
        data = np.zeros(128, dtype=np.float32)  # 假设FPGA输入缓冲区大小为128浮点数

        # 订单信息
        data[0] = 1 if order['direction'] == 'buy' else 0
        data[1] = order['price']
        data[2] = order['quantity']

        # 持仓信息
        data[3] = position['quantity']
        data[4] = position['avg_price']

        # 市场数据
        data[5] = market_data['prev_close']
        data[6] = market_data['current_price']
        data[7] = market_data['turnover_rate']

        return data

    def _parse_result(self, result_data: np.ndarray) -> Dict[str, Any]:
        """
        解析FPGA输出结果
        :param result_data: FPGA输出数据
        :return: 风险检查结果
        """
        return {
            'passed': bool(result_data[0]),
            'reason': int(result_data[1]),
            'limit_price': float(result_data[2]),
            'max_qty': int(result_data[3])
        }

    def _software_check(self, order, position, market_data) -> Dict[str, Any]:
        """
        软件降级检查
        """
        results = []
        for name, checker in self.software_checkers.items():
            result = checker.check(order, position, market_data)
            results.append(result)
            if not result['passed']:
                return result

        return {
            'passed': True,
            'reason': 0,
            'limit_price': order['price'],
            'max_qty': order['quantity']
        }

    def set_circuit_breaker_thresholds(self, thresholds: List[float]) -> bool:
        """
        设置熔断阈值(5%/7%/10%)
        :param thresholds: 阈值列表
        :return: 是否设置成功
        """
        if self.fallback_mode:
            return False

        try:
            # 写入FPGA寄存器
            addr = 0x10
            for threshold in thresholds:
                self.fpga_manager.write_register(addr, int(threshold * 10000))
                addr += 4
            return True
        except Exception as e:
            logger.error(f"设置熔断阈值失败: {str(e)}")
            return False

    def set_price_limit_thresholds(self, thresholds: Dict[str, float]) -> bool:
        """
        设置涨跌停阈值
        :param thresholds: 各板块阈值
        :return: 是否设置成功
        """
        if self.fallback_mode:
            return False

        try:
            # 主板阈值
            self.fpga_manager.write_register(
                0x20,
                int(thresholds.get('main_board', 0.10) * 10000)
            )
            # 科创板阈值
            self.fpga_manager.write_register(
                0x24,
                int(thresholds.get('star_market', 0.20) * 10000)
            )
            return True
        except Exception as e:
            logger.error(f"设置涨跌停阈值失败: {str(e)}")
            return False

    def close(self):
        """关闭FPGA资源"""
        if self.initialized and not self.fallback_mode:
            self.fpga_manager.disconnect()
            self.initialized = False
