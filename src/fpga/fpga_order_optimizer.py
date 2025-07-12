#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FPGA订单优化器
实现硬件加速的订单执行优化算法
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from src.fpga.fpga_manager import FPGAManager
from src.utils.logger import get_logger
from src.trading.execution.optimizer import BaseOrderOptimizer

logger = get_logger(__name__)

class FpgaOrderOptimizer(BaseOrderOptimizer):
    def __init__(self, config: Dict[str, Any]):
        """
        初始化FPGA订单优化器
        :param config: 配置参数
        """
        super().__init__(config)
        self.fpga_manager = FPGAManager(config.get("fpga", {}))
        self.fallback_mode = False
        self.initialized = False

        # 软件降级方案
        self.software_optimizer = BaseOrderOptimizer(config)

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
            if not self.fpga_manager.load_program("order_optimizer"):
                logger.error("FPGA程序加载失败")
                self.fallback_mode = True
                return False

            # 初始化FPGA内存
            self._init_fpga_memory()

            self.initialized = True
            logger.info("FPGA订单优化器初始化成功")
            return True

        except Exception as e:
            logger.error(f"FPGA初始化异常: {str(e)}")
            self.fallback_mode = True
            return False

    def _init_fpga_memory(self):
        """初始化FPGA内存区域"""
        # 配置寄存器
        self.fpga_manager.write_register(0x00, 0x01)  # 启用所有功能

        # 加载市场参数
        self._load_market_params()

    def _load_market_params(self):
        """加载市场参数到FPGA"""
        params = {
            "tick_size": 0.01,  # 最小价格变动单位
            "lot_size": 100,    # 最小交易单位
            "max_impact": 0.0015,  # 最大冲击成本限制
            "slippage_control": 0.5  # 滑点控制系数
        }

        # 将参数编码并写入FPGA
        encoded_params = np.array([
            params["tick_size"],
            params["lot_size"],
            params["max_impact"],
            params["slippage_control"]
        ], dtype=np.float32)

        self.fpga_manager.write_buffer(0, encoded_params)

    def optimize(self,
                order: Dict[str, Any],
                order_book: Dict[str, Any]) -> Dict[str, Any]:
        """
        优化订单执行(FPGA加速)
        :param order: 原始订单
        :param order_book: 订单簿数据
        :return: 优化后的订单
        """
        if not self.initialized or self.fallback_mode:
            return self.software_optimizer.optimize(order, order_book)

        try:
            # 准备输入数据
            input_data = self._prepare_input_data(order, order_book)

            # 写入FPGA输入缓冲区
            self.fpga_manager.write_buffer(1, input_data['order'])
            self.fpga_manager.write_buffer(2, input_data['order_book'])

            # 启动计算
            self.fpga_manager.start_calculation()

            # 读取结果
            result_data = self.fpga_manager.read_buffer(3)

            # 解析结果
            return self._parse_result(result_data, order)

        except Exception as e:
            logger.error(f"FPGA优化异常: {str(e)}")
            return self.software_optimizer.optimize(order, order_book)

    def _prepare_input_data(self,
                          order: Dict[str, Any],
                          order_book: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        准备FPGA输入数据
        :param order: 原始订单
        :param order_book: 订单簿数据
        :return: 打包后的输入数据
        """
        # 转换订单数据
        order_data = np.zeros(8, dtype=np.float32)  # 假设订单数据需要8个float32
        order_data[0] = order.get('price', 0.0)
        order_data[1] = order.get('quantity', 0)
        order_data[2] = order.get('urgency', 0.5)  # 默认紧急度0.5
        order_data[3] = order.get('side') == 'buy' and 1.0 or -1.0  # 买卖方向

        # 转换订单簿数据
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])

        # 假设FPGA需要前5档行情
        book_depth = 5
        book_data = np.zeros(book_depth * 4, dtype=np.float32)  # 每档价格+数量

        for i in range(book_depth):
            # 买盘
            if i < len(bids):
                book_data[i*2] = bids[i][0]  # 价格
                book_data[i*2+1] = bids[i][1]  # 数量

            # 卖盘
            if i < len(asks):
                book_data[book_depth*2 + i*2] = asks[i][0]  # 价格
                book_data[book_depth*2 + i*2+1] = asks[i][1]  # 数量

        return {
            'order': order_data,
            'order_book': book_data
        }

    def _parse_result(self,
                     result_data: np.ndarray,
                     original_order: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析FPGA输出结果
        :param result_data: FPGA输出数据
        :param original_order: 原始订单
        :return: 优化后的订单
        """
        optimized_order = original_order.copy()

        # 假设结果数据包含优化后的价格和数量
        optimized_order['price'] = float(result_data[0])
        optimized_order['quantity'] = int(result_data[1])
        optimized_order['strategy'] = 'fpga_optimized'

        # 添加优化元数据
        optimized_order['metadata'] = {
            'impact_cost': float(result_data[2]),
            'slippage': float(result_data[3]),
            'execution_prob': float(result_data[4])
        }

        return optimized_order

    def batch_optimize(self,
                      orders: List[Dict[str, Any]],
                      order_books: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量优化订单执行(FPGA加速)
        :param orders: 原始订单列表
        :param order_books: 订单簿数据列表
        :return: 优化后的订单列表
        """
        if not self.initialized or self.fallback_mode:
            return self.software_optimizer.batch_optimize(orders, order_books)

        try:
            # 准备批量输入数据
            batch_size = len(orders)
            order_data = np.zeros((batch_size, 8), dtype=np.float32)
            book_data = np.zeros((batch_size, 20), dtype=np.float32)  # 5档买卖

            for i in range(batch_size):
                input_data = self._prepare_input_data(orders[i], order_books[i])
                order_data[i] = input_data['order']
                book_data[i] = input_data['order_book']

            # 写入FPGA批量输入缓冲区
            self.fpga_manager.write_batch_buffer(1, order_data)
            self.fpga_manager.write_batch_buffer(2, book_data)

            # 启动批量计算
            self.fpga_manager.start_batch_calculation(batch_size)

            # 读取批量结果
            batch_output = self.fpga_manager.read_batch_buffer(3, batch_size)

            # 解析批量结果
            results = []
            for i in range(batch_size):
                results.append(self._parse_result(batch_output[i], orders[i]))

            return results

        except Exception as e:
            logger.error(f"FPGA批量优化异常: {str(e)}")
            return self.software_optimizer.batch_optimize(orders, order_books)

    def close(self):
        """关闭FPGA资源"""
        if self.initialized and not self.fallback_mode:
            self.fpga_manager.disconnect()
            self.initialized = False
