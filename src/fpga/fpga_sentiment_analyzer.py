#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FPGA情感分析加速器
实现硬件加速的中文财经文本情感分析
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from src.fpga.fpga_manager import FPGAManager
from src.utils.logger import get_logger
from src.features.sentiment.analyzer import BaseSentimentAnalyzer

logger = get_logger(__name__)

class FpgaSentimentAnalyzer(BaseSentimentAnalyzer):
    def __init__(self, config: Dict[str, Any]):
        """
        初始化FPGA情感分析器
        :param config: 配置参数
        """
        super().__init__(config)
        self.fpga_manager = FPGAManager(config.get("fpga", {}))
        self.fallback_mode = False
        self.initialized = False

        # 软件降级方案
        self.software_analyzer = BaseSentimentAnalyzer(config)

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
            if not self.fpga_manager.load_program("sentiment_analyzer"):
                logger.error("FPGA程序加载失败")
                self.fallback_mode = True
                return False

            # 初始化FPGA内存
            self._init_fpga_memory()

            self.initialized = True
            logger.info("FPGA情感分析器初始化成功")
            return True

        except Exception as e:
            logger.error(f"FPGA初始化异常: {str(e)}")
            self.fallback_mode = True
            return False

    def _init_fpga_memory(self):
        """初始化FPGA内存区域"""
        # 配置寄存器
        self.fpga_manager.write_register(0x00, 0x01)  # 启用所有功能

        # 加载财经关键词词典
        self._load_finance_dictionary()

    def _load_finance_dictionary(self):
        """加载财经领域关键词到FPGA"""
        finance_terms = [
            # 政策关键词
            "货币政策", "财政政策", "降准", "降息", "LPR",
            # 行业术语
            "市盈率", "市净率", "ROE", "毛利率", "现金流",
            # 情感词
            "超预期", "不及预期", "利好", "利空", "看好", "看空"
        ]

        # 将关键词编码并写入FPGA
        encoded_terms = [term.encode('utf-8') for term in finance_terms]
        self.fpga_manager.write_dictionary(0, encoded_terms)

    def analyze(self,
               text_data: str,
               features: List[str] = None) -> Dict[str, float]:
        """
        分析文本情感(FPGA加速)
        :param text_data: 待分析文本
        :param features: 需要提取的特征列表
        :return: 情感分析结果
        """
        if not self.initialized or self.fallback_mode:
            return self.software_analyzer.analyze(text_data, features)

        try:
            # 准备输入数据
            input_data = self._prepare_input_data(text_data)

            # 写入FPGA输入缓冲区
            self.fpga_manager.write_buffer(0, input_data)

            # 启动计算
            self.fpga_manager.start_calculation()

            # 读取结果
            result_data = self.fpga_manager.read_buffer(1)

            # 解析结果
            return self._parse_result(result_data, features)

        except Exception as e:
            logger.error(f"FPGA分析异常: {str(e)}")
            return self.software_analyzer.analyze(text_data, features)

    def _prepare_input_data(self, text_data: str) -> np.ndarray:
        """
        准备FPGA输入数据
        :param text_data: 待分析文本
        :return: 打包后的numpy数组
        """
        # 将文本数据编码为FPGA可处理的格式
        encoded_text = text_data.encode('utf-8')
        data = np.zeros(1024, dtype=np.uint8)  # 假设FPGA输入缓冲区大小为1024字节

        # 填充文本数据
        data[:len(encoded_text)] = list(encoded_text)

        return data

    def _parse_result(self,
                     result_data: np.ndarray,
                     requested_features: List[str] = None) -> Dict[str, float]:
        """
        解析FPGA输出结果
        :param result_data: FPGA输出数据
        :param requested_features: 请求的特征列表
        :return: 情感分析结果
        """
        default_features = ["sentiment_score", "policy_impact", "industry_relevance"]
        features = requested_features if requested_features else default_features

        result = {
            "sentiment_score": float(result_data[0]),
            "policy_impact": float(result_data[1]),
            "industry_relevance": float(result_data[2]),
            "positive_words": int(result_data[3]),
            "negative_words": int(result_data[4]),
            "finance_terms": int(result_data[5])
        }

        # 只返回请求的特征
        return {k: v for k, v in result.items() if k in features}

    def batch_analyze(self,
                     text_list: List[str],
                     features: List[str] = None) -> List[Dict[str, float]]:
        """
        批量分析文本情感(FPGA加速)
        :param text_list: 文本列表
        :param features: 需要提取的特征列表
        :return: 情感分析结果列表
        """
        if not self.initialized or self.fallback_mode:
            return self.software_analyzer.batch_analyze(text_list, features)

        try:
            # 准备批量输入数据
            batch_input = np.zeros((len(text_list), 1024), dtype=np.uint8)
            for i, text in enumerate(text_list):
                encoded_text = text.encode('utf-8')
                batch_input[i, :len(encoded_text)] = list(encoded_text)

            # 写入FPGA批量输入缓冲区
            self.fpga_manager.write_batch_buffer(0, batch_input)

            # 启动批量计算
            self.fpga_manager.start_batch_calculation(len(text_list))

            # 读取批量结果
            batch_output = self.fpga_manager.read_batch_buffer(1, len(text_list))

            # 解析批量结果
            results = []
            for result in batch_output:
                results.append(self._parse_result(result, features))

            return results

        except Exception as e:
            logger.error(f"FPGA批量分析异常: {str(e)}")
            return self.software_analyzer.batch_analyze(text_list, features)

    def close(self):
        """关闭FPGA资源"""
        if self.initialized and not self.fallback_mode:
            self.fpga_manager.disconnect()
            self.initialized = False
