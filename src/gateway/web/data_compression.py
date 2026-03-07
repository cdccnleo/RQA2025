"""
数据压缩模块
用于减少存储空间和传输量
支持JSON数据压缩、浮点数数据压缩等
"""

import logging
import json
import zlib
import base64
from typing import Dict, List, Any, Optional
import struct

# 使用统一日志系统
logger = logging.getLogger(__name__)


class DataCompressor:
    """
    数据压缩器
    提供多种数据压缩方法
    """
    
    @staticmethod
    def compress_json(data: Any, level: int = 6) -> str:
        """
        压缩JSON数据
        
        Args:
            data: 要压缩的数据
            level: 压缩级别，1-9，默认6
        
        Returns:
            压缩后的Base64编码字符串
        """
        try:
            # 转换为JSON字符串
            json_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
            # 压缩
            compressed = zlib.compress(json_str.encode('utf-8'), level=level)
            # 编码为Base64
            return base64.b64encode(compressed).decode('utf-8')
        except Exception as e:
            logger.error(f"压缩JSON数据失败: {e}")
            return None
    
    @staticmethod
    def decompress_json(compressed_data: str) -> Any:
        """
        解压JSON数据
        
        Args:
            compressed_data: 压缩的Base64编码字符串
        
        Returns:
            解压后的数据
        """
        try:
            # 解码Base64
            compressed = base64.b64decode(compressed_data.encode('utf-8'))
            # 解压
            decompressed = zlib.decompress(compressed)
            # 解析JSON
            return json.loads(decompressed.decode('utf-8'))
        except Exception as e:
            logger.error(f"解压JSON数据失败: {e}")
            return None
    
    @staticmethod
    def compress_floats(float_list: List[float], precision: int = 4) -> str:
        """
        压缩浮点数列表
        
        Args:
            float_list: 浮点数列表
            precision: 精度，默认4位小数
        
        Returns:
            压缩后的Base64编码字符串
        """
        try:
            # 转换为固定精度
            scaled_floats = [round(f * (10 ** precision)) for f in float_list]
            # 计算最小值，用于差分编码
            min_val = min(scaled_floats)
            # 差分编码
            diffs = [scaled_floats[0] - min_val]
            for i in range(1, len(scaled_floats)):
                diffs.append(scaled_floats[i] - scaled_floats[i-1])
            # 打包为二进制
            # 计算需要的字节数
            max_diff = max(abs(d) for d in diffs)
            if max_diff < 2**8:
                # 使用1字节
                format_str = f'<i{len(diffs)}B'
                packed = struct.pack(format_str, min_val, *diffs)
            elif max_diff < 2**16:
                # 使用2字节
                format_str = f'<i{len(diffs)}h'
                packed = struct.pack(format_str, min_val, *diffs)
            else:
                # 使用4字节
                format_str = f'<i{len(diffs)}i'
                packed = struct.pack(format_str, min_val, *diffs)
            # 压缩
            compressed = zlib.compress(packed, level=6)
            # 编码为Base64
            return base64.b64encode(compressed).decode('utf-8')
        except Exception as e:
            logger.error(f"压缩浮点数列表失败: {e}")
            return None
    
    @staticmethod
    def decompress_floats(compressed_data: str, precision: int = 4) -> List[float]:
        """
        解压浮点数列表
        
        Args:
            compressed_data: 压缩的Base64编码字符串
            precision: 精度，默认4位小数
        
        Returns:
            解压后的浮点数列表
        """
        try:
            # 解码Base64
            compressed = base64.b64decode(compressed_data.encode('utf-8'))
            # 解压
            decompressed = zlib.decompress(compressed)
            # 解析二进制数据
            # 首先读取最小值（4字节）
            min_val = struct.unpack('<i', decompressed[:4])[0]
            remaining = decompressed[4:]
            # 计算剩余数据长度和格式
            data_len = len(remaining)
            if data_len % 1 == 0:
                # 1字节格式
                format_str = f'<{data_len}B'
                diffs = struct.unpack(format_str, remaining)
            elif data_len % 2 == 0:
                # 2字节格式
                format_str = f'<{data_len//2}h'
                diffs = struct.unpack(format_str, remaining)
            else:
                # 4字节格式
                format_str = f'<{data_len//4}i'
                diffs = struct.unpack(format_str, remaining)
            # 恢复差分编码
            scaled_floats = [min_val + diffs[0]]
            for i in range(1, len(diffs)):
                scaled_floats.append(scaled_floats[i-1] + diffs[i])
            # 恢复精度
            return [f / (10 ** precision) for f in scaled_floats]
        except Exception as e:
            logger.error(f"解压浮点数列表失败: {e}")
            return []
    
    @staticmethod
    def compress_trades(trades: List[Dict[str, Any]]) -> str:
        """
        压缩交易记录
        
        Args:
            trades: 交易记录列表
        
        Returns:
            压缩后的Base64编码字符串
        """
        try:
            # 转换为紧凑格式
            compact_trades = []
            for trade in trades:
                compact = {
                    't': trade.get('timestamp'),
                    's': trade.get('symbol'),
                    'q': trade.get('quantity'),
                    'p': trade.get('price'),
                    'd': trade.get('direction', 'BUY'),
                    'id': trade.get('trade_id')
                }
                compact_trades.append(compact)
            # 压缩
            return DataCompressor.compress_json(compact_trades)
        except Exception as e:
            logger.error(f"压缩交易记录失败: {e}")
            return None
    
    @staticmethod
    def decompress_trades(compressed_data: str) -> List[Dict[str, Any]]:
        """
        解压交易记录
        
        Args:
            compressed_data: 压缩的Base64编码字符串
        
        Returns:
            解压后的交易记录列表
        """
        try:
            # 解压
            compact_trades = DataCompressor.decompress_json(compressed_data)
            if not compact_trades:
                return []
            # 恢复原始格式
            trades = []
            for compact in compact_trades:
                trade = {
                    'timestamp': compact.get('t'),
                    'symbol': compact.get('s'),
                    'quantity': compact.get('q'),
                    'price': compact.get('p'),
                    'direction': compact.get('d', 'BUY'),
                    'trade_id': compact.get('id')
                }
                trades.append(trade)
            return trades
        except Exception as e:
            logger.error(f"解压交易记录失败: {e}")
            return []
    
    @staticmethod
    def compress_equity_curve(equity_curve: List[float]) -> str:
        """
        压缩权益曲线
        
        Args:
            equity_curve: 权益曲线数据
        
        Returns:
            压缩后的Base64编码字符串
        """
        return DataCompressor.compress_floats(equity_curve, precision=4)
    
    @staticmethod
    def decompress_equity_curve(compressed_data: str) -> List[float]:
        """
        解压权益曲线
        
        Args:
            compressed_data: 压缩的Base64编码字符串
        
        Returns:
            解压后的权益曲线数据
        """
        return DataCompressor.decompress_floats(compressed_data, precision=4)
    
    @staticmethod
    def calculate_compression_ratio(original: Any, compressed: str) -> float:
        """
        计算压缩率
        
        Args:
            original: 原始数据
            compressed: 压缩后的数据
        
        Returns:
            压缩率，0-1之间，越小表示压缩效果越好
        """
        try:
            # 计算原始大小
            original_str = json.dumps(original, ensure_ascii=False)
            original_size = len(original_str.encode('utf-8'))
            # 计算压缩后大小
            compressed_size = len(compressed.encode('utf-8'))
            # 计算压缩率
            return compressed_size / original_size
        except Exception as e:
            logger.error(f"计算压缩率失败: {e}")
            return 1.0


# 全局数据压缩器实例
data_compressor = DataCompressor()


# 工具函数
def compress_data(data: Any, data_type: str = 'json') -> str:
    """
    压缩数据
    
    Args:
        data: 要压缩的数据
        data_type: 数据类型，可选值：json, floats, trades, equity_curve
    
    Returns:
        压缩后的Base64编码字符串
    """
    if data_type == 'json':
        return DataCompressor.compress_json(data)
    elif data_type == 'floats':
        return DataCompressor.compress_floats(data)
    elif data_type == 'trades':
        return DataCompressor.compress_trades(data)
    elif data_type == 'equity_curve':
        return DataCompressor.compress_equity_curve(data)
    else:
        logger.error(f"不支持的数据类型: {data_type}")
        return None


def decompress_data(compressed_data: str, data_type: str = 'json') -> Any:
    """
    解压数据
    
    Args:
        compressed_data: 压缩的Base64编码字符串
        data_type: 数据类型，可选值：json, floats, trades, equity_curve
    
    Returns:
        解压后的数据
    """
    if data_type == 'json':
        return DataCompressor.decompress_json(compressed_data)
    elif data_type == 'floats':
        return DataCompressor.decompress_floats(compressed_data)
    elif data_type == 'trades':
        return DataCompressor.decompress_trades(compressed_data)
    elif data_type == 'equity_curve':
        return DataCompressor.decompress_equity_curve(compressed_data)
    else:
        logger.error(f"不支持的数据类型: {data_type}")
        return None


def should_compress(data: Any, threshold: int = 1024) -> bool:
    """
    判断是否应该压缩数据
    
    Args:
        data: 要判断的数据
        threshold: 阈值，字节数
    
    Returns:
        是否应该压缩
    """
    try:
        json_str = json.dumps(data, ensure_ascii=False)
        return len(json_str.encode('utf-8')) > threshold
    except Exception:
        return False


def compress_if_needed(data: Any, data_type: str = 'json', threshold: int = 1024) -> Dict[str, Any]:
    """
    根据需要压缩数据
    
    Args:
        data: 要处理的数据
        data_type: 数据类型
        threshold: 阈值
    
    Returns:
        包含压缩状态和数据的字典
    """
    if should_compress(data, threshold):
        compressed = compress_data(data, data_type)
        if compressed:
            return {
                'compressed': True,
                'data_type': data_type,
                'data': compressed
            }
    
    return {
        'compressed': False,
        'data_type': data_type,
        'data': data
    }


def decompress_if_needed(compressed_data: Dict[str, Any]) -> Any:
    """
    根据需要解压数据
    
    Args:
        compressed_data: 包含压缩状态和数据的字典
    
    Returns:
        解压后的数据
    """
    if compressed_data.get('compressed'):
        return decompress_data(compressed_data.get('data'), compressed_data.get('data_type', 'json'))
    return compressed_data.get('data')


# 测试函数
def test_compression():
    """
    测试压缩功能
    """
    # 测试JSON压缩
    test_data = {
        'strategy_id': 'test_strategy',
        'total_return': 0.123456,
        'sharpe_ratio': 1.2345,
        'max_drawdown': 0.05678,
        'equity_curve': [1.0, 1.01, 1.02, 1.015, 1.03, 1.04, 1.035, 1.05, 1.06, 1.07]
    }
    
    print("测试JSON压缩...")
    compressed = DataCompressor.compress_json(test_data)
    if compressed:
        print(f"压缩前大小: {len(json.dumps(test_data).encode('utf-8'))} 字节")
        print(f"压缩后大小: {len(compressed.encode('utf-8'))} 字节")
        print(f"压缩率: {len(compressed.encode('utf-8')) / len(json.dumps(test_data).encode('utf-8')):.2f}")
        
        decompressed = DataCompressor.decompress_json(compressed)
        if decompressed:
            print("解压成功!")
            print(f"原始数据: {test_data}")
            print(f"解压数据: {decompressed}")
        else:
            print("解压失败!")
    else:
        print("压缩失败!")
    
    # 测试浮点数压缩
    print("\n测试浮点数压缩...")
    float_list = [1.0, 1.01, 1.02, 1.015, 1.03, 1.04, 1.035, 1.05, 1.06, 1.07] * 100  # 1000个数据点
    compressed = DataCompressor.compress_floats(float_list)
    if compressed:
        print(f"压缩前大小: {len(json.dumps(float_list).encode('utf-8'))} 字节")
        print(f"压缩后大小: {len(compressed.encode('utf-8'))} 字节")
        print(f"压缩率: {len(compressed.encode('utf-8')) / len(json.dumps(float_list).encode('utf-8')):.2f}")
        
        decompressed = DataCompressor.decompress_floats(compressed)
        if decompressed:
            print("解压成功!")
            print(f"原始数据长度: {len(float_list)}")
            print(f"解压数据长度: {len(decompressed)}")
            print(f"第一个数据: {float_list[0]} vs {decompressed[0]}")
            print(f"最后一个数据: {float_list[-1]} vs {decompressed[-1]}")
        else:
            print("解压失败!")
    else:
        print("压缩失败!")


if __name__ == "__main__":
    test_compression()
