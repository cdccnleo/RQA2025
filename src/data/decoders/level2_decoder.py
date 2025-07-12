"""Level2行情解码器模块"""
import struct
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class Level2Decoder:
    """A股Level2行情解码器"""
    
    def __init__(self):
        self._protocol_version = "1.0"
        self._supported_markets = ["SH", "SZ"]
        
    def decode(self, raw_data: bytes) -> Dict:
        """
        解码原始Level2行情数据
        Args:
            raw_data: 原始字节数据
        Returns:
            dict包含解码后的行情数据:
            - symbol: 股票代码
            - price: 最新价
            - volume: 成交量
            - bids: 买档位列表(价格,数量)
            - asks: 卖档位列表(价格,数量)
        """
        try:
            # 检查数据头
            if not self._validate_header(raw_data):
                raise ValueError("Invalid data header")
                
            # 解析基础信息
            symbol = self._decode_symbol(raw_data[4:10])
            price = self._decode_price(raw_data[10:18])
            volume = self._decode_volume(raw_data[18:26])
            
            # 解析订单簿
            bids, asks = self._decode_order_book(raw_data[26:])
            
            return {
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "bids": bids,
                "asks": asks,
                "market": self._detect_market(symbol)
            }
        except Exception as e:
            logger.error(f"Decode error: {str(e)}")
            raise
            
    def _validate_header(self, data: bytes) -> bool:
        """验证数据头"""
        return len(data) > 26 and data[:2] == b'\xAA\x55'
        
    def _decode_symbol(self, data: bytes) -> str:
        """解码股票代码"""
        return data.decode('ascii').strip()
        
    def _decode_price(self, data: bytes) -> float:
        """解码价格"""
        return struct.unpack('!d', data)[0]
        
    def _decode_volume(self, data: bytes) -> int:
        """解码成交量"""
        return struct.unpack('!Q', data)[0]
        
    def _decode_order_book(self, data: bytes) -> Tuple[list, list]:
        """解码订单簿"""
        bids = []
        asks = []
        
        # 检查数据长度是否足够
        if len(data) < 4:
            return bids, asks
            
        # 解析买卖档位数量
        bid_count, ask_count = struct.unpack('!HH', data[:4])
        pos = 4
        
        # 解析买档位
        for _ in range(bid_count):
            if pos + 16 > len(data):
                break
            price, volume = struct.unpack('!dQ', data[pos:pos+16])
            bids.append((price, volume))
            pos += 16
            
        # 解析卖档位
        for _ in range(ask_count):
            if pos + 16 > len(data):
                break
            price, volume = struct.unpack('!dQ', data[pos:pos+16])
            asks.append((price, volume))
            pos += 16
            
        # 按价格排序
        bids.sort(reverse=True)  # 买档从高到低
        asks.sort()              # 卖档从低到高
            
        return bids, asks
        
    def decode_batch(self, raw_data_list: List[bytes]) -> List[Dict]:
        """
        批量解码Level2数据
        Args:
            raw_data_list: 原始数据列表
        Returns:
            解码后的数据列表
        """
        results = []
        for raw_data in raw_data_list:
            try:
                decoded = self.decode(raw_data)
                results.append(decoded)
            except Exception as e:
                logger.warning(f"Failed to decode data: {str(e)}")
                continue
        return results
        
    def _detect_market(self, symbol: str) -> str:
        """识别市场(上海/深圳)"""
        if symbol.startswith(('6', '9')):
            return "SH"
        elif symbol.startswith(('0', '3')):
            return "SZ"
        else:
            return "UNKNOWN"
