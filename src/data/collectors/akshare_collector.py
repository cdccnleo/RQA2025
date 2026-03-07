"""
AKShare 数据采集器
使用 AKShare 库获取股票历史数据并写入数据库
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AKShareCollector:
    """
    AKShare 数据采集器
    
    职责：
    1. 使用 AKShare 库获取股票历史数据
    2. 数据清洗和转换
    3. 将数据写入 akshare_stock_data 表
    """
    
    def __init__(self):
        """初始化 AKShare 采集器"""
        self._akshare_available = self._check_akshare()
        
        if self._akshare_available:
            logger.info("AKShare 采集器初始化成功")
        else:
            logger.warning("AKShare 库不可用，采集器将无法工作")
    
    def _check_akshare(self) -> bool:
        """检查 AKShare 库是否可用"""
        try:
            import akshare as ak
            return True
        except ImportError:
            logger.error("AKShare 库未安装，请运行: pip install akshare")
            return False
    
    def collect_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        采集股票历史数据
        
        Args:
            symbol: 股票代码 (如 "000001")
            start_date: 开始日期 (YYYY-MM-DD)，默认为30天前
            end_date: 结束日期 (YYYY-MM-DD)，默认为今天
            adjust: 复权类型 (qfq-前复权, hfq-后复权, 不复权)
            
        Returns:
            股票数据列表，每个元素是一个字典
        """
        if not self._akshare_available:
            logger.error("AKShare 库不可用，无法采集数据")
            return None
        
        try:
            import akshare as ak
            
            # 设置默认日期
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            logger.info(f"开始采集股票数据: {symbol}, 时间范围: {start_date} 到 {end_date}")
            
            # 转换日期格式 (YYYY-MM-DD -> YYYYMMDD)
            start_date_fmt = start_date.replace("-", "")
            end_date_fmt = end_date.replace("-", "")
            
            # 使用 AKShare 获取数据
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date_fmt,
                end_date=end_date_fmt,
                adjust=adjust
            )
            
            if df.empty:
                logger.warning(f"未获取到数据: {symbol}")
                return None
            
            logger.info(f"成功获取原始数据: {symbol}, 记录数: {len(df)}")
            
            # 转换数据格式
            data = self._convert_data(df, symbol)
            
            logger.info(f"数据转换完成: {symbol}, 记录数: {len(data)}")
            return data
            
        except Exception as e:
            logger.error(f"采集股票数据失败 {symbol}: {e}")
            return None
    
    def _convert_data(
        self,
        df,
        symbol: str
    ) -> List[Dict[str, Any]]:
        """
        转换数据格式
        
        Args:
            df: AKShare 返回的 DataFrame
            symbol: 股票代码
            
        Returns:
            标准格式的数据列表
        """
        data = []
        
        # AKShare 返回的列名（中文）
        # 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
        
        for _, row in df.iterrows():
            try:
                record = {
                    "symbol": symbol,
                    "date": row.get("日期") if "日期" in row else row.get("date"),
                    "open": float(row.get("开盘", 0)) if "开盘" in row else float(row.get("open", 0)),
                    "high": float(row.get("最高", 0)) if "最高" in row else float(row.get("high", 0)),
                    "low": float(row.get("最低", 0)) if "最低" in row else float(row.get("low", 0)),
                    "close": float(row.get("收盘", 0)) if "收盘" in row else float(row.get("close", 0)),
                    "volume": int(float(row.get("成交量", 0))) if "成交量" in row else int(float(row.get("volume", 0))),
                    "amount": float(row.get("成交额", 0)) if "成交额" in row else float(row.get("amount", 0)),
                    "amplitude": float(row.get("振幅", 0)) if "振幅" in row else float(row.get("amplitude", 0)),
                    "pct_change": float(row.get("涨跌幅", 0)) if "涨跌幅" in row else float(row.get("pct_change", 0)),
                    "change_amount": float(row.get("涨跌额", 0)) if "涨跌额" in row else float(row.get("change_amount", 0)),
                    "turnover": float(row.get("换手率", 0)) if "换手率" in row else float(row.get("turnover", 0))
                }
                data.append(record)
            except Exception as e:
                logger.warning(f"转换数据行失败: {e}, row={row.to_dict()}")
                continue
        
        return data
    
    def save_to_database(self, data: List[Dict[str, Any]], symbol: str) -> bool:
        """
        将数据保存到数据库
        
        Args:
            data: 股票数据列表
            symbol: 股票代码
            
        Returns:
            是否保存成功
        """
        if not data:
            logger.warning(f"没有数据需要保存: {symbol}")
            return False
        
        try:
            from src.gateway.web.postgresql_persistence import get_db_connection
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # 准备插入语句（使用 UPSERT 语法）
            # 注意：列名使用数据库实际结构（带_price后缀）
            insert_query = """
                INSERT INTO akshare_stock_data (
                    source_id, symbol, date, open_price, high_price, low_price, close_price, 
                    volume, amount, amplitude, pct_change, change, turnover_rate, data_source
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_id, symbol, date, data_type) DO UPDATE SET
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume = EXCLUDED.volume,
                    amount = EXCLUDED.amount,
                    amplitude = EXCLUDED.amplitude,
                    pct_change = EXCLUDED.pct_change,
                    change = EXCLUDED.change,
                    turnover_rate = EXCLUDED.turnover_rate,
                    collected_at = CURRENT_TIMESTAMP
            """
            
            # 批量插入数据
            records_to_insert = []
            for record in data:
                values = (
                    'akshare_stock_a',  # source_id
                    record.get('symbol'),
                    record.get('date'),
                    record.get('open'),
                    record.get('high'),
                    record.get('low'),
                    record.get('close'),
                    record.get('volume'),
                    record.get('amount'),
                    record.get('amplitude'),
                    record.get('pct_change'),
                    record.get('change_amount'),  # 映射到 change 列
                    record.get('turnover'),  # 映射到 turnover_rate 列
                    'akshare'  # data_source
                )
                records_to_insert.append(values)
            
            # 执行批量插入
            cursor.executemany(insert_query, records_to_insert)
            conn.commit()
            
            logger.info(f"成功保存 {len(data)} 条数据到数据库: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"保存数据到数据库失败 {symbol}: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                # 将连接返回到连接池，而不是关闭
                from src.gateway.web.postgresql_persistence import return_db_connection
                return_db_connection(conn)

    def collect_and_save(self, symbol: str, start_date: str, end_date: str) -> bool:
        """
        采集并保存股票数据（一键操作）
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            
        Returns:
            是否成功
        """
        import time
        start_time = time.time()
        
        try:
            # 转换日期格式
            start_date_fmt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            end_date_fmt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
            
            # 采集数据
            data = self.collect_stock_data(symbol, start_date_fmt, end_date_fmt)
            
            if not data:
                logger.warning(f"未采集到数据: {symbol}")
                # 记录错误指标
                self._record_metrics(symbol, 0, 0, 1.0, False)
                return False
            
            # 保存到数据库
            success = self.save_to_database(data, symbol)
            
            # 计算采集延迟
            collection_time_ms = (time.time() - start_time) * 1000
            
            if success:
                logger.info(f"成功采集并保存 {len(data)} 条数据: {symbol}")
                # 记录成功指标
                self._record_metrics(symbol, collection_time_ms, len(data), 0.0, True)
            else:
                logger.error(f"保存数据失败: {symbol}")
                # 记录错误指标
                self._record_metrics(symbol, collection_time_ms, len(data), 1.0, False)
            
            return success
            
        except Exception as e:
            collection_time_ms = (time.time() - start_time) * 1000
            logger.error(f"采集并保存数据失败 {symbol}: {e}")
            # 记录错误指标
            self._record_metrics(symbol, collection_time_ms, 0, 1.0, False)
            return False
    
    def _record_metrics(self, symbol: str, latency_ms: float, record_count: int, error_rate: float, success: bool):
        """
        记录性能指标到 PerformanceMonitor
        
        Args:
            symbol: 股票代码
            latency_ms: 采集延迟（毫秒）
            record_count: 记录数
            error_rate: 错误率（0.0-1.0）
            success: 是否成功
        """
        try:
            from src.data.monitoring.performance_monitor import PerformanceMonitor
            performance_monitor = PerformanceMonitor()
            
            # 记录延迟指标
            performance_monitor.record_metric(
                f"data_source_akshare_stock_a_latency",
                latency_ms,
                "ms",
                {"symbol": symbol, "source": "akshare", "success": success}
            )
            
            # 记录吞吐量指标
            if record_count > 0:
                performance_monitor.record_metric(
                    f"data_source_akshare_stock_a_throughput",
                    record_count,
                    "records",
                    {"symbol": symbol, "source": "akshare"}
                )
            
            # 记录错误率指标
            performance_monitor.record_metric(
                f"data_source_akshare_stock_a_error_rate",
                error_rate,
                "%",
                {"symbol": symbol, "source": "akshare", "success": success}
            )
            
            logger.debug(f"✅ 已记录性能指标: {symbol}, 延迟={latency_ms:.2f}ms, 数据量={record_count}, 错误率={error_rate}")
        except Exception as metric_error:
            logger.debug(f"记录性能指标失败: {metric_error}")