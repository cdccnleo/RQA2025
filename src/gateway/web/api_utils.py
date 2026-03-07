"""
API工具函数模块
包含通用的API工具函数和辅助方法
"""

import time
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def get_data_source_config_manager_instance():
    """获取数据源配置管理器实例（占位符函数）"""
    # TODO: 实现实际的数据源配置管理器
    logger.warning("数据源配置管理器尚未实现")
    return None


async def persist_collected_data(source_id: str, data: List[Dict[str, Any]], metadata: Dict[str, Any], source_config: Dict[str, Any]) -> Dict[str, Any]:
    """持久化采集的数据（优先使用PostgreSQL，失败时回退到文件存储）"""

    # 对于股票数据，优先使用PostgreSQL存储
    source_type = source_config.get("type", "")
    logger.debug(f"持久化数据: source_id={source_id}, source_type={source_type}, data_count={len(data) if data else 0}")

    # 根据数据源类型选择相应的持久化函数
    if data:
        try:
            logger.info(f"尝试PostgreSQL持久化: {source_id} ({source_type})")

            pg_result = None

            if source_type == "股票数据":
                from .postgresql_persistence import persist_akshare_data_to_postgresql
                pg_result = persist_akshare_data_to_postgresql(
                    source_id=source_id,
                    data=data,
                    metadata=metadata,
                    source_config=source_config
                )
            elif source_type == "指数数据":
                from .postgresql_persistence import persist_akshare_index_data_to_postgresql
                pg_result = persist_akshare_index_data_to_postgresql(
                    source_id=source_id,
                    data=data,
                    metadata=metadata,
                    source_config=source_config
                )
            elif source_type == "基金数据":
                from .postgresql_persistence import persist_akshare_fund_data_to_postgresql
                pg_result = persist_akshare_fund_data_to_postgresql(
                    source_id=source_id,
                    data=data,
                    metadata=metadata,
                    source_config=source_config
                )
            elif source_type == "宏观经济":
                from .postgresql_persistence import persist_akshare_macro_data_to_postgresql
                pg_result = persist_akshare_macro_data_to_postgresql(
                    source_id=source_id,
                    data=data,
                    metadata=metadata,
                    source_config=source_config
                )
            elif source_type == "财经新闻":
                from .postgresql_persistence import persist_akshare_news_data_to_postgresql
                pg_result = persist_akshare_news_data_to_postgresql(
                    source_id=source_id,
                    data=data,
                    metadata=metadata,
                    source_config=source_config
                )
            elif source_type == "另类数据":
                from .postgresql_persistence import persist_akshare_alternative_data_to_postgresql
                pg_result = persist_akshare_alternative_data_to_postgresql(
                    source_id=source_id,
                    data=data,
                    metadata=metadata,
                    source_config=source_config
                )
            elif source_id == "akshare_stock_basic":
                # 股票基本信息持久化
                from .postgresql_persistence import save_stock_basic_info
                import pandas as pd
                # 将数据转换为DataFrame格式
                stock_basic_df = pd.DataFrame(data)
                pg_result = save_stock_basic_info(stock_basic_df)
            # 其他数据类型（债券、期货、外汇等）暂时使用文件存储

            if pg_result and pg_result.get("success"):
                logger.info(f"PostgreSQL持久化成功: {source_id}, 插入{pg_result.get('inserted_count', 0)}条记录")
                return pg_result
            elif pg_result:
                error_msg = pg_result.get('error', 'unknown')
                logger.warning(f"PostgreSQL持久化失败，尝试文件存储: {error_msg}")
                # 继续执行文件存储作为后备方案

        except ImportError as e:
            logger.warning(f"PostgreSQL持久化模块不可用，使用文件存储: {e}")
        except Exception as e:
            logger.error(f"PostgreSQL持久化异常，使用文件存储: {e}", exc_info=True)

    # 文件存储（后备方案或非股票数据）
    try:
        logger.info(f"开始文件存储: {source_id}, 数据量: {len(data)}")

        # 创建数据存储目录
        data_dir = Path("data/collected")
        data_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        timestamp = int(time.time())
        filename = f"{source_id}_{timestamp}.json"
        filepath = data_dir / filename

        # 准备存储数据
        storage_data = {
            "source_id": source_id,
            "source_config": source_config,
            "metadata": {
                **metadata,
                "persistence_timestamp": time.time(),
                "data_points": len(data)
            },
            "data": data,
            "collected_at": time.time(),
            "data_points": len(data)
        }

        # 再次确保数据可JSON序列化
        import json
        try:
            json.dumps(storage_data, default=str)
            logger.info("数据JSON序列化检查通过")
        except Exception as e:
            logger.error(f"数据不可JSON序列化: {e}")

            # 调试：找出哪些数据包含Timestamp对象
            for i, item in enumerate(data):
                if i >= 3:  # 只检查前3个项目
                    break
                for key, value in item.items():
                    if hasattr(value, 'isoformat'):
                        logger.error(f"项目{i}的字段{key}包含Timestamp: {type(value)} - {value}")
                    elif hasattr(value, 'strftime'):
                        logger.error(f"项目{i}的字段{key}包含datetime: {type(value)} - {value}")

            raise e

        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(storage_data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"文件存储持久化成功: {source_id} -> {filepath}")

        # 生成样本文件
        try:
            await generate_sample_file(source_id, source_type, data)
            logger.info(f"样本文件生成成功: {source_id}")
        except Exception as sample_error:
            logger.warning(f"样本文件生成失败: {sample_error}")

        return {
            "success": True,
            "storage_type": "file",
            "storage_id": filename,
            "processing_time": 0.0,
            "message": f"数据已保存到文件: {filepath}",
            "filepath": str(filepath),
            "fallback_reason": "PostgreSQL不可用或非股票数据类型"
        }

    except Exception as e:
        error_msg = f"数据持久化异常: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg
        }


async def generate_sample_file(source_id: str, source_type: str, data: List[Dict[str, Any]]):
    """
    从采集到的数据生成一个CSV样本文件，用于前端展示。
    文件命名约定: {source_id}_{source_type}_{timestamp}.csv
    """
    if not data:
        logger.warning(f"没有数据可用于生成 {source_id} 的样本文件。")
        return

    from pathlib import Path
    import pandas as pd
    from datetime import datetime

    samples_dir = Path("data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 移除source_type中的特殊字符，确保文件名合法
    safe_source_type = "".join(c for c in source_type if c.isalnum() or c in ('_', '-')).strip()
    filename = f"{source_id}_{safe_source_type}_{timestamp}.csv"
    filepath = samples_dir / filename

    try:
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')  # 使用utf-8-sig支持中文
        logger.info(f"成功为数据源 {source_id} 生成样本文件: {filepath}")
    except Exception as e:
        logger.error(f"为数据源 {source_id} 生成样本文件失败: {e}", exc_info=True)


async def broadcast_data_source_change(event_type: str, source_id: str, data: dict):
    """广播数据源变更事件"""
    try:
        from .websocket_manager import manager
        message = {
            "type": "data_source_event",
            "event": event_type,
            "source_id": source_id,
            "data": data,
            "timestamp": time.time()
        }
        await manager.broadcast("data_source", message)
    except Exception as e:
        logger.error(f"广播数据源变更事件失败: {e}")


def generate_stock_data_sample(source_name: str) -> Dict[str, Any]:
    """生成股票数据样本"""
    import random
    from datetime import datetime, timedelta

    base_price = random.uniform(10, 500)
    current_date = datetime.now()

    sample_data = {
        "symbol": "000001" if "MiniQMT" in source_name else ("600000" if "东方财富" in source_name else "000002"),
        "name": "平安银行" if "000001" in str(base_price) else ("浦发银行" if "600000" in str(base_price) else "万科A"),
        "date": (current_date - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d"),
        "open": round(base_price * random.uniform(0.95, 1.05), 2),
        "high": round(base_price * random.uniform(1.01, 1.08), 2),
        "low": round(base_price * random.uniform(0.92, 0.99), 2),
        "close": round(base_price * random.uniform(0.95, 1.05), 2),
        "volume": random.randint(100000, 10000000),
        "amount": round(random.uniform(1000000, 100000000), 2),
        "source": source_name,
        "last_update": current_date.strftime("%Y-%m-%d %H:%M:%S"),
        "data_points": random.randint(1000, 5000),
        "quality_score": round(random.uniform(85, 98), 1)
    }

    sample_data["change"] = round(sample_data["close"] - sample_data["open"], 2)
    sample_data["change_percent"] = round((sample_data["change"] / sample_data["open"]) * 100, 2)

    return sample_data


def generate_crypto_data_sample(source_name: str) -> Dict[str, Any]:
    """生成加密货币数据样本"""
    import random
    from datetime import datetime, timedelta

    cryptos = ["BTC", "ETH", "ADA", "SOL", "DOT"]
    crypto = random.choice(cryptos)

    base_price = {"BTC": 40000, "ETH": 2500, "ADA": 0.5, "SOL": 30, "DOT": 8}[crypto]

    current_date = datetime.now()

    sample_data = {
        "symbol": crypto,
        "name": f"{crypto}/USDT",
        "date": (current_date - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d"),
        "open": round(base_price * random.uniform(0.95, 1.05), 4),
        "high": round(base_price * random.uniform(1.01, 1.08), 4),
        "low": round(base_price * random.uniform(0.92, 0.99), 4),
        "close": round(base_price * random.uniform(0.95, 1.05), 4),
        "volume": round(random.uniform(1000, 50000), 2),
        "amount": round(random.uniform(1000000, 50000000), 2),
        "source": source_name,
        "last_update": current_date.strftime("%Y-%m-%d %H:%M:%S"),
        "data_points": random.randint(1000, 5000),
        "quality_score": round(random.uniform(85, 98), 1)
    }

    sample_data["change"] = round(sample_data["close"] - sample_data["open"], 4)
    sample_data["change_percent"] = round((sample_data["change"] / sample_data["open"]) * 100, 2)

    return sample_data


def generate_data_sample(source_config: Dict[str, Any]) -> Dict[str, Any]:
    """根据数据源配置生成数据样本"""
    source_type = source_config.get("type", "")
    source_name = source_config.get("name", "")

    if source_type == "股票数据":
        return generate_stock_data_sample(source_name)
    elif source_type in ["加密货币", "数字货币"]:
        return generate_crypto_data_sample(source_name)
    else:
        # 默认返回股票数据样本
        return generate_stock_data_sample(source_name)

async def generate_data_samples():
    """生成所有数据源的样本文件"""
    try:
        from pathlib import Path
        import pandas as pd
        import json

        collected_dir = Path("data/collected")
        samples_dir = Path("data/samples")

        if not collected_dir.exists():
            logger.warning("collected目录不存在，跳过样本生成")
            return

        samples_dir.mkdir(parents=True, exist_ok=True)

        # 查找所有JSON格式的采集文件
        collected_files = list(collected_dir.glob("*.json"))

        logger.info(f"找到 {len(collected_files)} 个采集文件，开始生成样本...")

        for json_file in collected_files:
            try:
                # 解析文件名获取数据源ID
                filename = json_file.stem  # 不含扩展名的文件名
                if "_" not in filename:
                    continue

                # 从文件名提取数据源ID (格式: source_id_timestamp)
                parts = filename.split("_")
                if len(parts) < 2:
                    continue

                # 找到时间戳前的部分作为数据源ID
                timestamp_part = parts[-1]
                if timestamp_part.isdigit():
                    source_id = "_".join(parts[:-1])
                else:
                    continue

                logger.debug(f"处理数据源 {source_id} 的样本文件: {json_file}")

                # 读取采集数据
                with open(json_file, 'r', encoding='utf-8') as f:
                    collected_data = json.load(f)

                data_list = collected_data.get("data", [])
                if not data_list:
                    logger.warning(f"数据源 {source_id} 的采集文件无数据")
                    continue

                # 转换为DataFrame进行处理
                df = pd.DataFrame(data_list)

                # 生成样本文件名 (格式: source_id_数据类型_timestamp.csv)
                source_config = collected_data.get("source_config", {})
                source_name = source_config.get("name", source_id)
                source_type = source_config.get("type", "unknown")

                # 转换为中文类型名称
                type_mapping = {
                    "股票数据": "股票",
                    "财经新闻": "财经新闻",
                    "宏观经济": "宏观经济",
                    "债券数据": "债券",
                    "期货数据": "期货",
                    "外汇数据": "外汇",
                    "指数数据": "指数",
                    "基金数据": "基金"
                }
                type_name = type_mapping.get(source_type, source_type)

                sample_filename = f"{source_id}_{type_name}_{timestamp_part}"

                # 生成CSV样本文件
                csv_path = samples_dir / f"{sample_filename}.csv"
                df.head(50).to_csv(csv_path, index=False, encoding='utf-8-sig')

                # 生成JSON样本文件
                json_path = samples_dir / f"{sample_filename}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "source_id": source_id,
                        "source_name": source_name,
                        "source_type": source_type,
                        "sample_count": len(df),
                        "generated_at": time.time(),
                        "data": df.head(10).to_dict('records')
                    }, f, ensure_ascii=False, indent=2)

                # 生成Excel样本文件 (如果有openpyxl)
                try:
                    excel_path = samples_dir / f"{sample_filename}.xlsx"
                    df.head(50).to_excel(excel_path, index=False, engine='openpyxl')
                except ImportError:
                    logger.debug("openpyxl未安装，跳过Excel样本生成")

                logger.info(f"生成数据源 {source_id} 的样本文件: {sample_filename}")

            except Exception as e:
                logger.error(f"处理样本文件 {json_file} 失败: {e}")
                continue

        logger.info("样本文件生成完成")

    except Exception as e:
        logger.error(f"生成数据样本失败: {e}")
