"""
配置管理模块
处理数据源配置、策略配置等的加载、保存和管理
"""

import json
import os
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import typing

logger = logging.getLogger(__name__)


# 数据源配置数据模型
class DataSourceConfig(BaseModel):
    id: str
    name: str
    type: str
    url: str
    api_key: Optional[str] = None
    rate_limit: str = "5次/分钟"
    enabled: bool = True
    last_test: Optional[str] = None
    status: str = "未测试"
    config: Optional[Dict[str, typing.Any]] = None


# 环境感知的配置文件路径
def _get_config_file_path():
    """根据环境获取配置文件路径"""
    env = os.getenv("RQA_ENV", "development").lower()

    # 获取项目根目录的绝对路径
    # 从 src/gateway/web/config_manager.py 向上4级到项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src/gateway/web
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # RQA2025
    print(f"DEBUG: 项目根目录: {project_root}")

    if env == "production":
        # 生产环境也使用主配置文件，确保配置一致性
        config_file = os.path.join(project_root, "data", "data_sources_config.json")
    elif env == "testing":
        # 测试环境使用测试目录
        config_file = os.path.join(project_root, "data", "testing", "data_sources_config.json")
    else:
        # 开发环境使用默认目录
        config_file = os.path.join(project_root, "data", "data_sources_config.json")

    print(f"DEBUG: 配置文件绝对路径: {config_file}")
    print(f"DEBUG: 配置文件是否存在: {os.path.exists(config_file)}")
    return config_file


DATA_SOURCES_CONFIG_FILE = _get_config_file_path()


def load_data_sources() -> List[Dict]:
    """加载数据源配置 - 使用 data_source_config_manager 以利用缓存机制"""
    try:
        # 使用 data_source_config_manager 获取数据源，利用其缓存机制避免重复加载
        from .data_source_config_manager import get_data_source_config_manager
        config_manager = get_data_source_config_manager()
        data_sources = config_manager.get_data_sources()
        
        logger.debug(f"从 data_source_config_manager 获取了 {len(data_sources)} 个数据源")
        return data_sources
    except Exception as e:
        logger.error(f"从 data_source_config_manager 获取数据源失败: {e}")
        # 降级方案：直接从PostgreSQL或文件系统加载
        return _load_data_sources_from_postgresql_or_file()


def _load_data_sources_from_postgresql_or_file() -> List[Dict]:
    """降级方案：直接从PostgreSQL或文件系统加载数据源配置"""
    try:
        # 优先从PostgreSQL加载配置
        try:
            from .postgresql_persistence import get_db_connection, return_db_connection
            
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    
                    # 查询数据源配置
                    cursor.execute("""
                        SELECT config_data
                        FROM data_source_configs
                        WHERE config_key = %s
                        ORDER BY updated_at DESC
                        LIMIT 1
                    """, (f"data_sources_{os.getenv('RQA_ENV', 'development').lower()}",))
                    
                    row = cursor.fetchone()
                    if row:
                        config_data = json.loads(row[0])
                        data_sources = config_data.get('data_sources', [])
                        logger.debug(f"降级方案：从PostgreSQL加载了 {len(data_sources)} 个数据源")
                        
                        # 检查并修复null ID
                        for source in data_sources:
                            id_value = source.get('id')
                            name = source.get('name', 'unknown')
                            
                            # 强制修复任何形式的null ID
                            needs_fix = (
                                id_value is None or
                                str(id_value).lower() in ['null', 'none', ''] or
                                id_value == 'null' or
                                id_value == 'None'
                            )

                            if needs_fix:
                                if '新浪财经' in name:
                                    source['id'] = 'sinafinance'
                                elif '宏观经济' in name:
                                    source['id'] = 'macrodata'
                                elif '财联社' in name:
                                    source['id'] = 'cls'
                                else:
                                    source['id'] = name.lower().replace(' ', '_').replace('（', '_').replace('）', '_')
                        
                        # 当从PostgreSQL加载配置成功后，同步更新本地文件
                        try:
                            config_file = _get_config_file_path()
                            os.makedirs(os.path.dirname(config_file), exist_ok=True)
                            
                            # 检查现有文件格式
                            file_format_is_list = False
                            if os.path.exists(config_file):
                                try:
                                    with open(config_file, 'r', encoding='utf-8') as f:
                                        existing_data = json.load(f)
                                        file_format_is_list = isinstance(existing_data, list)
                                except:
                                    pass
                            
                            # 保存配置到本地文件
                            if file_format_is_list:
                                with open(config_file, 'w', encoding='utf-8') as f:
                                    json.dump(data_sources, f, ensure_ascii=False, indent=2)
                            else:
                                with open(config_file, 'w', encoding='utf-8') as f:
                                    json.dump(config_data, f, ensure_ascii=False, indent=2)
                            logger.debug(f"降级方案：从PostgreSQL加载配置后，同步更新了本地文件")
                        except Exception as e:
                            logger.debug(f"降级方案：同步更新本地文件失败: {e}")
                        
                        return data_sources
                finally:
                    return_db_connection(conn)
        except Exception as e:
            logger.debug(f"降级方案：从PostgreSQL加载配置失败，尝试从文件系统加载: {e}")
        
        # 从文件系统加载配置
        config_file = _get_config_file_path()
        logger.debug(f"降级方案：从文件加载配置: {config_file}")

        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                raw_content = f.read()
                
                config_data = json.loads(raw_content)

                # 处理不同格式的配置文件
                if isinstance(config_data, dict):
                    data_sources = config_data.get('data_sources', [])
                elif isinstance(config_data, list):
                    data_sources = config_data
                else:
                    data_sources = []

            logger.debug(f"降级方案：从文件加载了 {len(data_sources)} 个数据源")

            # 检查并修复null ID - 多重保护
            for source in data_sources:
                id_value = source.get('id')
                name = source.get('name', 'unknown')

                # 强制修复任何形式的null ID
                needs_fix = (
                    id_value is None or
                    str(id_value).lower() in ['null', 'none', ''] or
                    id_value == 'null' or
                    id_value == 'None'
                )

                if needs_fix:
                    if '新浪财经' in name:
                        source['id'] = 'sinafinance'
                    elif '宏观经济' in name:
                        source['id'] = 'macrodata'
                    elif '财联社' in name:
                        source['id'] = 'cls'
                    else:
                        source['id'] = name.lower().replace(' ', '_').replace('（', '_').replace('）', '_')

            # 当从文件加载配置后，也更新本地文件以确保文件与内存中的配置一致
            try:
                config_file = _get_config_file_path()
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                
                # 保存修复后的配置到本地文件
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(data_sources, f, ensure_ascii=False, indent=2)
                logger.debug(f"降级方案：保存修复后的配置到本地文件")
            except Exception as e:
                logger.debug(f"降级方案：保存修复后的配置到本地文件失败: {e}")

            return data_sources
        else:
            logger.warning(f"降级方案：配置文件不存在: {config_file}")
            return []

    except Exception as e:
        logger.error(f"降级方案：加载数据源配置失败: {e}")
        return []


def _get_default_data_sources() -> List[Dict]:
    """获取默认数据源配置（仅用于开发/测试环境）"""
    return [
        {
            "id": "alpha-vantage",
            "name": "Alpha Vantage",
            "type": "股票数据",
            "url": "https://www.alphavantage.co",
            "rate_limit": "5次/分钟",
            "enabled": True,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "binance",
            "name": "Binance API",
            "type": "加密货币",
            "url": "https://api.binance.com",
            "rate_limit": "10次/分钟",
            "enabled": True,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "yahoo",
            "name": "Yahoo Finance",
            "type": "市场指数",
            "url": "https://finance.yahoo.com",
            "rate_limit": "5次/分钟",
            "enabled": True,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "newsapi",
            "name": "NewsAPI",
            "type": "新闻数据",
            "url": "https://newsapi.org",
            "rate_limit": "100次/天",
            "enabled": True,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "miniqmt",
            "name": "MiniQMT",
            "type": "本地交易",
            "url": "http://localhost:8888",
            "rate_limit": "无限制",
            "enabled": True,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "fred",
            "name": "FRED API",
            "type": "宏观经济",
            "url": "https://fred.stlouisfed.org",
            "rate_limit": "无限制",
            "enabled": True,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "coingecko",
            "name": "CoinGecko",
            "type": "加密货币",
            "url": "https://api.coingecko.com",
            "rate_limit": "10-50次/分钟",
            "enabled": True,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "emweb",
            "name": "东方财富",
            "type": "行情数据",
            "url": "https://emweb.securities.com.cn",
            "rate_limit": "5次/分钟",
            "enabled": True,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "ths",
            "name": "同花顺",
            "type": "行情数据",
            "url": "https://data.10jqka.com.cn",
            "rate_limit": "5次/分钟",
            "enabled": True,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "xueqiu",
            "name": "雪球",
            "type": "社区数据",
            "url": "https://xueqiu.com",
            "rate_limit": "60次/小时",
            "enabled": False,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "wind",
            "name": "Wind",
            "type": "专业数据",
            "url": "https://www.wind.com.cn",
            "rate_limit": "按协议",
            "enabled": False,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "bloomberg",
            "name": "Bloomberg",
            "type": "专业数据",
            "url": "https://www.bloomberg.com",
            "rate_limit": "按协议",
            "enabled": False,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "qqfinance",
            "name": "腾讯财经",
            "type": "财经新闻",
            "url": "https://finance.qq.com",
            "rate_limit": "10次/分钟",
            "enabled": False,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "sinafinance",
            "name": "新浪财经",
            "type": "财经新闻",
            "url": "https://finance.sina.com.cn",
            "rate_limit": "10次/分钟",
            "enabled": False,
            "last_test": None,
            "status": "未测试"
        }
    ]


def save_data_sources(sources: List[Dict]):
    """保存数据源配置到文件，带生产环境保护，并同步到DataSourceConfigManager"""
    env = os.getenv("RQA_ENV", "development").lower()

    # 生产环境保护：检查是否正在用默认数据覆盖生产数据
    if env == "production":
        try:
            # 检查现有文件是否存在且不为空
            if os.path.exists(DATA_SOURCES_CONFIG_FILE):
                with open(DATA_SOURCES_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

                # 如果现有数据不为空，但新数据看起来像是默认数据，则拒绝保存
                if (len(existing_data) > 0 and len(sources) > 0 and
                    _is_likely_default_data(sources) and not _is_likely_default_data(existing_data)):
                    print("生产环境保护：拒绝用默认数据覆盖现有生产配置")
                    print("如果需要重置配置，请手动删除配置文件后重启")
                    return
        except Exception as e:
            print(f"生产环境数据保护检查失败: {e}")

    try:
        os.makedirs(os.path.dirname(DATA_SOURCES_CONFIG_FILE), exist_ok=True)

        # 创建备份
        if os.path.exists(DATA_SOURCES_CONFIG_FILE):
            backup_file = f"{DATA_SOURCES_CONFIG_FILE}.backup"
            import shutil
            shutil.copy2(DATA_SOURCES_CONFIG_FILE, backup_file)
            print(f"创建配置文件备份: {backup_file}")

        # 保存前再次检查数据
        logger.info(f"准备保存数据源配置 ({len(sources)} 个数据源) 到文件: {DATA_SOURCES_CONFIG_FILE}")
        for i, source in enumerate(sources):
            logger.debug(f"  数据源 {i}: id={source.get('id')}, name={source.get('name')}, enabled={source.get('enabled')}")

        # 步骤1: 先更新DataSourceConfigManager内存，确保内存状态最新
        try:
            from .data_source_config_manager import get_data_source_config_manager
            manager = get_data_source_config_manager()
            if manager:
                # 直接更新配置管理器中的数据源列表（先更新内存）
                manager.config_manager.set('data_sources.core.data_sources', sources)
                # 更新元数据
                from datetime import datetime
                manager.config_manager.set('data_sources.metadata.last_updated', datetime.now().isoformat())
                
                # 记录每个数据源的enabled状态，用于验证
                for source in sources:
                    logger.debug(f"同步数据源到管理器内存: id={source.get('id')}, name={source.get('name')}, enabled={source.get('enabled')}")
                
                logger.info(f"数据源配置已同步到DataSourceConfigManager内存 ({len(sources)} 个数据源)")
        except Exception as e:
            logger.warning(f"同步到DataSourceConfigManager内存失败: {e}")

        # 步骤2: 保存到文件系统（使用统一的文件路径）
        with open(DATA_SOURCES_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(sources, f, ensure_ascii=False, indent=2)

        logger.info(f"数据源配置已保存到文件: {DATA_SOURCES_CONFIG_FILE} ({len(sources)} 个数据源)")

        # 步骤3: 通过DataSourceConfigManager保存到PostgreSQL（如果可用）
        try:
            if manager and manager.auto_save:
                # save_config会从内存读取数据并保存到文件和PostgreSQL
                # 由于我们已经更新了内存，这里会保存最新的状态
                manager.save_config()
                logger.info(f"数据源配置已通过DataSourceConfigManager保存到PostgreSQL ({len(sources)} 个数据源)")
        except Exception as e:
            logger.warning(f"保存到PostgreSQL失败（使用文件系统）: {e}")

    except Exception as e:
        print(f"保存数据源配置失败: {e}")
        logger.error(f"保存数据源配置失败: {e}", exc_info=True)


def _is_likely_default_data(data: List[Dict]) -> bool:
    """检查数据是否看起来像是默认配置数据"""
    if not data or len(data) == 0:
        return False

    # 检查是否所有数据源都是"未测试"状态和None的last_test
    # 这通常表示是默认数据而不是生产使用后的数据
    all_untested = all(
        item.get("status") == "未测试" and item.get("last_test") is None
        for item in data
    )

    return all_untested


def get_data_source_config_manager_instance():
    """获取数据源配置管理器实例"""
    try:
        from .data_source_config_manager import get_data_source_config_manager
        return get_data_source_config_manager()
    except Exception as e:
        logger.warning(f"获取数据源配置管理器实例失败: {e}")
        return None
