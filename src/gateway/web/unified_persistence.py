"""
统一持久化模块
实现双存储机制：优先PostgreSQL数据库，降级文件系统
支持策略开发流程各环节的数据持久化
"""

import logging
import json
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# 使用统一日志系统
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# 导入数据压缩模块
try:
    from .data_compression import (
        compress_if_needed,
        decompress_if_needed,
        data_compressor
    )
    COMPRESSION_AVAILABLE = True
    logger.info("数据压缩模块导入成功")
except Exception as e:
    logger.warning(f"数据压缩模块导入失败: {e}")
    COMPRESSION_AVAILABLE = False
    
    # 定义空函数作为降级
    def compress_if_needed(data, data_type='json', threshold=1024):
        return {'compressed': False, 'data_type': data_type, 'data': data}
    
    def decompress_if_needed(compressed_data):
        return compressed_data.get('data')

# 延迟导入数据库连接模块
db_available = False
try:
    from .postgresql_persistence import get_db_connection, return_db_connection
    db_available = True
except ImportError:
    logger.warning("PostgreSQL持久化模块不可用，将使用文件系统存储")

# 数据存储根目录
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../data")


class UnifiedPersistence:
    """
    统一持久化类
    实现双存储机制：优先PostgreSQL，降级文件系统
    """
    
    def __init__(self, table_name: str, file_dir: str):
        """
        初始化持久化管理器
        
        Args:
            table_name: 数据库表名
            file_dir: 文件系统存储目录
        """
        self.table_name = table_name
        self.file_dir = os.path.join(DATA_DIR, file_dir)
        self.ensure_directory()
    
    def ensure_directory(self):
        """
        确保文件系统存储目录存在
        """
        os.makedirs(self.file_dir, exist_ok=True)
    
    def save(self, data: Dict[str, Any], primary_key: str = "id") -> bool:
        """
        保存数据，优先使用数据库，失败后降级到文件系统
        
        Args:
            data: 要保存的数据
            primary_key: 主键字段名
        
        Returns:
            是否成功保存
        """
        item_id = data.get(primary_key, 'unknown')
        logger.info(f"[UnifiedPersistence] 开始保存数据到表 {self.table_name}, ID: {item_id}")
        logger.info(f"[UnifiedPersistence] 数据字段: {list(data.keys())}")
        logger.info(f"[UnifiedPersistence] stats字段: {data.get('stats')}")
        
        try:
            # 尝试保存到数据库
            if db_available:
                logger.info(f"[UnifiedPersistence] 尝试保存到数据库")
                if self._save_to_database(data, primary_key):
                    logger.info(f"[UnifiedPersistence] 数据已保存到数据库表 {self.table_name}")
                    return True
                else:
                    logger.warning(f"[UnifiedPersistence] 数据库保存返回False")
            else:
                logger.warning(f"[UnifiedPersistence] 数据库不可用")
        except Exception as e:
            logger.warning(f"[UnifiedPersistence] 数据库保存失败: {e}")
        
        # 降级到文件系统
        try:
            logger.info(f"[UnifiedPersistence] 尝试保存到文件系统")
            if self._save_to_filesystem(data, primary_key):
                logger.info(f"[UnifiedPersistence] 数据已保存到文件系统目录 {self.file_dir}")
                return True
            else:
                logger.warning(f"[UnifiedPersistence] 文件系统保存返回False")
        except Exception as e:
            logger.error(f"[UnifiedPersistence] 文件系统保存失败: {e}")
        
        logger.error(f"[UnifiedPersistence] 数据保存完全失败")
        return False
    
    def _save_to_database(self, data: Dict[str, Any], primary_key: str) -> bool:
        """
        保存数据到PostgreSQL数据库
        
        Args:
            data: 要保存的数据
            primary_key: 主键字段名
        
        Returns:
            是否成功保存
        """
        conn = None
        try:
            conn = get_db_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            # 构建插入/更新语句
            id_value = data.get(primary_key)
            if not id_value:
                logger.error(f"数据缺少主键字段 {primary_key}")
                return False
            
            # 检查记录是否存在
            cursor.execute(f"SELECT 1 FROM {self.table_name} WHERE {primary_key} = %s", (id_value,))
            exists = cursor.fetchone() is not None
            
            # 获取表的列信息
            def get_table_columns(cursor, table_name):
                """获取表的列信息"""
                cursor.execute(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                """)
                return [row[0] for row in cursor.fetchall()]
            
            # 获取表列
            table_columns = get_table_columns(cursor, self.table_name)
            
            # 准备数据，只保留表中存在的字段
            data_to_save = {}
            # 确保包含必要的字段
            if "updated_at" in table_columns:
                # 检查数据中的updated_at类型
                if "updated_at" in data:
                    updated_at = data["updated_at"]
                    # 如果是数字类型（时间戳），转换为datetime
                    if isinstance(updated_at, (int, float)):
                        data_to_save["updated_at"] = datetime.fromtimestamp(updated_at)
                    # 如果是字符串类型，尝试解析
                    elif isinstance(updated_at, str):
                        try:
                            # 尝试解析ISO格式时间
                            if updated_at.endswith('Z') or 'T' in updated_at:
                                data_to_save["updated_at"] = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                            # 尝试解析数字字符串
                            else:
                                data_to_save["updated_at"] = datetime.fromtimestamp(float(updated_at))
                        except:
                            data_to_save["updated_at"] = datetime.now()
                    else:
                        data_to_save["updated_at"] = updated_at
                else:
                    data_to_save["updated_at"] = datetime.now()
            
            if "created_at" in table_columns and "created_at" not in data:
                data_to_save["created_at"] = datetime.now()
            elif "created_at" in table_columns and "created_at" in data:
                created_at = data["created_at"]
                # 如果是数字类型（时间戳），转换为datetime
                if isinstance(created_at, (int, float)):
                    data_to_save["created_at"] = datetime.fromtimestamp(created_at)
                # 如果是字符串类型，尝试解析
                elif isinstance(created_at, str):
                    try:
                        # 尝试解析ISO格式时间
                        if created_at.endswith('Z') or 'T' in created_at:
                            data_to_save["created_at"] = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        # 尝试解析数字字符串
                        else:
                            data_to_save["created_at"] = datetime.fromtimestamp(float(created_at))
                    except:
                        data_to_save["created_at"] = datetime.now()
                else:
                    data_to_save["created_at"] = created_at
            
            # 只保留表中存在的字段，跳过已处理的时间字段
            logger.info(f"[_save_to_database] 表列: {table_columns}")
            for key, value in data.items():
                if key in table_columns and key not in ['updated_at', 'created_at']:
                    data_to_save[key] = value
                    logger.info(f"[_save_to_database] 字段 {key} 将保存到单独列")
            
            # 确保主键字段存在
            if primary_key not in data_to_save:
                data_to_save[primary_key] = id_value
            
            # 如果有 'data' 字段，存储完整的原始数据（包括nodes、connections、stats等）
            if 'data' in table_columns:
                # 保存完整数据到data字段，排除已单独保存的字段
                full_data = {k: v for k, v in data.items() if k not in data_to_save or k == primary_key}
                # 确保stats字段被包含
                if 'stats' in data and 'stats' not in data_to_save:
                    full_data['stats'] = data['stats']
                    logger.info(f"[_save_to_database] stats字段已添加到full_data")
                if full_data:
                    data_to_save['data'] = json.dumps(full_data)
                    logger.info(f"[_save_to_database] 保存到data字段的键: {list(full_data.keys())}")
                    logger.info(f"[_save_to_database] data字段中的stats: {full_data.get('stats')}")
            else:
                # 如果没有data字段，将stats保存到parameters字段中
                if 'stats' in data and 'parameters' in table_columns:
                    logger.info(f"[_save_to_database] 没有data字段，将stats保存到parameters")
                    # 获取现有的parameters或创建新的
                    params = data_to_save.get('parameters', {})
                    if isinstance(params, str):
                        try:
                            params = json.loads(params)
                        except:
                            params = {}
                    # 将stats保存到parameters._stats
                    params['_stats'] = data['stats']
                    data_to_save['parameters'] = params
                    logger.info(f"[_save_to_database] stats已保存到parameters._stats: {params['_stats']}")
            
            # 处理JSONB字段
            json_fields = []
            for key, value in data_to_save.items():
                if isinstance(value, (dict, list)):
                    json_fields.append(key)
                    data_to_save[key] = json.dumps(value)
            
            if not data_to_save:
                logger.warning(f"没有有效的字段可保存到表 {self.table_name}")
                return False
            
            if exists:
                # 更新现有记录
                update_fields = []
                update_values = []
                
                for key, value in data_to_save.items():
                    if key != primary_key:
                        update_fields.append(f"{key} = %s")
                        update_values.append(value)
                
                if update_fields:
                    update_values.append(id_value)
                    
                    query = f"""
                        UPDATE {self.table_name}
                        SET {', '.join(update_fields)}
                        WHERE {primary_key} = %s
                    """
                    cursor.execute(query, update_values)
            else:
                # 插入新记录
                columns = list(data_to_save.keys())
                values = list(data_to_save.values())
                placeholders = ["%s"] * len(values)
                
                if columns:
                    query = f"""
                        INSERT INTO {self.table_name} ({', '.join(columns)})
                        VALUES ({', '.join(placeholders)})
                    """
                    cursor.execute(query, values)
            
            conn.commit()
            cursor.close()
            return True
            
        except Exception as e:
            logger.warning(f"数据库操作失败: {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            return False
        finally:
            if conn:
                try:
                    return_db_connection(conn)
                except:
                    pass
    
    def _save_to_filesystem(self, data: Dict[str, Any], primary_key: str) -> bool:
        """
        保存数据到文件系统
        
        Args:
            data: 要保存的数据
            primary_key: 主键字段名
        
        Returns:
            是否成功保存
        """
        try:
            self.ensure_directory()
            
            id_value = data.get(primary_key)
            if not id_value:
                logger.error(f"数据缺少主键字段 {primary_key}")
                return False
            
            # 准备数据
            data_to_save = data.copy()
            data_to_save["saved_at"] = time.time()
            data_to_save["updated_at"] = time.time()
            if "created_at" not in data_to_save:
                data_to_save["created_at"] = time.time()
            
            # 压缩大型数据结构
            large_fields = ['equity_curve', 'trades', 'metrics']
            for field in large_fields:
                if field in data_to_save:
                    field_data = data_to_save[field]
                    if field == 'equity_curve' and isinstance(field_data, list):
                        compressed = compress_if_needed(field_data, 'equity_curve')
                    elif field == 'trades' and isinstance(field_data, list):
                        compressed = compress_if_needed(field_data, 'trades')
                    else:
                        compressed = compress_if_needed(field_data, 'json')
                    
                    if compressed['compressed']:
                        data_to_save[field] = compressed
            
            # 保存到文件
            filepath = os.path.join(self.file_dir, f"{id_value}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"文件系统保存失败: {e}")
            return False
    
    def load(self, item_id: str, primary_key: str = "id") -> Optional[Dict[str, Any]]:
        """
        加载数据，优先从数据库加载，失败后从文件系统加载
        
        Args:
            item_id: 要加载的数据ID
            primary_key: 主键字段名
        
        Returns:
            加载的数据，如果不存在返回None
        """
        try:
            # 尝试从数据库加载
            if db_available:
                data = self._load_from_database(item_id, primary_key)
                if data:
                    return data
        except Exception as e:
            logger.warning(f"数据库加载失败: {e}")
        
        # 降级到文件系统
        try:
            data = self._load_from_filesystem(item_id)
            if data:
                return data
        except Exception as e:
            logger.error(f"文件系统加载失败: {e}")
        
        return None
    
    def _load_from_database(self, item_id: str, primary_key: str) -> Optional[Dict[str, Any]]:
        """
        从PostgreSQL数据库加载数据
        
        Args:
            item_id: 要加载的数据ID
            primary_key: 主键字段名
        
        Returns:
            加载的数据，如果不存在返回None
        """
        conn = None
        try:
            conn = get_db_connection()
            if not conn:
                return None
            
            cursor = conn.cursor()
            
            # 查询数据
            cursor.execute(f"SELECT * FROM {self.table_name} WHERE {primary_key} = %s", (item_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # 获取列名
            columns = [desc[0] for desc in cursor.description]
            
            # 构建结果字典
            result = {}
            data_field_content = None
            
            for i, value in enumerate(row):
                column_name = columns[i]
                # 处理JSONB字段
                if isinstance(value, str):
                    try:
                        parsed_value = json.loads(value)
                        # 如果是data字段，保存内容用于后续合并
                        if column_name == 'data':
                            data_field_content = parsed_value
                        else:
                            result[column_name] = parsed_value
                        continue
                    except:
                        pass
                # 处理时间字段
                if isinstance(value, datetime):
                    result[column_name] = value.isoformat()
                else:
                    result[column_name] = value
            
            # 如果有data字段的内容，合并到结果中（data字段包含nodes、connections等完整数据）
            if data_field_content and isinstance(data_field_content, dict):
                # 合并data字段的内容，但不覆盖已有的非空字段
                for key, value in data_field_content.items():
                    # 特殊处理stats字段：如果result中的stats为空对象，也进行合并
                    if key == 'stats' and key in result:
                        existing = result[key]
                        if existing is None or (isinstance(existing, dict) and not existing):
                            result[key] = value
                    elif key not in result or result[key] is None:
                        result[key] = value
            
            # 如果没有data字段，从parameters._stats中读取stats
            if 'stats' not in result or not result['stats']:
                params = result.get('parameters', {})
                if isinstance(params, dict) and '_stats' in params:
                    result['stats'] = params['_stats']
                    logger.info(f"[_load_from_database] 从parameters._stats读取stats: {result['stats']}")
            
            cursor.close()
            return result
            
        except Exception as e:
            logger.warning(f"数据库加载失败: {e}")
            return None
        finally:
            if conn:
                try:
                    return_db_connection(conn)
                except:
                    pass
    
    def _load_from_filesystem(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        从文件系统加载数据
        
        Args:
            item_id: 要加载的数据ID
        
        Returns:
            加载的数据，如果不存在返回None
        """
        try:
            filepath = os.path.join(self.file_dir, f"{item_id}.json")
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 解压可能压缩的数据结构
            large_fields = ['equity_curve', 'trades', 'metrics']
            for field in large_fields:
                if field in data:
                    field_data = data[field]
                    if isinstance(field_data, dict) and field_data.get('compressed'):
                        data[field] = decompress_if_needed(field_data)
            
            return data
        except Exception as e:
            logger.error(f"文件系统加载失败: {e}")
            return None
    
    def list(self, filters: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        列出数据，优先从数据库获取，不足时从文件系统补充
        
        Args:
            filters: 过滤条件
            limit: 返回的最大结果数
        
        Returns:
            数据列表
        """
        results = []
        
        try:
            # 尝试从数据库获取
            if db_available:
                db_results = self._list_from_database(filters, limit)
                if db_results:
                    results.extend(db_results)
        except Exception as e:
            logger.warning(f"数据库列表查询失败: {e}")
        
        # 如果数据库结果不足，从文件系统补充
        if len(results) < limit:
            try:
                file_results = self._list_from_filesystem(filters, limit - len(results))
                # 去重 - 使用正确的字段名'id'而不是'backtest_id'
                existing_ids = {r.get('id') for r in results}
                for result in file_results:
                    if result.get('id') not in existing_ids:
                        results.append(result)
            except Exception as e:
                logger.error(f"文件系统列表查询失败: {e}")
        
        # 按创建时间排序
        def get_sort_key(item):
            """获取排序键，确保类型一致"""
            key = item.get('created_at', item.get('saved_at', 0))
            # 尝试将字符串转换为浮点数（时间戳）
            if isinstance(key, str):
                try:
                    # 尝试解析ISO格式时间
                    if key.endswith('Z') or 'T' in key:
                        return datetime.fromisoformat(key.replace('Z', '+00:00')).timestamp()
                    # 尝试解析纯数字字符串
                    return float(key)
                except (ValueError, TypeError):
                    return 0
            return key
        
        results.sort(key=get_sort_key, reverse=True)
        
        return results[:limit]
    
    def _list_from_database(self, filters: Optional[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """
        从PostgreSQL数据库列出数据
        
        Args:
            filters: 过滤条件
            limit: 返回的最大结果数
        
        Returns:
            数据列表
        """
        conn = None
        try:
            conn = get_db_connection()
            if not conn:
                return []
            
            cursor = conn.cursor()
            
            # 构建查询语句
            query = f"SELECT * FROM {self.table_name}"
            params = []
            
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(f"{key} = %s")
                    params.append(value)
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            # 获取列名
            columns = [desc[0] for desc in cursor.description]
            
            # 构建结果列表
            results = []
            for row in rows:
                result = {}
                data_field_content = None
                
                for i, value in enumerate(row):
                    column_name = columns[i]
                    # 处理JSONB字段
                    if isinstance(value, str):
                        try:
                            parsed_value = json.loads(value)
                            # 如果是data字段，保存内容用于后续合并
                            if column_name == 'data':
                                data_field_content = parsed_value
                            else:
                                result[column_name] = parsed_value
                            continue
                        except:
                            pass
                    # 处理时间字段
                    if isinstance(value, datetime):
                        result[column_name] = value.isoformat()
                    else:
                        result[column_name] = value
                
                # 如果有data字段的内容，合并到结果中（data字段包含nodes、connections等完整数据）
                if data_field_content and isinstance(data_field_content, dict):
                    # 合并data字段的内容，但不覆盖已有的非空字段
                    for key, value in data_field_content.items():
                        # 特殊处理stats字段：如果result中的stats为空对象，也进行合并
                        if key == 'stats' and key in result:
                            existing = result[key]
                            if existing is None or (isinstance(existing, dict) and not existing):
                                result[key] = value
                        elif key not in result or result[key] is None:
                            result[key] = value
                
                # 如果没有data字段，从parameters._stats中读取stats
                if 'stats' not in result or not result['stats']:
                    params = result.get('parameters', {})
                    if isinstance(params, dict) and '_stats' in params:
                        result['stats'] = params['_stats']
                
                results.append(result)
            
            cursor.close()
            return results
            
        except Exception as e:
            logger.warning(f"数据库列表查询失败: {e}")
            return []
        finally:
            if conn:
                try:
                    return_db_connection(conn)
                except:
                    pass
    
    def _list_from_filesystem(self, filters: Optional[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """
        从文件系统列出数据
        
        Args:
            filters: 过滤条件
            limit: 返回的最大结果数
        
        Returns:
            数据列表
        """
        try:
            results = []
            
            for filename in os.listdir(self.file_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.file_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 应用过滤条件
                        if filters:
                            match = True
                            for key, value in filters.items():
                                if data.get(key) != value:
                                    match = False
                                    break
                            if not match:
                                continue
                        
                        results.append(data)
                    except Exception as e:
                        logger.warning(f"加载文件 {filename} 失败: {e}")
            
            # 按保存时间排序
            def get_sort_key(item):
                key = item.get('saved_at') or item.get('created_at') or 0
                # 尝试将字符串转换为浮点数（时间戳）
                if isinstance(key, str):
                    try:
                        # 尝试解析ISO格式时间
                        if key.endswith('Z') or 'T' in key:
                            from datetime import datetime
                            return datetime.fromisoformat(key.replace('Z', '+00:00')).timestamp()
                        # 尝试解析纯数字字符串
                        return float(key)
                    except (ValueError, TypeError):
                        return 0
                return key
            
            results.sort(key=get_sort_key, reverse=True)
            
            return results[:limit]
        except Exception as e:
            logger.error(f"文件系统列表查询失败: {e}")
            return []
    
    def delete(self, item_id: str, primary_key: str = "id") -> bool:
        """
        删除数据，同时从数据库和文件系统删除
        
        Args:
            item_id: 要删除的数据ID
            primary_key: 主键字段名
        
        Returns:
            是否成功删除
        """
        success = False
        
        # 尝试从数据库删除
        try:
            if db_available:
                if self._delete_from_database(item_id, primary_key):
                    success = True
        except Exception as e:
            logger.warning(f"数据库删除失败: {e}")
        
        # 从文件系统删除
        try:
            if self._delete_from_filesystem(item_id):
                success = True
        except Exception as e:
            logger.error(f"文件系统删除失败: {e}")
        
        return success
    
    def batch_save(self, data_list: List[Dict[str, Any]], primary_key: str = "id", batch_size: int = 100) -> Dict[str, Any]:
        """
        批量保存数据，优先使用数据库批量操作，失败后降级到文件系统
        
        Args:
            data_list: 要保存的数据列表
            primary_key: 主键字段名
            batch_size: 每批处理的数据量
        
        Returns:
            批量操作结果，包含成功、失败和跳过的数量
        """
        start_time = time.time()
        total_processed = 0
        total_success = 0
        total_failed = 0
        total_skipped = 0
        
        if not data_list:
            return {
                "success": True,
                "total_processed": 0,
                "success_count": 0,
                "failed_count": 0,
                "skipped_count": 0,
                "processing_time": 0
            }
        
        try:
            # 尝试批量保存到数据库
            if db_available:
                result = self._batch_save_to_database(data_list, primary_key, batch_size)
                total_processed = result.get("total_processed", 0)
                total_success = result.get("success_count", 0)
                total_failed = result.get("failed_count", 0)
                total_skipped = result.get("skipped_count", 0)
                
                if total_processed > 0:
                    logger.info(f"批量保存到数据库完成: 处理 {total_processed} 条，成功 {total_success} 条，失败 {total_failed} 条，跳过 {total_skipped} 条")
                    return {
                        "success": total_failed == 0,
                        "total_processed": total_processed,
                        "success_count": total_success,
                        "failed_count": total_failed,
                        "skipped_count": total_skipped,
                        "processing_time": time.time() - start_time
                    }
        except Exception as e:
            logger.warning(f"数据库批量保存失败: {e}")
        
        # 降级到文件系统批量保存
        try:
            for data in data_list:
                try:
                    if self._save_to_filesystem(data, primary_key):
                        total_success += 1
                    else:
                        total_failed += 1
                except Exception as e:
                    logger.error(f"文件系统保存失败: {e}")
                    total_failed += 1
                finally:
                    total_processed += 1
            
            logger.info(f"批量保存到文件系统完成: 处理 {total_processed} 条，成功 {total_success} 条，失败 {total_failed} 条")
            return {
                "success": total_failed == 0,
                "total_processed": total_processed,
                "success_count": total_success,
                "failed_count": total_failed,
                "skipped_count": total_skipped,
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"文件系统批量保存失败: {e}")
            return {
                "success": False,
                "total_processed": 0,
                "success_count": 0,
                "failed_count": len(data_list),
                "skipped_count": 0,
                "processing_time": time.time() - start_time
            }
    
    def _batch_save_to_database(self, data_list: List[Dict[str, Any]], primary_key: str, batch_size: int) -> Dict[str, Any]:
        """
        批量保存数据到PostgreSQL数据库
        
        Args:
            data_list: 要保存的数据列表
            primary_key: 主键字段名
            batch_size: 每批处理的数据量
        
        Returns:
            批量操作结果
        """
        conn = None
        try:
            conn = get_db_connection()
            if not conn:
                return {"total_processed": 0, "success_count": 0, "failed_count": 0, "skipped_count": 0}
            
            cursor = conn.cursor()
            
            # 检查表是否存在
            cursor.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = '{self.table_name}'
                );
            """)
            
            if not cursor.fetchone()[0]:
                logger.warning(f"表 {self.table_name} 不存在，无法批量保存")
                return {"total_processed": 0, "success_count": 0, "failed_count": 0, "skipped_count": 0}
            
            # 获取表的列信息
            cursor.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{self.table_name}'
            """)
            table_columns = [row[0] for row in cursor.fetchall()]
            
            total_processed = 0
            total_success = 0
            total_failed = 0
            total_skipped = 0
            
            # 分批处理
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                batch_data = []
                
                for data in batch:
                    try:
                        # 检查主键
                        id_value = data.get(primary_key)
                        if not id_value:
                            total_skipped += 1
                            continue
                        
                        # 准备数据
                        data_to_save = {}
                        
                        # 处理时间字段
                        if "updated_at" in table_columns:
                            data_to_save["updated_at"] = datetime.now()
                        
                        if "created_at" in table_columns and "created_at" not in data:
                            data_to_save["created_at"] = datetime.now()
                        elif "created_at" in table_columns and "created_at" in data:
                            created_at = data["created_at"]
                            if isinstance(created_at, (int, float)):
                                data_to_save["created_at"] = datetime.fromtimestamp(created_at)
                            elif isinstance(created_at, str):
                                try:
                                    if created_at.endswith('Z') or 'T' in created_at:
                                        data_to_save["created_at"] = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                                    else:
                                        data_to_save["created_at"] = datetime.fromtimestamp(float(created_at))
                                except:
                                    data_to_save["created_at"] = datetime.now()
                            else:
                                data_to_save["created_at"] = created_at
                        
                        # 处理其他字段
                        for key, value in data.items():
                            if key in table_columns and key not in ['updated_at', 'created_at']:
                                # 处理JSON字段
                                if isinstance(value, (dict, list)):
                                    data_to_save[key] = json.dumps(value)
                                else:
                                    data_to_save[key] = value
                        
                        # 确保主键存在
                        if primary_key not in data_to_save:
                            data_to_save[primary_key] = id_value
                        
                        if data_to_save:
                            batch_data.append(data_to_save)
                    except Exception as e:
                        logger.warning(f"准备数据失败: {e}")
                        total_failed += 1
                    finally:
                        total_processed += 1
                
                # 执行批量插入/更新
                if batch_data:
                    # 使用UPSERT语句
                    columns = list(batch_data[0].keys())
                    placeholders = []
                    values = []
                    
                    for i, data in enumerate(batch_data):
                        row_placeholders = []
                        for col in columns:
                            row_placeholders.append(f"${len(values) + 1}")
                            values.append(data.get(col))
                        placeholders.append(f"({', '.join(row_placeholders)})")
                    
                    # 构建UPDATE子句
                    update_clauses = []
                    for col in columns:
                        if col != primary_key:
                            update_clauses.append(f"{col} = EXCLUDED.{col}")
                    
                    if columns and placeholders:
                        query = f"""
                            INSERT INTO {self.table_name} ({', '.join(columns)})
                            VALUES {', '.join(placeholders)}
                            ON CONFLICT ({primary_key})
                            DO UPDATE SET {', '.join(update_clauses)}
                        """
                        
                        try:
                            cursor.execute(query, values)
                            total_success += len(batch_data)
                        except Exception as e:
                            logger.warning(f"批量执行失败: {e}")
                            total_failed += len(batch_data)
            
            conn.commit()
            cursor.close()
            
            return {
                "total_processed": total_processed,
                "success_count": total_success,
                "failed_count": total_failed,
                "skipped_count": total_skipped
            }
            
        except Exception as e:
            logger.warning(f"数据库批量操作失败: {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            return {"total_processed": 0, "success_count": 0, "failed_count": 0, "skipped_count": 0}
        finally:
            if conn:
                try:
                    return_db_connection(conn)
                except:
                    pass
    
    def batch_delete(self, item_ids: List[str], primary_key: str = "id", batch_size: int = 100) -> Dict[str, Any]:
        """
        批量删除数据，同时从数据库和文件系统删除
        
        Args:
            item_ids: 要删除的数据ID列表
            primary_key: 主键字段名
            batch_size: 每批处理的数据量
        
        Returns:
            批量操作结果，包含成功和失败的数量
        """
        start_time = time.time()
        total_processed = 0
        total_success = 0
        total_failed = 0
        
        if not item_ids:
            return {
                "success": True,
                "total_processed": 0,
                "success_count": 0,
                "failed_count": 0,
                "processing_time": 0
            }
        
        try:
            # 尝试批量删除数据库中的数据
            if db_available:
                result = self._batch_delete_from_database(item_ids, primary_key, batch_size)
                total_processed = result.get("total_processed", 0)
                total_success = result.get("success_count", 0)
                total_failed = result.get("failed_count", 0)
                
                if total_processed > 0:
                    logger.info(f"批量删除数据库记录完成: 处理 {total_processed} 条，成功 {total_success} 条，失败 {total_failed} 条")
        except Exception as e:
            logger.warning(f"数据库批量删除失败: {e}")
        
        # 从文件系统删除
        for item_id in item_ids:
            try:
                if self._delete_from_filesystem(item_id):
                    total_success += 1
                else:
                    total_failed += 1
            except Exception as e:
                logger.error(f"文件系统删除失败: {e}")
                total_failed += 1
            finally:
                total_processed += 1
        
        return {
            "success": total_failed == 0,
            "total_processed": total_processed,
            "success_count": total_success,
            "failed_count": total_failed,
            "processing_time": time.time() - start_time
        }
    
    def _batch_delete_from_database(self, item_ids: List[str], primary_key: str, batch_size: int) -> Dict[str, Any]:
        """
        批量从PostgreSQL数据库删除数据
        
        Args:
            item_ids: 要删除的数据ID列表
            primary_key: 主键字段名
            batch_size: 每批处理的数据量
        
        Returns:
            批量操作结果
        """
        conn = None
        try:
            conn = get_db_connection()
            if not conn:
                return {"total_processed": 0, "success_count": 0, "failed_count": 0}
            
            cursor = conn.cursor()
            total_processed = 0
            total_success = 0
            total_failed = 0
            
            # 分批处理
            for i in range(0, len(item_ids), batch_size):
                batch = item_ids[i:i + batch_size]
                
                if batch:
                    # 构建IN语句
                    placeholders = [f"${j + 1}" for j in range(len(batch))]
                    query = f"DELETE FROM {self.table_name} WHERE {primary_key} IN ({', '.join(placeholders)})"
                    
                    try:
                        cursor.execute(query, batch)
                        deleted_count = cursor.rowcount
                        total_success += deleted_count
                        total_failed += len(batch) - deleted_count
                    except Exception as e:
                        logger.warning(f"批量删除失败: {e}")
                        total_failed += len(batch)
                    finally:
                        total_processed += len(batch)
            
            conn.commit()
            cursor.close()
            
            return {
                "total_processed": total_processed,
                "success_count": total_success,
                "failed_count": total_failed
            }
            
        except Exception as e:
            logger.warning(f"数据库批量删除失败: {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            return {"total_processed": 0, "success_count": 0, "failed_count": 0}
        finally:
            if conn:
                try:
                    return_db_connection(conn)
                except:
                    pass
    
    def _delete_from_database(self, item_id: str, primary_key: str) -> bool:
        """
        从PostgreSQL数据库删除数据
        
        Args:
            item_id: 要删除的数据ID
            primary_key: 主键字段名
        
        Returns:
            是否成功删除
        """
        conn = None
        try:
            conn = get_db_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            cursor.execute(f"DELETE FROM {self.table_name} WHERE {primary_key} = %s", (item_id,))
            conn.commit()
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.warning(f"数据库删除失败: {e}")
            return False
        finally:
            if conn:
                try:
                    return_db_connection(conn)
                except:
                    pass
    
    def _delete_from_filesystem(self, item_id: str) -> bool:
        """
        从文件系统删除数据
        
        Args:
            item_id: 要删除的数据ID
        
        Returns:
            是否成功删除
        """
        try:
            filepath = os.path.join(self.file_dir, f"{item_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
            return False
        except Exception as e:
            logger.error(f"文件系统删除失败: {e}")
            return False


# 全局持久化实例
def get_persistence_manager(table_name: str, file_dir: str) -> UnifiedPersistence:
    """
    获取持久化管理器实例
    
    Args:
        table_name: 数据库表名
        file_dir: 文件系统存储目录
    
    Returns:
        持久化管理器实例
    """
    return UnifiedPersistence(table_name, file_dir)


# 策略开发流程各环节的持久化管理器

def get_strategy_conception_persistence() -> UnifiedPersistence:
    """
    获取策略构思持久化管理器
    """
    return get_persistence_manager('strategy_conceptions', 'strategy_conceptions')


def get_strategy_management_persistence() -> UnifiedPersistence:
    """
    获取策略管理持久化管理器
    """
    return get_persistence_manager('strategy_management', 'strategy_management')


def get_strategy_optimization_persistence() -> UnifiedPersistence:
    """
    获取策略优化持久化管理器
    """
    return get_persistence_manager('strategy_optimizations', 'strategy_optimizations')


def get_strategy_performance_persistence() -> UnifiedPersistence:
    """
    获取策略性能评估持久化管理器
    """
    return get_persistence_manager('strategy_performance_evaluations', 'strategy_performance')


def get_strategy_lifecycle_persistence() -> UnifiedPersistence:
    """
    获取策略部署持久化管理器
    """
    return get_persistence_manager('strategy_lifecycle', 'strategy_lifecycle')


def get_strategy_execution_persistence() -> UnifiedPersistence:
    """
    获取策略执行监控持久化管理器
    """
    return get_persistence_manager('strategy_execution_monitor', 'strategy_execution')


def get_backtest_persistence() -> UnifiedPersistence:
    """
    获取回测结果持久化管理器
    """
    return get_persistence_manager('backtest_results', 'backtest_results')
