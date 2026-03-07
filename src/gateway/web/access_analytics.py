"""
访问统计和使用分析模块
用于记录用户行为、页面访问和操作统计
"""

import logging
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# 使用统一日志系统
logger = logging.getLogger(__name__)

# 延迟导入数据库连接模块
db_available = False
try:
    from .postgresql_persistence import get_db_connection, return_db_connection
    db_available = True
except ImportError:
    logger.warning("PostgreSQL持久化模块不可用，将使用内存存储")

# 内存存储作为降级方案
memory_storage = {
    "page_views": [],
    "user_actions": [],
    "api_calls": []
}


class AccessAnalytics:
    """
    访问统计和使用分析类
    负责记录和分析用户行为
    """
    
    @staticmethod
    def ensure_analytics_tables() -> bool:
        """
        确保分析表存在
        
        Returns:
            是否成功创建表
        """
        if not db_available:
            return False
        
        conn = None
        try:
            conn = get_db_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            # 创建页面访问表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS page_views (
                    id UUID PRIMARY KEY,
                    page_url VARCHAR(255) NOT NULL,
                    user_agent TEXT,
                    ip_address VARCHAR(50),
                    referer TEXT,
                    session_id VARCHAR(100),
                    duration INTEGER,  -- 停留时间（秒）
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # 创建用户操作表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_actions (
                    id UUID PRIMARY KEY,
                    action_type VARCHAR(50) NOT NULL,
                    page_url VARCHAR(255),
                    element_id VARCHAR(100),
                    event_data JSONB,
                    session_id VARCHAR(100),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # 创建API调用表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_calls (
                    id UUID PRIMARY KEY,
                    endpoint VARCHAR(255) NOT NULL,
                    method VARCHAR(10) NOT NULL,
                    status_code INTEGER,
                    response_time INTEGER,  -- 响应时间（毫秒）
                    user_agent TEXT,
                    ip_address VARCHAR(50),
                    session_id VARCHAR(100),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # 创建索引
            cursor.execute("""
                -- 页面访问表索引
                CREATE INDEX IF NOT EXISTS idx_page_views_url ON page_views(page_url);
                CREATE INDEX IF NOT EXISTS idx_page_views_timestamp ON page_views(timestamp);
                CREATE INDEX IF NOT EXISTS idx_page_views_session ON page_views(session_id);
                
                -- 用户操作表索引
                CREATE INDEX IF NOT EXISTS idx_user_actions_type ON user_actions(action_type);
                CREATE INDEX IF NOT EXISTS idx_user_actions_timestamp ON user_actions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_user_actions_session ON user_actions(session_id);
                
                -- API调用表索引
                CREATE INDEX IF NOT EXISTS idx_api_calls_endpoint ON api_calls(endpoint);
                CREATE INDEX IF NOT EXISTS idx_api_calls_status ON api_calls(status_code);
                CREATE INDEX IF NOT EXISTS idx_api_calls_timestamp ON api_calls(timestamp);
            """)
            
            conn.commit()
            cursor.close()
            
            logger.info("分析表创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建分析表失败: {e}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return False
        finally:
            if conn:
                return_db_connection(conn)
    
    @staticmethod
    def record_page_view(page_url: str, user_agent: str = None, ip_address: str = None, 
                        referer: str = None, session_id: str = None, duration: int = None) -> bool:
        """
        记录页面访问
        
        Args:
            page_url: 页面URL
            user_agent: 用户代理
            ip_address: IP地址
            referer: 来源URL
            session_id: 会话ID
            duration: 停留时间（秒）
        
        Returns:
            是否成功记录
        """
        try:
            record_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()
            
            if db_available:
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO page_views (id, page_url, user_agent, ip_address, referer, session_id, duration, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (record_id, page_url, user_agent, ip_address, referer, session_id, duration, timestamp))
                    conn.commit()
                    cursor.close()
                    return_db_connection(conn)
                    return True
            
            # 降级到内存存储
            memory_storage["page_views"].append({
                "id": record_id,
                "page_url": page_url,
                "user_agent": user_agent,
                "ip_address": ip_address,
                "referer": referer,
                "session_id": session_id,
                "duration": duration,
                "timestamp": timestamp.isoformat()
            })
            
            # 限制内存存储大小
            if len(memory_storage["page_views"]) > 1000:
                memory_storage["page_views"] = memory_storage["page_views"][-500:]
            
            return True
            
        except Exception as e:
            logger.error(f"记录页面访问失败: {e}")
            return False
    
    @staticmethod
    def record_user_action(action_type: str, page_url: str = None, element_id: str = None, 
                          event_data: Dict[str, Any] = None, session_id: str = None) -> bool:
        """
        记录用户操作
        
        Args:
            action_type: 操作类型
            page_url: 页面URL
            element_id: 元素ID
            event_data: 事件数据
            session_id: 会话ID
        
        Returns:
            是否成功记录
        """
        try:
            record_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()
            
            if db_available:
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO user_actions (id, action_type, page_url, element_id, event_data, session_id, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (record_id, action_type, page_url, element_id, event_data, session_id, timestamp))
                    conn.commit()
                    cursor.close()
                    return_db_connection(conn)
                    return True
            
            # 降级到内存存储
            memory_storage["user_actions"].append({
                "id": record_id,
                "action_type": action_type,
                "page_url": page_url,
                "element_id": element_id,
                "event_data": event_data,
                "session_id": session_id,
                "timestamp": timestamp.isoformat()
            })
            
            # 限制内存存储大小
            if len(memory_storage["user_actions"]) > 2000:
                memory_storage["user_actions"] = memory_storage["user_actions"][-1000:]
            
            return True
            
        except Exception as e:
            logger.error(f"记录用户操作失败: {e}")
            return False
    
    @staticmethod
    def record_api_call(endpoint: str, method: str, status_code: int = None, response_time: int = None, 
                       user_agent: str = None, ip_address: str = None, session_id: str = None) -> bool:
        """
        记录API调用
        
        Args:
            endpoint: API端点
            method: HTTP方法
            status_code: 状态码
            response_time: 响应时间（毫秒）
            user_agent: 用户代理
            ip_address: IP地址
            session_id: 会话ID
        
        Returns:
            是否成功记录
        """
        try:
            record_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()
            
            if db_available:
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO api_calls (id, endpoint, method, status_code, response_time, user_agent, ip_address, session_id, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (record_id, endpoint, method, status_code, response_time, user_agent, ip_address, session_id, timestamp))
                    conn.commit()
                    cursor.close()
                    return_db_connection(conn)
                    return True
            
            # 降级到内存存储
            memory_storage["api_calls"].append({
                "id": record_id,
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "response_time": response_time,
                "user_agent": user_agent,
                "ip_address": ip_address,
                "session_id": session_id,
                "timestamp": timestamp.isoformat()
            })
            
            # 限制内存存储大小
            if len(memory_storage["api_calls"]) > 2000:
                memory_storage["api_calls"] = memory_storage["api_calls"][-1000:]
            
            return True
            
        except Exception as e:
            logger.error(f"记录API调用失败: {e}")
            return False
    
    @staticmethod
    def get_page_view_stats(start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        获取页面访问统计
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
        
        Returns:
            页面访问统计
        """
        if db_available:
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    
                    query = """
                        SELECT 
                            page_url,
                            COUNT(*) as view_count,
                            AVG(duration) as avg_duration
                        FROM page_views
                    """
                    
                    where_clause = []
                    params = []
                    
                    if start_time:
                        where_clause.append("timestamp >= %s")
                        params.append(start_time)
                    if end_time:
                        where_clause.append("timestamp <= %s")
                        params.append(end_time)
                    
                    if where_clause:
                        query += " WHERE " + " AND ".join(where_clause)
                    
                    query += " GROUP BY page_url ORDER BY view_count DESC LIMIT 20"
                    
                    cursor.execute(query, params)
                    
                    results = []
                    for row in cursor.fetchall():
                        results.append({
                            "page_url": row[0],
                            "view_count": row[1],
                            "avg_duration": row[2]
                        })
                    
                    cursor.close()
                    return_db_connection(conn)
                    
                    return {
                        "total_views": sum(r["view_count"] for r in results),
                        "pages": results
                    }
                    
                except Exception as e:
                    logger.error(f"获取页面访问统计失败: {e}")
                    return_db_connection(conn)
        
        # 降级到内存存储
        filtered_views = memory_storage["page_views"]
        if start_time:
            filtered_views = [v for v in filtered_views if datetime.fromisoformat(v["timestamp"]) >= start_time]
        if end_time:
            filtered_views = [v for v in filtered_views if datetime.fromisoformat(v["timestamp"]) <= end_time]
        
        # 按页面URL分组
        page_stats = {}
        for view in filtered_views:
            page_url = view["page_url"]
            if page_url not in page_stats:
                page_stats[page_url] = {
                    "view_count": 0,
                    "total_duration": 0,
                    "durations": []
                }
            page_stats[page_url]["view_count"] += 1
            if view["duration"]:
                page_stats[page_url]["total_duration"] += view["duration"]
                page_stats[page_url]["durations"].append(view["duration"])
        
        # 构建结果
        results = []
        for page_url, stats in page_stats.items():
            avg_duration = None
            if stats["durations"]:
                avg_duration = sum(stats["durations"]) / len(stats["durations"])
            results.append({
                "page_url": page_url,
                "view_count": stats["view_count"],
                "avg_duration": avg_duration
            })
        
        results.sort(key=lambda x: x["view_count"], reverse=True)
        
        return {
            "total_views": sum(r["view_count"] for r in results),
            "pages": results[:20]
        }
    
    @staticmethod
    def get_user_action_stats(start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        获取用户操作统计
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
        
        Returns:
            用户操作统计
        """
        if db_available:
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    
                    query = """
                        SELECT 
                            action_type,
                            COUNT(*) as action_count
                        FROM user_actions
                    """
                    
                    where_clause = []
                    params = []
                    
                    if start_time:
                        where_clause.append("timestamp >= %s")
                        params.append(start_time)
                    if end_time:
                        where_clause.append("timestamp <= %s")
                        params.append(end_time)
                    
                    if where_clause:
                        query += " WHERE " + " AND ".join(where_clause)
                    
                    query += " GROUP BY action_type ORDER BY action_count DESC"
                    
                    cursor.execute(query, params)
                    
                    results = []
                    for row in cursor.fetchall():
                        results.append({
                            "action_type": row[0],
                            "action_count": row[1]
                        })
                    
                    cursor.close()
                    return_db_connection(conn)
                    
                    return {
                        "total_actions": sum(r["action_count"] for r in results),
                        "actions": results
                    }
                    
                except Exception as e:
                    logger.error(f"获取用户操作统计失败: {e}")
                    return_db_connection(conn)
        
        # 降级到内存存储
        filtered_actions = memory_storage["user_actions"]
        if start_time:
            filtered_actions = [a for a in filtered_actions if datetime.fromisoformat(a["timestamp"]) >= start_time]
        if end_time:
            filtered_actions = [a for a in filtered_actions if datetime.fromisoformat(a["timestamp"]) <= end_time]
        
        # 按操作类型分组
        action_stats = {}
        for action in filtered_actions:
            action_type = action["action_type"]
            if action_type not in action_stats:
                action_stats[action_type] = 0
            action_stats[action_type] += 1
        
        # 构建结果
        results = []
        for action_type, count in action_stats.items():
            results.append({
                "action_type": action_type,
                "action_count": count
            })
        
        results.sort(key=lambda x: x["action_count"], reverse=True)
        
        return {
            "total_actions": sum(r["action_count"] for r in results),
            "actions": results
        }
    
    @staticmethod
    def get_api_call_stats(start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        获取API调用统计
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
        
        Returns:
            API调用统计
        """
        if db_available:
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    
                    query = """
                        SELECT 
                            endpoint,
                            method,
                            COUNT(*) as call_count,
                            AVG(response_time) as avg_response_time,
                            COUNT(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 END) as success_count,
                            COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count
                        FROM api_calls
                    """
                    
                    where_clause = []
                    params = []
                    
                    if start_time:
                        where_clause.append("timestamp >= %s")
                        params.append(start_time)
                    if end_time:
                        where_clause.append("timestamp <= %s")
                        params.append(end_time)
                    
                    if where_clause:
                        query += " WHERE " + " AND ".join(where_clause)
                    
                    query += " GROUP BY endpoint, method ORDER BY call_count DESC LIMIT 20"
                    
                    cursor.execute(query, params)
                    
                    results = []
                    for row in cursor.fetchall():
                        results.append({
                            "endpoint": row[0],
                            "method": row[1],
                            "call_count": row[2],
                            "avg_response_time": row[3],
                            "success_count": row[4],
                            "error_count": row[5]
                        })
                    
                    cursor.close()
                    return_db_connection(conn)
                    
                    return {
                        "total_calls": sum(r["call_count"] for r in results),
                        "endpoints": results
                    }
                    
                except Exception as e:
                    logger.error(f"获取API调用统计失败: {e}")
                    return_db_connection(conn)
        
        # 降级到内存存储
        filtered_calls = memory_storage["api_calls"]
        if start_time:
            filtered_calls = [c for c in filtered_calls if datetime.fromisoformat(c["timestamp"]) >= start_time]
        if end_time:
            filtered_calls = [c for c in filtered_calls if datetime.fromisoformat(c["timestamp"]) <= end_time]
        
        # 按端点和方法分组
        api_stats = {}
        for call in filtered_calls:
            key = f"{call['endpoint']}:{call['method']}"
            if key not in api_stats:
                api_stats[key] = {
                    "endpoint": call["endpoint"],
                    "method": call["method"],
                    "call_count": 0,
                    "total_response_time": 0,
                    "response_times": [],
                    "success_count": 0,
                    "error_count": 0
                }
            
            api_stats[key]["call_count"] += 1
            if call["response_time"]:
                api_stats[key]["total_response_time"] += call["response_time"]
                api_stats[key]["response_times"].append(call["response_time"])
            if call["status_code"]:
                if 200 <= call["status_code"] < 300:
                    api_stats[key]["success_count"] += 1
                elif call["status_code"] >= 400:
                    api_stats[key]["error_count"] += 1
        
        # 构建结果
        results = []
        for stats in api_stats.values():
            avg_response_time = None
            if stats["response_times"]:
                avg_response_time = sum(stats["response_times"]) / len(stats["response_times"])
            results.append({
                "endpoint": stats["endpoint"],
                "method": stats["method"],
                "call_count": stats["call_count"],
                "avg_response_time": avg_response_time,
                "success_count": stats["success_count"],
                "error_count": stats["error_count"]
            })
        
        results.sort(key=lambda x: x["call_count"], reverse=True)
        
        return {
            "total_calls": sum(r["call_count"] for r in results),
            "endpoints": results[:20]
        }


# 全局访问分析实例
access_analytics = AccessAnalytics()


# 工具函数
def track_page_view(page_url: str, **kwargs) -> bool:
    """
    跟踪页面访问
    
    Args:
        page_url: 页面URL
        **kwargs: 其他参数
    
    Returns:
        是否成功跟踪
    """
    return AccessAnalytics.record_page_view(page_url, **kwargs)


def track_user_action(action_type: str, **kwargs) -> bool:
    """
    跟踪用户操作
    
    Args:
        action_type: 操作类型
        **kwargs: 其他参数
    
    Returns:
        是否成功跟踪
    """
    return AccessAnalytics.record_user_action(action_type, **kwargs)


def track_api_call(endpoint: str, method: str, **kwargs) -> bool:
    """
    跟踪API调用
    
    Args:
        endpoint: API端点
        method: HTTP方法
        **kwargs: 其他参数
    
    Returns:
        是否成功跟踪
    """
    return AccessAnalytics.record_api_call(endpoint, method, **kwargs)


def get_analytics_summary(start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, Any]:
    """
    获取分析摘要
    
    Args:
        start_time: 开始时间
        end_time: 结束时间
    
    Returns:
        分析摘要
    """
    return {
        "page_views": AccessAnalytics.get_page_view_stats(start_time, end_time),
        "user_actions": AccessAnalytics.get_user_action_stats(start_time, end_time),
        "api_calls": AccessAnalytics.get_api_call_stats(start_time, end_time)
    }


# 初始化函数
def initialize_analytics():
    """
    初始化分析模块
    """
    # 确保分析表存在
    AccessAnalytics.ensure_analytics_tables()
    logger.info("访问统计和使用分析模块初始化成功")


# 测试函数
def test_analytics():
    """
    测试分析功能
    """
    print("测试页面访问跟踪...")
    AccessAnalytics.record_page_view(
        page_url="/strategy-performance-evaluation",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        session_id="test-session-1",
        duration=120
    )
    
    print("测试用户操作跟踪...")
    AccessAnalytics.record_user_action(
        action_type="refresh",
        page_url="/strategy-performance-evaluation",
        element_id="refresh-button",
        session_id="test-session-1"
    )
    
    print("测试API调用跟踪...")
    AccessAnalytics.record_api_call(
        endpoint="/api/v1/strategy/performance/comparison",
        method="GET",
        status_code=200,
        response_time=150,
        session_id="test-session-1"
    )
    
    print("\n测试统计数据获取...")
    summary = get_analytics_summary()
    print(f"页面访问统计: {summary['page_views']}")
    print(f"用户操作统计: {summary['user_actions']}")
    print(f"API调用统计: {summary['api_calls']}")


if __name__ == "__main__":
    initialize_analytics()
    test_analytics()
