"""
数据源健康检测器
定期自动测试所有数据源连接，自动禁用不健康的数据源
符合架构设计：使用ServiceContainer进行依赖管理
"""

import asyncio
import logging
import aiohttp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import asyncpg

logger = logging.getLogger(__name__)


class HealthStatusEnum(str, Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class HealthStatus:
    """健康状态数据类"""
    source_id: str
    status: HealthStatusEnum
    response_time_ms: int
    message: str
    check_time: datetime
    consecutive_failures: int = 0


@dataclass
class HealthCheckConfig:
    """健康检测配置"""
    check_interval_seconds: int = 300  # 5分钟
    timeout_seconds: int = 10
    max_concurrent_checks: int = 5
    auto_disable_threshold: int = 3  # 连续失败3次自动禁用
    response_time_threshold_ms: int = 10000  # 10秒


class DataSourceHealthChecker:
    """数据源健康检测器
    
    定期自动测试所有数据源连接，自动禁用不健康的数据源
    """
    
    def __init__(self, config: Optional[HealthCheckConfig] = None):
        self.config = config or HealthCheckConfig()
        self._running = False
        self._check_task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._db_pool: Optional[asyncpg.Pool] = None
    
    async def _get_db_pool(self) -> asyncpg.Pool:
        """获取数据库连接池"""
        if self._db_pool is None:
            # 从环境变量获取数据库密码
            import os
            db_password = os.environ.get('DB_PASSWORD', 'SecurePass123!')
            self._db_pool = await asyncpg.create_pool(
                host="rqa2025-postgres",
                port=5432,
                database="rqa2025_prod",
                user="rqa2025_admin",
                password=db_password,
                min_size=2,
                max_size=5
            )
        return self._db_pool
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def check_health(self, source_id: str, source_config: Dict[str, Any]) -> HealthStatus:
        """检测单个数据源健康状态
        
        Args:
            source_id: 数据源ID
            source_config: 数据源配置
            
        Returns:
            健康状态
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            url = source_config.get('url', '')
            if not url:
                return HealthStatus(
                    source_id=source_id,
                    status=HealthStatusEnum.ERROR,
                    response_time_ms=0,
                    message='URL配置为空',
                    check_time=datetime.now()
                )
            
            # 测试连接
            session = await self._get_session()
            async with session.get(url) as response:
                response_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
                
                if response.status == 200:
                    status = HealthStatusEnum.HEALTHY
                    message = '连接正常'
                else:
                    status = HealthStatusEnum.UNHEALTHY
                    message = f'HTTP {response.status}'
                
                return HealthStatus(
                    source_id=source_id,
                    status=status,
                    response_time_ms=response_time,
                    message=message,
                    check_time=datetime.now()
                )
                
        except asyncio.TimeoutError:
            return HealthStatus(
                source_id=source_id,
                status=HealthStatusEnum.TIMEOUT,
                response_time_ms=self.config.timeout_seconds * 1000,
                message='连接超时',
                check_time=datetime.now()
            )
        except Exception as e:
            return HealthStatus(
                source_id=source_id,
                status=HealthStatusEnum.ERROR,
                response_time_ms=0,
                message=str(e),
                check_time=datetime.now()
            )
    
    async def check_all_health(self) -> List[HealthStatus]:
        """检测所有数据源健康状态
        
        Returns:
            所有数据源的健康状态列表
        """
        from src.gateway.web.data_source_config_manager import get_data_source_config_manager
        
        config_manager = get_data_source_config_manager()
        sources = config_manager.get_data_sources()
        
        results = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_checks)
        
        async def check_with_limit(source: Dict[str, Any]) -> HealthStatus:
            async with semaphore:
                source_id = source.get('id', '')
                return await self.check_health(source_id, source)
        
        # 并行检测所有数据源
        tasks = [check_with_limit(source) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                source_id = sources[i].get('id', f'unknown_{i}')
                valid_results.append(HealthStatus(
                    source_id=source_id,
                    status=HealthStatusEnum.ERROR,
                    response_time_ms=0,
                    message=str(result),
                    check_time=datetime.now()
                ))
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _save_health_log(self, health_status: HealthStatus):
        """保存健康检测日志到数据库"""
        try:
            logger.info(f"开始保存健康检测日志: {health_status.source_id}")
            pool = await self._get_db_pool()
            logger.info(f"获取数据库连接池成功")
            async with pool.acquire() as conn:
                logger.info(f"获取数据库连接成功")
                # 查询连续失败次数
                row = await conn.fetchrow(
                    """
                    SELECT consecutive_failures 
                    FROM data_source_health_log 
                    WHERE source_id = $1 
                    ORDER BY check_time DESC 
                    LIMIT 1
                    """,
                    health_status.source_id
                )
                
                consecutive_failures = 0
                if row and health_status.status != HealthStatusEnum.HEALTHY:
                    consecutive_failures = row['consecutive_failures'] + 1
                elif health_status.status == HealthStatusEnum.HEALTHY:
                    consecutive_failures = 0
                elif row:
                    consecutive_failures = row['consecutive_failures'] + 1
                
                # 插入日志
                logger.info(f"准备插入日志: {health_status.source_id}, {health_status.status.value}")
                await conn.execute(
                    """
                    INSERT INTO data_source_health_log 
                    (source_id, check_time, status, response_time_ms, error_message, consecutive_failures)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    health_status.source_id,
                    health_status.check_time,
                    health_status.status.value,
                    health_status.response_time_ms,
                    health_status.message if health_status.status != HealthStatusEnum.HEALTHY else None,
                    consecutive_failures
                )
                logger.info(f"日志插入成功: {health_status.source_id}")
                
                health_status.consecutive_failures = consecutive_failures
                
        except Exception as e:
            logger.error(f"保存健康检测日志失败: {e}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise  # 重新抛出异常以便上层处理
    
    async def auto_disable_unhealthy(self, health_statuses: List[HealthStatus]):
        """自动禁用不健康的数据源
        
        Args:
            health_statuses: 健康状态列表
        """
        from src.gateway.web.data_source_config_manager import get_data_source_config_manager
        
        config_manager = get_data_source_config_manager()
        
        for status in health_statuses:
            if status.consecutive_failures >= self.config.auto_disable_threshold:
                try:
                    # 自动禁用数据源
                    source = config_manager.get_data_source(status.source_id)
                    if source and source.get('enabled', True):
                        source['enabled'] = False
                        source['auto_disabled'] = True
                        source['auto_disabled_reason'] = f"连续{status.consecutive_failures}次检测失败"
                        source['auto_disabled_at'] = datetime.now().isoformat()
                        config_manager.update_data_source(status.source_id, source)
                        
                        logger.warning(
                            f"数据源 {status.source_id} 已被自动禁用，"
                            f"原因: 连续{status.consecutive_failures}次检测失败"
                        )
                except Exception as e:
                    logger.error(f"自动禁用数据源 {status.source_id} 失败: {e}")
    
    async def _check_loop(self):
        """健康检测循环"""
        while self._running:
            try:
                logger.info("开始数据源健康检测...")
                
                # 检测所有数据源
                results = await self.check_all_health()
                
                # 保存日志并检查告警
                for result in results:
                    await self._save_health_log(result)
                    
                    # 检查并发送告警
                    try:
                        from src.gateway.web.datasource_alert_manager import get_alert_manager
                        alert_manager = get_alert_manager()
                        await alert_manager.check_and_alert(result)
                    except Exception as e:
                        logger.error(f"告警检查失败: {e}")
                
                # 自动禁用不健康的数据源
                await self.auto_disable_unhealthy(results)
                
                healthy_count = sum(1 for r in results if r.status == HealthStatusEnum.HEALTHY)
                logger.info(
                    f"健康检测完成: 总计 {len(results)} 个数据源，"
                    f"正常 {healthy_count} 个，异常 {len(results) - healthy_count} 个"
                )
                
            except Exception as e:
                logger.error(f"健康检测循环异常: {e}")
            
            # 等待下一次检测
            await asyncio.sleep(self.config.check_interval_seconds)
    
    def start(self):
        """启动健康检测服务"""
        if not self._running:
            self._running = True
            self._check_task = asyncio.create_task(self._check_loop())
            logger.info(f"数据源健康检测服务已启动，检测间隔: {self.config.check_interval_seconds}秒")
    
    def stop(self):
        """停止健康检测服务"""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
        if self._session and not self._session.closed:
            asyncio.create_task(self._session.close())
        logger.info("数据源健康检测服务已停止")
    
    async def get_health_history(
        self, 
        source_id: Optional[str] = None, 
        limit: int = 100,
        hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """获取健康检测历史
        
        Args:
            source_id: 数据源ID（可选）
            limit: 返回记录数限制
            hours: 最近几小时的数据
            
        Returns:
            健康检测历史记录
        """
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                query = """
                    SELECT log_id, source_id, check_time, status, 
                           response_time_ms, error_message, consecutive_failures
                    FROM data_source_health_log
                    WHERE 1=1
                """
                params = []
                
                if source_id:
                    query += f" AND source_id = ${len(params) + 1}"
                    params.append(source_id)
                
                if hours:
                    query += f" AND check_time >= ${len(params) + 1}"
                    params.append(datetime.now() - timedelta(hours=hours))
                
                query += " ORDER BY check_time DESC LIMIT {}".format(limit)
                
                rows = await conn.fetch(query, *params)
                
                return [
                    {
                        'log_id': row['log_id'],
                        'source_id': row['source_id'],
                        'check_time': row['check_time'].isoformat(),
                        'status': row['status'],
                        'response_time_ms': row['response_time_ms'],
                        'error_message': row['error_message'],
                        'consecutive_failures': row['consecutive_failures']
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"获取健康检测历史失败: {e}")
            return []
    
    async def get_latest_health(self, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取最新健康状态
        
        Args:
            source_id: 数据源ID（可选，不传则返回所有数据源的最新状态）
            
        Returns:
            最新健康状态列表
        """
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                if source_id:
                    # 获取指定数据源的最新状态
                    row = await conn.fetchrow(
                        """
                        SELECT log_id, source_id, check_time, status, 
                               response_time_ms, error_message, consecutive_failures
                        FROM data_source_health_log
                        WHERE source_id = $1
                        ORDER BY check_time DESC
                        LIMIT 1
                        """,
                        source_id
                    )
                    return [dict(row)] if row else []
                else:
                    # 获取所有数据源的最新状态
                    rows = await conn.fetch(
                        """
                        SELECT DISTINCT ON (source_id)
                            log_id, source_id, check_time, status, 
                            response_time_ms, error_message, consecutive_failures
                        FROM data_source_health_log
                        ORDER BY source_id, check_time DESC
                        """
                    )
                    return [dict(row) for row in rows]
                    
        except Exception as e:
            logger.error(f"获取最新健康状态失败: {e}")
            return []


# 全局健康检测器实例
_health_checker: Optional[DataSourceHealthChecker] = None


def get_health_checker(config: Optional[HealthCheckConfig] = None) -> DataSourceHealthChecker:
    """获取健康检测器实例（单例模式）"""
    global _health_checker
    if _health_checker is None:
        _health_checker = DataSourceHealthChecker(config)
    return _health_checker
