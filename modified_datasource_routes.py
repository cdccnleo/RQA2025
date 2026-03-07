"""
数据源路由模块
包含数据源配置、测试、采集等相关API端点
符合架构设计：使用EventBus进行事件通信，使用ServiceContainer进行依赖管理，使用BusinessProcessOrchestrator进行业务流程编排
"""

import time
import socket
import asyncio
import logging
import aiohttp
from typing import List, Dict
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

# 移除循环导入，改为函数内部导入
# from .api import load_data_sources, save_data_sources
# from .websocket_manager import websocket_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# 导出导入路由（必须在带参数的路由之前定义）
@router.get("/api/v1/data/sources/export")
async def export_data_sources_config():
    """导出数据源配置"""
    try:
        from src.gateway.web.data_source_config_manager import get_data_source_config_manager
        config_manager = get_data_source_config_manager()
        config_data = config_manager.export_config()
        logger.info(f"配置导出成功，数据源数量: {len(config_data.get('data_sources', []))}")
        return config_data
    except Exception as e:
        logger.error(f"导出配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")


@router.post("/api/v1/data/sources/import")
async def import_data_sources_config(config_data: dict):
    """导入数据源配置"""
    try:
        from src.gateway.web.data_source_config_manager import get_data_source_config_manager
        config_manager = get_data_source_config_manager()
        success = config_manager.import_config(config_data)
        if success:
            logger.info(f"配置导入成功，数据源数量: {len(config_data.get('data_sources', []))}")
            return {"success": True, "message": "配置导入成功"}
        else:
            logger.error("配置导入失败")
            raise HTTPException(status_code=400, detail="配置导入失败")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导入配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"导入失败: {str(e)}")

# 全局服务容器（延迟初始化，符合架构设计）
_container = None


def _get_container():
    """获取服务容器实例（单例模式，符合架构设计）"""
    global _container
    if _container is None:
        try:
            from src.core.container.container import DependencyContainer
            _container = DependencyContainer()
            
            # 注册事件总线（符合架构设计：事件驱动通信）
            try:
                from src.core.event_bus.core import EventBus
                event_bus = EventBus()
                event_bus.initialize()
                _container.register(
                    "event_bus",
                    service=event_bus,
                    lifecycle="singleton"
                )
                logger.info("事件总线已注册到服务容器")
            except Exception as e:
                logger.warning(f"注册事件总线失败: {e}")
            
            # 注册业务流程编排器（符合架构设计：业务流程管理）
            try:
                from src.core.orchestration.orchestrator_refactored import BusinessProcessOrchestrator
                _container.register(
                    "business_process_orchestrator",
                    factory=lambda: BusinessProcessOrchestrator(),
                    lifecycle="singleton"
                )
                logger.info("业务流程编排器已注册到服务容器")
            except Exception as e:
                logger.debug(f"注册业务流程编排器失败（可选）: {e}")
            
            logger.info("服务容器初始化成功")
        except Exception as e:
            logger.error(f"服务容器初始化失败: {e}")
            return None
    return _container


def _get_event_bus():
    """通过服务容器获取事件总线实例（符合架构设计）"""
    container = _get_container()
    if container:
        try:
            event_bus = container.resolve("event_bus")
            return event_bus
        except Exception as e:
            logger.debug(f"获取事件总线失败: {e}")
            return None
    return None


def _get_orchestrator():
    """通过服务容器获取业务流程编排器实例（符合架构设计）"""
    container = _get_container()
    if container:
        try:
            orchestrator = container.resolve("business_process_orchestrator")
            return orchestrator
        except Exception as e:
            logger.debug(f"获取业务流程编排器失败: {e}")
            return None
    return None


@router.get("/api/v1/data/status")
async def data_status():
    """数据服务状态"""
    from src.gateway.web.config_manager import load_data_sources
    sources = load_data_sources()
    return {
        "service": "data",
        "status": "healthy",
        "data_sources": len(sources),
        "active_sources": len([s for s in sources if s.get("enabled", True)]),
        "processed_records": 0,
        "last_update": time.time()
    }


@router.get("/api/v1/data/scheduler/dashboard")
async def get_scheduler_dashboard():
    """获取调度器监控面板数据"""
    try:
        try:
            from src.core.orchestration.business_process.service_scheduler import get_data_collection_scheduler
            import psutil
            import time
            from datetime import datetime

            scheduler = get_data_collection_scheduler()
            status = scheduler.get_status()


            # 获取系统负载信息
            try:
                cpu_usage = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                memory_usage = memory.percent
            except Exception as e:
                cpu_usage = 0.0
                memory_usage = 0.0

            # 计算运行时间
            uptime_seconds = 0
            if status.get("startup_time"):
                try:
                    startup_time = datetime.fromisoformat(status["startup_time"])
                    current_time = datetime.now()
                    uptime_seconds = (current_time - startup_time.replace(tzinfo=None)).total_seconds()
                except:
                    uptime_seconds = 0

            # 格式化运行时间
            if uptime_seconds > 0:
                hours = int(uptime_seconds // 3600)
                minutes = int((uptime_seconds % 3600) // 60)
                uptime_str = f"{hours}h {minutes}m"
            else:
                uptime_str = "0h 0m"

            return {
                "scheduler": {
                    "running": scheduler.is_running(),
                    "uptime": uptime_str,
                    "uptime_seconds": uptime_seconds,
                    "active_sources": status.get("enabled_sources_count", 0),
                    "total_sources": status.get("enabled_sources_count", 0),  # 暂时使用相同值
                    "last_check": status.get("check_interval", 30),
                    "concurrent_limit": getattr(scheduler, 'max_concurrent_tasks', 3),
                    "active_tasks": len(getattr(scheduler, 'active_tasks', set())),
                    "startup_path": status.get("startup_path", "unknown")
                },
                "performance": {
                    "cpu_usage": round(cpu_usage, 1),
                    "memory_usage": round(memory_usage, 1)
                },
                "recent_activity": status.get("last_collection_times", {}),
                "timestamp": time.time()
            }
        except ImportError:
            # 模块不存在时返回默认仪表板数据
            import time
            return {
                "scheduler": {
                    "running": False,
                    "uptime": "0h 0m",
                    "uptime_seconds": 0,
                    "active_sources": 0,
                    "total_sources": 0,
                    "last_check": 30,
                    "concurrent_limit": 3,
                    "active_tasks": 0,
                    "startup_path": "unknown"
                },
                "performance": {
                    "cpu_usage": 0.0,
                    "memory_usage": 0.0
                },
                "recent_activity": {},
                "timestamp": time.time(),
                "message": "调度器模块未初始化"
            }

    except Exception as e:
        logger.error(f"获取调度器监控面板数据失败: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取监控数据失败: {str(e)}")


@router.post("/api/v1/data/scheduler/control")
async def control_scheduler(request: dict):
    """控制调度器启动/停止"""
    try:
        action = request.get("action", "status")
        from src.core.orchestration.business_process.service_scheduler import (
            get_data_collection_scheduler,
            start_data_collection_scheduler,
            stop_data_collection_scheduler
        )

        scheduler = get_data_collection_scheduler()

        if action == "start":
            success = await start_data_collection_scheduler("manual")
            return {"success": success, "action": "start", "running": scheduler.is_running()}
        elif action == "stop":
            success = await stop_data_collection_scheduler()
            return {"success": success, "action": "stop", "running": scheduler.is_running()}
        else:
            return {"success": True, "action": "status", "running": scheduler.is_running()}

    except Exception as e:
        logger.error(f"调度器控制操作失败: {e}")
        raise HTTPException(status_code=500, detail=f"控制操作失败: {str(e)}")


@router.get("/api/v1/data/sources")
async def get_data_sources():
    """获取所有数据源配置"""
    try:
        # 使用绝对导入避免相对导入问题
        from src.gateway.web.config_manager import load_data_sources
        sources = load_data_sources()
        active_count = len([s for s in sources if s.get("enabled", True)])
        return {
            "data": sources,
            "data_sources": sources,  # 兼容前端期望的字段名
            "total": len(sources),
            "active": active_count,  # 添加活跃数据源数量
            "message": "数据源加载成功"
        }
    except Exception as e:
        logger.error(f"获取数据源失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取数据源失败: {str(e)}")

@router.post("/api/v1/data/sources")
async def create_or_get_data_sources(request: Request):
    """创建新的数据源配置或获取所有数据源（兼容前端的临时方案）"""
    print("🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨 create_or_get_data_sources 被调用！🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨")
    try:

        # 手动解析请求body
        try:
            body = await request.json()
            print(f"📥 解析到的请求body: {body}")
        except Exception as e:
            print(f"📥 请求body解析失败: {e}")
            body = {}

        # 使用绝对导入避免相对导入问题
        from src.gateway.web.config_manager import load_data_sources, save_data_sources

        # 检查是否是获取请求（前端的临时方案）
        print(f"🎯 检查条件: body={body}, isinstance(body, dict)={isinstance(body, dict) if body else False}, action={body.get('action') if body else None}")
        if body and isinstance(body, dict) and body.get('action') == 'get_all':
            # 这是一个获取请求，返回所有数据源
            print("🎯 检测到获取请求，开始加载数据源...")
            sources = load_data_sources()
            print(f"🎯 成功加载 {len(sources)} 个数据源")
            if sources:
                print(f"🎯 示例数据源: {sources[0]}")
            return {"data": sources, "total": len(sources), "message": "数据源加载成功"}

        # 正常的创建请求
        print(f"🎯 进入创建请求分支, body={body}")
        if not body:
            raise HTTPException(status_code=400, detail="缺少数据源配置")

        # 设置默认值
        new_source = body.copy()
        new_source.setdefault("enabled", True)
        new_source.setdefault("rate_limit", "100次/分钟")
        new_source.setdefault("last_test", None)
        new_source.setdefault("status", "未测试")

        # 添加到数据源列表
        sources = load_data_sources()
        sources.append(new_source)
        save_data_sources(sources)

        logger.info(f"数据源添加成功: id={new_source.get('id')}, name={new_source.get('name')}")

        source_id = new_source.get("id") or new_source.get("name", "unknown")
        
        # 可选：使用BusinessProcessOrchestrator管理数据源创建业务流程（符合架构设计）
        orchestrator = _get_orchestrator()
        process_id = None
        if orchestrator:
            try:
                from src.core.orchestration.orchestrator_refactored import BusinessProcessState, ProcessConfig
                process_id = f"data_source_create_{source_id}_{int(time.time())}"
                process_config = ProcessConfig(
                    process_id=process_id,
                    name=f"Data Source Creation: {source_id}",
                    initial_state=BusinessProcessState.DATA_SOURCE_CONFIGURATION,
                    parameters={
                        "source_id": source_id,
                        "action": "create",
                        "data_source": new_source
                    }
                )
                orchestrator.start_process(process_config)
                logger.debug(f"已启动数据源创建业务流程: {process_id}")
            except Exception as e:
                logger.debug(f"启动数据源创建业务流程失败（可选功能）: {e}")
        
        # 发布配置变更事件到EventBus（符合架构设计：事件驱动通信）
        event_bus = _get_event_bus()
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                # 使用CONFIG_UPDATED事件类型发布数据源配置变更事件
                event_bus.publish(
                    EventType.CONFIG_UPDATED,
                    {
                        "source_id": source_id,
                        "action": "data_source_created",
                        "data_source": new_source,
                        "config_type": "data_source",
                        "process_id": process_id,
                        "timestamp": time.time()
                    },
                    source="datasource_routes"
                )
                logger.debug(f"已发布数据源配置变更事件 (创建): {source_id}")
            except Exception as e:
                logger.debug(f"发布配置变更事件失败: {e}")
        
        # 可选：使用BusinessProcessOrchestrator更新流程状态为完成（符合架构设计）
        if orchestrator and process_id:
            try:
                from src.core.orchestration.orchestrator_refactored import BusinessProcessState
                orchestrator.update_process_state(
                    process_id,
                    BusinessProcessState.COMPLETED,
                    metrics={
                        "source_id": source_id,
                        "action": "create",
                        "timestamp": time.time()
                    }
                )
                logger.debug(f"已更新数据源创建业务流程状态: {process_id}")
            except Exception as e:
                logger.debug(f"更新数据源创建业务流程状态失败（可选功能）: {e}")
        
        # WebSocket广播数据源创建事件
        await broadcast_data_source_change("data_source_created", source_id, new_source)

        return {
            "success": True,
            "message": f"数据源 {new_source.get('name', '未知')} 创建成功",
            "data_source": new_source,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"创建数据源失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建数据源失败: {str(e)}")


@router.options("/api/v1/data/sources")
async def options_data_sources():
    """处理CORS预检请求"""
    return {"message": "CORS preflight OK"}


@router.get("/api/v1/data/sources")
async def get_data_sources_api():
    """获取所有数据源配置"""
    print("🚨🚨🚨 get_data_sources_api 函数被调用了！🚨🚨🚨")
    try:
        # 使用绝对导入避免相对导入问题
        from src.gateway.web.config_manager import load_data_sources
        sources = load_data_sources()
        print(f"🎯 加载到 {len(sources)} 个数据源")

        # 最终修复：确保所有数据源都有有效的字符串ID
        import sys
        print("FORCED DEBUG: 开始API层最终ID修复...", file=sys.stderr)
        for source in sources:
            id_value = source.get('id')
            name = source.get('name', 'unknown')
            print(f"FORCED DEBUG: 检查数据源 {name}: id={repr(id_value)} (type: {type(id_value)})", file=sys.stderr)

            if id_value is None or str(id_value).lower() in ['null', 'none']:
                if '新浪财经' in name:
                    source['id'] = 'sinafinance'
                elif '宏观经济' in name:
                    source['id'] = 'macrodata'
                elif '财联社' in name:
                    source['id'] = 'cls'
                else:
                    source['id'] = name.lower().replace(' ', '_')
                print(f"FORCED DEBUG: ✅ API层修复数据源 {name} 的ID: {repr(id_value)} -> {source['id']}", file=sys.stderr)
            else:
                print(f"FORCED DEBUG: ✅ 数据源 {name} 的ID已经是有效的: {source['id']}", file=sys.stderr)

        print(f"FORCED DEBUG: API返回数据源数量: {len(sources)}", file=sys.stderr)
        for i, source in enumerate(sources):
            print(f"FORCED DEBUG: 最终API数据源 {i}: id={repr(source.get('id'))}, name={source.get('name')}", file=sys.stderr)

        response = {
            "data_sources": sources,
            "total": len(sources),
            "active": len([s for s in sources if s.get("enabled", True)]),
            "timestamp": time.time()
        }

        return response
    except Exception as e:
        logger.error(f"获取数据源列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/data/sources/{source_id}")
async def get_data_source_api(source_id: str):
    """获取指定的数据源配置"""
    logger.info(f"get_data_source_api 被调用，参数: {source_id}")
    try:
        # 使用绝对导入避免相对导入问题
        from src.gateway.web.config_manager import load_data_sources
        sources = load_data_sources()
        logger.info(f"加载了 {len(sources)} 个数据源")

        for source in sources:
            current_id = source.get("id")
            print(f"DEBUG: 检查数据源: {current_id}")
            if current_id == source_id:
                print(f"DEBUG: 找到匹配的数据源: {current_id}")
                # 添加缓存控制头，防止浏览器缓存
                return JSONResponse(content=source, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

        print(f"DEBUG: 未找到数据源: {source_id}")
        raise HTTPException(status_code=404, detail=f"数据源 {source_id} 不存在")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取数据源失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/data/sources/{source_id}/sample")
async def get_data_source_sample_api(source_id: str, data_type: str = None):
    """获取数据源的最新样本数据（直接从PostgreSQL数据库查询）"""
    logger.info(f"get_data_source_sample_api 被调用，参数: {source_id}")
    try:
        import time
        from datetime import datetime
        from decimal import Decimal
        from src.gateway.web.config_manager import load_data_sources

        # 检查数据源是否存在
        sources = load_data_sources()
        source_config = None
        for source in sources:
            if source.get("id") == source_id:
                source_config = source
                break

        if not source_config:
            raise HTTPException(status_code=404, detail=f"数据源 {source_id} 不存在")

        # 直接从PostgreSQL查询最新数据
        try:
            from src.gateway.web.postgresql_persistence import query_latest_stock_data_from_postgresql

            # 查询最新的10条股票数据，支持按数据类型过滤
            sample_data_list = query_latest_stock_data_from_postgresql(source_id, limit=10, data_type=data_type)

            if sample_data_list and len(sample_data_list) > 0:
                # 处理Decimal对象，确保JSON序列化兼容
                from decimal import Decimal
                processed_data = []
                for record in sample_data_list:
                    processed_record = {}
                    for key, value in record.items():
                        if isinstance(value, Decimal):
                            processed_record[key] = float(value)
                        elif isinstance(value, (int, float)) and (value != value or value in [float('inf'), float('-inf')]):
                            # 处理NaN和Infinity
                            processed_record[key] = None
                        else:
                            processed_record[key] = value
                    processed_data.append(processed_record)

                # 返回格式化的响应
                sample_data = {
                    "source_id": source_id,
                    "source_name": source_config.get('name', source_id),
                    "source_type": source_config.get('type', 'unknown'),
                    "sample_count": len(processed_data),
                    "total_count": len(processed_data),  # 简化处理，实际总数可以通过另一个查询获取
                    "generated_at": int(time.time()),
                    "data": processed_data,
                    "message": f"从PostgreSQL获取最新数据样本（共{len(processed_data)}条记录）"
                }
                logger.info(f"成功从PostgreSQL获取数据源 {source_id} 的样本数据: {len(sample_data_list)} 条记录")
            else:
                # 没有找到数据
                sample_data = {
                    "source_id": source_id,
                    "source_name": source_config.get('name', source_id),
                    "sample_count": 0,
                    "total_count": 0,
                    "generated_at": int(time.time()),
                    "data": [],
                    "message": "数据库中暂无该数据源的样本数据"
                }
                logger.info(f"数据源 {source_id} 在PostgreSQL中没有找到样本数据")

        except ImportError as e:
            logger.error(f"无法导入PostgreSQL查询函数: {e}")
            return JSONResponse(content={
                "source_id": source_id,
                "source_name": source_config.get('name', source_id),
                "sample_count": 0,
                "total_count": 0,
                "generated_at": int(time.time()),
                "data": [],
                "message": "数据库查询功能不可用"
            }, status_code=503)
        except Exception as db_error:
            logger.error(f"从PostgreSQL查询样本数据失败 {source_id}: {db_error}")
            return JSONResponse(content={
                "source_id": source_id,
                "source_name": source_config.get('name', source_id),
                "sample_count": 0,
                "total_count": 0,
                "generated_at": int(time.time()),
                "data": [],
                "message": f"数据库查询失败: {str(db_error)}"
            }, status_code=500)

        # 添加缓存控制头，防止浏览器缓存
        return JSONResponse(content=sample_data, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取数据源样本失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取样本失败: {str(e)}")


@router.put("/api/v1/data/sources/{source_id}")
async def update_data_source_api(source_id: str, updated_source: dict):
    """更新数据源配置"""
    try:
        # 使用绝对导入避免相对导入问题
        from src.gateway.web.config_manager import load_data_sources, save_data_sources
        sources = load_data_sources()
        found = False
        for i, source in enumerate(sources):
            if source["id"] == source_id:
                # 记录更新前后的状态，特别是enabled字段
                old_enabled = source.get("enabled", True)
                new_enabled = updated_source.get("enabled", old_enabled)  # 如果未提供，保持原值
                
                # 合并更新：保留原有字段，只更新提供的字段
                updated_source_copy = source.copy()  # 先复制原有配置
                updated_source_copy.update(updated_source)  # 用新值更新
                updated_source_copy["id"] = source_id  # 强制使用URL中的ID，防止用户修改ID
                
                final_enabled = updated_source_copy.get('enabled', old_enabled)
                logger.info(f"更新数据源 {source_id} ({source.get('name', 'unknown')}): enabled={old_enabled} -> {final_enabled}")
                logger.debug(f"更新详情: 接收到的更新数据={updated_source}, 最终enabled状态={final_enabled}")
                
                sources[i] = updated_source_copy
                found = True
                logger.info(f"更新数据源 {source_id}: ID保持不变, enabled状态已更新为 {final_enabled}")
                break

        if not found:
            raise HTTPException(status_code=404, detail=f"数据源 {source_id} 不存在")

        # 确保更新后的数据源有正确的ID
        logger.info(f"更新前数据源列表检查:")
        for i, source in enumerate(sources):
            logger.info(f"  {i}: {source.get('name')} - id={repr(source.get('id'))}")

        for source in sources:
            if source.get('id') is None or str(source.get('id')).lower() in ['null', 'none']:
                name = source.get('name', 'unknown')
                if '新浪财经' in name:
                    source['id'] = 'sinafinance'
                elif '宏观经济' in name:
                    source['id'] = 'macrodata'
                elif '财联社' in name:
                    source['id'] = 'cls'
                else:
                    source['id'] = name.lower().replace(' ', '_')
                logger.info(f"🔧 更新时修复数据源 {name} 的ID -> {source['id']}")

        logger.info(f"更新后数据源列表检查:")
        for i, source in enumerate(sources):
            logger.info(f"  {i}: {source.get('name')} - id={repr(source.get('id'))}")

        save_data_sources(sources)
        logger.info(f"数据源 {source_id} 更新成功，最终ID: {updated_source.get('id')}")

        # 通知调度器重新加载配置
        try:
            from src.core.orchestration.business_process.service_scheduler import get_data_collection_scheduler
            scheduler = get_data_collection_scheduler()
            if scheduler.reload_data_sources():
                logger.info(f"已通知调度器重新加载数据源配置")
            else:
                logger.warning(f"调度器重新加载配置失败")
        except Exception as e:
            logger.warning(f"通知调度器重新加载配置异常: {e}")

        # 可选：使用BusinessProcessOrchestrator管理数据源更新业务流程（符合架构设计）
        orchestrator = _get_orchestrator()
        process_id = None
        if orchestrator:
            try:
                from src.core.orchestration.orchestrator_refactored import BusinessProcessState, ProcessConfig
                process_id = f"data_source_update_{source_id}_{int(time.time())}"
                process_config = ProcessConfig(
                    process_id=process_id,
                    name=f"Data Source Update: {source_id}",
                    initial_state=BusinessProcessState.DATA_SOURCE_CONFIGURATION,
                    parameters={
                        "source_id": source_id,
                        "action": "update",
                        "data_source": updated_source
                    }
                )
                orchestrator.start_process(process_config)
                logger.debug(f"已启动数据源更新业务流程: {process_id}")
            except Exception as e:
                logger.debug(f"启动数据源更新业务流程失败（可选功能）: {e}")

        # 发布配置变更事件到EventBus（符合架构设计：事件驱动通信）
        event_bus = _get_event_bus()
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                # 使用CONFIG_UPDATED事件类型发布数据源配置变更事件
                event_bus.publish(
                    EventType.CONFIG_UPDATED,
                    {
                        "source_id": source_id,
                        "action": "data_source_updated",
                        "data_source": updated_source,
                        "config_type": "data_source",
                        "process_id": process_id,
                        "timestamp": time.time()
                    },
                    source="datasource_routes"
                )
                logger.debug(f"已发布数据源配置变更事件 (更新): {source_id}")
            except Exception as e:
                logger.debug(f"发布配置变更事件失败: {e}")
        
        # 可选：使用BusinessProcessOrchestrator更新流程状态为完成（符合架构设计）
        if orchestrator and process_id:
            try:
                from src.core.orchestration.orchestrator_refactored import BusinessProcessState
                orchestrator.update_process_state(
                    process_id,
                    BusinessProcessState.COMPLETED,
                    metrics={
                        "source_id": source_id,
                        "action": "update",
                        "timestamp": time.time()
                    }
                )
                logger.debug(f"已更新数据源更新业务流程状态: {process_id}")
            except Exception as e:
                logger.debug(f"更新数据源更新业务流程状态失败（可选功能）: {e}")
        
        # WebSocket广播数据源更新事件
        await broadcast_data_source_change("data_source_updated", source_id, updated_source)

        return {"message": f"数据源 {source_id} 更新成功", "source": updated_source}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新数据源失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")


@router.delete("/api/v1/data/sources/{source_id}")
async def delete_data_source_api(source_id: str):
    """删除数据源配置"""
    try:
        # 使用绝对导入避免相对导入问题
        from src.gateway.web.config_manager import load_data_sources, save_data_sources
        sources = load_data_sources()

        # 查找并删除数据源
        for i, source in enumerate(sources):
            if source["id"] == source_id:
                deleted_source = sources.pop(i)
                save_data_sources(sources)

                # 可选：使用BusinessProcessOrchestrator管理数据源删除业务流程（符合架构设计）
                orchestrator = _get_orchestrator()
                process_id = None
                if orchestrator:
                    try:
                        from src.core.orchestration.orchestrator_refactored import BusinessProcessState, ProcessConfig
                        process_id = f"data_source_delete_{source_id}_{int(time.time())}"
                        process_config = ProcessConfig(
                            process_id=process_id,
                            name=f"Data Source Deletion: {source_id}",
                            initial_state=BusinessProcessState.DATA_SOURCE_CONFIGURATION,
                            parameters={
                                "source_id": source_id,
                                "action": "delete",
                                "deleted_source": deleted_source,
                                "remaining_count": len(sources)
                            }
                        )
                        orchestrator.start_process(process_config)
                        logger.debug(f"已启动数据源删除业务流程: {process_id}")
                    except Exception as e:
                        logger.debug(f"启动数据源删除业务流程失败（可选功能）: {e}")

                # 发布配置变更事件到EventBus（符合架构设计：事件驱动通信）
                event_bus = _get_event_bus()
                if event_bus:
                    try:
                        from src.core.event_bus.types import EventType
                        # 使用CONFIG_UPDATED事件类型发布数据源配置变更事件
                        event_bus.publish(
                            EventType.CONFIG_UPDATED,
                            {
                                "source_id": source_id,
                                "action": "data_source_deleted",
                                "deleted_source": deleted_source,
                                "config_type": "data_source",
                                "remaining_count": len(sources),
                                "process_id": process_id,
                                "timestamp": time.time()
                            },
                            source="datasource_routes"
                        )
                        logger.debug(f"已发布数据源配置变更事件 (删除): {source_id}")
                    except Exception as e:
                        logger.debug(f"发布配置变更事件失败: {e}")
                
                # 可选：使用BusinessProcessOrchestrator更新流程状态为完成（符合架构设计）
                if orchestrator and process_id:
                    try:
                        from src.core.orchestration.orchestrator_refactored import BusinessProcessState
                        orchestrator.update_process_state(
                            process_id,
                            BusinessProcessState.COMPLETED,
                            metrics={
                                "source_id": source_id,
                                "action": "delete",
                                "remaining_count": len(sources),
                                "timestamp": time.time()
                            }
                        )
                        logger.debug(f"已更新数据源删除业务流程状态: {process_id}")
                    except Exception as e:
                        logger.debug(f"更新数据源删除业务流程状态失败（可选功能）: {e}")

                # WebSocket广播数据源删除事件
                await broadcast_data_source_change("data_source_deleted", source_id, deleted_source)

                return {
                    "success": True,
                    "message": f"数据源 {source_id} 已删除",
                    "deleted_source": deleted_source,
                    "remaining_count": len(sources),
                    "timestamp": time.time()
                }

        raise HTTPException(status_code=404, detail=f"数据源 {source_id} 不存在")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除数据源失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@router.get("/test-akshare")
async def test_akshare_endpoint():
    """测试AKShare路由是否工作"""
    return {"message": "AKShare路由工作正常", "timestamp": "2026-01-01"}

@router.get("/routes")
async def list_routes():
    """列出所有路由"""
    routes = []
    for route in router.routes:
        if hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": getattr(route, 'methods', []),
                "name": getattr(route, 'name', '')
            })
    return {"routes": routes}

# 移除有问题的路由，让app级别的路由处理


@router.get("/api/v1/data-sources/metrics")
async def get_data_sources_metrics():
    """获取数据源性能指标 - 使用真实监控数据，不返回硬编码估算值"""
    try:
        # 使用绝对导入避免相对导入问题
        from src.gateway.web.config_manager import load_data_sources
        sources = load_data_sources()

        # 尝试从性能监控器获取真实数据
        performance_monitor = None
        try:
            from src.data.monitoring.performance_monitor import PerformanceMonitor
            # 尝试获取已存在的监控器实例或创建新实例
            performance_monitor = PerformanceMonitor()
        except Exception as e:
            logger.debug(f"性能监控器不可用: {e}")

        # 初始化指标结构
        metrics = {
            "total_sources": len(sources),
            "active_sources": len([s for s in sources if s.get("enabled", True)]),
            "disabled_sources": len([s for s in sources if not s.get("enabled", True)]),
            "latency_data": {},
            "throughput_data": {},
            "error_rates": {},
            "availability": {},
            "last_updated": {},
            "health_scores": {},
            "performance_trends": {},
            "system_metrics": {
                "total_uptime": 0,
                "avg_response_time": 0,
                "error_count": 0,
                "success_count": 0,
                "note": "量化交易系统要求使用真实监控数据。如果指标为空，表示监控系统尚未收集到数据。"
            },
            "timestamp": time.time()
        }

        # 从真实监控系统获取性能数据
        for source in sources:
            source_id = source["id"]
            source_name = source.get("name", "unknown")
            is_enabled = source.get("enabled", True)
            last_test = source.get("last_test")
            status = source.get("status", "未测试")

            # 对禁用数据源进行过滤：不参与明细指标，不输出处理日志（仅在计数中体现）
            if not is_enabled:
                continue

            logger.info(f"处理数据源 {source_name}: id={repr(source_id)}, enabled={is_enabled}, status={status}")

            if is_enabled and performance_monitor:
                try:
                    # 尝试从监控器获取该数据源的性能指标
                    # 使用数据源ID作为指标名称的一部分
                    latency_metric_name = f"data_source_{source_id}_latency"
                    throughput_metric_name = f"data_source_{source_id}_throughput"
                    error_rate_metric_name = f"data_source_{source_id}_error_rate"
                    
                    # 获取最近24小时的指标历史
                    latency_history = performance_monitor.get_metric_history(latency_metric_name, hours=24)
                    throughput_history = performance_monitor.get_metric_history(throughput_metric_name, hours=24)
                    error_rate_history = performance_monitor.get_metric_history(error_rate_metric_name, hours=24)
                    
                    # 计算平均值
                    if latency_history:
                        avg_latency = sum(m.value for m in latency_history) / len(latency_history)
                        metrics["latency_data"][source_id] = avg_latency
                    
                    if throughput_history:
                        avg_throughput = sum(m.value for m in throughput_history) / len(throughput_history)
                        metrics["throughput_data"][source_id] = avg_throughput
                    
                    if error_rate_history:
                        avg_error_rate = sum(m.value for m in error_rate_history) / len(error_rate_history)
                        metrics["error_rates"][source_id] = avg_error_rate
                        # 计算可用性
                        metrics["availability"][source_id] = max(0, 1.0 - avg_error_rate)
                        # 计算健康分数
                        metrics["health_scores"][source_id] = max(0, min(100, (1.0 - avg_error_rate) * 100))
                    
                    # 性能趋势分析
                    if latency_history and len(latency_history) >= 2:
                        recent_latency = latency_history[-1].value
                        previous_latency = latency_history[-2].value
                        if recent_latency < previous_latency * 0.95:
                            metrics["performance_trends"][source_id] = "改善"
                        elif recent_latency > previous_latency * 1.05:
                            metrics["performance_trends"][source_id] = "恶化"
                        else:
                            metrics["performance_trends"][source_id] = "稳定"
                    elif status == "连接正常":
                        metrics["performance_trends"][source_id] = "稳定"
                    else:
                        metrics["performance_trends"][source_id] = "未知"
                        
                except Exception as e:
                    logger.debug(f"从监控器获取数据源 {source_id} 的性能指标失败: {e}")
                    # 如果监控器没有该数据源的数据，不填充任何值（保持为空）
                    # 量化交易系统要求：不使用估算值，只返回真实监控数据
                    pass
            
            # 最后更新时间（来自数据源配置，仅对启用数据源输出到明细）
            if last_test:
                metrics["last_updated"][source_id] = last_test
            else:
                metrics["last_updated"][source_id] = "从未测试"

        # 计算平均值指标（仅基于真实数据）
        if metrics["latency_data"]:
            latency_values = list(metrics["latency_data"].values())
            avg_latency = sum(latency_values) / len(latency_values) if latency_values else 0
        else:
            avg_latency = 0

        if metrics["error_rates"]:
            error_rate_values = list(metrics["error_rates"].values())
            avg_error_rate = sum(error_rate_values) / len(error_rate_values) if error_rate_values else 0
        else:
            avg_error_rate = 0

        if metrics["throughput_data"]:
            throughput_values = list(metrics["throughput_data"].values())
            avg_throughput = sum(throughput_values) / len(throughput_values) if throughput_values else 0
        else:
            avg_throughput = 0

        # 更新system_metrics
        metrics["system_metrics"].update({
            "avg_latency": avg_latency,
            "avg_error_rate": avg_error_rate,
            "avg_throughput": avg_throughput,
            "avg_response_time": avg_latency if avg_latency > 0 else 0,
            "error_count": 0,  # TODO: 从监控系统获取真实错误计数
            "success_count": 0,  # TODO: 从监控系统获取真实成功计数
            "total_uptime": 0,  # TODO: 从监控系统获取真实运行时间
            "note": "量化交易系统要求使用真实监控数据。如果指标为空，表示监控系统尚未收集到该数据源的性能数据。"
        })

        return metrics

    except Exception as e:
        logger.error(f"获取数据源性能指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能指标失败: {str(e)}")


@router.post("/api/v1/data/sources/{source_id}/test")
async def test_data_source(source_id: str):
    """测试数据源连接"""
    import time
    from datetime import datetime

    print(f"🚀🚀🚀 test_data_source 被调用: {source_id} 🚀🚀🚀")

    # 获取数据源配置
    from src.gateway.web.config_manager import load_data_sources, save_data_sources
    print(f"📋 加载数据源配置...")
    sources = load_data_sources()
    print(f"📋 找到 {len(sources)} 个数据源")

    # 检查是否是AKShare数据源
    is_akshare = "akshare" in source_id.lower() or source_id == "akshare_news_wallstreet"
    print(f"🔍 检查AKShare条件: {'akshare' in source_id.lower()} or {source_id == 'akshare_news_wallstreet'} = {is_akshare}")
    source = None
    for s in sources:
        if s.get("id") == source_id:
            source = s
            break

    print(f"🔍 查找数据源 {source_id}: {'找到' if source else '未找到'}")

    if not source:
        return {
            "source_id": source_id,
            "success": False,
            "status": "数据源不存在",
            "last_test": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": f"数据源 {source_id} 不存在",
            "timestamp": time.time()
        }

    # 特殊处理AKShare数据源
    if "akshare" in source_id.lower() or source_id == "akshare_news_wallstreet":
        print(f"DEBUG: 检测到AKShare数据源: {source_id}，开始真实API测试")
        try:
            # 导入AKShare库
            import akshare
            import asyncio

            # 获取AKShare函数名
            config = source.get("config", {})
            akshare_function = config.get("akshare_function", "")

            # 特殊处理 akshare_stock_basic 数据源
            if source_id == "akshare_stock_basic":
                print(f"DEBUG: 特殊处理 akshare_stock_basic 数据源，强制使用 stock_zh_a_spot 函数")
                akshare_function = "stock_zh_a_spot"

            if not akshare_function:
                return {
                    "source_id": source_id,
                    "success": False,
                    "status": "配置错误",
                    "last_test": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "message": f"AKShare数据源 {source_id} 缺少 akshare_function 配置",
                    "timestamp": time.time()
                }

            # 检查函数是否存在
            if not hasattr(akshare, akshare_function):
                return {
                    "source_id": source_id,
                    "success": False,
                    "status": "函数不存在",
                    "last_test": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "message": f"AKShare函数 {akshare_function} 不存在",
                    "timestamp": time.time()
                }

            # 调用AKShare函数进行真实测试
            print(f"DEBUG: 调用AKShare函数: {akshare_function}")
            akshare_func = getattr(akshare, akshare_function)

            # 根据不同函数传递不同参数
            if akshare_function == "news_economic_baidu":
                # 对于新闻函数，尝试获取少量数据进行测试
                data = await asyncio.to_thread(akshare_func, date="20241107")
            else:
                # 其他函数尝试无参数调用
                data = await asyncio.to_thread(akshare_func)

            # 检查数据是否获取成功
            if data is not None and not data.empty and len(data) > 0:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 更新数据源状态
                source["last_test"] = current_time
                source["status"] = "连接正常"
                save_data_sources(sources)

                print(f"DEBUG: AKShare数据源 {source_id} 测试成功，获取{len(data)}条数据")

                # 自动采集数据样本用于前端显示
                try:
                    from .api_utils import persist_collected_data
                    from .data_collectors import collect_data_via_data_layer

                    # 采集完整数据集
                    collected_data = await collect_data_via_data_layer(source)
                    if collected_data and len(collected_data) > 0:
                        # 持久化数据
                        metadata = {
                            "collection_timestamp": time.time(),
                            "test_collection": True,
                            "data_count": len(collected_data)
                        }
                        persist_result = await persist_collected_data(source_id, collected_data, metadata, source)
                        print(f"DEBUG: 数据样本采集完成: {len(collected_data)}条记录")
                    else:
                        print("DEBUG: 未采集到数据样本")
                except Exception as sample_error:
                    print(f"DEBUG: 数据样本采集失败: {sample_error}")

                return {
                    "source_id": source_id,
                    "success": True,
                    "status": "连接正常",
                    "last_test": current_time,
                    "message": f"AKShare API测试成功 - 获取{len(data)}条数据",
                    "timestamp": time.time(),
                    "data_sample": data.head(3).to_dict('records') if len(data) > 3 else data.to_dict('records')
                }
            else:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 更新数据源状态
                source["last_test"] = current_time
                source["status"] = "数据获取失败"
                save_data_sources(sources)

                return {
                    "source_id": source_id,
                    "success": False,
                    "status": "数据获取失败",
                    "last_test": current_time,
                    "message": "AKShare API调用成功但未获取到数据",
                    "timestamp": time.time()
                }

        except Exception as e:
            print(f"DEBUG: AKShare数据源 {source_id} 测试异常: {e}")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 更新数据源状态
            source["last_test"] = current_time
            source["status"] = f"连接异常: {str(e)[:50]}"
            save_data_sources(sources)

            return {
                "source_id": source_id,
                "success": False,
                "status": f"连接异常: {str(e)[:50]}",
                "last_test": current_time,
                "message": f"AKShare API测试失败: {str(e)}",
                "timestamp": time.time()
            }

    # 其他数据源使用HTTP连接测试
    try:
        import aiohttp
        source_url = source.get("url", "")
        if not source_url:
            return {
                "source_id": source_id,
                "success": False,
                "status": "配置错误",
                "last_test": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": f"数据源 {source_id} 缺少URL配置",
                "timestamp": time.time()
            }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.head(source_url, allow_redirects=True) as response:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if response.status < 400:
                    # 更新数据源状态
                    source["last_test"] = current_time
                    source["status"] = f"HTTP {response.status} - 连接正常"
                    save_data_sources(sources)

                    return {
                        "source_id": source_id,
                        "success": True,
                        "status": f"HTTP {response.status} - 连接正常",
                        "last_test": current_time,
                        "message": f"连接测试完成：HTTP {response.status} - 连接正常",
                        "timestamp": time.time()
                    }
                else:
                    # 更新数据源状态
                    source["last_test"] = current_time
                    source["status"] = f"HTTP {response.status} - 服务错误"
                    save_data_sources(sources)

                    return {
                        "source_id": source_id,
                        "success": False,
                        "status": f"HTTP {response.status} - 服务错误",
                        "last_test": current_time,
                        "message": f"连接测试失败：HTTP {response.status} - 服务错误",
                        "timestamp": time.time()
                    }

    except Exception as e:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 更新数据源状态
        source["last_test"] = current_time
        source["status"] = f"连接异常: {str(e)[:50]}"
        save_data_sources(sources)

        return {
            "source_id": source_id,
            "success": False,
            "status": f"连接异常: {str(e)[:50]}",
            "last_test": current_time,
            "message": f"连接测试异常: {str(e)}",
            "timestamp": time.time()
        }


async def broadcast_data_source_change(event_type: str, source_id: str, data: dict):
    """广播数据源变更事件"""
    try:
        # 根据事件类型生成用户友好的消息
        message_text = ""
        if event_type == "data_source_created":
            message_text = f"数据源 {data.get('name', source_id)} 创建成功"
        elif event_type == "data_source_updated":
            enabled_status = "启用" if data.get('enabled', True) else "禁用"
            message_text = f"数据源 {data.get('name', source_id)} 已{enabled_status}"
        elif event_type == "data_source_deleted":
            message_text = f"数据源 {data.get('name', source_id)} 已删除"
        elif event_type == "data_source_tested":
            status = data.get("status", "unknown") if data else "unknown"
            message_text = f"数据源 {data.get('name', source_id)} 连接测试完成: {status}"

        # 发送前端期望的消息格式
        message = {
            "type": event_type,  # 前端期望直接使用事件类型
            "source_id": source_id,
            "data": data,
            "message": message_text,
            "timestamp": time.time()
        }

        # 尝试使用WebSocket广播
        try:
            from src.gateway.web.websocket_manager import websocket_manager
            if websocket_manager:
                await websocket_manager.broadcast_message(message)
                logger.debug(f"已广播数据源变更事件: {event_type} - {source_id}")
        except Exception as ws_error:
            logger.debug(f"WebSocket广播失败（可选功能）: {ws_error}")

        # 尝试使用EventBus发布事件
        event_bus = _get_event_bus()
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                event_bus.publish(
                    EventType.DATA_SOURCE_CHANGED,
                    {
                        "source_id": source_id,
                        "event_type": event_type,
                        "data": data,
                        "message": message_text,
                        "timestamp": time.time()
                    },
                    source="datasource_routes"
                )
                logger.debug(f"已发布数据源变更事件到EventBus: {event_type} - {source_id}")
            except Exception as e:
                logger.debug(f"EventBus发布失败（可选功能）: {e}")

    except Exception as e:
        logger.debug(f"广播数据源变更事件失败: {e}")


# 健康检查路由
@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "data",
        "timestamp": time.time()
    }
