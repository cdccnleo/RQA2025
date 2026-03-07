"""
策略路由模块
包含策略构思、模板、验证等相关API端点
"""

import os
import json
import time
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter()

# 导入统一持久化模块
try:
    from .unified_persistence import get_strategy_conception_persistence
    persistence = get_strategy_conception_persistence()
    logger.info("策略构思持久化模块初始化成功")
except ImportError as e:
    logger.error(f"导入统一持久化模块失败: {e}")
    raise


STRATEGY_CONCEPTION_DIR = "data/strategy_conceptions"


def load_strategy_conceptions() -> List[Dict]:
    """加载策略构思配置（优先从数据库加载，带超时回退）"""
    import concurrent.futures
    
    def _load_with_timeout():
        try:
            return persistence.list(limit=100)
        except Exception as e:
            logger.warning(f"数据库加载失败，回退到文件系统: {e}")
            return []
    
    try:
        # 使用线程池执行，设置5秒超时
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_load_with_timeout)
            conceptions = future.result(timeout=5.0)
            
            # 如果数据库返回空，尝试从文件系统加载
            if not conceptions:
                conceptions = _load_from_filesystem_only()
            
            return conceptions
    except concurrent.futures.TimeoutError:
        logger.error("加载策略构思超时，使用文件系统回退")
        return _load_from_filesystem_only()
    except Exception as e:
        logger.error(f"加载策略构思配置失败: {e}")
        return _load_from_filesystem_only()


def save_strategy_conception(conception_data: dict, auto_increment_version: bool = True):
    """保存策略构思配置（优先保存到数据库）
    
    Args:
        conception_data: 策略构思数据
        auto_increment_version: 是否自动增加版本号，默认为True
    """
    try:
        # 确保策略ID
        strategy_id = conception_data.get("id", f"strategy_{int(time.time())}")
        conception_data["id"] = strategy_id
        
        logger.info(f"[Save Strategy] 开始保存策略 {strategy_id}")
        logger.info(f"[Save Strategy] 请求数据包含的字段: {list(conception_data.keys())}")
        logger.info(f"[Save Strategy] stats字段内容: {conception_data.get('stats')}")

        # 处理版本号
        if auto_increment_version:
            current_version = conception_data.get("version", 1)
            if isinstance(current_version, str):
                try:
                    current_version = int(current_version)
                except (ValueError, TypeError):
                    current_version = 1
            conception_data["version"] = current_version + 1

        # 保存数据
        logger.info(f"[Save Strategy] 调用persistence.save保存数据")
        success = persistence.save(conception_data)
        if success:
            logger.info(f"[Save Strategy] 策略 {strategy_id} 保存成功")
            return {"success": True, "strategy_id": strategy_id, "message": "策略构思保存成功"}
        else:
            logger.error(f"[Save Strategy] 策略 {strategy_id} 保存失败")
            raise Exception("保存失败")

    except Exception as e:
        logger.error(f"[Save Strategy] 保存策略构思配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")


def validate_strategy_conception(conception_data: any) -> dict:
    """验证策略构思配置"""
    errors = []
    warnings = []

    # 确保conception_data是字典类型
    if not isinstance(conception_data, dict):
        errors.append("无效的策略构思数据格式")
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "complexity_score": 0,
            "complexity_level": "低",
            "estimated_days": 0,
            "node_count": 0,
            "connection_count": 0,
            "parameter_count": 0
        }

    # 基本信息验证
    if not conception_data.get("name"):
        errors.append("策略名称不能为空")

    if not conception_data.get("type"):
        errors.append("策略类型不能为空")

    # 节点验证
    nodes = conception_data.get("nodes", [])
    if not nodes:
        errors.append("策略至少需要一个节点")
    else:
        # 确保nodes是列表
        if not isinstance(nodes, list):
            errors.append("节点配置格式无效")
        else:
            node_types = []
            for node in nodes:
                if isinstance(node, dict):
                    node_types.append(node.get("type"))
            required_types = ["data_source", "trade"]

            for required_type in required_types:
                if required_type not in node_types:
                    errors.append(f"缺少必需的节点类型: {required_type}")

            # 检查节点连接性
            connections = conception_data.get("connections", [])
            if len(nodes) > 1 and (not connections or not isinstance(connections, list)):
                warnings.append("建议为多个节点建立连接关系")

    # 参数验证
    parameters = conception_data.get("parameters", {})
    if isinstance(parameters, dict):
        for param_name, param_config in parameters.items():
            if isinstance(param_config, dict):
                param_type = param_config.get("type")
                param_value = param_config.get("value", param_config.get("default"))

                if param_type == "number":
                    min_val = param_config.get("min")
                    max_val = param_config.get("max")

                    # 确保参数值和范围值类型一致
                    if min_val is not None and param_value is not None:
                        try:
                            if float(param_value) < float(min_val):
                                errors.append(f"参数 {param_name} 值 {param_value} 小于最小值 {min_val}")
                        except (ValueError, TypeError):
                            pass

                    if max_val is not None and param_value is not None:
                        try:
                            if float(param_value) > float(max_val):
                                errors.append(f"参数 {param_name} 值 {param_value} 大于最大值 {max_val}")
                        except (ValueError, TypeError):
                            pass

    # 复杂度评分
    complexity_score = len(nodes) * 0.3 + len(connections) * 0.4 + len(parameters) * 0.3
    complexity_level = "低"
    if complexity_score > 3:
        complexity_level = "中"
    if complexity_score > 6:
        complexity_level = "高"

    # 开发时间估算 (天)
    estimated_days = max(1, int(complexity_score * 2))

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "complexity_score": round(complexity_score, 1),
        "complexity_level": complexity_level,
        "estimated_days": estimated_days,
        "node_count": len(nodes),
        "connection_count": len(connections),
        "parameter_count": len(parameters)
    }


@router.get("/api/v1/strategy/conception/templates")
async def get_strategy_conception_templates():
    """获取策略构思模板列表"""
    templates = {
        "trend_following": {
            "name": "趋势跟踪策略",
            "description": "基于技术指标识别市场趋势并跟随交易",
            "parameters": {
                "trend_period": {"type": "number", "default": 20, "min": 5, "max": 100, "label": "趋势周期"},
                "entry_threshold": {"type": "number", "default": 0.02, "min": 0.001, "max": 0.1, "step": 0.001, "label": "入场阈值"},
                "exit_threshold": {"type": "number", "default": 0.01, "min": 0.001, "max": 0.05, "step": 0.001, "label": "出场阈值"}
            },
            "required_nodes": ["data_source", "feature", "trade", "risk"],
            "estimated_complexity": "中",
            "estimated_days": 7
        },
        "mean_reversion": {
            "name": "均值回归策略",
            "description": "利用价格偏离均值的回归特性进行交易",
            "parameters": {
                "lookback_period": {"type": "number", "default": 20, "min": 5, "max": 100, "label": "回望周期"},
                "deviation_threshold": {"type": "number", "default": 2.0, "min": 0.5, "max": 5.0, "step": 0.1, "label": "偏离阈值"},
                "holding_period": {"type": "number", "default": 5, "min": 1, "max": 20, "label": "持仓周期"}
            },
            "required_nodes": ["data_source", "feature", "model", "trade", "risk"],
            "estimated_complexity": "中",
            "estimated_days": 8
        },
        "ml_based": {
            "name": "机器学习策略",
            "description": "使用机器学习算法进行价格预测和交易决策",
            "parameters": {
                "model_type": {"type": "select", "options": ["random_forest", "xgboost", "neural_network"], "default": "random_forest", "label": "模型类型"},
                "training_period": {"type": "number", "default": 252, "min": 30, "max": 1000, "label": "训练周期"},
                "prediction_horizon": {"type": "number", "default": 5, "min": 1, "max": 20, "label": "预测周期"},
                "feature_count": {"type": "number", "default": 10, "min": 3, "max": 50, "label": "特征数量"}
            },
            "required_nodes": ["data_source", "feature", "model", "trade", "risk"],
            "estimated_complexity": "高",
            "estimated_days": 14
        },
        "arbitrage": {
            "name": "套利策略",
            "description": "利用不同市场或相关资产间的价差进行套利",
            "parameters": {
                "spread_threshold": {"type": "number", "default": 0.005, "min": 0.001, "max": 0.05, "step": 0.001, "label": "价差阈值"},
                "max_holding_time": {"type": "number", "default": 300, "min": 60, "max": 3600, "label": "最大持仓时间(秒)"},
                "correlation_threshold": {"type": "number", "default": 0.8, "min": 0.5, "max": 0.99, "step": 0.01, "label": "相关性阈值"}
            },
            "required_nodes": ["data_source", "model", "trade", "risk"],
            "estimated_complexity": "高",
            "estimated_days": 12
        }
    }

    return {
        "templates": templates,
        "count": len(templates),
        "timestamp": time.time()
    }


@router.get("/api/v1/strategy/conceptions")
async def get_strategy_conceptions_list():
    """获取所有已保存的策略构思（包含生命周期信息）- 优先从PostgreSQL数据库加载"""
    import asyncio
    
    try:
        # 优先从数据库加载，带超时控制
        conceptions = await asyncio.wait_for(
            asyncio.to_thread(load_strategy_conceptions),
            timeout=8.0  # 8秒超时
        )
    except asyncio.TimeoutError:
        logger.error("从数据库加载策略构思超时，回退到文件系统")
        # 超时后从文件系统加载
        conceptions = await asyncio.to_thread(_load_from_filesystem_only)
    except Exception as e:
        logger.error(f"从数据库加载策略构思失败: {e}，回退到文件系统")
        conceptions = await asyncio.to_thread(_load_from_filesystem_only)
    
    # 为每个策略构思添加生命周期信息（直接从文件读取）
    try:
        import json
        import os
        lifecycle_dir = "data/lifecycles"
        
        for conception in conceptions:
            strategy_id = conception.get('id')
            if strategy_id:
                # 直接从文件读取生命周期状态
                lifecycle_file = os.path.join(lifecycle_dir, f"{strategy_id}.json")
                if os.path.exists(lifecycle_file):
                    try:
                        with open(lifecycle_file, 'r', encoding='utf-8') as f:
                            lifecycle_data = json.load(f)
                            conception['lifecycle_stage'] = lifecycle_data.get('current_status', 'draft')
                            conception['lifecycle_updated_at'] = lifecycle_data.get('updated_at')
                    except Exception as e:
                        logger.debug(f"读取生命周期文件失败 {strategy_id}: {e}")
                        conception['lifecycle_stage'] = 'draft'
                else:
                    conception['lifecycle_stage'] = 'draft'
    except Exception as e:
        logger.debug(f"获取生命周期信息失败: {e}")

    return {
        "conceptions": conceptions,
        "count": len(conceptions),
        "timestamp": time.time()
    }


def _load_from_filesystem_only() -> List[Dict]:
    """仅从文件系统加载策略构思（绕过数据库）"""
    try:
        conceptions = []
        if os.path.exists(STRATEGY_CONCEPTION_DIR):
            for filename in os.listdir(STRATEGY_CONCEPTION_DIR):
                if filename.endswith('.json'):
                    filepath = os.path.join(STRATEGY_CONCEPTION_DIR, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if data and isinstance(data, dict):
                                conceptions.append(data)
                    except Exception as e:
                        logger.debug(f"读取文件失败 {filepath}: {e}")
        
        # 按创建时间排序
        conceptions.sort(key=lambda x: x.get('created_at', 0), reverse=True)
        return conceptions
    except Exception as e:
        logger.error(f"从文件系统加载策略构思失败: {e}")
        return []


@router.get("/api/v1/strategy/conceptions/{strategy_id}")
async def get_strategy_conception(strategy_id: str):
    """获取指定的策略构思配置（优先从数据库加载）"""
    try:
        conception = persistence.load(strategy_id)
        if conception:
            return conception
        raise HTTPException(status_code=404, detail=f"策略构思 {strategy_id} 不存在")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取策略构思失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.post("/api/v1/strategy/conceptions")
async def create_strategy_conception(conception_data: dict):
    """创建新的策略构思"""
    try:
        # 添加基本信息
        if not conception_data.get("id"):
            conception_data["id"] = f"strategy_{int(time.time())}"

        conception_data["created_at"] = time.time()
        conception_data["updated_at"] = time.time()
        conception_data["version"] = 1

        # 保存到文件
        result = save_strategy_conception(conception_data)

        return {
            "success": True,
            "message": "策略构思创建成功",
            "strategy_id": result["strategy_id"],
            "data": conception_data,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"创建策略构思失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建失败: {str(e)}")


@router.put("/api/v1/strategy/conceptions/{strategy_id}")
async def update_strategy_conception(strategy_id: str, conception_data: dict):
    """更新策略构思配置"""
    try:
        # 确保ID一致
        conception_data["id"] = strategy_id

        # 保存更新
        result = save_strategy_conception(conception_data)

        return {
            "success": True,
            "message": "策略构思更新成功",
            "strategy_id": strategy_id,
            "data": conception_data,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"更新策略构思失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")


@router.delete("/api/v1/strategy/conceptions/{strategy_id}")
async def delete_strategy_conception(strategy_id: str):
    """删除策略构思配置（同时从数据库和文件系统删除）"""
    try:
        # 先检查策略是否存在
        existing = persistence.load(strategy_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"策略构思 {strategy_id} 不存在")
        
        # 删除策略
        success = persistence.delete(strategy_id)
        if success:
            return {
                "success": True,
                "message": f"策略构思 {strategy_id} 已删除",
                "strategy_id": strategy_id,
                "timestamp": time.time()
            }
        else:
            raise Exception("删除失败")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除策略构思失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@router.post("/api/v1/strategy/conceptions/{strategy_id}/validate")
async def validate_strategy_conception_api(strategy_id: str, conception_data: dict = None):
    """验证策略构思配置"""
    try:
        if conception_data is None:
            # 如果没有提供数据，从已保存的配置中加载
            conception_data = await get_strategy_conception(strategy_id)

        validation_result = validate_strategy_conception(conception_data)

        return {
            "strategy_id": strategy_id,
            "validation": validation_result,
            "timestamp": time.time()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"验证策略构思失败: {e}")
        raise HTTPException(status_code=500, detail=f"验证失败: {str(e)}")


@router.post("/api/v1/strategy/conceptions/validate")
async def validate_strategy_conception_new(conception_data: dict):
    """验证新的策略构思配置（未保存的）"""
    try:
        validation_result = validate_strategy_conception(conception_data)

        return {
            "validation": validation_result,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"验证策略构思失败: {e}")
        raise HTTPException(status_code=500, detail=f"验证失败: {str(e)}")
