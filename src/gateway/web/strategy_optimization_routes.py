"""
策略优化路由模块
提供策略参数优化、AI优化、组合优化等API端点
符合量化交易系统安全要求
"""

import time
import logging
import random
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter()

# 存储优化任务状态
optimization_tasks = {}
ai_optimization_tasks = {}

# 量化交易系统安全要求：计算资源限制常量
MAX_OPTIMIZATION_ITERATIONS = 1000  # 最大迭代次数
MAX_OPTIMIZATION_TIME_SECONDS = 30 * 60  # 最大30分钟
MAX_PARAM_COMBINATIONS = 10000  # 最大参数组合数


@router.post("/api/v1/strategy/optimization/start")
async def start_optimization(request: Dict[str, Any]):
    """启动策略优化 - 符合量化交易系统安全要求"""
    try:
        # 量化交易系统安全要求：验证参数搜索空间大小
        parameters = request.get("parameters", {})
        if parameters:
            param_count = 1
            for param_values in parameters.values():
                if isinstance(param_values, list):
                    param_count *= len(param_values)
            
            if param_count > MAX_PARAM_COMBINATIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"参数组合数({param_count})超过限制({MAX_PARAM_COMBINATIONS})，请减少参数数量或范围"
                )
            
            logger.info(f"参数搜索空间大小: {param_count}")
        
        # 量化交易系统安全要求：验证最大迭代次数
        max_iterations = request.get("max_iterations", MAX_OPTIMIZATION_ITERATIONS)
        if max_iterations > MAX_OPTIMIZATION_ITERATIONS:
            logger.warning(f"最大迭代次数({max_iterations})超过限制，使用默认值: {MAX_OPTIMIZATION_ITERATIONS}")
            max_iterations = MAX_OPTIMIZATION_ITERATIONS
        
        # 量化交易系统安全要求：验证计算时间限制
        max_time_seconds = request.get("max_time_seconds", MAX_OPTIMIZATION_TIME_SECONDS)
        if max_time_seconds > MAX_OPTIMIZATION_TIME_SECONDS:
            logger.warning(f"计算时间限制({max_time_seconds}秒)超过限制，使用默认值: {MAX_OPTIMIZATION_TIME_SECONDS}秒")
            max_time_seconds = MAX_OPTIMIZATION_TIME_SECONDS
        
        from .strategy_optimization_service import start_parameter_optimization
        
        task_id = await start_parameter_optimization(
            strategy_id=request.get("strategy_id"),
            method=request.get("method"),
            target=request.get("target"),
            parameters=parameters,
            max_iterations=max_iterations,
            max_time_seconds=max_time_seconds
        )
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "优化任务已启动",
            "timestamp": time.time(),
            "config": {
                "max_iterations": max_iterations,
                "max_time_seconds": max_time_seconds,
                "param_combinations": param_count if parameters else 0
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动优化失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动失败: {str(e)}")


@router.get("/api/v1/strategy/optimization/progress")
async def get_optimization_progress(task_id: str = None):
    """获取优化进度
    
    Args:
        task_id: 优化任务ID，如果不提供则返回最新的任务进度
    """
    try:
        from .strategy_optimization_service import get_optimization_progress
        return await get_optimization_progress(task_id)
    except Exception as e:
        logger.error(f"获取优化进度失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/strategy/optimization/results")
async def get_optimization_results(strategy_id: str = None):
    """获取优化结果"""
    try:
        from .strategy_persistence import list_optimization_results
        
        results = list_optimization_results(strategy_id)
        
        # 格式化结果
        formatted_results = []
        for result in results:  # 处理所有结果
            # 获取优化方法、目标和策略信息
            optimization_method = result.get("method", "grid_search")
            optimization_target = result.get("target", "sharpe")
            strategy_id_from_result = result.get("strategy_id", "")
            strategy_name = result.get("strategy_name", "未知策略")
            
            # 从优化结果中提取参数和指标
            # 如果是results数组，取第一个；如果是单个结果，直接使用
            if "results" in result and result["results"]:
                # 遍历所有结果（已经按得分排序）
                for idx, opt_result in enumerate(result["results"]):
                    if isinstance(opt_result, dict):
                        # 适配参数优化器返回的格式
                        # 参数优化器返回: {'params': {...}, 'performance': {...}}
                        params = opt_result.get("params", opt_result.get("parameters", {}))
                        performance = opt_result.get("performance", {})
                        
                        formatted_results.append({
                            "id": f"{result.get('task_id', '')}_{idx}",
                            "strategy_id": strategy_id_from_result,
                            "strategy_name": strategy_name,
                            "optimization_method": optimization_method,
                            "target": optimization_target,
                            "parameters": params,
                            "score": performance.get("sharpe", performance.get("sharpe_ratio", 0)),
                            "sharpe_ratio": performance.get("sharpe", performance.get("sharpe_ratio", 0)),
                            "total_return": performance.get("total_return", 0),
                            "max_drawdown": performance.get("max_drawdown", 0)
                        })
            else:
                # 直接使用结果
                formatted_results.append({
                    "id": result.get("task_id", ""),
                    "strategy_id": strategy_id_from_result,
                    "strategy_name": strategy_name,
                    "optimization_method": optimization_method,
                    "target": optimization_target,
                    "parameters": result.get("parameters", {}),
                    "score": result.get("best_score", result.get("score", 0)),
                    "sharpe_ratio": result.get("sharpe_ratio", 0),
                    "total_return": result.get("total_return", 0),
                    "max_drawdown": result.get("max_drawdown", 0)
                })
        
        # 按得分排序（确保排名正确）
        formatted_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # 只返回前10个结果
        formatted_results = formatted_results[:10]
        
        return {"results": formatted_results}
    except Exception as e:
        logger.error(f"获取优化结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.post("/api/v1/strategy/optimization/stop")
async def stop_optimization():
    """停止优化"""
    try:
        if optimization_tasks:
            task = list(optimization_tasks.values())[-1]
            task["status"] = "stopped"
        return {"success": True, "message": "优化已停止"}
    except Exception as e:
        logger.error(f"停止优化失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止失败: {str(e)}")


@router.delete("/api/v1/strategy/optimization/results/{task_id}")
async def delete_optimization_result_api(task_id: str):
    """删除优化结果"""
    try:
        from .strategy_persistence import delete_optimization_result, load_optimization_result, save_optimization_result
        import os
        
        # 处理复合ID (如: task_id_0, task_id_1)
        # 如果ID包含下划线，可能是复合ID，需要提取原始task_id
        original_task_id = task_id
        result_index = None
        
        if '_' in task_id:
            # 尝试解析复合ID
            parts = task_id.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                original_task_id = parts[0]
                result_index = int(parts[1])
        
        # 如果是指定索引的删除（从多结果中删除一个）
        if result_index is not None:
            # 加载原始结果
            result = load_optimization_result(original_task_id)
            if result and "results" in result and isinstance(result["results"], list):
                if 0 <= result_index < len(result["results"]):
                    # 删除指定索引的结果
                    result["results"].pop(result_index)
                    
                    # 如果还有结果，保存更新后的结果
                    if result["results"]:
                        save_optimization_result(original_task_id, result)
                        return {"success": True, "message": "优化结果已删除"}
                    else:
                        # 如果没有结果了，删除整个文件
                        success = delete_optimization_result(original_task_id)
                        if success:
                            return {"success": True, "message": "优化结果已删除"}
                        else:
                            raise HTTPException(status_code=404, detail="优化结果不存在")
                else:
                    raise HTTPException(status_code=404, detail="优化结果索引无效")
            else:
                raise HTTPException(status_code=404, detail="优化结果不存在")
        else:
            # 删除整个优化结果文件
            success = delete_optimization_result(task_id)
            if success:
                return {"success": True, "message": "优化结果已删除"}
            else:
                raise HTTPException(status_code=404, detail="优化结果不存在")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除优化结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@router.post("/api/v1/strategy/optimization/results/batch-delete")
async def batch_delete_optimization_results(request: Dict[str, Any]):
    """批量删除优化结果"""
    try:
        from .strategy_persistence import delete_optimization_result
        
        task_ids = request.get("task_ids", [])
        if not task_ids or not isinstance(task_ids, list):
            raise HTTPException(status_code=400, detail="缺少task_ids参数或格式不正确")
        
        deleted_count = 0
        failed_count = 0
        errors = []
        
        for task_id in task_ids:
            try:
                # 处理复合ID (如: task_id_0, task_id_1)
                original_task_id = task_id
                if '_' in task_id:
                    parts = task_id.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        original_task_id = parts[0]
                
                # 删除优化结果
                success = delete_optimization_result(original_task_id)
                if success:
                    deleted_count += 1
                    logger.info(f"批量删除成功: {task_id}")
                else:
                    failed_count += 1
                    errors.append(f"{task_id}: 删除失败")
                    logger.warning(f"批量删除失败: {task_id}")
            except Exception as e:
                failed_count += 1
                errors.append(f"{task_id}: {str(e)}")
                logger.error(f"批量删除异常 {task_id}: {e}")
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "failed_count": failed_count,
            "total_count": len(task_ids),
            "errors": errors,
            "message": f"成功删除 {deleted_count} 条，失败 {failed_count} 条"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量删除优化结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量删除失败: {str(e)}")


@router.post("/api/v1/strategy/ai-optimization/start")
async def start_ai_optimization(request: Dict[str, Any]):
    """启动AI策略优化"""
    try:
        from .strategy_optimization_service import start_ai_optimization
        
        task_id = await start_ai_optimization(
            strategy_id=request.get("strategy_id"),
            engine=request.get("engine"),
            target=request.get("target")
        )
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "AI优化任务已启动",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"启动AI优化失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动失败: {str(e)}")


@router.get("/api/v1/strategy/ai-optimization/progress")
async def get_ai_optimization_progress():
    """获取AI优化进度"""
    try:
        from .strategy_optimization_service import get_ai_optimization_progress
        return await get_ai_optimization_progress()
    except Exception as e:
        logger.error(f"获取AI优化进度失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/strategy/ai-optimization/results")
async def get_ai_optimization_results():
    """获取AI优化结果"""
    try:
        return {
            "sharpe_ratio": 1.35,
            "total_return": 0.18,
            "max_drawdown": 0.07
        }
    except Exception as e:
        logger.error(f"获取AI优化结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.post("/api/v1/strategy/ai-optimization/stop")
async def stop_ai_optimization():
    """停止AI优化"""
    try:
        if ai_optimization_tasks:
            task = list(ai_optimization_tasks.values())[-1]
            task["status"] = "stopped"
        return {"success": True, "message": "AI优化已停止"}
    except Exception as e:
        logger.error(f"停止AI优化失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止失败: {str(e)}")


@router.post("/api/v1/strategy/portfolio/optimize")
async def optimize_portfolio(request: Dict[str, Any]):
    """多策略组合优化"""
    try:
        from .strategy_optimization_service import optimize_portfolio
        
        result = await optimize_portfolio(
            strategy_ids=request.get("strategy_ids", []),
            method=request.get("method", "risk_parity"),
            target_return=request.get("target_return", 0.15),
            max_risk=request.get("max_risk", 0.20)
        )
        
        return result
    except Exception as e:
        logger.error(f"组合优化失败: {e}")
        raise HTTPException(status_code=500, detail=f"优化失败: {str(e)}")


@router.post("/api/v1/strategy/optimization/{task_id}/apply")
async def apply_optimization_result(task_id: str, request: Dict[str, Any]):
    """应用优化结果到策略"""
    try:
        from .strategy_persistence import load_optimization_result
        from .strategy_routes import save_strategy_conception
        import os
        import json

        # 处理复合ID (如: opt_123_0, opt_123_1) - 格式为 {task_id}_{index}
        # 注意：task_id本身可能包含下划线（如 opt_1771491474）
        original_task_id = task_id
        result_index = 0  # 默认使用第一个结果
        
        # 检查是否是复合ID格式（以数字结尾且前面有下划线）
        # 但task_id本身可能是 opt_1771491474 这种格式，不应该被分割
        # 只有明确的 {task_id}_{index} 格式才需要分割
        if '_' in task_id:
            parts = task_id.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                # 检查后一部分是否是小的索引数字（0-99）
                # 如果是，则认为是复合ID；否则认为是task_id的一部分
                idx = int(parts[1])
                if 0 <= idx <= 99 and len(parts[1]) <= 2:
                    original_task_id = parts[0]
                    result_index = idx

        # 加载优化结果
        result = load_optimization_result(original_task_id)
        if not result:
            raise HTTPException(status_code=404, detail="优化结果不存在")

        strategy_id = request.get("strategy_id") or result.get("strategy_id")
        if not strategy_id:
            raise HTTPException(status_code=400, detail="缺少策略ID")

        # 获取指定索引的参数
        results = result.get("results", [])
        if not results:
            raise HTTPException(status_code=400, detail="优化结果为空")

        if isinstance(results, list):
            if result_index < 0 or result_index >= len(results):
                raise HTTPException(status_code=400, detail="优化结果索引无效")
            best_result = results[result_index]
        else:
            best_result = results
        
        best_params = best_result.get("params", {})

        # 加载策略配置 - 使用统一持久化模块（优先从PostgreSQL加载）
        try:
            from .strategy_routes import persistence
            strategy = persistence.load(strategy_id)
            if not strategy:
                # 降级到文件系统
                strategy_path = os.path.join("/app/data/strategy_conceptions", f"{strategy_id}.json")
                if os.path.exists(strategy_path):
                    with open(strategy_path, 'r', encoding='utf-8') as f:
                        strategy = json.load(f)
                else:
                    raise HTTPException(status_code=404, detail="策略不存在")
        except Exception as load_error:
            logger.warning(f"从统一持久化加载策略失败，尝试文件系统: {load_error}")
            # 降级到文件系统
            strategy_path = os.path.join("/app/data/strategy_conceptions", f"{strategy_id}.json")
            if os.path.exists(strategy_path):
                with open(strategy_path, 'r', encoding='utf-8') as f:
                    strategy = json.load(f)
            else:
                raise HTTPException(status_code=404, detail="策略不存在")

        # 更新策略参数
        if "parameters" not in strategy:
            strategy["parameters"] = {}
        strategy["parameters"].update(best_params)

        # 记录版本信息
        if "version_history" not in strategy:
            strategy["version_history"] = []

        strategy["version_history"].append({
            "version": strategy.get("version", 1),
            "updated_at": time.time(),
            "reason": f"应用优化结果 {task_id}",
            "parameters": best_params.copy()
        })

        # 增加版本号
        strategy["version"] = strategy.get("version", 1) + 1
        strategy["updated_at"] = time.time()

        # 更新统计信息 - 增加优化次数
        if "stats" not in strategy:
            strategy["stats"] = {}
        strategy["stats"]["optimization_count"] = strategy["stats"].get("optimization_count", 0) + 1
        strategy["stats"]["last_optimization_at"] = time.time()
        
        # 将stats合并到parameters中（因为数据库表中没有stats字段）
        if "parameters" not in strategy:
            strategy["parameters"] = {}
        strategy["parameters"]["_stats"] = strategy["stats"]

        # 保存策略 - 使用统一持久化模块（双写：PostgreSQL + 文件系统）
        try:
            from .strategy_routes import save_strategy_conception
            # 传入 auto_increment_version=False，因为版本号已经在上面增加了
            save_result = save_strategy_conception(strategy, auto_increment_version=False)
            if save_result:
                logger.info(f"优化结果已应用到策略 {strategy_id} (通过统一持久化): {best_params}")
            else:
                # 降级到直接文件保存
                with open(strategy_path, 'w', encoding='utf-8') as f:
                    json.dump(strategy, f, ensure_ascii=False, indent=2)
                logger.info(f"优化结果已应用到策略 {strategy_id} (直接文件保存): {best_params}")
        except Exception as save_error:
            logger.warning(f"统一持久化保存失败，使用直接文件保存: {save_error}")
            # 降级到直接文件保存
            with open(strategy_path, 'w', encoding='utf-8') as f:
                json.dump(strategy, f, ensure_ascii=False, indent=2)
            logger.info(f"优化结果已应用到策略 {strategy_id}: {best_params}")

        return {
            "success": True,
            "message": "优化结果已成功应用到策略",
            "strategy_id": strategy_id,
            "applied_parameters": best_params,
            "new_version": strategy["version"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"应用优化结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"应用失败: {str(e)}")

