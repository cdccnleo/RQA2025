"""
路由健康检查模块
在应用启动时验证所有路由是否正确注册
优化版：支持路由智能分类（必需/可选/实验性）和动态发现
"""

import logging
import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from fastapi import FastAPI

logger = logging.getLogger(__name__)


class RoutePriority(Enum):
    """路由优先级枚举"""
    REQUIRED = "required"      # 必需路由：缺失时报告ERROR
    OPTIONAL = "optional"       # 可选路由：缺失时报告WARNING
    EXPERIMENTAL = "experimental"  # 实验性路由：缺失时报告INFO或忽略


class RouteHealthChecker:
    """路由健康检查器（优化版：智能分类 + 动态发现）"""
    
    def __init__(self, app: FastAPI, enable_dynamic_discovery: bool = True):
        self.app = app
        self.enable_dynamic_discovery = enable_dynamic_discovery
        # 路由分类：必需路由（必需）、可选路由（可选）、实验性路由（实验性）
        self.expected_routes: Dict[str, Tuple[List[str], RoutePriority]] = {
            # 必需路由：核心功能，缺失时报告ERROR
            "数据源": (
                [
                    "/api/v1/data/sources",
                    "/api/v1/data-sources/metrics",
                ],
                RoutePriority.REQUIRED
            ),
            # 可选路由：增强功能，缺失时报告WARNING
            # 注意：这里定义的是路由器中的路径（不包含前缀），因为健康检查器会自动添加前缀进行匹配
            "数据质量": (
                [
                    "/data/lake/stats",
                    "/data/cache/warmup",
                    "/data/lake/datasets",
                    "/data/lake/datasets/{dataset_name}",
                    "/data/performance/recommendations",
                    "/data/quality/repair",
                    "/data/cache/clear/{level}",
                    "/data/performance/metrics",
                    "/data/performance/alerts",
                    "/data/quality/recommendations",
                    "/data/cache/stats",
                    "/data/quality/issues",
                    "/data/quality/metrics",
                ],
                RoutePriority.OPTIONAL
            ),
            "特征工程": (
                [
                    "/features/engineering/tasks",
                    "/features/engineering/tasks/{task_id}/stop",
                    "/features/engineering/tasks/{task_id}",
                    "/features/engineering/features",
                    "/features/engineering/features/{feature_name}",
                    "/features/engineering/indicators",
                ],
                RoutePriority.OPTIONAL
            ),
            "模型训练": (
                [
                    "/ml/training/jobs",
                    "/ml/training/jobs/{job_id}",
                    "/ml/training/jobs/{job_id}/stop",
                    "/ml/training/metrics",
                ],
                RoutePriority.OPTIONAL
            ),
            "策略性能": (
                [
                    "/strategy/performance/comparison",
                    "/strategy/performance/metrics",
                    "/strategy/performance/{strategy_id}",
                ],
                RoutePriority.OPTIONAL
            ),
            "交易信号": (
                [
                    "/trading/signals/realtime",
                    "/trading/signals/stats",
                    "/trading/signals/distribution",
                ],
                RoutePriority.OPTIONAL
            ),
            "订单路由": (
                [
                    "/trading/routing/decisions",
                    "/trading/routing/stats",
                    "/trading/routing/performance",
                ],
                RoutePriority.OPTIONAL
            ),
            "风险报告": (
                [
                    "/risk/reporting/history/{report_id}/download",
                    "/risk/reporting/templates/{template_id}",
                    "/risk/reporting/history/{report_id}",
                    "/risk/reporting/history",
                    "/risk/reporting/stats",
                    "/risk/reporting/templates",
                    "/risk/reporting/tasks/{task_id}/cancel",
                    "/risk/reporting/tasks",
                ],
                RoutePriority.OPTIONAL
            ),
        }
        # WebSocket路由：可选功能
        self.expected_websocket_routes: List[str] = [
            "/ws/feature-engineering",
            "/ws/model-training",
            "/ws/trading-signals",
            "/ws/order-routing",
        ]
        
        # 动态发现的路由（如果启用）
        if self.enable_dynamic_discovery:
            self._discovered_routes = self._discover_routes()
            # 合并动态发现的路由到预期路由中
            self._merge_discovered_routes()
    
    def _discover_routes(self) -> Dict[str, Tuple[List[str], RoutePriority]]:
        """
        动态发现路由
        
        通过扫描路由文件，自动发现定义的路由
        
        Returns:
            Dict[str, Tuple[List[str], RoutePriority]]: 发现的路由，按类别分组（路径列表 + 优先级）
        """
        discovered = {}
        routes_dir = Path(__file__).parent
        
        # 路由文件模式映射
        route_file_patterns = {
            r'.*data.*management.*routes\.py': ('数据质量', RoutePriority.OPTIONAL),
            r'.*feature.*engineering.*routes\.py': ('特征工程', RoutePriority.OPTIONAL),
            r'.*model.*training.*routes\.py': ('模型训练', RoutePriority.OPTIONAL),
            r'.*strategy.*performance.*routes\.py': ('策略性能', RoutePriority.OPTIONAL),
            r'.*trading.*signal.*routes\.py': ('交易信号', RoutePriority.OPTIONAL),
            r'.*order.*routing.*routes\.py': ('订单路由', RoutePriority.OPTIONAL),
            r'.*risk.*reporting.*routes\.py': ('风险报告', RoutePriority.OPTIONAL),
            r'.*datasource.*routes\.py': ('数据源', RoutePriority.REQUIRED),
        }
        
        try:
            for route_file in routes_dir.glob("*_routes.py"):
                file_name = route_file.name
                
                # 匹配路由文件模式
                matched_category = None
                matched_priority = RoutePriority.OPTIONAL
                for pattern, (category, priority) in route_file_patterns.items():
                    if re.match(pattern, file_name, re.IGNORECASE):
                        matched_category = category
                        matched_priority = priority
                        break
                
                if not matched_category:
                    continue
                
                # 解析路由文件，提取路由路径
                routes = self._extract_routes_from_file(route_file)
                if routes:
                    if matched_category not in discovered:
                        discovered[matched_category] = (routes, matched_priority)
                    else:
                        # 合并路由
                        existing_routes, _ = discovered[matched_category]
                        discovered[matched_category] = (existing_routes + routes, matched_priority)
        
        except Exception as e:
            logger.debug(f"动态路由发现失败: {e}")
        
        return discovered
    
    def _extract_routes_from_file(self, route_file: Path) -> List[str]:
        """
        从路由文件中提取路由路径

        Args:
            route_file: 路由文件路径

        Returns:
            List[str]: 提取的路由路径列表
        """
        routes = []

        try:
            with open(route_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 改进的正则表达式，支持多行和复杂格式
            # 匹配 @router.get("/path") 或 @router.post("/path") 等
            route_pattern = r'@router\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\'](?:\s*,.*?)?\)'
            matches = re.findall(route_pattern, content, re.DOTALL)

            for method, path in matches:
                # 标准化路径
                if not path.startswith('/'):
                    path = '/' + path

                # 移除重复的斜杠
                path = re.sub(r'/+', '/', path)

                # 添加到路由列表（去重）
                if path not in routes:
                    routes.append(path)

            # 如果正则表达式没有匹配到，尝试更简单的方法
            if not routes:
                # 查找所有包含 @router. 的行
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('@router.') and '(' in line:
                        # 提取路径参数
                        start = line.find('("') or line.find("('")
                        if start != -1:
                            start += 2  # 跳过引号
                            end = line.find('")', start) or line.find("')", start)
                            if end != -1:
                                path = line[start:end]
                                if not path.startswith('/'):
                                    path = '/' + path
                                path = re.sub(r'/+', '/', path)
                                if path not in routes:
                                    routes.append(path)

        except Exception as e:
            logger.debug(f"从文件 {route_file} 提取路由失败: {e}")

        return routes
    
    def _merge_discovered_routes(self) -> None:
        """合并动态发现的路由到预期路由中"""
        # 优先使用预定义路由，动态发现作为补充
        # 只有当预定义路由为空时才使用动态发现的路由
        for category, (discovered_paths, priority) in self._discovered_routes.items():
            if category in self.expected_routes:
                # 如果类别已存在，检查预定义路由是否为空
                existing_paths, existing_priority = self.expected_routes[category]
                if not existing_paths and discovered_paths:
                    # 只有当预定义路由为空时才使用动态发现的路由
                    logger.debug(f"为 {category} 使用动态发现的路由: {len(discovered_paths)} 个")
                    self.expected_routes[category] = (discovered_paths, existing_priority)
            else:
                # 新类别，直接添加动态发现的路由
                logger.debug(f"添加新的动态发现类别 {category}: {len(discovered_paths)} 个路由")
                self.expected_routes[category] = (discovered_paths, priority)
    
    def check_routes(self) -> Dict[str, Any]:
        """检查所有路由是否正确注册（支持智能分类：必需/可选/实验性）"""
        results = {
            "total_routes": len(self.app.routes),
            "checked_routes": {},
            "missing_routes": {},
            "extra_routes": [],
            "health_status": "healthy",
            "errors": [],
            "warnings": [],
            "info": [],
        }

        # 获取所有已注册的路由路径
        registered_paths = set()
        route_details = []
        for i, route in enumerate(self.app.routes):
            route_type = type(route).__name__
            if hasattr(route, 'path'):
                registered_paths.add(route.path)
                methods = getattr(route, 'methods', set())
                method_str = list(methods)[0] if methods else 'UNKNOWN'
                route_details.append(f"[{i}] {method_str} {route.path} ({route_type})")
            else:
                # 调试：检查没有path属性的路由
                name = getattr(route, 'name', 'unknown')
                route_details.append(f"[{i}] NO_PATH: {route_type} - {name}")

        # 调试信息
        logger.debug(f"总路由数: {len(self.app.routes)}")
        logger.debug(f"有path属性的路由数: {len(registered_paths)}")
        logger.debug("路由详情 (前30个):")
        for detail in route_details[:30]:
            logger.debug(f"  {detail}")
        if len(route_details) > 30:
            logger.debug(f"  ... 还有 {len(route_details) - 30} 个路由")
        
        def _coerce_expected_entry(raw_entry: Any) -> Tuple[List[Any], RoutePriority]:
            """
            兼容新旧 expected_routes 结构：
            - 新结构: (List[str], RoutePriority)
            - 旧结构: List[str]
            """
            if isinstance(raw_entry, tuple) and len(raw_entry) == 2 and isinstance(raw_entry[1], RoutePriority):
                return raw_entry[0], raw_entry[1]
            if isinstance(raw_entry, list):
                return raw_entry, RoutePriority.OPTIONAL
            # 兜底：无法识别的结构，按可选处理
            return [raw_entry], RoutePriority.OPTIONAL

        def _normalize_paths(raw_paths: List[Any], category_name: str) -> List[str]:
            """
            防御性处理：扁平化/过滤非字符串路径，避免出现 unhashable type: 'list'。
            """
            normalized: List[str] = []
            for item in raw_paths:
                if isinstance(item, str):
                    normalized.append(item)
                    continue
                # 处理嵌套容器（list/tuple/set）导致的 path 为 list 的情况
                if isinstance(item, (list, tuple, set)):
                    for sub in item:
                        if isinstance(sub, str):
                            normalized.append(sub)
                        else:
                            results["info"].append(
                                f"{category_name} 模块预期路由包含非字符串项，已忽略: {type(sub).__name__}"
                            )
                    continue
                results["info"].append(
                    f"{category_name} 模块预期路由包含非字符串项，已忽略: {type(item).__name__}"
                )
            # 去重并保持稳定顺序
            seen: Set[str] = set()
            deduped: List[str] = []
            for p in normalized:
                if p not in seen:
                    seen.add(p)
                    deduped.append(p)
            return deduped

        # 检查预期路由（按优先级分类，兼容新旧结构）
        # 为预期路径添加前缀（因为路由器注册时使用了前缀）
        prefix_map = {
            "数据源": "",  # 数据源路由器可能没有前缀或不同前缀
            "数据质量": "/api/v1",
            "特征工程": "/api/v1",
            "模型训练": "/api/v1",
            "策略性能": "/api/v1",
            "交易信号": "/api/v1",
            "订单路由": "/api/v1",
            "风险报告": "/api/v1",
        }

        for category, raw_entry in self.expected_routes.items():
            expected_paths_raw, priority = _coerce_expected_entry(raw_entry)
            expected_paths = _normalize_paths(expected_paths_raw, category)

            # 为路径添加前缀
            prefix = prefix_map.get(category, "/api/v1")
            prefixed_paths = [f"{prefix}{path}" for path in expected_paths]

            category_results = {
                "expected": len(prefixed_paths),
                "found": 0,
                "missing": [],
                "found_paths": [],
                "priority": priority.value,
            }

            for path in prefixed_paths:
                if path in registered_paths:
                    category_results["found"] += 1
                    category_results["found_paths"].append(path)
                else:
                    category_results["missing"].append(path)
                    results["missing_routes"][category] = results["missing_routes"].get(category, [])
                    results["missing_routes"][category].append(path)
            
            results["checked_routes"][category] = category_results
            
            if category_results["missing"]:
                missing_count = len(category_results["missing"])
                missing_paths = ", ".join(category_results["missing"])

                if priority == RoutePriority.REQUIRED:
                    # 必需路由缺失：ERROR级别，并标记整体不健康
                    results["health_status"] = "unhealthy"
                    results["errors"].append(
                        f"{category}模块缺少 {missing_count} 个必需路由: {missing_paths}"
                    )
                elif priority == RoutePriority.OPTIONAL:
                    # 可选路由缺失：WARNING级别（不影响整体健康状态）
                    results["warnings"].append(
                        f"{category}模块缺少 {missing_count} 个可选路由: {missing_paths}"
                    )
                else:
                    # 实验性路由缺失：INFO级别
                    results["info"].append(
                        f"{category}模块缺少 {missing_count} 个实验性路由: {missing_paths}"
                    )
        
        # 检查WebSocket路由（可选功能）
        websocket_results = {
            "expected": len(self.expected_websocket_routes),
            "found": 0,
            "missing": [],
            "found_paths": [],
            "priority": RoutePriority.OPTIONAL.value
        }
        
        for ws_path in self.expected_websocket_routes:
            if ws_path in registered_paths:
                websocket_results["found"] += 1
                websocket_results["found_paths"].append(ws_path)
            else:
                websocket_results["missing"].append(ws_path)
        
        results["checked_routes"]["WebSocket"] = websocket_results
        
        # WebSocket路由缺失：WARNING级别（可选功能）
        if websocket_results["missing"]:
            warning_msg = f"WebSocket路由缺少 {len(websocket_results['missing'])} 个: {', '.join(websocket_results['missing'])}"
            results["warnings"].append(warning_msg)
        
        return results
    
    def print_health_report(self) -> None:
        """打印健康检查报告"""
        results = self.check_routes()
        
        print("\n" + "=" * 80)
        print("  路由健康检查报告")
        print("=" * 80)
        print(f"总路由数: {results['total_routes']}")
        print(f"健康状态: {results['health_status'].upper()}")
        print()
        
        # 打印每个模块的检查结果
        for category, category_results in results["checked_routes"].items():
            missing_count = len(category_results.get("missing", []))
            priority = category_results.get("priority", "unknown")
            
            if missing_count == 0:
                status_icon = "✅"
            elif priority == RoutePriority.REQUIRED.value:
                status_icon = "❌"
            elif priority == RoutePriority.OPTIONAL.value:
                status_icon = "⚠️"
            else:
                status_icon = "ℹ️"
            
            print(f"{status_icon} {category} ({priority}):")
            print(f"   预期: {category_results['expected']} | "
                  f"已注册: {category_results['found']} | "
                  f"缺失: {missing_count}")
            
            if category_results.get("missing"):
                print(f"   缺失路由:")
                for missing_path in category_results["missing"]:
                    print(f"     - {missing_path}")
            print()
        
        # 打印错误信息（按优先级分类）
        if results["errors"]:
            print("❌ 必需路由缺失（ERROR级别）:")
            for error in results["errors"]:
                print(f"   - {error}")
            print()
        
        if results["warnings"]:
            print("⚠️  可选路由缺失（WARNING级别）:")
            for warning in results["warnings"]:
                print(f"   - {warning}")
            print()
        
        if results["info"]:
            print("ℹ️  实验性路由缺失（INFO级别）:")
            for info in results["info"]:
                print(f"   - {info}")
            print()
        
        # 打印总结（仅必需路由缺失会导致整体 unhealthy）
        total_expected = sum(r.get("expected", 0) for r in results["checked_routes"].values())
        total_found = sum(r.get("found", 0) for r in results["checked_routes"].values())
        required_missing = len(results["errors"])
        optional_missing = len(results["warnings"])
        
        print("=" * 80)
        print(f"总结: {total_found}/{total_expected} 个预期路由已注册")
        if results["health_status"] == "healthy":
            print("✅ 所有必需路由健康检查通过！")
            if optional_missing:
                print(f"⚠️  有 {optional_missing} 条可选路由缺失告警（不影响核心功能）")
        else:
            print(f"❌ 路由健康检查发现问题：{required_missing} 条必需路由缺失")
            if optional_missing:
                print(f"⚠️  另外有 {optional_missing} 条可选路由缺失告警")
        print("=" * 80 + "\n")
        
        # 记录到日志
        if results["health_status"] == "healthy":
            logger.info(f"路由健康检查通过: {total_found}/{total_expected} 个路由已注册")
            if results["warnings"]:
                logger.warning(f"路由健康检查存在可选路由缺失告警: {len(results['warnings'])} 条")
        else:
            logger.error(f"路由健康检查失败: {len(results['errors'])} 个必需路由缺失")
            for error in results["errors"]:
                logger.error(f"   - {error}")
            if results["warnings"]:
                for warning in results["warnings"]:
                    logger.warning(f"   - {warning}")
    
    def validate_routes(self, strict: bool = True) -> bool:
        """
        验证路由，如果发现问题则返回False
        
        Args:
            strict: 如果为True，只有必需路由缺失时返回False；如果为False，任何路由缺失都返回False
        
        Returns:
            bool: 路由验证是否通过
        """
        results = self.check_routes()
        if strict:
            # 严格模式：只有必需路由缺失时返回False
            return len(results["errors"]) == 0
        else:
            # 非严格模式：任何路由缺失都返回False
            return results["health_status"] == "healthy"


def check_routes_health(app: FastAPI) -> Dict[str, Any]:
    """检查路由健康的便捷函数"""
    checker = RouteHealthChecker(app)
    return checker.check_routes()


def print_routes_health_report(app: FastAPI) -> None:
    """打印路由健康报告的便捷函数"""
    checker = RouteHealthChecker(app)
    checker.print_health_report()

