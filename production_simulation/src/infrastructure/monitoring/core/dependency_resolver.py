#!/usr/bin/env python3
"""
RQA2025 基础设施层依赖解析器

负责组件依赖关系的解析、验证和管理。
这是从ComponentRegistry中拆分出来的依赖管理组件。
"""

import logging
from typing import Dict, Any, Optional, List, Set
import threading

logger = logging.getLogger(__name__)


class DependencyResolver:
    """
    依赖解析器

    负责分析和验证组件之间的依赖关系，确保组件按正确的顺序启动和停止。
    """

    def __init__(self):
        """初始化依赖解析器"""
        self.dependency_graph: Dict[str, Set[str]] = {}  # 组件 -> 依赖它的组件
        self.reverse_graph: Dict[str, Set[str]] = {}     # 组件 -> 它依赖的组件
        self._lock = threading.RLock()

        logger.info("依赖解析器初始化完成")

    def add_component(self, component_name: str, dependencies: Optional[List[str]] = None):
        """
        添加组件及其依赖关系

        Args:
            component_name: 组件名称
            dependencies: 该组件依赖的其他组件列表
        """
        with self._lock:
            dependencies = dependencies or []

            # 更新依赖图
            self.dependency_graph[component_name] = set()
            self.reverse_graph[component_name] = set(dependencies)

            # 更新反向依赖关系
            for dep in dependencies:
                if dep not in self.dependency_graph:
                    self.dependency_graph[dep] = set()
                self.dependency_graph[dep].add(component_name)

            logger.debug(f"添加组件依赖关系: {component_name} -> {dependencies}")

    def remove_component(self, component_name: str):
        """
        移除组件及其依赖关系

        Args:
            component_name: 组件名称
        """
        with self._lock:
            if component_name in self.reverse_graph:
                # 移除该组件的依赖关系
                dependencies = self.reverse_graph[component_name]
                for dep in dependencies:
                    if dep in self.dependency_graph:
                        self.dependency_graph[dep].discard(component_name)

                # 移除该组件的被依赖关系
                if component_name in self.dependency_graph:
                    depended_by = self.dependency_graph[component_name]
                    for dependent in depended_by:
                        if dependent in self.reverse_graph:
                            self.reverse_graph[dependent].discard(component_name)

                # 清理图
                self.reverse_graph.pop(component_name, None)
                self.dependency_graph.pop(component_name, None)

                logger.debug(f"移除组件依赖关系: {component_name}")

    def get_startup_order(self, component_names: List[str]) -> List[str]:
        """
        获取组件启动顺序

        Args:
            component_names: 要启动的组件列表

        Returns:
            List[str]: 按依赖顺序排列的组件列表
        """
        with self._lock:
            # 构建子图
            subgraph = self._build_subgraph(component_names)

            # 拓扑排序
            try:
                order = self._topological_sort(subgraph)
                logger.debug(f"组件启动顺序: {order}")
                return order
            except ValueError as e:
                logger.error(f"依赖关系循环检测失败: {e}")
                raise

    def get_shutdown_order(self, component_names: List[str]) -> List[str]:
        """
        获取组件停止顺序（启动顺序的逆序）

        Args:
            component_names: 要停止的组件列表

        Returns:
            List[str]: 按依赖顺序排列的停止组件列表
        """
        startup_order = self.get_startup_order(component_names)
        return list(reversed(startup_order))

    def validate_dependencies(self, component_names: List[str]) -> Dict[str, Any]:
        """
        验证组件依赖关系

        Args:
            component_names: 要验证的组件列表

        Returns:
            Dict[str, Any]: 验证结果
        """
        with self._lock:
            issues = []
            warnings = []

            # 检查缺失依赖
            for component in component_names:
                if component not in self.reverse_graph:
                    warnings.append(f"组件 '{component}' 没有定义依赖关系")
                    continue

                dependencies = self.reverse_graph[component]
                for dep in dependencies:
                    if dep not in component_names and dep not in self.reverse_graph:
                        issues.append(f"组件 '{component}' 依赖 '{dep}' 未找到")

            # 检查循环依赖
            try:
                self.get_startup_order(component_names)
            except ValueError as e:
                issues.append(f"检测到循环依赖: {str(e)}")

            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'warnings': warnings
            }

    def get_dependency_info(self, component_name: str) -> Dict[str, Any]:
        """
        获取组件的依赖信息

        Args:
            component_name: 组件名称

        Returns:
            Dict[str, Any]: 依赖信息
        """
        with self._lock:
            if component_name not in self.reverse_graph:
                return {
                    'component': component_name,
                    'dependencies': [],
                    'depended_by': [],
                    'depth': 0
                }

            dependencies = list(self.reverse_graph[component_name])
            depended_by = list(self.dependency_graph.get(component_name, []))

            # 计算依赖深度
            depth = self._calculate_dependency_depth(component_name)

            return {
                'component': component_name,
                'dependencies': dependencies,
                'depended_by': depended_by,
                'depth': depth
            }

    def get_system_dependency_graph(self) -> Dict[str, Any]:
        """
        获取整个系统的依赖关系图

        Returns:
            Dict[str, Any]: 依赖关系图
        """
        with self._lock:
            return {
                'dependency_graph': {k: list(v) for k, v in self.dependency_graph.items()},
                'reverse_graph': {k: list(v) for k, v in self.reverse_graph.items()},
                'all_components': list(set(self.dependency_graph.keys()) | set(self.reverse_graph.keys())),
                'total_relationships': sum(len(deps) for deps in self.reverse_graph.values())
            }

    def find_critical_path(self, start_component: str, end_component: str) -> Optional[List[str]]:
        """
        查找两个组件之间的关键路径

        Args:
            start_component: 起始组件
            end_component: 目标组件

        Returns:
            Optional[List[str]]: 关键路径，如果不存在则返回None
        """
        with self._lock:
            # 使用DFS查找路径
            visited = set()
            path = []

            def dfs(current: str) -> Optional[List[str]]:
                if current in visited:
                    return None

                visited.add(current)
                path.append(current)

                if current == end_component:
                    return path.copy()

                # 尝试通过依赖关系继续搜索
                for dependent in self.dependency_graph.get(current, []):
                    result = dfs(dependent)
                    if result:
                        return result

                path.pop()
                return None

            return dfs(start_component)

    def _build_subgraph(self, component_names: List[str]) -> Dict[str, Set[str]]:
        """
        构建子图

        Args:
            component_names: 组件列表

        Returns:
            Dict[str, Set[str]]: 子图
        """
        subgraph = {}

        for component in component_names:
            if component in self.reverse_graph:
                # 只包含指定组件集合内的依赖
                subgraph[component] = self.reverse_graph[component] & set(component_names)

        return subgraph

    def _topological_sort(self, graph: Dict[str, Set[str]]) -> List[str]:
        """
        拓扑排序

        Args:
            graph: 依赖图

        Returns:
            List[str]: 排序后的组件列表

        Raises:
            ValueError: 检测到循环依赖
        """
        # 计算入度
        in_degree = {node: 0 for node in graph}
        for node in graph:
            for dependency in graph[node]:
                if dependency in in_degree:
                    in_degree[dependency] += 1

        # 初始化队列
        queue = [node for node in in_degree if in_degree[node] == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            # 更新依赖该节点的节点的入度
            if current in graph:
                for dependent in self.dependency_graph.get(current, []):
                    if dependent in in_degree:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            queue.append(dependent)

        # 检查是否有循环依赖
        if len(result) != len(graph):
            remaining = set(graph.keys()) - set(result)
            raise ValueError(f"检测到循环依赖，剩余组件: {remaining}")

        return result

    def _calculate_dependency_depth(self, component_name: str, visited: Optional[Set[str]] = None) -> int:
        """
        计算组件的依赖深度

        Args:
            component_name: 组件名称
            visited: 已访问的组件集合

        Returns:
            int: 依赖深度
        """
        if visited is None:
            visited = set()

        if component_name in visited:
            return 0  # 防止循环

        visited.add(component_name)

        if component_name not in self.reverse_graph:
            return 0

        dependencies = self.reverse_graph[component_name]
        if not dependencies:
            return 0

        # 递归计算最大深度
        max_depth = 0
        for dep in dependencies:
            depth = self._calculate_dependency_depth(dep, visited.copy())
            max_depth = max(max_depth, depth)

        return max_depth + 1

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取依赖解析器的健康状态

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        try:
            graph_info = self.get_system_dependency_graph()

            issues = []

            # 检查孤立组件
            isolated_components = []
            for component in graph_info['all_components']:
                deps = len(self.reverse_graph.get(component, []))
                depended_by = len(self.dependency_graph.get(component, []))
                if deps == 0 and depended_by == 0:
                    isolated_components.append(component)

            if isolated_components:
                issues.append(f"发现孤立组件: {isolated_components}")

            # 检查深度过大的依赖链
            deep_dependencies = []
            for component in graph_info['all_components']:
                depth = self._calculate_dependency_depth(component)
                if depth > 5:  # 依赖深度超过5层
                    deep_dependencies.append(f"{component}({depth})")

            if deep_dependencies:
                issues.append(f"依赖链过深: {deep_dependencies}")

            return {
                'status': 'healthy' if not issues else 'warning',
                'total_components': len(graph_info['all_components']),
                'total_relationships': graph_info['total_relationships'],
                'issues': issues,
                'last_check': 'now'
            }

        except Exception as e:
            logger.error(f"获取健康状态失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


# 全局依赖解析器实例
global_dependency_resolver = DependencyResolver()
