#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 读取文件
with open('src/infrastructure/utils/components/unified_query.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 在__init__方法中添加_adapters初始化
# 找到self.storage_adapters = self._initialize_storage_adapters()这一行并在其后添加
for i, line in enumerate(lines):
    if 'self.storage_adapters = self._initialize_storage_adapters()' in line:
        # 在下一行插入
        lines.insert(i + 1, "        self._adapters = self.storage_adapters  # 为测试兼容性添加别名\n")
        lines.insert(i + 2, "        \n")
        lines.insert(i + 3, "        # 查询统计\n")
        lines.insert(i + 4, "        self._query_stats = {\n")
        lines.insert(i + 5, "            'total_queries': 0,\n")
        lines.insert(i + 6, "            'successful_queries': 0,\n")
        lines.insert(i + 7, "            'failed_queries': 0,\n")
        lines.insert(i + 8, "            'total_execution_time': 0.0,\n")
        lines.insert(i + 9, "            'cache_hits': 0,\n")
        lines.insert(i + 10, "            'cache_misses': 0\n")
        lines.insert(i + 11, "        }\n")
        break

# 在shutdown方法之前添加新方法（第895行之前）
new_methods = """
    def register_adapter(self, storage_type: StorageType, adapter: Any) -> None:
        \"\"\"
        注册存储适配器
        
        Args:
            storage_type: 存储类型
            adapter: 适配器实例
        \"\"\"
        self._adapters[storage_type] = adapter
        self.storage_adapters[storage_type] = adapter
        logger.info(f"已注册适配器: {storage_type.value}")

    def unregister_adapter(self, storage_type: StorageType) -> None:
        \"\"\"
        取消注册存储适配器
        
        Args:
            storage_type: 存储类型
        \"\"\"
        if storage_type in self._adapters:
            del self._adapters[storage_type]
        if storage_type in self.storage_adapters:
            del self.storage_adapters[storage_type]
        logger.info(f"已取消注册适配器: {storage_type.value}")

    def get_registered_adapters(self) -> List[StorageType]:
        \"\"\"
        获取已注册的适配器列表
        
        Returns:
            已注册的存储类型列表
        \"\"\"
        return list(self._adapters.keys())

    def get_query_stats(self) -> Dict[str, Any]:
        \"\"\"
        获取查询统计信息
        
        Returns:
            查询统计字典
        \"\"\"
        stats = self._query_stats.copy()
        if stats['total_queries'] > 0:
            stats['avg_execution_time'] = stats['total_execution_time'] / stats['total_queries']
        else:
            stats['avg_execution_time'] = 0.0
        
        if stats['cache_hits'] + stats['cache_misses'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        else:
            stats['cache_hit_rate'] = 0.0
        
        stats['active_connections'] = len(self._adapters)
        return stats

    def validate_query_request(self, request: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        \"\"\"
        验证查询请求
        
        Args:
            request: 查询请求字典
            
        Returns:
            (是否有效, 错误信息)
        \"\"\"
        # 检查必需字段
        required_fields = ['query_type', 'storage_type']
        for field in required_fields:
            if field not in request:
                return False, f"缺少必需字段: {field}"
        
        # 验证查询类型
        query_type = request.get('query_type')
        valid_query_types = ['realtime', 'historical', 'aggregated', 'cross_storage']
        if query_type not in valid_query_types:
            return False, f"无效的查询类型: {query_type}"
        
        # 验证存储类型
        storage_type = request.get('storage_type')
        valid_storage_types = ['influxdb', 'parquet', 'redis', 'hybrid']
        if storage_type not in valid_storage_types:
            return False, f"无效的存储类型: {storage_type}"
        
        return True, None

"""

# 找到shutdown方法的位置并在其前面插入新方法
for i, line in enumerate(lines):
    if 'def shutdown(self):' in line and i > 100:  # 确保是UnifiedQueryInterface的shutdown
        lines.insert(i, new_methods)
        break

# 写回文件
with open('src/infrastructure/utils/components/unified_query.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("成功添加缺失的方法")

