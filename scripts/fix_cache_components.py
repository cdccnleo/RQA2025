#!/usr/bin/env python3
"""
修复cache_components.py文件
"""

content_to_append = """

class CacheComponentFactory:
    \"\"\"CacheComponent工厂\"\"\"

    # 支持的组件ID列表
    SUPPORTED_COMPONENT_IDS = [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85, 91]

    @staticmethod
    def create_component(component_id: int) -> 'CacheComponent':
        \"\"\"创建指定ID的组件\"\"\"
        if component_id not in CacheComponentFactory.SUPPORTED_COMPONENT_IDS:
            raise ValueError(f"不支持的组件ID: {component_id}。支持的ID: {CacheComponentFactory.SUPPORTED_COMPONENT_IDS}")

        return CacheComponent(component_id)

    @staticmethod
    def get_available_components() -> List[int]:
        \"\"\"获取所有可用的组件ID\"\"\"
        return sorted(list(CacheComponentFactory.SUPPORTED_COMPONENT_IDS))

    @staticmethod
    def create_all_components() -> Dict[int, 'CacheComponent']:
        \"\"\"创建所有可用组件\"\"\"
        return {
            component_id: CacheComponent(component_id)
            for component_id in CacheComponentFactory.SUPPORTED_COMPONENT_IDS
        }

    @staticmethod
    def get_component_info() -> Dict[str, Any]:
        \"\"\"获取组件工厂信息\"\"\"
        return {
            "factory_name": "CacheComponentFactory",
            "version": "2.0.0",
            "total_components": len(CacheComponentFactory.SUPPORTED_COMPONENT_IDS),
            "supported_ids": sorted(list(CacheComponentFactory.SUPPORTED_COMPONENT_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一cache_templates工厂，替代原有的模板化文件"
        }


# 向后兼容：创建旧的组件实例
def create_cache_component_1(): return CacheComponentFactory.create_component(1)
def create_cache_component_7(): return CacheComponentFactory.create_component(7)
def create_cache_component_13(): return CacheComponentFactory.create_component(13)
def create_cache_component_19(): return CacheComponentFactory.create_component(19)
def create_cache_component_25(): return CacheComponentFactory.create_component(25)
def create_cache_component_31(): return CacheComponentFactory.create_component(31)
def create_cache_component_37(): return CacheComponentFactory.create_component(37)
def create_cache_component_43(): return CacheComponentFactory.create_component(43)
def create_cache_component_49(): return CacheComponentFactory.create_component(49)
def create_cache_component_55(): return CacheComponentFactory.create_component(55)
def create_cache_component_61(): return CacheComponentFactory.create_component(61)
def create_cache_component_67(): return CacheComponentFactory.create_component(67)
def create_cache_component_73(): return CacheComponentFactory.create_component(73)
def create_cache_component_79(): return CacheComponentFactory.create_component(79)
def create_cache_component_85(): return CacheComponentFactory.create_component(85)
def create_cache_component_91(): return CacheComponentFactory.create_component(91)


__all__ = [
    "ICacheComponent",
    "CacheComponent",
    "CacheComponentFactory",
    "create_cache_component_1",
    "create_cache_component_7",
    "create_cache_component_13",
    "create_cache_component_19",
    "create_cache_component_25",
    "create_cache_component_31",
    "create_cache_component_37",
    "create_cache_component_43",
    "create_cache_component_49",
    "create_cache_component_55",
    "create_cache_component_61",
    "create_cache_component_67",
    "create_cache_component_73",
    "create_cache_component_79",
    "create_cache_component_85",
    "create_cache_component_91",
]
"""

with open('src/infrastructure/cache/cache_components.py', 'a', encoding='utf-8') as f:
    f.write(content_to_append)

print("Successfully appended CacheComponentFactory to cache_components.py")
