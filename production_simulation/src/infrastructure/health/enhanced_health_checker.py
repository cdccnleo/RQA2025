"""
增强健康检查器模块（别名模块）
提供向后兼容的导入路径

实际实现在 components/enhanced_health_checker.py 中
"""

try:
    from .components.enhanced_health_checker import EnhancedHealthChecker as _CoreEnhancedHealthChecker
    from .models.health_result import HealthCheckResult, CheckType

    class EnhancedHealthChecker(_CoreEnhancedHealthChecker):
        """向后兼容包装：默认返回字典结果"""

        def check_health(self, check_type: CheckType = CheckType.BASIC):
            result = super().check_health(check_type)
            if isinstance(result, HealthCheckResult):
                return result.to_dict()
            return result

        def check_health_result(self, check_type: CheckType = CheckType.BASIC) -> HealthCheckResult:
            """提供显式返回HealthCheckResult对象的接口"""
            return super().check_health(check_type)

except ImportError:
    # 提供基础实现
    class EnhancedHealthChecker:
        pass

__all__ = ['EnhancedHealthChecker']

