"""
示例生成器

负责为API端点生成请求和响应示例。

重构前: APIDocumentationEnhancer中的示例生成逻辑 (~80行)
重构后: ExampleGenerator独立组件 (~70行)
"""

from typing import Dict, Any


class ExampleGenerator:
    """
    示例生成器
    
    职责：
    - 生成请求示例
    - 生成响应示例
    - 生成错误示例
    """
    
    def generate_request_example(self, endpoint) -> Dict[str, Any]:
        """
        生成请求示例
        
        Args:
            endpoint: 端点文档对象
        
        Returns:
            Dict[str, Any]: 请求示例
        """
        example = {}
        
        # 从参数生成示例
        for param in endpoint.parameters:
            if param.example:
                example[param.name] = param.example
            elif param.default is not None:
                example[param.name] = param.default
        
        return example
    
    def generate_success_response_example(self, endpoint) -> Dict[str, Any]:
        """生成成功响应示例"""
        return {
            "success": True,
            "message": "操作成功",
            "data": self._generate_data_example(endpoint),
            "timestamp": "2025-10-23T22:00:00Z",
            "request_id": "req_abc123"
        }
    
    def generate_error_response_example(self, status_code: int) -> Dict[str, Any]:
        """生成错误响应示例"""
        error_messages = {
            400: "请求参数错误",
            401: "未授权，请先登录",
            403: "权限不足",
            404: "资源不存在",
            429: "请求过于频繁",
            500: "服务器内部错误"
        }
        
        return {
            "success": False,
            "message": error_messages.get(status_code, "未知错误"),
            "error": {
                "code": f"E{status_code}",
                "message": error_messages.get(status_code, "未知错误"),
                "details": {}
            },
            "timestamp": "2025-10-23T22:00:00Z",
            "request_id": "req_error_123"
        }
    
    def _generate_data_example(self, endpoint) -> Any:
        """根据端点类型生成数据示例"""
        path_lower = endpoint.path.lower()
        
        if 'market' in path_lower or 'kline' in path_lower:
            return {
                "symbol": "BTC/USDT",
                "price": 45000.00,
                "volume": 123.45,
                "timestamp": "2025-10-23T22:00:00Z"
            }
        elif 'order' in path_lower:
            return {
                "order_id": "ORD123456",
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": 1.0,
                "price": 45000.00,
                "status": "filled"
            }
        elif 'feature' in path_lower:
            return {
                "features": [
                    {"name": "MACD", "value": 123.45},
                    {"name": "RSI", "value": 65.30}
                ]
            }
        else:
            return {}

