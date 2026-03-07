"""
api_documentation_enhancer 模块

提供 api_documentation_enhancer 相关功能和接口。
"""

import json

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API文档增强器
完善接口参数文档和标准化响应格式
"""


@dataclass
class APIParameterDocumentation:
    """API参数文档"""
    name: str
    type: str
    required: bool
    description: str
    example: Any = None
    default: Any = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[str] = field(default_factory=list)


@dataclass
class APIResponseDocumentation:
    """API响应文档"""
    status_code: int
    description: str
    content_type: str = "application/json"
    schema: Dict[str, Any] = field(default_factory=dict)
    examples: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class APIEndpointDocumentation:
    """API端点文档"""
    path: str
    method: str
    summary: str
    description: str
    parameters: List[APIParameterDocumentation] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: List[APIResponseDocumentation] = field(default_factory=list)
    authentication: List[str] = field(default_factory=list)
    rate_limits: Dict[str, Any] = field(default_factory=dict)
    error_codes: List[Dict[str, Any]] = field(default_factory=list)
    examples: Dict[str, Any] = field(default_factory=dict)
    changelog: List[Dict[str, Any]] = field(default_factory=list)


class APIDocumentationEnhancer:
    """API文档增强器"""

    def __init__(self):
        self.endpoints: Dict[str, APIEndpointDocumentation] = {}
        self.common_responses = self._create_common_responses()
        self.error_codes = self._create_error_codes()

    def _create_common_responses(self) -> Dict[str, APIResponseDocumentation]:
        """创建通用响应"""
        return {
            "success": APIResponseDocumentation(
                status_code=200,
                description="请求成功",
                schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "example": True},
                        "message": {"type": "string", "example": "操作成功"},
                        "data": {"type": "object"},
                        "timestamp": {"type": "string", "format": "date-time"},
                        "request_id": {"type": "string"}
                    },
                    "required": ["success", "timestamp"]
                },
                examples={
                    "success_response": {
                        "success": True,
                        "message": "操作成功",
                        "data": {"key": "value"},
                        "timestamp": "2025-02-07T10:00:00Z",
                        "request_id": "req_123456789"
                    }
                }
            ),
            "created": APIResponseDocumentation(
                status_code=201,
                description="资源创建成功",
                headers={"Location": "/api/v1/resource/123"},
                examples={
                    "created_response": {
                        "success": True,
                        "message": "资源创建成功",
                        "data": {"id": 123, "name": "新资源"},
                        "timestamp": "2025-02-07T10:00:00Z"
                    }
                }
            ),
            "bad_request": APIResponseDocumentation(
                status_code=400,
                description="请求参数错误",
                schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "example": False},
                        "message": {"type": "string", "example": "请求参数无效"},
                        "error": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string", "example": "VALIDATION_ERROR"},
                                "message": {"type": "string", "example": "字段 'name' 不能为空"},
                                "details": {"type": "object"}
                            }
                        },
                        "timestamp": {"type": "string", "format": "date-time"}
                    }
                }
            ),
            "unauthorized": APIResponseDocumentation(
                status_code=401,
                description="未授权访问",
                examples={
                    "unauthorized_response": {
                        "success": False,
                        "message": "未授权访问",
                        "error": {
                            "code": "UNAUTHORIZED",
                            "message": "请提供有效的访问令牌"
                        }
                    }
                }
            ),
            "forbidden": APIResponseDocumentation(
                status_code=403,
                description="访问被拒绝",
                examples={
                    "forbidden_response": {
                        "success": False,
                        "message": "访问被拒绝",
                        "error": {
                            "code": "FORBIDDEN",
                            "message": "您没有权限执行此操作"
                        }
                    }
                }
            ),
            "not_found": APIResponseDocumentation(
                status_code=404,
                description="资源未找到",
                examples={
                    "not_found_response": {
                        "success": False,
                        "message": "资源未找到",
                        "error": {
                            "code": "NOT_FOUND",
                            "message": "请求的资源不存在"
                        }
                    }
                }
            ),
            "rate_limited": APIResponseDocumentation(
                status_code=429,
                description="请求频率过高",
                headers={"Retry-After": "60"},
                examples={
                    "rate_limited_response": {
                        "success": False,
                        "message": "请求频率过高，请稍后再试",
                        "error": {
                            "code": "RATE_LIMITED",
                            "message": "超出每分钟100次请求的限制"
                        }
                    }
                }
            ),
            "internal_error": APIResponseDocumentation(
                status_code=500,
                description="服务器内部错误",
                examples={
                    "internal_error_response": {
                        "success": False,
                        "message": "服务器内部错误",
                        "error": {
                            "code": "INTERNAL_ERROR",
                            "message": "服务器处理请求时发生错误，请稍后重试"
                        }
                    }
                }
            )
        }

    def _create_error_codes(self) -> List[Dict[str, Any]]:
        """创建错误码定义"""
        return [
            {
                "code": "VALIDATION_ERROR",
                "message": "请求参数验证失败",
                "description": "请求中的参数不符合API规范要求",
                "category": "客户端错误"
            },
            {
                "code": "AUTHENTICATION_FAILED",
                "message": "身份验证失败",
                "description": "提供的凭据无效或已过期",
                "category": "认证错误"
            },
            {
                "code": "AUTHORIZATION_FAILED",
                "message": "权限验证失败",
                "description": "用户没有执行此操作的权限",
                "category": "授权错误"
            },
            {
                "code": "RESOURCE_NOT_FOUND",
                "message": "资源未找到",
                "description": "请求的资源不存在",
                "category": "资源错误"
            },
            {
                "code": "RESOURCE_CONFLICT",
                "message": "资源冲突",
                "description": "请求与现有资源状态冲突",
                "category": "资源错误"
            },
            {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "请求频率超限",
                "description": "在指定时间内的请求次数超过限制",
                "category": "限制错误"
            },
            {
                "code": "SERVICE_UNAVAILABLE",
                "message": "服务不可用",
                "description": "服务暂时不可用，请稍后重试",
                "category": "服务错误"
            },
            {
                "code": "DATABASE_ERROR",
                "message": "数据库错误",
                "description": "数据库操作失败",
                "category": "数据错误"
            },
            {
                "code": "EXTERNAL_SERVICE_ERROR",
                "message": "外部服务错误",
                "description": "依赖的外部服务发生错误",
                "category": "外部错误"
            }
        ]

    def add_endpoint(self, endpoint: APIEndpointDocumentation):
        """添加端点文档"""
        key = f"{endpoint.method.upper()} {endpoint.path}"
        self.endpoints[key] = endpoint

    def enhance_endpoint_documentation(self, endpoint_key: str):
        """增强端点文档"""
        if endpoint_key not in self.endpoints:
            raise ValueError(f"端点 {endpoint_key} 不存在")

        endpoint = self.endpoints[endpoint_key]

        # 增强参数文档
        self._enhance_parameters(endpoint)

        # 标准化响应
        self._standardize_responses(endpoint)

        # 添加错误码信息
        self._add_error_codes(endpoint)

        # 生成使用示例
        self._generate_examples(endpoint)

    def _enhance_parameters(self, endpoint: APIEndpointDocumentation):
        """增强参数文档"""
        for param in endpoint.parameters:
            # 添加详细的类型信息
            param.constraints = self._generate_constraints(param)

            # 生成验证规则描述
            param.validation_rules = self._generate_validation_rules(param)

            # 生成示例值
            if param.example is None:
                param.example = self._generate_example_value(param)

    def _generate_constraints(self, param: APIParameterDocumentation) -> Dict[str, Any]:
        """生成参数约束"""
        constraints = {}

        if param.type == "string":
            constraints["minLength"] = 0
            constraints["maxLength"] = 1000
        elif param.type == "integer":
            constraints["minimum"] = 0
            constraints["maximum"] = 999999
        elif param.type == "number":
            constraints["minimum"] = 0.0
            constraints["maximum"] = 999999.99

        # 根据参数名添加特定约束
        if "email" in param.name.lower():
            constraints["pattern"] = r"^[^@]+@[^@]+\.[^@]+$"
            constraints["format"] = "email"
        elif "phone" in param.name.lower():
            constraints["pattern"] = r"^\+?[\d\s\-\(\)]+$"
        elif "url" in param.name.lower():
            constraints["format"] = "uri"
        elif "date" in param.name.lower():
            constraints["format"] = "date"
        elif "datetime" in param.name.lower():
            constraints["format"] = "date-time"

        return constraints

    def _generate_validation_rules(self, param: APIParameterDocumentation) -> List[str]:
        """生成验证规则"""
        rules = []

        if param.required:
            rules.append("必填字段")

        if param.type == "string":
            rules.append(f"字符串类型，最大长度 {param.constraints.get('maxLength', 1000)}")
            if "pattern" in param.constraints:
                rules.append("必须匹配指定格式")
        elif param.type == "integer":
            min_val = param.constraints.get("minimum", 0)
            max_val = param.constraints.get("maximum", 999999)
            rules.append(f"整数类型，范围: {min_val} - {max_val}")
        elif param.type == "number":
            min_val = param.constraints.get("minimum", 0.0)
            max_val = param.constraints.get("maximum", 999999.99)
            rules.append(f"数值类型，范围: {min_val} - {max_val}")

        if "email" in param.name.lower():
            rules.append("必须是有效的邮箱地址格式")
        elif "phone" in param.name.lower():
            rules.append("必须是有效的电话号码格式")
        elif "url" in param.name.lower():
            rules.append("必须是有效的URL格式")

        return rules

    def _generate_example_value(self, param: APIParameterDocumentation) -> Any:
        """生成示例值"""
        if param.type == "string":
            if "email" in param.name.lower():
                return "user@example.com"
            elif "phone" in param.name.lower():
                return "+86-138-0013-8000"
            elif "url" in param.name.lower():
                return "https://api.example.com"
            elif "symbol" in param.name.lower():
                return "BTC/USDT"
            elif "name" in param.name.lower():
                return "示例名称"
            else:
                return "示例字符串"
        elif param.type == "integer":
            if "port" in param.name.lower():
                return 8080
            elif "limit" in param.name.lower():
                return 100
            else:
                return 42
        elif param.type == "number":
            if "price" in param.name.lower():
                return 45000.50
            elif "amount" in param.name.lower():
                return 1.5
            else:
                return 123.45
        elif param.type == "boolean":
            return True
        else:
            return None

    def _standardize_responses(self, endpoint: APIEndpointDocumentation):
        """标准化响应"""
        # 确保所有端点都有标准的成功响应
        has_success_response = any(r.status_code == 200 for r in endpoint.responses)
        if not has_success_response and endpoint.method.upper() in ["GET", "POST", "PUT", "DELETE"]:
            if endpoint.method.upper() == "GET":
                endpoint.responses.append(self.common_responses["success"])
            elif endpoint.method.upper() == "POST":
                endpoint.responses.append(self.common_responses["created"])
            elif endpoint.method.upper() in ["PUT", "PATCH"]:
                endpoint.responses.append(self.common_responses["success"])
            elif endpoint.method.upper() == "DELETE":
                endpoint.responses.append(self.common_responses["success"])

        # 确保所有端点都有标准的错误响应
        standard_errors = [400, 401, 403, 404, 429, 500]
        existing_codes = {r.status_code for r in endpoint.responses}

        for code in standard_errors:
            if code not in existing_codes:
                if code == 400:
                    endpoint.responses.append(self.common_responses["bad_request"])
                elif code == 401:
                    endpoint.responses.append(self.common_responses["unauthorized"])
                elif code == 403:
                    endpoint.responses.append(self.common_responses["forbidden"])
                elif code == 404:
                    endpoint.responses.append(self.common_responses["not_found"])
                elif code == 429:
                    endpoint.responses.append(self.common_responses["rate_limited"])
                elif code == 500:
                    endpoint.responses.append(self.common_responses["internal_error"])

    def _add_error_codes(self, endpoint: APIEndpointDocumentation):
        """添加错误码信息"""
        # 根据端点类型添加相关的错误码
        if endpoint.method.upper() == "GET":
            endpoint.error_codes.extend([
                self.error_codes[0],  # VALIDATION_ERROR
                self.error_codes[3],  # RESOURCE_NOT_FOUND
                self.error_codes[7],  # DATABASE_ERROR
                self.error_codes[8]   # EXTERNAL_SERVICE_ERROR
            ])
        elif endpoint.method.upper() in ["POST", "PUT", "PATCH"]:
            endpoint.error_codes.extend([
                self.error_codes[0],  # VALIDATION_ERROR
                self.error_codes[1],  # AUTHENTICATION_FAILED
                self.error_codes[2],  # AUTHORIZATION_FAILED
                self.error_codes[4],  # RESOURCE_CONFLICT
                self.error_codes[7]   # DATABASE_ERROR
            ])
        elif endpoint.method.upper() == "DELETE":
            endpoint.error_codes.extend([
                self.error_codes[1],  # AUTHENTICATION_FAILED
                self.error_codes[2],  # AUTHORIZATION_FAILED
                self.error_codes[3],  # RESOURCE_NOT_FOUND
                self.error_codes[7]   # DATABASE_ERROR
            ])

    def _generate_examples(self, endpoint: APIEndpointDocumentation):
        """生成使用示例"""
        examples = {}

        # 生成请求示例
        if endpoint.parameters:
            request_example = {}
            for param in endpoint.parameters:
                if param.required:
                    request_example[param.name] = param.example
            if request_example:
                examples["request"] = request_example

        # 生成成功响应示例
        success_response = next((r for r in endpoint.responses if r.status_code == 200), None)
        if success_response and success_response.examples:
            examples["success_response"] = success_response.examples.get(
                "success_response",
                {"success": True, "message": "操作成功"}
            )

        # 生成错误响应示例
        error_response = next((r for r in endpoint.responses if r.status_code >= 400), None)
        if error_response and error_response.examples:
            examples["error_response"] = error_response.examples.get(
                "bad_request_response",
                {"success": False, "message": "请求参数错误"}
            )

        endpoint.examples = examples

    def generate_enhanced_documentation(self, output_file: str):
        """生成增强版文档"""
        documentation = {
            "title": "RQA2025 API 增强文档",
            "version": "1.0.0",
            "description": "RQA2025 量化交易系统 API 完整增强文档",
            "generated_at": datetime.now().isoformat(),
            "endpoints": {},
            "error_codes": self.error_codes,
            "common_responses": {}
        }

        # 转换通用响应
        for name, response in self.common_responses.items():
            documentation["common_responses"][name] = {
                "status_code": response.status_code,
                "description": response.description,
                "schema": response.schema,
                "examples": response.examples,
                "headers": response.headers
            }

        # 转换端点文档
        for key, endpoint in self.endpoints.items():
            documentation["endpoints"][key] = {
                "path": endpoint.path,
                "method": endpoint.method,
                "summary": endpoint.summary,
                "description": endpoint.description,
                "parameters": [
                    {
                        "name": param.name,
                        "type": param.type,
                        "required": param.required,
                        "description": param.description,
                        "example": param.example,
                        "default": param.default,
                        "constraints": param.constraints,
                        "validation_rules": param.validation_rules
                    }
                    for param in endpoint.parameters
                ],
                "responses": [
                    {
                        "status_code": resp.status_code,
                        "description": resp.description,
                        "content_type": resp.content_type,
                        "schema": resp.schema,
                        "examples": resp.examples,
                        "headers": resp.headers
                    }
                    for resp in endpoint.responses
                ],
                "authentication": endpoint.authentication,
                "rate_limits": endpoint.rate_limits,
                "error_codes": endpoint.error_codes,
                "examples": endpoint.examples,
                "changelog": endpoint.changelog
            }

        # 保存文档
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documentation, f, indent=2, ensure_ascii=False)

        print(f"增强版API文档已生成: {output_file}")
        return documentation


class RQAApiDocEnhancer(APIDocumentationEnhancer):
    """RQA API文档增强器"""

    def __init__(self):
        super().__init__()
        self._load_rqa_endpoints()

    def _load_rqa_endpoints(self):
        """加载RQA系统的端点"""
        # 数据服务端点
        self._add_data_endpoints()

        # 特征工程端点
        self._add_feature_endpoints()

        # 交易服务端点
        self._add_trading_endpoints()

        # 监控服务端点
        self._add_monitoring_endpoints()

    def _add_data_endpoints(self):
        """添加数据服务端点"""
        # 市场数据获取
        market_data = APIEndpointDocumentation(
            path="/api/v1/data/market/{symbol}",
            method="GET",
            summary="获取市场数据",
            description="获取指定交易对的市场数据，包括K线、成交量等信息",
            parameters=[
                APIParameterDocumentation(
                    name="symbol",
                    type="string",
                    required=True,
                    description="交易对符号，例如：BTC/USDT, ETH/USDT"
                ),
                APIParameterDocumentation(
                    name="interval",
                    type="string",
                    required=False,
                    description="时间间隔",
                    default="1h",
                    constraints={"enum": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]}
                ),
                APIParameterDocumentation(
                    name="limit",
                    type="integer",
                    required=False,
                    description="返回数据条数",
                    default=100,
                    constraints={"minimum": 1, "maximum": 1000}
                ),
                APIParameterDocumentation(
                    name="start_time",
                    type="string",
                    required=False,
                    description="开始时间，ISO 8601格式",
                    constraints={"format": "date-time"}
                ),
                APIParameterDocumentation(
                    name="end_time",
                    type="string",
                    required=False,
                    description="结束时间，ISO 8601格式",
                    constraints={"format": "date-time"}
                )
            ],
            authentication=["bearer", "api_key"],
            rate_limits={
                "per_minute": 60,
                "per_hour": 1000,
                "description": "每分钟最多60次请求，每小时最多1000次请求"
            }
        )
        self.add_endpoint(market_data)

        # 数据验证
        data_validation = APIEndpointDocumentation(
            path="/api/v1/data/validate",
            method="POST",
            summary="数据质量验证",
            description="验证上传数据的质量、完整性和一致性",
            request_body={
                "description": "数据验证请求",
                "schema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "待验证的数据数组"
                        },
                        "data_type": {
                            "type": "string",
                            "enum": ["market_data", "order_book", "trade_data"],
                            "description": "数据类型"
                        },
                        "validation_rules": {
                            "type": "object",
                            "description": "自定义验证规则"
                        }
                    },
                    "required": ["data", "data_type"]
                }
            },
            authentication=["bearer"],
            rate_limits={"per_minute": 30}
        )
        self.add_endpoint(data_validation)

    def _add_feature_endpoints(self):
        """添加特征工程端点"""
        # 技术指标计算
        feature_compute = APIEndpointDocumentation(
            path="/api/v1/features/compute",
            method="POST",
            summary="技术指标计算",
            description="为市场数据计算各种技术指标",
            request_body={
                "schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "交易对符号"},
                        "data": {"type": "array", "items": {"type": "object"}, "description": "市场数据"},
                        "indicators": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "要计算的指标列表",
                            "example": ["SMA", "EMA", "RSI", "MACD"]
                        },
                        "parameters": {"type": "object", "description": "指标参数"}
                    },
                    "required": ["symbol", "data", "indicators"]
                }
            },
            authentication=["bearer"],
            rate_limits={"per_minute": 20}
        )
        self.add_endpoint(feature_compute)

        # 情感分析
        sentiment_analysis = APIEndpointDocumentation(
            path="/api/v1/features/sentiment",
            method="POST",
            summary="新闻情感分析",
            description="对新闻文本进行情感分析，判断市场情绪",
            request_body={
                "schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "新闻文本内容"},
                        "language": {"type": "string", "default": "zh", "description": "文本语言"},
                        "context": {"type": "object", "description": "上下文信息"}
                    },
                    "required": ["text"]
                }
            },
            authentication=["bearer"],
            rate_limits={"per_minute": 50}
        )
        self.add_endpoint(sentiment_analysis)

    def _add_trading_endpoints(self):
        """添加交易服务端点"""
        # 策略执行
        strategy_execute = APIEndpointDocumentation(
            path="/api/v1/trading/strategy/{strategy_id}/execute",
            method="POST",
            summary="执行交易策略",
            description="执行指定的量化交易策略",
            parameters=[
                APIParameterDocumentation(
                    name="strategy_id",
                    type="string",
                    required=True,
                    description="策略唯一标识符"
                )
            ],
            request_body={
                "schema": {
                    "type": "object",
                    "properties": {
                        "capital": {"type": "number", "description": "投入资本", "minimum": 0},
                        "parameters": {"type": "object", "description": "策略参数"},
                        "risk_level": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "风险等级"
                        },
                        "dry_run": {"type": "boolean", "description": "是否为试运行", "default": False}
                    }
                }
            },
            authentication=["bearer"],
            rate_limits={"per_minute": 10}
        )
        self.add_endpoint(strategy_execute)

        # 投资组合查询
        portfolio_get = APIEndpointDocumentation(
            path="/api/v1/trading/portfolio",
            method="GET",
            summary="获取投资组合",
            description="获取当前投资组合的状态和持仓信息",
            authentication=["bearer"],
            rate_limits={"per_minute": 30}
        )
        self.add_endpoint(portfolio_get)

    def _add_monitoring_endpoints(self):
        """添加监控服务端点"""
        # 系统健康检查
        health_check = APIEndpointDocumentation(
            path="/api/v1/monitoring/health",
            method="GET",
            summary="系统健康检查",
            description="检查系统各组件的健康状态",
            authentication=[],  # 公开接口
            rate_limits={"per_minute": 60}
        )
        self.add_endpoint(health_check)

        # 系统指标
        system_metrics = APIEndpointDocumentation(
            path="/api/v1/monitoring/metrics",
            method="GET",
            summary="系统性能指标",
            description="获取系统的性能指标数据",
            authentication=["bearer"],
            rate_limits={"per_minute": 30}
        )
        self.add_endpoint(system_metrics)

    def enhance_all_endpoints(self):
        """增强所有端点的文档"""
        for endpoint_key in list(self.endpoints.keys()):
            try:
                self.enhance_endpoint_documentation(endpoint_key)
                print(f"✓ 增强端点文档: {endpoint_key}")
            except Exception as e:
                print(f"✗ 增强端点文档失败 {endpoint_key}: {e}")


if __name__ == "__main__":
    # 创建RQA API文档增强器
    print("初始化RQA API文档增强器...")

    enhancer = RQAApiDocEnhancer()

    print(f"加载了 {len(enhancer.endpoints)} 个API端点")

    # 增强所有端点文档
    print("开始增强API文档...")
    enhancer.enhance_all_endpoints()

    # 生成增强版文档
    output_file = "docs/api/enhanced_rqa_api_documentation.json"
    documentation = enhancer.generate_enhanced_documentation(output_file)

    print(f"\\n📊 文档统计:")
    print(f"   📋 总端点数: {len(documentation['endpoints'])}")
    print(f"   📋 通用响应: {len(documentation['common_responses'])}")
    print(f"   📋 错误码: {len(documentation['error_codes'])}")
    print(f"   📄 输出文件: {output_file}")

    print("\\n🎉 API文档增强完成！")
