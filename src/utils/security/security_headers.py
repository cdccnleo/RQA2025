#!/usr/bin/env python3
"""
安全HTTP头部中间件
添加安全相关的HTTP响应头部
"""

from fastapi import Request, Response
from fastapi.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    安全HTTP头部中间件
    添加OWASP推荐的安全头部
    """
    
    def __init__(
        self,
        app,
        csp_policy: Optional[str] = None,
        allow_iframe: bool = False
    ):
        super().__init__(app)
        self.csp_policy = csp_policy or self._default_csp_policy()
        self.allow_iframe = allow_iframe
    
    def _default_csp_policy(self) -> str:
        """默认内容安全策略"""
        return (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "media-src 'self'; "
            "object-src 'none'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )
    
    async def dispatch(self, request: Request, call_next):
        """处理请求并添加安全头部"""
        response = await call_next(request)
        
        # 添加安全头部
        headers = response.headers
        
        # 1. X-Content-Type-Options
        # 防止MIME类型嗅探
        headers['X-Content-Type-Options'] = 'nosniff'
        
        # 2. X-Frame-Options
        # 防止点击劫持攻击
        if self.allow_iframe:
            headers['X-Frame-Options'] = 'SAMEORIGIN'
        else:
            headers['X-Frame-Options'] = 'DENY'
        
        # 3. X-XSS-Protection
        # 启用浏览器XSS过滤
        headers['X-XSS-Protection'] = '1; mode=block'
        
        # 4. Strict-Transport-Security (HSTS)
        # 强制HTTPS连接
        headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
        
        # 5. Content-Security-Policy
        # 内容安全策略
        headers['Content-Security-Policy'] = self.csp_policy
        
        # 6. Referrer-Policy
        # 控制Referrer信息
        headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # 7. Permissions-Policy
        # 权限策略
        headers['Permissions-Policy'] = (
            'accelerometer=(), '
            'camera=(), '
            'geolocation=(), '
            'gyroscope=(), '
            'magnetometer=(), '
            'microphone=(), '
            'payment=(), '
            'usb=()'
        )
        
        # 8. X-Permitted-Cross-Domain-Policies
        # 禁止Adobe Flash跨域策略
        headers['X-Permitted-Cross-Domain-Policies'] = 'none'
        
        # 9. Cache-Control (敏感页面)
        # 防止缓存敏感信息
        if self._is_sensitive_path(request.url.path):
            headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            headers['Pragma'] = 'no-cache'
            headers['Expires'] = '0'
        
        # 10. 移除服务器信息
        # 隐藏服务器类型和版本
        headers.pop('Server', None)
        headers.pop('X-Powered-By', None)
        
        return response
    
    def _is_sensitive_path(self, path: str) -> bool:
        """检查是否为敏感路径"""
        sensitive_paths = [
            '/login',
            '/logout',
            '/auth',
            '/admin',
            '/api/token',
            '/api/keys',
            '/user/profile',
            '/settings'
        ]
        return any(path.startswith(sensitive) for sensitive in sensitive_paths)


class CORSSecurityMiddleware(BaseHTTPMiddleware):
    """
    安全的CORS中间件
    限制跨域访问，只允许特定来源
    """
    
    def __init__(
        self,
        app,
        allowed_origins: Optional[list] = None,
        allow_credentials: bool = False
    ):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ['https://rqa2025.com']
        self.allow_credentials = allow_credentials
    
    async def dispatch(self, request: Request, call_next):
        """处理CORS请求"""
        origin = request.headers.get('origin')
        
        response = await call_next(request)
        
        # 检查来源是否允许
        if origin in self.allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-API-Key'
            response.headers['Access-Control-Max-Age'] = '86400'
            
            if self.allow_credentials:
                response.headers['Access-Control-Allow-Credentials'] = 'true'
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    简单的速率限制中间件
    防止暴力破解和DDoS攻击
    """
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_size: int = 10
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        # 简化的速率限制存储（生产环境应使用Redis）
        self.request_counts = {}
    
    async def dispatch(self, request: Request, call_next):
        """检查速率限制"""
        client_ip = request.client.host
        
        # 检查是否超过限制
        current_count = self.request_counts.get(client_ip, 0)
        if current_count > self.requests_per_minute:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # 更新计数
        self.request_counts[client_ip] = current_count + 1
        
        response = await call_next(request)
        
        # 添加速率限制头部
        response.headers['X-RateLimit-Limit'] = str(self.requests_per_minute)
        response.headers['X-RateLimit-Remaining'] = str(
            max(0, self.requests_per_minute - self.request_counts[client_ip])
        )
        
        return response


def setup_security_middleware(app):
    """
    设置所有安全中间件
    
    用法:
        from fastapi import FastAPI
        from src.utils.security.security_headers import setup_security_middleware
        
        app = FastAPI()
        setup_security_middleware(app)
    """
    # 添加安全头部中间件
    app.add_middleware(SecurityHeadersMiddleware)
    
    # 添加CORS中间件
    app.add_middleware(
        CORSSecurityMiddleware,
        allowed_origins=['https://rqa2025.com', 'https://app.rqa2025.com'],
        allow_credentials=True
    )
    
    # 添加速率限制中间件
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=100,
        burst_size=20
    )
    
    print("✅ 安全中间件已启用")
    print("   - SecurityHeadersMiddleware")
    print("   - CORSSecurityMiddleware")
    print("   - RateLimitMiddleware")


# 示例用法
if __name__ == "__main__":
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    
    app = FastAPI()
    
    # 设置安全中间件
    setup_security_middleware(app)
    
    @app.get("/")
    def root():
        return {"message": "Hello, Secure World!"}
    
    @app.get("/test-headers")
    def test_headers(request: Request):
        """测试安全头部"""
        return JSONResponse({
            "message": "Check response headers for security headers"
        })
    
    print("\n=== 安全头部中间件测试 ===")
    print("启动服务器: uvicorn security_headers:app --reload")
    print("访问: http://localhost:8000/test-headers")
    print("检查响应头部中的安全头部")
