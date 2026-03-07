"""
HTTP相关常量定义
"""


class HTTPConstants:
    """HTTP状态码和相关常量"""
    
    # 成功状态码
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    
    # 重定向状态码
    MOVED_PERMANENTLY = 301
    FOUND = 302
    NOT_MODIFIED = 304
    
    # 客户端错误状态码
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429
    
    # 服务器错误状态码
    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504
    
    # HTTP方法
    METHOD_GET = 'GET'
    METHOD_POST = 'POST'
    METHOD_PUT = 'PUT'
    METHOD_DELETE = 'DELETE'
    METHOD_PATCH = 'PATCH'
    METHOD_HEAD = 'HEAD'
    METHOD_OPTIONS = 'OPTIONS'
    
    # 内容类型
    CONTENT_TYPE_JSON = 'application/json'
    CONTENT_TYPE_XML = 'application/xml'
    CONTENT_TYPE_FORM = 'application/x-www-form-urlencoded'
    CONTENT_TYPE_MULTIPART = 'multipart/form-data'
    CONTENT_TYPE_TEXT = 'text/plain'
    CONTENT_TYPE_HTML = 'text/html'
    
    # 默认端口
    DEFAULT_HTTP_PORT = 80
    DEFAULT_HTTPS_PORT = 443
    DEFAULT_API_PORT = 5000
    DEFAULT_ADMIN_PORT = 8080

