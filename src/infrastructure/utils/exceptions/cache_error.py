class CacheError(Exception):
    """
    缓存操作异常基类

    Attributes:
        message: 错误信息
        original_exception: 原始异常(如果有)
    """

    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.message = message
        self.original_exception = original_exception

    def __str__(self):
        if self.original_exception:
            return f"{self.message} (原始异常: {str(self.original_exception)})"
        return self.message
