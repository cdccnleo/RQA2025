"""基础设施层模块使用示例"""
import logging
import yaml
from pathlib import Path
from src.infrastructure.config.config_manager import ConfigManager
from src.infrastructure.config.config_validator import ConfigValidator
from src.infrastructure.error.error_handler import ErrorHandler
from src.infrastructure.error.retry_handler import RetryHandler
from pydantic import BaseModel, Field

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. 配置管理示例
def config_management_example():
    """配置管理使用示例"""
    print("\n=== 配置管理示例 ===")

    # 创建临时配置目录和文件
    config_dir = Path("tmp_config")
    config_dir.mkdir(exist_ok=True)

    # 写入基础配置
    base_config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "timeout": 30
        }
    }
    with open(config_dir / "base.yaml", "w") as f:
        yaml.dump(base_config, f)

    # 使用ConfigManager
    with ConfigManager(config_dir=str(config_dir)) as manager:
        print(f"数据库主机: {manager.get('database.host')}")
        print(f"数据库端口: {manager.get('database.port')}")

        # 模拟配置热更新
        base_config["database"]["port"] = 5433
        with open(config_dir / "base.yaml", "w") as f:
            yaml.dump(base_config, f)

        # 等待配置更新
        import time
        time.sleep(0.5)

        print(f"更新后的端口: {manager.get('database.port')}")

# 2. 配置验证示例
def config_validation_example():
    """配置验证使用示例"""
    print("\n=== 配置验证示例 ===")

    # 定义配置模式
    class DatabaseConfig(BaseModel):
        host: str = Field(min_length=1)
        port: int = Field(gt=0, lt=65536)
        timeout: int = Field(ge=0, le=300)

    # 创建验证器并注册模式
    validator = ConfigValidator()
    validator.register_schema("database", DatabaseConfig)

    # 测试验证
    test_configs = [
        {"host": "db.example.com", "port": 5432, "timeout": 30},  # 有效
        {"host": "", "port": 5432, "timeout": 30},  # 无效host
        {"host": "db.example.com", "port": 70000, "timeout": 30}  # 无效port
    ]

    for config in test_configs:
        try:
            validated = validator.validate("database", config)
            print(f"配置验证通过: {validated}")
        except Exception as e:
            print(f"配置验证失败: {e}")

# 3. 错误处理示例
def error_handling_example():
    """错误处理使用示例"""
    print("\n=== 错误处理示例 ===")

    # 创建错误处理器
    handler = ErrorHandler()

    # 注册自定义处理器
    def handle_value_error(e):
        print(f"处理ValueError: {e}")
        return "default_value"

    handler.register_handler(ValueError, handle_value_error)

    # 测试处理
    try:
        raise ValueError("测试错误")
    except Exception as e:
        result = handler.handle(e)
        print(f"处理结果: {result}")

# 4. 重试机制示例
def retry_example():
    """重试机制使用示例"""
    print("\n=== 重试机制示例 ===")

    # 自定义异常
    class TemporaryError(Exception):
        pass

    # 创建重试处理器
    retry_handler = RetryHandler(
        max_attempts=3,
        initial_delay=0.5,
        retry_exceptions=[TemporaryError]
    )

    # 模拟可能失败的操作
    attempt = 0

    @retry_handler
    def unstable_operation():
        nonlocal attempt
        attempt += 1
        if attempt < 3:
            print(f"第{attempt}次尝试失败")
            raise TemporaryError("临时错误")
        return "操作成功"

    # 执行操作
    result = unstable_operation()
    print(f"最终结果: {result} (尝试次数: {attempt})")

if __name__ == "__main__":
    config_management_example()
    config_validation_example()
    error_handling_example()
    retry_example()
