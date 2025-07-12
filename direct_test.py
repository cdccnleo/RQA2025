import os
import sys
import logging
from pathlib import Path

# 设置绝对路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 手动创建日志管理器类
class SimpleLogManager:
    def __init__(self, app_name="test_app", log_dir="test_logs"):
        self.logger = logging.getLogger(app_name)
        self.logger.setLevel(logging.DEBUG)

        # 确保日志目录存在
        log_path = os.path.abspath(log_dir)
        print(f"[DEBUG] 尝试创建日志目录: {log_path}")
        try:
            os.makedirs(log_path, exist_ok=True)
            print(f"[DEBUG] 目录已创建/存在: {log_path}")
        except Exception as e:
            print(f"[ERROR] 创建目录失败: {e}")
            raise

        # 文件处理器
        log_file = os.path.join(log_path, f"{app_name}.log")
        print(f"[DEBUG] 尝试创建日志文件: {log_file}")
        try:
            file_handler = logging.FileHandler(
                filename=log_file,
                encoding='utf-8',
                mode='a'
            )
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)
            print(f"[DEBUG] 文件处理器已添加: {log_file}")
        except Exception as e:
            print(f"[ERROR] 创建文件处理器失败: {e}")
            raise

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

# 测试日志
log_manager = SimpleLogManager()
log_manager.debug("DEBUG测试消息")
log_manager.info("INFO测试消息")
log_manager.warning("WARNING测试消息")

print(f"请检查 {os.path.abspath('test_logs/test_app.log')}")
