# 导入一致性检查报告
==================================================

src\infrastructure\utils\__init__.py:
  deprecated_imports:
    - Line 19: from src.infrastructure.utils.logger import get_logger

src\utils\date_utils.py:
  deprecated_imports:
    - Line 10: from src.infrastructure.utils.date_utils import convert_timezone as _convert_timezone

src\utils\logger.py:
  deprecated_imports:
    - Line 11: from src.infrastructure.utils.logger import get_logger

==================================================
检查文件数: 3
有问题的文件数: 3
总问题数: 3

⚠️  发现 3 个文件需要修复