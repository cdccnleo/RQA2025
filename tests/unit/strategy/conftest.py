import sys
from pathlib import Path

# 添加项目根目录和src路径到Python路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_PATH = PROJECT_ROOT / 'src'

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
