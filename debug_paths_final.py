import sys
import os
from pathlib import Path

print("Current working directory:", os.getcwd())
print("Script location:", __file__)
print("Script dir:", os.path.dirname(__file__))

project_root = Path(__file__).resolve().parent
src_path = project_root / "src"

print(f"Calculated project root: {project_root}")
print(f"Calculated src path: {src_path}")
print(f"src exists: {src_path.exists()}")

print("\nBefore adding to sys.path:")
for i, p in enumerate(sys.path[:5]):
    print(f"  {i}: {p}")

sys.path.insert(0, str(src_path))

print("\nAfter adding to sys.path:")
for i, p in enumerate(sys.path[:5]):
    print(f"  {i}: {p}")

print("\nTesting imports:")
try:
    import src
    print("✅ import src - SUCCESS")
except ImportError as e:
    print(f"❌ import src - FAILED: {e}")

try:
    import src.risk
    print("✅ import src.risk - SUCCESS")
except ImportError as e:
    print(f"❌ import src.risk - FAILED: {e}")

try:
    from src.risk.models import risk_manager
    print("✅ from src.risk.models import risk_manager - SUCCESS")
except ImportError as e:
    print(f"❌ from src.risk.models import risk_manager - FAILED: {e}")
