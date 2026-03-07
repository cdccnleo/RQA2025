#!/usr/bin/env python3
"""
快速测试修复效果
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def test_fpga():
    try:
        from src.features.acceleration.fpga.fpga_manager import FPGAManager
        manager = FPGAManager()
        result = manager.get_accelerator()
        return result is not None
    except:
        return False

def test_infra():
    try:
        from src.infrastructure.init_infrastructure import Infrastructure
        infra1 = Infrastructure()
        infra2 = Infrastructure()
        return infra1 is infra2
    except:
        return False

def test_event_bus():
    try:
        from src.core.event_bus import get_event_bus
        bus = get_event_bus()
        return bus is not None
    except:
        return False

if __name__ == "__main__":
    print("快速测试修复效果:")
    print(f"FPGA修复: {'✅' if test_fpga() else '❌'}")
    print(f"基础设施修复: {'✅' if test_infra() else '❌'}")
    print(f"事件总线修复: {'✅' if test_event_bus() else '❌'}")