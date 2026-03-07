#!/usr/bin/env python3
"""
检查策略 model_strategy_1771503574 的状态
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gateway.web.execution_persistence import load_execution_state
from gateway.web.strategy_lifecycle import get_strategy_lifecycle

def check_strategy_status(strategy_id):
    print(f"=== 策略 {strategy_id} 状态检查 ===\n")
    
    # 检查执行状态
    print("1. 执行状态:")
    exec_state = load_execution_state(strategy_id)
    if exec_state:
        print(f"   - 状态: {exec_state.get('status', 'unknown')}")
        print(f"   - 名称: {exec_state.get('name', 'N/A')}")
        print(f"   - 类型: {exec_state.get('type', 'N/A')}")
        if 'stopped_at' in exec_state:
            print(f"   - 停止时间: {exec_state.get('stopped_at')}")
        if 'stop_reason' in exec_state:
            print(f"   - 停止原因: {exec_state.get('stop_reason')}")
    else:
        print("   - 执行状态不存在")
    
    # 检查生命周期状态
    print("\n2. 生命周期状态:")
    lifecycle = get_strategy_lifecycle(strategy_id)
    if lifecycle:
        print(f"   - 当前状态: {lifecycle.current_status.value}")
        print(f"   - 策略名称: {lifecycle.strategy_name}")
        print(f"   - 创建时间: {lifecycle.created_at}")
        print(f"   - 更新时间: {lifecycle.updated_at}")
        if lifecycle.archived_at:
            print(f"   - 归档时间: {lifecycle.archived_at}")
        print(f"   - 事件数量: {len(lifecycle.events)}")
        if lifecycle.events:
            print("   - 最近事件:")
            for event in lifecycle.events[-3:]:
                print(f"     * {event.event_type}: {event.from_status} -> {event.to_status}")
    else:
        print("   - 生命周期不存在")
    
    print("\n=== 检查完成 ===")

if __name__ == "__main__":
    check_strategy_status("model_strategy_1771503574")
