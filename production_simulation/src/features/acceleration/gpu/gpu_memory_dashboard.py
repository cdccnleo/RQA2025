import time
import os
from typing import List

try:
    import torch
except ImportError:
    torch = None


def clear():

    os.system("cls" if os.name == "nt" else "clear")


def get_gpu_stats() -> List[dict]:

    stats = []
    if torch and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
            reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
            total = props.total_memory / 1024 / 1024
            percent = allocated / total * 100 if total > 0 else 0
            stats.append(
                {
                    "index": i,
                    "name": props.name,
                    "allocated": allocated,
                    "reserved": reserved,
                    "total": total,
                    "percent": percent,
                    "multi_process": props.multi_processor_count,
                    "capability": f"{props.major}.{props.minor}",
                    "clock": props.clock_rate,
                }
            )
    return stats


def print_dashboard():

    stats = get_gpu_stats()
    if not stats:
        print("未检测到可用GPU或未安装torch")
        return
    print("GPU 显存监控看板 (每2秒刷新)")
    print("=" * 60)
    for s in stats:
        print(f"GPU {s['index']} | {s['name']}")
        print(
            f"  显存: {s['allocated']:.1f}MB / {s['total']:.1f}MB  ({s['percent']:.1f}%)  预留: {s['reserved']:.1f}MB"
        )
        print(
            f"  多处理器: {s['multi_process']}  计算能力: {s['capability']}  主频: {s['clock']} MHz"
        )
        print("-" * 60)


def run_dashboard():

    try:
        while True:
            clear()
            print_dashboard()
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n退出GPU显存监控看板")


if __name__ == "__main__":
    run_dashboard()
