import streamlit as st
import time
import pandas as pd

try:
    import torch
except ImportError:
    torch = None


def get_gpu_stats():

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
                    "GPU": f"{i} | {props.name}",
                    "Allocated(MB)": allocated,
                    "Reserved(MB)": reserved,
                    "Total(MB)": total,
                    "Usage(%)": percent,
                    "MultiProcessor": props.multi_processor_count,
                    "Capability": f"{props.major}.{props.minor}",
                    "Clock(MHz)": props.clock_rate,
                }
            )
    return stats


def main():

    st.set_page_config(page_title="GPU显存监控Web看板", layout="wide")
    st.title("GPU显存监控Web看板")
    st.caption("每2秒自动刷新 | 支持多卡 | 依赖PyTorch")
    placeholder = st.empty()
    while True:
        stats = get_gpu_stats()
        if not stats:
            placeholder.warning("未检测到可用GPU或未安装torch")
        else:
            df = pd.DataFrame(stats)
            placeholder.dataframe(df, use_container_width=True)
            st.bar_chart(df.set_index("GPU")[["Allocated(MB)", "Total(MB)"]])
            st.line_chart(df.set_index("GPU")["Usage(%)"])
        time.sleep(2)


if __name__ == "__main__":
    main()
