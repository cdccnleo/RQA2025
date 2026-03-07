#!/bin/bash
# 动态宇宙管理系统启动脚本
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LOG_LEVEL=INFO
python src/main.py
