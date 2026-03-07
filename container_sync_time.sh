#!/bin/bash

# RQA2025 容器时间同步脚本
# 用于Docker容器内的时间同步

echo "=== RQA2025 容器时间同步工具 ==="
echo ""

# 检查当前时间
echo "容器当前时间:"
date
echo ""

# 检查时区设置
echo "当前时区设置:"
cat /etc/timezone 2>/dev/null || echo "时区文件不存在"
echo ""

# 检查NTP服务状态
echo "检查NTP服务状态:"
if command -v systemctl &> /dev/null; then
    systemctl status ntp 2>/dev/null || systemctl status chronyd 2>/dev/null || echo "NTP服务未运行"
elif command -v service &> /dev/null; then
    service ntp status 2>/dev/null || service chrony status 2>/dev/null || echo "NTP服务未运行"
else
    echo "NTP服务管理命令不可用"
fi
echo ""

# 同步时间
echo "正在同步时间..."
if command -v ntpd &> /dev/null; then
    echo "使用ntpd进行时间同步..."
    ntpd -q
    if [ $? -eq 0 ]; then
        echo "NTP时间同步成功!"
    else
        echo "NTP时间同步失败"
    fi
elif command -v chronyc &> /dev/null; then
    echo "使用chronyd进行时间同步..."
    chronyc makestep
    if [ $? -eq 0 ]; then
        echo "Chrony时间同步成功!"
    else
        echo "Chrony时间同步失败"
    fi
else
    echo "未找到NTP客户端，尝试手动设置时间..."
    # 使用curl获取网络时间（需要网络连接）
    if command -v curl &> /dev/null; then
        echo "从网络获取时间..."
        # 这里可以添加网络时间获取逻辑
        echo "网络时间同步功能待实现"
    fi
fi

echo ""
echo "同步后的时间:"
date

echo ""
echo "=== 容器时间同步完成 ==="