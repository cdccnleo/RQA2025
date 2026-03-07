#!/bin/bash
# RQA2025 性能监控脚本

echo "📊 执行RQA2025性能监控..."

# 收集系统指标
echo "🔍 收集系统性能指标..."

# CPU使用率
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
echo "CPU使用率: ${CPU_USAGE}%"

# 内存使用率
MEM_TOTAL=$(free -m | grep '^Mem:' | awk '{print $2}')
MEM_USED=$(free -m | grep '^Mem:' | awk '{print $3}')
MEM_USAGE=$(echo "scale=2; $MEM_USED / $MEM_TOTAL * 100" | bc)
echo "内存使用率: ${MEM_USAGE}%"

# 磁盘I/O
DISK_IO=$(iostat -d 1 1 | grep -A 1 "Device:" | tail -1 | awk '{print $2}')
echo "磁盘I/O: ${DISK_IO} tps"

# 网络I/O
NET_RX=$(cat /proc/net/dev | grep eth0 | awk '{print $2}')
NET_TX=$(cat /proc/net/dev | grep eth0 | awk '{print $10}')
echo "网络接收: ${NET_RX} bytes"
echo "网络发送: ${NET_TX} bytes"

# 收集应用指标
echo "🔍 收集应用性能指标..."

# 请求响应时间 (如果有metrics端点)
if curl -s http://localhost:8000/metrics > /dev/null 2>&1; then
    RESPONSE_TIME=$(curl -s -w "%{time_total}" -o /dev/null http://localhost:8000/health | awk '{print $1 * 1000}')
    echo "应用响应时间: ${RESPONSE_TIME}ms"
fi

# 数据库连接数
# 这里添加数据库连接数检查

# 活跃线程数
THREAD_COUNT=$(ps -o nlwp= -C python | awk '{sum += $1} END {print sum}')
echo "活跃线程数: ${THREAD_COUNT}"

# 生成性能报告
REPORT_FILE="/var/log/rqa2025/performance_$(date +%Y%m%d_%H%M%S).log"
cat > "$REPORT_FILE" << EOF
RQA2025性能监控报告 - $(date)
=====================================
系统指标:
  CPU使用率: ${CPU_USAGE}%
  内存使用率: ${MEM_USAGE}%
  磁盘I/O: ${DISK_IO} tps

应用指标:
  响应时间: ${RESPONSE_TIME}ms
  活跃线程: ${THREAD_COUNT}

网络指标:
  接收: ${NET_RX} bytes
  发送: ${NET_TX} bytes
EOF

echo "📄 性能报告已保存: $REPORT_FILE"
echo "✅ 性能监控完成"
