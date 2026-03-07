# RQA2025 时间同步指南

## 概述

本项目包含时间同步工具，用于确保系统和Docker容器的时间准确性。时间同步对于量化交易系统至关重要，可以避免数据时间戳不一致的问题。

## 当前状态

- **系统时间**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
- **时区**: 东八区 (Asia/Shanghai)
- **Docker配置**: 已配置NTP服务器 (阿里云、NIST、Windows时间服务器)

## 同步工具

### 1. Windows系统时间同步 (`sync_time.ps1`)

**功能**:
- 检查当前系统时间
- 查询时间同步服务状态
- 强制同步系统时间
- 检查Docker容器状态和时间

**使用方法**:
```powershell
# 在项目根目录运行
.\sync_time.ps1
```

**手动同步命令**:
```powershell
# 检查时间同步状态
w32tm /query /status

# 强制同步时间
w32tm /resync /force

# 配置时间服务器
w32tm /config /manualpeerlist:"time.windows.com,0x1" /syncfromflags:manual /reliable:YES /update
```

### 2. 容器时间同步 (`container_sync_time.sh`)

**功能**:
- 检查容器内时间
- 同步NTP时间
- 支持ntpd和chronyd

**使用方法**:
```bash
# 进入容器后运行
chmod +x container_sync_time.sh
./container_sync_time.sh
```

## Docker时间同步配置

### Dockerfile中的NTP配置

```dockerfile
# 安装NTP
RUN apt-get update && apt-get install -y ntp tzdata

# 配置NTP服务器
RUN echo "server ntp.aliyun.com iburst" >> /etc/ntp.conf && \
    echo "server time.nist.gov iburst" >> /etc/ntp.conf && \
    echo "server time.windows.com iburst" >> /etc/ntp.conf

# 创建时间同步脚本
RUN echo '#!/bin/bash
echo "同步系统时间..."
ntpd -q
if [ $? -eq 0 ]; then
    echo "时间同步成功"
    date
else
    echo "时间同步失败"
fi' > /usr/local/bin/sync_time.sh && \
chmod +x /usr/local/bin/sync_time.sh
```

### Docker Compose配置

```yaml
environment:
  - TZ=Asia/Shanghai  # 设置时区
```

## 启动和使用

### 1. 启动容器前的时间同步

```bash
# Windows系统同步
.\sync_time.ps1

# 启动Docker容器
docker-compose up -d
```

### 2. 容器启动后的时间同步

```bash
# 进入主应用容器
docker exec -it rqa2025-app bash

# 运行时间同步
./container_sync_time.sh
```

### 3. 检查容器时间

```bash
# 检查所有容器时间
docker ps --format "{{.Names}}" | xargs -I {} docker exec {} date
```

## 故障排除

### 常见问题

1. **时间不同步**
   - 检查网络连接
   - 确认NTP服务器可达
   - 重启时间同步服务

2. **Docker容器时间不同步**
   - 使用 `--privileged` 模式运行容器
   - 或在容器内安装并配置NTP

3. **时区设置错误**
   - 确保环境变量 `TZ=Asia/Shanghai`
   - 检查 `/etc/timezone` 文件

### 手动同步步骤

```bash
# 系统级同步
sudo ntpdate -u ntp.aliyun.com

# 或使用chrony
sudo chronyc makestep

# 重启NTP服务
sudo systemctl restart ntp
```

## 监控和维护

- 定期检查时间同步状态
- 设置定时任务自动同步
- 监控时间偏差日志

## 技术说明

- NTP服务器: ntp.aliyun.com, time.nist.gov, time.windows.com
- 时区: Asia/Shanghai (UTC+8)
- 同步间隔: 建议每小时同步一次
- 时间精度: 毫秒级