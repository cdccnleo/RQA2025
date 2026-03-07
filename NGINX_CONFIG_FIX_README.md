# Nginx配置修复说明

## 问题描述

在生产环境中，历史数据采集监控API返回404错误：

```
INFO: 172.18.0.7:58116 - "GET /api/v1/monitoring/historical-collection/status HTTP/1.0" 404 Not Found
INFO: 172.18.0.7:58132 - "GET /api/v1/monitoring/historical-collection/scheduler/status HTTP/1.0" 404 Not Found
INFO: 172.18.0.7:58142 - "POST /api/v1/monitoring/historical-collection/scheduler/start HTTP/1.0" 404 Not Found
```

## 问题原因

生产环境的nginx配置（`nginx/nginx.conf`）缺少以下关键配置：

1. **WebSocket代理配置** - 历史数据采集监控页面需要WebSocket进行实时通信
2. **CORS配置** - API和WebSocket请求需要跨域支持
3. **OPTIONS预检请求处理** - 前端发送的预检请求未被正确处理

## 修复内容

### 1. 添加WebSocket代理配置

```nginx
# WebSocket endpoints for real-time monitoring
location /ws/ {
    proxy_pass http://rqa2025_app;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # WebSocket specific settings
    proxy_buffering off;
    proxy_cache off;

    # Timeout settings for WebSocket
    proxy_connect_timeout 7d;
    proxy_send_timeout 7d;
    proxy_read_timeout 7d;
}
```

### 2. 添加CORS配置

```nginx
# CORS headers for API and WebSocket
add_header Access-Control-Allow-Origin * always;
add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS, HEAD, PATCH" always;
add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization,X-Requested-With,Accept,Accept-Encoding,Accept-Language,X-CSRFToken" always;
add_header Access-Control-Allow-Credentials false always;
```

### 3. 添加OPTIONS预检请求处理

```nginx
# Handle OPTIONS preflight requests
if ($request_method = 'OPTIONS') {
    add_header Access-Control-Allow-Origin *;
    add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS, HEAD, PATCH";
    add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization,X-Requested-With,Accept,Accept-Encoding,Accept-Language,X-CSRFToken";
    add_header Access-Control-Max-Age 86400;
    add_header Content-Type text/plain;
    add_header Content-Length 0;
    return 204;
}
```

## 应用修复

### 方法1：重启nginx容器（推荐）

**Linux/macOS:**
```bash
./scripts/restart_nginx.sh
```

**Windows PowerShell:**
```powershell
.\scripts\restart_nginx.ps1
```

### 方法2：手动重启

```bash
# 重启nginx容器
docker restart rqa2025-nginx

# 等待启动完成
sleep 5

# 检查健康状态
curl http://localhost/health
```

### 方法3：重新构建整个系统

```bash
# 停止所有服务
docker-compose -f docker-compose.prod.yml down

# 重新启动所有服务
docker-compose -f docker-compose.prod.yml up -d

# 检查服务状态
docker-compose -f docker-compose.prod.yml ps
```

## 验证修复

重启nginx后，验证以下端点是否正常工作：

### API端点
```bash
# 历史数据采集监控状态
curl http://localhost/api/v1/monitoring/historical-collection/status

# 调度器状态
curl http://localhost/api/v1/monitoring/historical-collection/scheduler/status

# 启动调度器
curl -X POST http://localhost/api/v1/monitoring/historical-collection/scheduler/start
```

### WebSocket连接
```javascript
// 在浏览器控制台测试
const ws = new WebSocket('ws://localhost/ws/historical-collection');
ws.onopen = () => console.log('WebSocket连接成功');
ws.onmessage = (event) => console.log('收到消息:', event.data);
```

### 监控页面
访问：`http://localhost/data-collection-monitor.html`

应该能看到：
- 历史数据采集监控区域正常显示
- WebSocket连接成功（无错误提示）
- 实时数据更新正常

## 配置对比

### 修复前的问题配置
- ❌ 缺少WebSocket代理 (`/ws/` 路径)
- ❌ 缺少CORS头
- ❌ 缺少OPTIONS预检处理

### 修复后的正确配置
- ✅ 完整的API代理 (`/api/`)
- ✅ WebSocket代理 (`/ws/`)
- ✅ CORS支持
- ✅ OPTIONS预检处理
- ✅ 适当的超时设置

## 注意事项

1. **配置生效时间**: nginx配置更新后需要重启容器才会生效
2. **WebSocket调试**: 如果WebSocket仍有问题，检查浏览器开发者工具的网络标签
3. **防火墙**: 确保宿主机防火墙允许80端口访问
4. **日志检查**: 如仍有问题，检查nginx日志：`docker logs rqa2025-nginx`

## 相关文件

- `nginx/nginx.conf` - 生产环境nginx配置
- `web-static/nginx.conf` - 开发环境nginx配置
- `scripts/restart_nginx.sh` - Linux/macOS重启脚本
- `scripts/restart_nginx.ps1` - Windows重启脚本