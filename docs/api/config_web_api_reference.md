# 配置管理Web服务 API接口说明

## 认证与会话

### POST /api/login
- 参数：username, password
- 返回：session_id, user信息

## 仪表盘与配置

### GET /api/dashboard
- 需认证
- 返回：同步状态、配置统计、系统状态等

### GET /api/config
- 需认证
- 返回：当前配置、配置树结构

### PUT /api/config/{path}
- 需认证，需write权限
- 参数：path, value
- 返回：更新结果

### POST /api/config/validate
- 需认证
- 参数：original_config, new_config
- 返回：变更验证结果

### POST /api/config/encrypt
- 需认证
- 参数：config
- 返回：加密后的配置

### POST /api/config/decrypt
- 需认证
- 参数：config
- 返回：解密后的配置

## 同步与冲突

### GET /api/sync/nodes
- 需认证
- 返回：同步节点列表

### POST /api/sync
- 需认证，需sync权限
- 参数：target_nodes（可选）
- 返回：同步结果

### GET /api/sync/history
- 需认证
- 参数：limit（可选）
- 返回：同步历史

### GET /api/sync/conflicts
- 需认证
- 返回：冲突列表

### POST /api/sync/conflicts/resolve
- 需认证，需sync权限
- 参数：strategy
- 返回：冲突解决结果

## 健康检查

### GET /api/health
- 返回：服务健康状态

## 错误码说明
- 401 未认证/会话无效
- 403 权限不足
- 400 请求参数错误
- 500 服务器内部错误