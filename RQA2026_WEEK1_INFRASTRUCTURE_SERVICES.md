# 🏗️ RQA2026概念验证阶段 - Week 1 基础设施基础服务建设

**执行周期**: 2024年12月11日 - 2024年12月13日 (CTO负责，1.5天)
**任务目标**: 构建日志监控、配置管理和安全框架
**核心价值**: 为三大引擎提供稳定的基础设施支撑

---

## 🎯 基础设施服务目标

### 功能目标
```
1. 日志系统: 结构化日志收集、存储和分析
2. 配置管理: 集中式配置管理和秘钥管理
3. 监控告警: 实时监控和智能告警机制
4. 安全认证: 统一身份认证和权限控制
5. 健康检查: 系统健康状态监控和服务发现
```

### 性能目标
```
- 日志处理延迟: < 10ms
- 配置更新延迟: < 5s
- 监控数据延迟: < 1s
- 认证响应时间: < 50ms
- 系统可用性: > 99.9%
```

---

## 🏗️ 基础设施架构设计

### 整体架构图
```
┌─────────────────────────────────────────────────────────────┐
│                  Infrastructure Services                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Log Service│  │Config Service│ │Monitor&Alert│         │
│  │             │  │             │  │             │         │
│  │ • Collection │  │ • Centralized│  │ • Metrics    │         │
│  │ • Storage   │  │ • Hot Reload│  │ • Alerting   │         │
│  │ • Analysis  │  │ • Encryption│  │ • Dashboard  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │Auth Service │  │Health Check │  │Service Mesh │         │
│  │             │  │             │  │             │         │
│  │ • JWT Auth  │  │ • Endpoint  │  │ • Discovery │         │
│  │ • RBAC      │  │ • Dependency │  │ • Load Bal  │         │
│  │ • SSO       │  │ • Auto Rec  │  │ • Circuit Br│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件设计

#### 1. Logging Service (日志服务)
```python
class LoggingService:
    """日志服务 - 统一日志管理"""
    
    def __init__(self):
        self.loggers = {}  # service_name -> logger
        self.handlers = {}  # handler_name -> handler
        self.formatters = {}  # formatter_name -> formatter
        self.log_queue = asyncio.Queue()
        self.processor_task = None
    
    def get_logger(self, service_name: str, config: LogConfig) -> Logger:
        """获取或创建服务日志器"""
        if service_name in self.loggers:
            return self.loggers[service_name]
        
        # 创建日志器
        logger = logging.getLogger(service_name)
        logger.setLevel(self._get_log_level(config.level))
        
        # 配置处理器
        for handler_config in config.handlers:
            handler = self._create_handler(handler_config)
            logger.addHandler(handler)
        
        # 配置格式化器
        if config.formatter:
            formatter = self._create_formatter(config.formatter)
            for handler in logger.handlers:
                handler.setFormatter(formatter)
        
        self.loggers[service_name] = logger
        return logger
    
    def _create_handler(self, config: HandlerConfig) -> logging.Handler:
        """创建日志处理器"""
        handler_type = config.type
        
        if handler_type == "console":
            handler = logging.StreamHandler()
        elif handler_type == "file":
            handler = logging.FileHandler(config.filename)
        elif handler_type == "rotating_file":
            handler = logging.handlers.RotatingFileHandler(
                config.filename, 
                maxBytes=config.max_bytes, 
                backupCount=config.backup_count
            )
        elif handler_type == "syslog":
            handler = logging.handlers.SysLogHandler(address=config.address)
        elif handler_type == "elasticsearch":
            handler = ElasticsearchHandler(
                hosts=config.hosts,
                index=config.index,
                auth=config.auth
            )
        else:
            raise ValueError(f"Unsupported handler type: {handler_type}")
        
        handler.setLevel(self._get_log_level(config.level))
        return handler
    
    def _create_formatter(self, config: FormatterConfig) -> logging.Formatter:
        """创建格式化器"""
        if config.format_type == "json":
            return JSONFormatter(
                include_fields=config.include_fields,
                exclude_fields=config.exclude_fields
            )
        elif config.format_type == "structured":
            return StructuredFormatter(
                format_string=config.format_string,
                date_format=config.date_format
            )
        else:
            return logging.Formatter(config.format_string, config.date_format)
    
    async def log_async(self, level: str, message: str, 
                       service: str, extra: dict = None):
        """异步日志记录"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "service": service,
            "message": message,
            "extra": extra or {},
            "hostname": socket.gethostname(),
            "pid": os.getpid()
        }
        
        await self.log_queue.put(log_entry)
    
    async def start_processing(self):
        """启动日志处理任务"""
        self.processor_task = asyncio.create_task(self._process_logs())
    
    async def stop_processing(self):
        """停止日志处理"""
        if self.processor_task:
            self.processor_task.cancel()
            await self.processor_task
    
    async def _process_logs(self):
        """日志处理循环"""
        batch_size = 100
        batch_timeout = 1.0  # 1秒
        
        while True:
            try:
                batch = []
                start_time = time.time()
                
                # 收集批次
                while len(batch) < batch_size:
                    try:
                        log_entry = await asyncio.wait_for(
                            self.log_queue.get(), 
                            timeout=batch_timeout
                        )
                        batch.append(log_entry)
                        batch_timeout = min(0.1, batch_timeout)  # 减少超时时间
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    await self._write_batch(batch)
                    
            except Exception as e:
                print(f"Log processing error: {e}", file=sys.stderr)
    
    async def _write_batch(self, batch: List[dict]):
        """批量写入日志"""
        # 这里可以实现不同的存储策略
        # 1. 写入文件
        # 2. 发送到Elasticsearch
        # 3. 发送到Kafka
        # 4. 写入数据库
        
        for entry in batch:
            # 控制台输出 (开发环境)
            print(f"[{entry['timestamp']}] {entry['service']} {entry['level']}: {entry['message']}")
            
            # 可以添加更多输出目标
            if self.elasticsearch_handler:
                await self._write_to_elasticsearch(entry)
    
    async def query_logs(self, service: str = None, level: str = None, 
                        start_time: datetime = None, end_time: datetime = None,
                        limit: int = 100) -> List[dict]:
        """查询日志"""
        # 这里实现日志查询逻辑
        # 可以从Elasticsearch、数据库等查询
        
        # 简化的内存查询 (生产环境需要持久化存储)
        matching_logs = []
        
        for log_entry in self.recent_logs:  # 假设有recent_logs缓存
            if service and log_entry.get('service') != service:
                continue
            if level and log_entry.get('level') != level:
                continue
            if start_time and datetime.fromisoformat(log_entry['timestamp']) < start_time:
                continue
            if end_time and datetime.fromisoformat(log_entry['timestamp']) > end_time:
                continue
            
            matching_logs.append(log_entry)
            if len(matching_logs) >= limit:
                break
        
        return matching_logs
    
    def _get_log_level(self, level_str: str) -> int:
        """转换日志级别字符串为数字"""
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        return levels.get(level_str.upper(), logging.INFO)
```

#### 2. Config Manager (配置管理器)
```python
class ConfigManager:
    """配置管理器 - 集中式配置管理"""
    
    def __init__(self, config_source: str = "file"):
        self.config_source = config_source
        self.configs = {}  # service_name -> config
        self.watchers = {}  # service_name -> list of callbacks
        self.encryption = ConfigEncryption()
        self.validator = ConfigValidator()
    
    async def load_config(self, service_name: str, config_path: str = None) -> dict:
        """加载服务配置"""
        if self.config_source == "file":
            config = await self._load_from_file(service_name, config_path)
        elif self.config_source == "consul":
            config = await self._load_from_consul(service_name)
        elif self.config_source == "etcd":
            config = await self._load_from_etcd(service_name)
        else:
            raise ValueError(f"Unsupported config source: {self.config_source}")
        
        # 验证配置
        await self.validator.validate_config(service_name, config)
        
        # 解密敏感信息
        config = await self.encryption.decrypt_config(config)
        
        self.configs[service_name] = config
        return config
    
    async def get_config(self, service_name: str, key: str = None) -> Any:
        """获取配置值"""
        if service_name not in self.configs:
            raise ConfigNotFoundError(f"Config not found for service: {service_name}")
        
        config = self.configs[service_name]
        
        if key is None:
            return config
        
        # 支持点分隔的键路径
        keys = key.split('.')
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                raise ConfigKeyNotFoundError(f"Config key not found: {key}")
        
        return value
    
    async def set_config(self, service_name: str, key: str, value: Any):
        """设置配置值"""
        if service_name not in self.configs:
            self.configs[service_name] = {}
        
        # 支持点分隔的键路径
        keys = key.split('.')
        config = self.configs[service_name]
        
        # 导航到父级字典
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        # 设置值
        config[keys[-1]] = value
        
        # 加密敏感信息
        if self._is_sensitive_key(key):
            config[keys[-1]] = await self.encryption.encrypt_value(value)
        
        # 保存配置
        await self._save_config(service_name)
        
        # 通知监听者
        await self._notify_watchers(service_name, key, value)
    
    async def watch_config(self, service_name: str, key: str, callback: Callable):
        """监听配置变化"""
        if service_name not in self.watchers:
            self.watchers[service_name] = {}
        
        if key not in self.watchers[service_name]:
            self.watchers[service_name][key] = []
        
        self.watchers[service_name][key].append(callback)
    
    async def _notify_watchers(self, service_name: str, key: str, value: Any):
        """通知配置监听者"""
        if service_name in self.watchers and key in self.watchers[service_name]:
            tasks = []
            for callback in self.watchers[service_name][key]:
                tasks.append(callback(key, value))
            
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _load_from_file(self, service_name: str, config_path: str) -> dict:
        """从文件加载配置"""
        if not config_path:
            config_path = f"config/{service_name}.yaml"
        
        if not os.path.exists(config_path):
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
        
        return config or {}
    
    async def _load_from_consul(self, service_name: str) -> dict:
        """从Consul加载配置"""
        # 实现Consul配置加载逻辑
        # 这里需要consul客户端
        pass
    
    async def _load_from_etcd(self, service_name: str) -> dict:
        """从etcd加载配置"""
        # 实现etcd配置加载逻辑
        # 这里需要etcd客户端
        pass
    
    async def _save_config(self, service_name: str):
        """保存配置"""
        if self.config_source == "file":
            await self._save_to_file(service_name)
        elif self.config_source == "consul":
            await self._save_to_consul(service_name)
        elif self.config_source == "etcd":
            await self._save_to_etcd(service_name)
    
    async def _save_to_file(self, service_name: str):
        """保存配置到文件"""
        config_path = f"config/{service_name}.yaml"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.configs[service_name], f, default_flow_style=False)
    
    def _is_sensitive_key(self, key: str) -> bool:
        """检查是否为敏感配置项"""
        sensitive_keys = [
            'password', 'secret', 'token', 'key', 'credential',
            'database.password', 'api.secret', 'jwt.secret'
        ]
        
        key_lower = key.lower()
        return any(sensitive in key_lower for sensitive in sensitive_keys)
```

#### 3. Auth Service (认证服务)
```python
class AuthService:
    """认证服务 - 统一身份认证和权限控制"""
    
    def __init__(self):
        self.jwt_manager = JWTManager()
        self.user_store = UserStore()
        self.role_manager = RoleManager()
        self.permission_cache = {}  # user_id -> permissions
    
    async def authenticate(self, username: str, password: str) -> AuthResult:
        """用户认证"""
        try:
            # 验证用户凭据
            user = await self.user_store.get_user_by_username(username)
            if not user:
                return AuthResult(success=False, error="User not found")
            
            # 验证密码
            if not self._verify_password(password, user.hashed_password):
                return AuthResult(success=False, error="Invalid password")
            
            # 检查用户状态
            if not user.is_active:
                return AuthResult(success=False, error="User is inactive")
            
            # 生成访问令牌
            access_token = await self.jwt_manager.generate_access_token(user)
            refresh_token = await self.jwt_manager.generate_refresh_token(user)
            
            # 缓存用户权限
            permissions = await self.role_manager.get_user_permissions(user.id)
            self.permission_cache[user.id] = permissions
            
            return AuthResult(
                success=True,
                user=user,
                access_token=access_token,
                refresh_token=refresh_token,
                permissions=permissions
            )
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return AuthResult(success=False, error="Authentication failed")
    
    async def authorize(self, user_id: str, resource: str, action: str) -> bool:
        """权限验证"""
        try:
            # 获取用户权限
            permissions = self.permission_cache.get(user_id)
            if permissions is None:
                permissions = await self.role_manager.get_user_permissions(user_id)
                self.permission_cache[user_id] = permissions
            
            # 检查权限
            required_permission = f"{resource}:{action}"
            return required_permission in permissions
            
        except Exception as e:
            logger.error(f"Authorization error: {e}")
            return False
    
    async def validate_token(self, token: str) -> TokenValidationResult:
        """验证访问令牌"""
        try:
            payload = await self.jwt_manager.verify_token(token)
            
            # 检查令牌是否过期
            if payload.get('exp') and datetime.fromtimestamp(payload['exp']) < datetime.now():
                return TokenValidationResult(valid=False, error="Token expired")
            
            # 检查用户是否仍然有效
            user_id = payload.get('user_id')
            user = await self.user_store.get_user_by_id(user_id)
            if not user or not user.is_active:
                return TokenValidationResult(valid=False, error="User invalid")
            
            return TokenValidationResult(
                valid=True,
                user_id=user_id,
                username=payload.get('username'),
                roles=payload.get('roles', [])
            )
            
        except jwt.ExpiredSignatureError:
            return TokenValidationResult(valid=False, error="Token expired")
        except jwt.InvalidTokenError:
            return TokenValidationResult(valid=False, error="Invalid token")
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return TokenValidationResult(valid=False, error="Validation failed")
    
    async def refresh_token(self, refresh_token: str) -> TokenRefreshResult:
        """刷新访问令牌"""
        try:
            # 验证刷新令牌
            payload = await self.jwt_manager.verify_refresh_token(refresh_token)
            
            # 获取用户
            user_id = payload.get('user_id')
            user = await self.user_store.get_user_by_id(user_id)
            if not user or not user.is_active:
                return TokenRefreshResult(success=False, error="User invalid")
            
            # 生成新的访问令牌
            access_token = await self.jwt_manager.generate_access_token(user)
            
            return TokenRefreshResult(
                success=True,
                access_token=access_token
            )
            
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return TokenRefreshResult(success=False, error="Refresh failed")
    
    async def create_user(self, user_data: UserCreateRequest) -> UserCreateResult:
        """创建新用户"""
        try:
            # 检查用户名是否已存在
            existing_user = await self.user_store.get_user_by_username(user_data.username)
            if existing_user:
                return UserCreateResult(success=False, error="Username already exists")
            
            # 创建用户
            hashed_password = self._hash_password(user_data.password)
            user = User(
                username=user_data.username,
                email=user_data.email,
                hashed_password=hashed_password,
                full_name=user_data.full_name,
                is_active=True,
                created_at=datetime.now()
            )
            
            user_id = await self.user_store.create_user(user)
            
            # 分配默认角色
            await self.role_manager.assign_role(user_id, "user")
            
            return UserCreateResult(success=True, user_id=user_id)
            
        except Exception as e:
            logger.error(f"User creation error: {e}")
            return UserCreateResult(success=False, error="Creation failed")
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return bcrypt.checkpw(
            plain_password.encode('utf-8'), 
            hashed_password.encode('utf-8')
        )
    
    def _hash_password(self, password: str) -> str:
        """哈希密码"""
        return bcrypt.hashpw(
            password.encode('utf-8'), 
            bcrypt.gensalt()
        ).decode('utf-8')
```

#### 4. Health Checker (健康检查器)
```python
class HealthChecker:
    """健康检查器 - 系统健康监控"""
    
    def __init__(self):
        self.checks = {}  # check_name -> check_function
        self.last_results = {}  # check_name -> last_result
        self.dependencies = {}  # service -> list of dependencies
    
    def register_check(self, name: str, check_func: Callable, 
                      interval: int = 30, timeout: int = 10):
        """注册健康检查"""
        self.checks[name] = {
            'function': check_func,
            'interval': interval,
            'timeout': timeout,
            'last_run': 0,
            'running': False
        }
    
    def add_dependency(self, service: str, dependency: str):
        """添加服务依赖关系"""
        if service not in self.dependencies:
            self.dependencies[service] = []
        self.dependencies[service].append(dependency)
    
    async def check_health(self, service: str = None) -> HealthStatus:
        """执行健康检查"""
        if service:
            return await self._check_single_service(service)
        else:
            return await self._check_all_services()
    
    async def _check_single_service(self, service: str) -> HealthStatus:
        """检查单个服务健康状态"""
        if service not in self.checks:
            return HealthStatus(
                service=service,
                status="unknown",
                message="No health check registered"
            )
        
        check_info = self.checks[service]
        
        # 检查是否需要运行
        current_time = time.time()
        if current_time - check_info['last_run'] < check_info['interval']:
            # 返回缓存结果
            cached_result = self.last_results.get(service)
            if cached_result:
                return cached_result
        
        # 防止并发运行
        if check_info['running']:
            return HealthStatus(
                service=service,
                status="checking",
                message="Health check in progress"
            )
        
        check_info['running'] = True
        
        try:
            # 执行健康检查
            check_func = check_info['function']
            result = await asyncio.wait_for(
                check_func(),
                timeout=check_info['timeout']
            )
            
            # 更新结果
            status = HealthStatus(
                service=service,
                status=result.get('status', 'unknown'),
                message=result.get('message', ''),
                details=result.get('details', {}),
                timestamp=datetime.now()
            )
            
            self.last_results[service] = status
            check_info['last_run'] = current_time
            
            return status
            
        except asyncio.TimeoutError:
            return HealthStatus(
                service=service,
                status="timeout",
                message=f"Health check timed out after {check_info['timeout']}s"
            )
        
        except Exception as e:
            return HealthStatus(
                service=service,
                status="error",
                message=f"Health check failed: {str(e)}"
            )
        
        finally:
            check_info['running'] = False
    
    async def _check_all_services(self) -> dict:
        """检查所有服务健康状态"""
        results = {}
        
        # 并发生成所有健康检查任务
        tasks = []
        for service in self.checks.keys():
            tasks.append(self._check_single_service(service))
        
        # 等待所有检查完成
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 整理结果
        for service, result in zip(self.checks.keys(), check_results):
            if isinstance(result, Exception):
                results[service] = HealthStatus(
                    service=service,
                    status="error",
                    message=f"Check execution failed: {str(result)}"
                )
            else:
                results[service] = result
        
        # 计算整体状态
        overall_status = self._calculate_overall_status(results)
        
        return {
            'overall_status': overall_status,
            'services': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_overall_status(self, service_results: dict) -> str:
        """计算整体健康状态"""
        if not service_results:
            return "unknown"
        
        statuses = [result.status for result in service_results.values()]
        
        if any(status in ["critical", "error"] for status in statuses):
            return "unhealthy"
        elif any(status == "warning" for status in statuses):
            return "degraded"
        elif all(status == "healthy" for status in statuses):
            return "healthy"
        else:
            return "unknown"
    
    async def get_service_status(self, service: str) -> HealthStatus:
        """获取服务健康状态"""
        return self.last_results.get(service)
    
    async def get_dependency_status(self, service: str) -> dict:
        """获取服务依赖状态"""
        if service not in self.dependencies:
            return {}
        
        dependency_statuses = {}
        for dependency in self.dependencies[service]:
            status = await self.get_service_status(dependency)
            dependency_statuses[dependency] = status
        
        return dependency_statuses
    
    async def start_monitoring(self):
        """启动持续监控"""
        asyncio.create_task(self._continuous_monitoring())
    
    async def _continuous_monitoring(self):
        """持续监控循环"""
        while True:
            try:
                # 检查所有服务
                health_status = await self.check_health()
                
                # 记录不健康的服务
                unhealthy_services = []
                for service, status in health_status.get('services', {}).items():
                    if status.status not in ['healthy', 'unknown']:
                        unhealthy_services.append(f"{service}: {status.status}")
                
                if unhealthy_services:
                    logger.warning(f"Unhealthy services detected: {', '.join(unhealthy_services)}")
                
                # 等待下一次检查
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger.error(f"Continuous monitoring error: {e}")
                await asyncio.sleep(30)  # 出错后等待30秒重试
```

#### 5. Metrics Collector (指标收集器)
```python
class MetricsCollector:
    """指标收集器 - 系统监控指标收集"""
    
    def __init__(self):
        self.metrics = {}  # metric_name -> metric_data
        self.collectors = {}  # collector_name -> collector
        self.exporters = {}  # exporter_name -> exporter
    
    def register_metric(self, name: str, metric_type: str, 
                       description: str = "", labels: dict = None):
        """注册指标"""
        self.metrics[name] = {
            'type': metric_type,
            'description': description,
            'labels': labels or {},
            'values': [],
            'timestamps': []
        }
    
    def record_metric(self, name: str, value: float, 
                     labels: dict = None, timestamp: datetime = None):
        """记录指标值"""
        if name not in self.metrics:
            raise MetricNotFoundError(f"Metric not registered: {name}")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        metric = self.metrics[name]
        metric['values'].append(value)
        metric['timestamps'].append(timestamp)
        
        # 更新标签
        if labels:
            metric['labels'].update(labels)
        
        # 限制历史数据长度 (避免内存溢出)
        max_history = 1000
        if len(metric['values']) > max_history:
            metric['values'] = metric['values'][-max_history:]
            metric['timestamps'] = metric['timestamps'][-max_history:]
    
    def get_metric(self, name: str, start_time: datetime = None, 
                  end_time: datetime = None) -> dict:
        """获取指标数据"""
        if name not in self.metrics:
            raise MetricNotFoundError(f"Metric not registered: {name}")
        
        metric = self.metrics[name]
        
        # 过滤时间范围
        if start_time or end_time:
            filtered_values = []
            filtered_timestamps = []
            
            for value, timestamp in zip(metric['values'], metric['timestamps']):
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                
                filtered_values.append(value)
                filtered_timestamps.append(timestamp)
            
            return {
                'name': name,
                'type': metric['type'],
                'description': metric['description'],
                'labels': metric['labels'],
                'values': filtered_values,
                'timestamps': filtered_timestamps
            }
        
        return metric
    
    def register_collector(self, name: str, collector: Callable):
        """注册指标收集器"""
        self.collectors[name] = collector
    
    async def collect_system_metrics(self):
        """收集系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        self.record_metric('system.cpu.usage', cpu_percent, 
                          labels={'unit': 'percent'})
        
        # 内存使用率
        memory = psutil.virtual_memory()
        self.record_metric('system.memory.usage', memory.percent,
                          labels={'unit': 'percent'})
        self.record_metric('system.memory.used', memory.used,
                          labels={'unit': 'bytes'})
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        self.record_metric('system.disk.usage', disk.percent,
                          labels={'mount': '/', 'unit': 'percent'})
        
        # 网络流量
        network = psutil.net_io_counters()
        self.record_metric('system.network.bytes_sent', network.bytes_sent,
                          labels={'unit': 'bytes'})
        self.record_metric('system.network.bytes_recv', network.bytes_recv,
                          labels={'unit': 'bytes'})
    
    async def collect_application_metrics(self, service_name: str):
        """收集应用指标"""
        # 这里可以添加应用特定的指标收集逻辑
        # 例如: 请求数、响应时间、错误率等
        
        # 示例指标
        self.record_metric(f'{service_name}.requests.total', 1,
                          labels={'method': 'GET', 'endpoint': '/health'})
        self.record_metric(f'{service_name}.response.time', 0.05,
                          labels={'method': 'GET', 'endpoint': '/health', 'unit': 'seconds'})
    
    def register_exporter(self, name: str, exporter: Callable):
        """注册指标导出器"""
        self.exporters[name] = exporter
    
    async def export_metrics(self, format: str = 'prometheus') -> str:
        """导出指标数据"""
        if format == 'prometheus':
            return self._export_prometheus_format()
        elif format == 'json':
            return self._export_json_format()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_prometheus_format(self) -> str:
        """导出Prometheus格式"""
        lines = []
        
        for name, metric in self.metrics.items():
            # 指标类型
            lines.append(f"# HELP {name} {metric['description']}")
            lines.append(f"# TYPE {name} {metric['type']}")
            
            # 最新值
            if metric['values']:
                latest_value = metric['values'][-1]
                labels_str = ""
                if metric['labels']:
                    labels_list = [f'{k}="{v}"' for k, v in metric['labels'].items()]
                    labels_str = f"{{{','.join(labels_list)}}}"
                
                lines.append(f"{name}{labels_str} {latest_value}")
        
        return "\n".join(lines)
    
    def _export_json_format(self) -> str:
        """导出JSON格式"""
        export_data = {}
        
        for name, metric in self.metrics.items():
            export_data[name] = {
                'type': metric['type'],
                'description': metric['description'],
                'labels': metric['labels'],
                'latest_value': metric['values'][-1] if metric['values'] else None,
                'latest_timestamp': metric['timestamps'][-1].isoformat() if metric['timestamps'] else None
            }
        
        return json.dumps(export_data, indent=2)
    
    async def start_collection(self, interval: int = 60):
        """启动指标收集"""
        asyncio.create_task(self._continuous_collection(interval))
    
    async def _continuous_collection(self, interval: int):
        """持续收集指标"""
        while True:
            try:
                # 收集系统指标
                await self.collect_system_metrics()
                
                # 收集应用指标
                for collector_name, collector in self.collectors.items():
                    try:
                        await collector()
                    except Exception as e:
                        logger.error(f"Collector {collector_name} failed: {e}")
                
                # 等待下一次收集
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)  # 出错后等待30秒重试
```

---

## 🧪 测试与验证

### 单元测试
```python
class TestInfrastructureServices:
    
    @pytest.mark.asyncio
    async def test_logging_service(self):
        """测试日志服务"""
        logging_service = LoggingService()
        
        # 获取日志器
        logger = logging_service.get_logger("test_service", LogConfig())
        assert logger is not None
        
        # 异步日志记录
        await logging_service.log_async("INFO", "Test message", "test_service")
        
        # 验证日志记录成功
        logs = await logging_service.query_logs(service="test_service")
        assert len(logs) > 0
    
    @pytest.mark.asyncio
    async def test_config_manager(self):
        """测试配置管理器"""
        config_manager = ConfigManager()
        
        # 加载配置
        config = await config_manager.load_config("test_service")
        
        # 设置配置值
        await config_manager.set_config("test_service", "database.host", "localhost")
        
        # 获取配置值
        host = await config_manager.get_config("test_service", "database.host")
        assert host == "localhost"
    
    @pytest.mark.asyncio
    async def test_auth_service(self):
        """测试认证服务"""
        auth_service = AuthService()
        
        # 创建用户
        create_result = await auth_service.create_user(UserCreateRequest(
            username="testuser",
            password="testpass123",
            email="test@example.com"
        ))
        assert create_result.success
        
        # 用户认证
        auth_result = await auth_service.authenticate("testuser", "testpass123")
        assert auth_result.success
        assert auth_result.access_token is not None
    
    @pytest.mark.asyncio
    async def test_health_checker(self):
        """测试健康检查器"""
        health_checker = HealthChecker()
        
        # 注册健康检查
        async def mock_check():
            return {"status": "healthy", "message": "OK"}
        
        health_checker.register_check("test_service", mock_check)
        
        # 执行健康检查
        status = await health_checker.check_health("test_service")
        assert status.status == "healthy"
    
    def test_metrics_collector(self):
        """测试指标收集器"""
        metrics_collector = MetricsCollector()
        
        # 注册指标
        metrics_collector.register_metric(
            "test.metric", 
            "gauge", 
            "Test metric",
            labels={"unit": "count"}
        )
        
        # 记录指标值
        metrics_collector.record_metric("test.metric", 42.0)
        
        # 获取指标数据
        metric_data = metrics_collector.get_metric("test.metric")
        assert metric_data["values"][-1] == 42.0
```

---

## 📊 验收标准

### 功能验收标准
```
✅ 日志服务:
- 支持多种日志级别和格式化器
- 支持异步日志记录和批量处理
- 支持结构化日志和上下文信息
- 支持日志查询和过滤功能

✅ 配置管理:
- 支持多种配置源 (文件、Consul、etcd)
- 支持配置热更新和监听机制
- 支持敏感信息加密存储
- 支持配置验证和类型检查

✅ 认证授权:
- 支持JWT令牌认证和刷新
- 支持基于角色的权限控制 (RBAC)
- 支持用户管理和密码安全
- 支持令牌过期和撤销机制

✅ 健康检查:
- 支持自定义健康检查函数
- 支持依赖关系检查
- 支持持续监控和状态缓存
- 支持健康状态查询API

✅ 指标收集:
- 支持多种指标类型 (Counter、Gauge、Histogram)
- 支持指标标签和元数据
- 支持多种导出格式 (Prometheus、JSON)
- 支持持续收集和历史数据
```

### 性能验收标准
```
✅ 性能指标:
- 日志记录延迟 < 10ms
- 配置读取延迟 < 5ms
- 认证验证延迟 < 50ms
- 健康检查延迟 < 100ms
- 指标收集开销 < 5% CPU

✅ 可扩展性:
- 支持并发日志记录 > 10000/min
- 支持配置监听器数量 > 100
- 支持健康检查服务数量 > 50
- 支持指标数量 > 1000

✅ 稳定性:
- 系统可用性 > 99.9%
- 内存使用稳定无泄漏
- 磁盘空间使用可控
- 网络连接稳定可靠

✅ 安全性:
- 敏感信息加密存储
- 访问令牌安全传输
- 配置访问权限控制
- 日志信息脱敏处理
```

---

## 🚀 部署配置

### Docker配置
```dockerfile
# Infrastructure Services Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create directories
RUN mkdir -p /app/logs /app/config

# Expose ports
EXPOSE 8080 8081 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8080/health || exit 1

# Start services
CMD ["python", "-m", "src.infrastructure.main"]
```

### Kubernetes配置
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: infrastructure-services
  labels:
    app: infrastructure-services
spec:
  replicas: 2
  selector:
    matchLabels:
      app: infrastructure-services
  template:
    metadata:
      labels:
        app: infrastructure-services
    spec:
      containers:
      - name: infrastructure
        image: rqa2026/infrastructure:latest
        ports:
        - containerPort: 8080
          name: api
        - containerPort: 8081
          name: metrics
        - containerPort: 8082
          name: health
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: CONFIG_SOURCE
          value: "consul"
        - name: CONSUL_URL
          value: "http://consul:8500"
        livenessProbe:
          httpGet:
            path: /health
            port: 8082
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8082
          initialDelaySeconds: 15
          periodSeconds: 10
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: config-volume
        configMap:
          name: infrastructure-config
      - name: logs-volume
        persistentVolumeClaim:
          claimName: infrastructure-logs-pvc
```

---

## 📊 监控与告警

### 监控指标
```
基础设施服务指标:
- infrastructure.log.entries_processed: 日志条目处理数
- infrastructure.config.updates: 配置更新次数
- infrastructure.auth.requests: 认证请求数
- infrastructure.health.checks: 健康检查次数
- infrastructure.metrics.collection_time: 指标收集时间

系统资源指标:
- system.cpu.usage: CPU使用率
- system.memory.usage: 内存使用率
- system.disk.usage: 磁盘使用率
- system.network.io: 网络IO

应用性能指标:
- app.response.time: 响应时间
- app.error.rate: 错误率
- app.throughput: 吞吐量
- app.active_connections: 活跃连接数
```

### 告警规则
```yaml
groups:
- name: infrastructure_alerts
  rules:
  - alert: HighCPUUsage
    expr: system_cpu_usage > 90
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "CPU使用率过高"
      description: "基础设施服务CPU使用率超过90%"

  - alert: HighMemoryUsage
    expr: system_memory_usage > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "内存使用率过高"
      description: "基础设施服务内存使用率超过85%"

  - alert: ServiceUnhealthy
    expr: up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "服务不健康"
      description: "基础设施服务健康检查失败"

  - alert: ConfigUpdateFailed
    expr: increase(infrastructure_config_update_errors[5m]) > 0
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "配置更新失败"
      description: "基础设施配置更新出现错误"
```

---

## 🎯 Sprint 1交付物

### 代码交付物
```
✅ 核心框架代码:
- src/infrastructure/
  ├── logging_service.py         # 日志服务
  ├── config_manager.py          # 配置管理器
  ├── auth_service.py            # 认证服务
  ├── health_checker.py          # 健康检查器
  ├── metrics_collector.py       # 指标收集器
  └── models.py                  # 数据模型

✅ 测试代码:
- tests/infrastructure/
  ├── unit/                      # 单元测试
  ├── integration/              # 集成测试
  └── performance/              # 性能测试

✅ 部署配置:
- deployment/docker/
- deployment/k8s/
- deployment/aws/
- deployment/monitoring/
```

### 文档交付物
```
✅ 技术文档:
- docs/infrastructure/
  ├── architecture.md            # 架构设计
  ├── logging.md                 # 日志服务
  ├── configuration.md           # 配置管理
  ├── authentication.md          # 认证授权
  ├── monitoring.md              # 监控告警

✅ 使用指南:
- README.md                     # 项目说明
- DEVELOPMENT.md                # 开发指南
- DEPLOYMENT.md                 # 部署文档
- API_REFERENCE.md              # API参考
- MONITORING_GUIDE.md           # 监控指南
```

---

*生成时间: 2024年12月10日*
*执行状态: Week 1 基础设施基础服务建设计划制定完成*




