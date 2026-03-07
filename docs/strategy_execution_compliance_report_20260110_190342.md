# 策略执行监控仪表盘架构符合性检查报告

**检查时间**: 2026-01-10T19:03:42.055934

## 检查摘要

- **总检查项**: 46
- **通过**: 46 ✅
- **失败**: 0 ❌
- **警告**: 0 ⚠️
- **未实现**: 0 📋
- **通过率**: 100.00%

## 1. 前端功能模块检查

### dashboard_exists ✅

- **文件**: web-static/strategy-execution-monitor.html
- **状态**: passed

### statistics_cards ✅

- **文件**: web-static/strategy-execution-monitor.html
- **状态**: passed
- **消息**: 统计卡片模块（运行中策略、平均延迟、今日信号数、总交易数）: 找到 8/4 个必需模式
- **匹配情况**: 8/4

### api_integration ✅

- **文件**: web-static/strategy-execution-monitor.html
- **状态**: passed
- **消息**: API集成（/strategy/execution/status, /strategy/execution/metrics, /strategy/realtime/signals）: 找到 12/2 个必需模式
- **匹配情况**: 12/2

### websocket_integration ✅

- **文件**: web-static/strategy-execution-monitor.html
- **状态**: passed
- **消息**: WebSocket实时更新集成（/ws/execution-status）: 找到 4/2 个必需模式
- **匹配情况**: 4/2

### chart_rendering ✅

- **文件**: web-static/strategy-execution-monitor.html
- **状态**: passed
- **消息**: 图表和可视化渲染（延迟趋势图、吞吐量趋势图）: 找到 19/3 个必需模式
- **匹配情况**: 19/3

### function_modules ✅

- **文件**: web-static/strategy-execution-monitor.html
- **状态**: passed
- **消息**: 功能模块完整性（策略执行状态表格、实时交易信号列表、策略操作功能）: 找到 7/4 个必需模式
- **匹配情况**: 7/4

## 2. 后端API端点检查

### api_endpoints ✅

- **文件**: src/gateway/web/strategy_execution_routes.py
- **状态**: passed
- **消息**: API端点实现（GET /strategy/execution/status, GET /strategy/execution/metrics, POST /strategy/execution/{strategy_id}/start, POST /strategy/execution/{strategy_id}/pause）: 找到 4/2 个必需模式
- **匹配情况**: 4/2

### service_layer_usage ✅

- **文件**: src/gateway/web/strategy_execution_routes.py
- **状态**: passed
- **消息**: 服务层封装使用: 找到 17/2 个必需模式
- **匹配情况**: 17/2

### unified_logger ✅

- **文件**: src/gateway/web/strategy_execution_routes.py
- **状态**: passed
- **消息**: 统一日志系统使用: 找到 4/1 个必需模式
- **匹配情况**: 4/1

### event_bus ✅

- **文件**: src/gateway/web/strategy_execution_routes.py
- **状态**: passed
- **消息**: 事件总线集成（发布EXECUTION_STARTED, EXECUTION_COMPLETED事件）: 找到 30/2 个必需模式
- **匹配情况**: 30/2

### business_orchestrator ✅

- **文件**: src/gateway/web/strategy_execution_routes.py
- **状态**: passed
- **消息**: BusinessProcessOrchestrator业务流程编排（start_process, update_process_state）: 找到 31/2 个必需模式
- **匹配情况**: 31/2

### websocket_broadcast ✅

- **文件**: src/gateway/web/strategy_execution_routes.py
- **状态**: passed
- **消息**: WebSocket实时广播（manager.broadcast）: 找到 9/1 个必需模式
- **匹配情况**: 9/1

### service_container ✅

- **文件**: src/gateway/web/strategy_execution_routes.py
- **状态**: passed
- **消息**: ServiceContainer依赖注入: 找到 8/1 个必需模式
- **匹配情况**: 8/1

## 3. 服务层实现检查

### unified_logger ✅

- **文件**: src/gateway/web/strategy_execution_service.py
- **状态**: passed
- **消息**: 统一日志系统使用: 找到 4/1 个必需模式
- **匹配情况**: 4/1

### adapter_factory_usage ✅

- **文件**: src/gateway/web/strategy_execution_service.py
- **状态**: passed
- **消息**: 统一适配器工厂使用（策略层和交易层）: 找到 4/2 个必需模式
- **匹配情况**: 4/2

### strategy_adapter ✅

- **文件**: src/gateway/web/strategy_execution_service.py
- **状态**: passed
- **消息**: 策略层适配器获取: 找到 21/1 个必需模式
- **匹配情况**: 21/1

### trading_adapter ✅

- **文件**: src/gateway/web/strategy_execution_service.py
- **状态**: passed
- **消息**: 交易层适配器获取: 找到 12/1 个必需模式
- **匹配情况**: 12/1

### fallback_mechanism ✅

- **文件**: src/gateway/web/strategy_execution_service.py
- **状态**: passed
- **消息**: 降级服务机制（当策略层适配器不可用时的降级处理）: 找到 11/2 个必需模式
- **匹配情况**: 11/2

### realtime_engine ✅

- **文件**: src/gateway/web/strategy_execution_service.py
- **状态**: passed
- **消息**: 实时策略引擎封装: 找到 15/2 个必需模式
- **匹配情况**: 15/2

### persistence_integration ✅

- **文件**: src/gateway/web/strategy_execution_service.py
- **状态**: passed
- **消息**: 持久化集成: 找到 11/2 个必需模式
- **匹配情况**: 11/2

## 4. 持久化实现检查

### file_persistence ✅

- **文件**: src/gateway/web/execution_persistence.py
- **状态**: passed
- **消息**: 文件系统持久化（JSON格式）: 找到 17/3 个必需模式
- **匹配情况**: 17/3

### postgresql_persistence ✅

- **文件**: src/gateway/web/execution_persistence.py
- **状态**: passed
- **消息**: PostgreSQL持久化: 找到 9/2 个必需模式
- **匹配情况**: 9/2

### dual_storage ✅

- **文件**: src/gateway/web/execution_persistence.py
- **状态**: passed
- **消息**: 双重存储机制（PostgreSQL优先，文件系统降级）: 找到 12/2 个必需模式
- **匹配情况**: 12/2

### crud_operations ✅

- **文件**: src/gateway/web/execution_persistence.py
- **状态**: passed
- **消息**: 执行状态CRUD操作: 找到 7/4 个必需模式
- **匹配情况**: 7/4

### unified_logger ✅

- **文件**: src/gateway/web/execution_persistence.py
- **状态**: passed
- **消息**: 统一日志系统使用: 找到 5/1 个必需模式
- **匹配情况**: 5/1

## 5. 架构设计符合性检查

### infrastructure_logger ✅

- **文件**: src/gateway/web/strategy_execution_routes.py
- **状态**: passed
- **消息**: 基础设施层统一日志系统集成: 找到 2/1 个必需模式
- **匹配情况**: 2/1

### event_bus_usage ✅

- **文件**: src/gateway/web/strategy_execution_routes.py
- **状态**: passed
- **消息**: EventBus事件驱动通信: 找到 30/2 个必需模式
- **匹配情况**: 30/2

### service_container ✅

- **文件**: src/gateway/web/strategy_execution_routes.py
- **状态**: passed
- **消息**: ServiceContainer依赖注入: 找到 8/1 个必需模式
- **匹配情况**: 8/1

### business_orchestrator ✅

- **文件**: src/gateway/web/strategy_execution_routes.py
- **状态**: passed
- **消息**: BusinessProcessOrchestrator业务流程编排: 找到 29/1 个必需模式
- **匹配情况**: 29/1

### adapter_factory_usage ✅

- **文件**: src/gateway/web/strategy_execution_service.py
- **状态**: passed
- **消息**: 统一适配器工厂使用（策略层和交易层）: 找到 4/2 个必需模式
- **匹配情况**: 4/2

### strategy_layer_access ✅

- **文件**: src/gateway/web/strategy_execution_service.py
- **状态**: passed
- **消息**: 策略层组件访问（通过适配器）: 找到 25/1 个必需模式
- **匹配情况**: 25/1

### trading_layer_access ✅

- **文件**: src/gateway/web/strategy_execution_service.py
- **状态**: passed
- **消息**: 交易层组件访问（通过适配器）: 找到 13/1 个必需模式
- **匹配情况**: 13/1

## 6. 策略层集成检查

### adapter_factory_usage ✅

- **文件**: src/gateway/web/strategy_execution_service.py
- **状态**: passed
- **消息**: 通过统一适配器工厂访问策略层: 找到 12/1 个必需模式
- **匹配情况**: 12/1

### strategy_adapter_access ✅

- **文件**: src/gateway/web/strategy_execution_service.py
- **状态**: passed
- **消息**: 策略层适配器获取: 找到 21/1 个必需模式
- **匹配情况**: 21/1

### realtime_engine_usage ✅

- **文件**: src/gateway/web/strategy_execution_service.py
- **状态**: passed
- **消息**: 实时策略引擎使用（通过适配器或降级方案）: 找到 15/1 个必需模式
- **匹配情况**: 15/1

### execution_state_integration ✅

- **文件**: src/gateway/web/strategy_execution_service.py
- **状态**: passed
- **消息**: 策略执行状态集成（从实时引擎获取策略执行状态）: 找到 3/1 个必需模式
- **匹配情况**: 3/1

### performance_metrics_integration ✅

- **文件**: src/gateway/web/strategy_execution_service.py
- **状态**: passed
- **消息**: 策略性能指标集成（延迟、吞吐量、信号数等）: 找到 2/1 个必需模式
- **匹配情况**: 2/1

## 7. 交易层集成检查

### adapter_factory_usage ✅

- **文件**: src/gateway/web/strategy_execution_service.py
- **状态**: passed
- **消息**: 通过统一适配器工厂访问交易层: 找到 12/1 个必需模式
- **匹配情况**: 12/1

### trading_adapter_access ✅

- **文件**: src/gateway/web/strategy_execution_service.py
- **状态**: passed
- **消息**: 交易层适配器获取: 找到 12/1 个必需模式
- **匹配情况**: 12/1

### realtime_signals_integration ✅

- **文件**: src/gateway/web/strategy_execution_routes.py
- **状态**: passed
- **消息**: 实时交易信号集成（从交易信号服务获取最近信号）: 找到 5/1 个必需模式
- **匹配情况**: 5/1

## 8. WebSocket实时更新检查

### websocket_endpoint ✅

- **文件**: src/gateway/web/websocket_routes.py
- **状态**: passed
- **消息**: WebSocket端点注册（/ws/execution-status）: 找到 2/1 个必需模式
- **匹配情况**: 2/1

### websocket_manager ✅

- **文件**: src/gateway/web/websocket_manager.py
- **状态**: passed
- **消息**: 执行状态WebSocket广播实现: 找到 8/2 个必需模式
- **匹配情况**: 8/2

### frontend_websocket ✅

- **文件**: web-static/strategy-execution-monitor.html
- **状态**: passed
- **消息**: 前端WebSocket消息处理: 找到 5/3 个必需模式
- **匹配情况**: 5/3

## 9. 业务流程编排检查

### orchestrator_usage ✅

- **文件**: src/gateway/web/strategy_execution_routes.py
- **状态**: passed
- **消息**: BusinessProcessOrchestrator使用: 找到 29/2 个必需模式
- **匹配情况**: 29/2

### process_management ✅

- **文件**: src/gateway/web/strategy_execution_routes.py
- **状态**: passed
- **消息**: 流程状态管理（策略执行流程状态管理）: 找到 53/2 个必需模式
- **匹配情况**: 53/2

### event_publishing ✅

- **文件**: src/gateway/web/strategy_execution_routes.py
- **状态**: passed
- **消息**: 执行流程事件发布（EXECUTION_STARTED, EXECUTION_COMPLETED, SIGNAL_GENERATED）: 找到 8/2 个必需模式
- **匹配情况**: 8/2

## 详细检查结果

```json
{
  "timestamp": "2026-01-10T19:03:42.055934",
  "frontend_modules": {
    "dashboard_exists": {
      "file": "web-static/strategy-execution-monitor.html",
      "exists": true,
      "status": "passed"
    },
    "statistics_cards": {
      "status": "passed",
      "message": "统计卡片模块（运行中策略、平均延迟、今日信号数、总交易数）: 找到 8/4 个必需模式",
      "found_patterns": [
        [
          "running-strategies|avg-latency|today-signals|total-trades",
          8
        ]
      ],
      "found_count": 8,
      "required_count": 4,
      "file": "web-static/strategy-execution-monitor.html"
    },
    "api_integration": {
      "status": "passed",
      "message": "API集成（/strategy/execution/status, /strategy/execution/metrics, /strategy/realtime/signals）: 找到 12/2 个必需模式",
      "found_patterns": [
        [
          "/strategy/execution/status|/strategy/execution/metrics|/strategy/realtime/signals",
          3
        ],
        [
          "fetch\\(|getApiBaseUrl",
          9
        ]
      ],
      "found_count": 12,
      "required_count": 2,
      "file": "web-static/strategy-execution-monitor.html"
    },
    "websocket_integration": {
      "status": "passed",
      "message": "WebSocket实时更新集成（/ws/execution-status）: 找到 4/2 个必需模式",
      "found_patterns": [
        [
          "/ws/execution-status|connectWebSocket|ws\\.onmessage",
          4
        ]
      ],
      "found_count": 4,
      "required_count": 2,
      "file": "web-static/strategy-execution-monitor.html"
    },
    "chart_rendering": {
      "status": "passed",
      "message": "图表和可视化渲染（延迟趋势图、吞吐量趋势图）: 找到 19/3 个必需模式",
      "found_patterns": [
        [
          "latencyChart|throughputChart|Chart\\.js|new Chart",
          19
        ]
      ],
      "found_count": 19,
      "required_count": 3,
      "file": "web-static/strategy-execution-monitor.html"
    },
    "function_modules": {
      "status": "passed",
      "message": "功能模块完整性（策略执行状态表格、实时交易信号列表、策略操作功能）: 找到 7/4 个必需模式",
      "found_patterns": [
        [
          "策略执行列表|最近信号|toggleStrategy|viewStrategyDetails",
          7
        ]
      ],
      "found_count": 7,
      "required_count": 4,
      "file": "web-static/strategy-execution-monitor.html"
    }
  },
  "backend_apis": {
    "api_endpoints": {
      "status": "passed",
      "message": "API端点实现（GET /strategy/execution/status, GET /strategy/execution/metrics, POST /strategy/execution/{strategy_id}/start, POST /strategy/execution/{strategy_id}/pause）: 找到 4/2 个必需模式",
      "found_patterns": [
        [
          "@router\\.get\\(.*/strategy/execution/status|@router\\.get\\(.*/strategy/execution/metrics",
          2
        ],
        [
          "@router\\.post\\(.*/strategy/execution/.*start|@router\\.post\\(.*/strategy/execution/.*pause",
          2
        ]
      ],
      "found_count": 4,
      "required_count": 2,
      "file": "src/gateway/web/strategy_execution_routes.py"
    },
    "service_layer_usage": {
      "status": "passed",
      "message": "服务层封装使用: 找到 17/2 个必需模式",
      "found_patterns": [
        [
          "from \\.strategy_execution_service import|get_strategy_execution_status|get_execution_metrics|start_strategy|pause_strategy",
          17
        ]
      ],
      "found_count": 17,
      "required_count": 2,
      "file": "src/gateway/web/strategy_execution_routes.py"
    },
    "unified_logger": {
      "status": "passed",
      "message": "统一日志系统使用: 找到 4/1 个必需模式",
      "found_patterns": [
        [
          "get_unified_logger|统一日志",
          4
        ]
      ],
      "found_count": 4,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_routes.py"
    },
    "event_bus": {
      "status": "passed",
      "message": "事件总线集成（发布EXECUTION_STARTED, EXECUTION_COMPLETED事件）: 找到 30/2 个必需模式",
      "found_patterns": [
        [
          "EventBus|event_bus|\\.publish\\(|EventType\\.EXECUTION_STARTED|EventType\\.EXECUTION_COMPLETED",
          30
        ]
      ],
      "found_count": 30,
      "required_count": 2,
      "file": "src/gateway/web/strategy_execution_routes.py"
    },
    "business_orchestrator": {
      "status": "passed",
      "message": "BusinessProcessOrchestrator业务流程编排（start_process, update_process_state）: 找到 31/2 个必需模式",
      "found_patterns": [
        [
          "BusinessProcessOrchestrator|orchestrator|业务流程|_get_orchestrator|start_process|update_process_state",
          31
        ]
      ],
      "found_count": 31,
      "required_count": 2,
      "file": "src/gateway/web/strategy_execution_routes.py"
    },
    "websocket_broadcast": {
      "status": "passed",
      "message": "WebSocket实时广播（manager.broadcast）: 找到 9/1 个必需模式",
      "found_patterns": [
        [
          "websocket_manager|_get_websocket_manager|manager\\.broadcast|broadcast.*execution",
          9
        ]
      ],
      "found_count": 9,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_routes.py"
    },
    "service_container": {
      "status": "passed",
      "message": "ServiceContainer依赖注入: 找到 8/1 个必需模式",
      "found_patterns": [
        [
          "ServiceContainer|DependencyContainer|container\\.resolve|_get_container",
          8
        ]
      ],
      "found_count": 8,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_routes.py"
    }
  },
  "service_layer": {
    "unified_logger": {
      "status": "passed",
      "message": "统一日志系统使用: 找到 4/1 个必需模式",
      "found_patterns": [
        [
          "get_unified_logger|统一日志",
          4
        ]
      ],
      "found_count": 4,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_service.py"
    },
    "adapter_factory_usage": {
      "status": "passed",
      "message": "统一适配器工厂使用（策略层和交易层）: 找到 4/2 个必需模式",
      "found_patterns": [
        [
          "get_unified_adapter_factory|BusinessLayerType\\.STRATEGY|BusinessLayerType\\.TRADING",
          4
        ]
      ],
      "found_count": 4,
      "required_count": 2,
      "file": "src/gateway/web/strategy_execution_service.py"
    },
    "strategy_adapter": {
      "status": "passed",
      "message": "策略层适配器获取: 找到 21/1 个必需模式",
      "found_patterns": [
        [
          "_get_strategy_adapter|strategy_adapter|策略层适配器",
          21
        ]
      ],
      "found_count": 21,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_service.py"
    },
    "trading_adapter": {
      "status": "passed",
      "message": "交易层适配器获取: 找到 12/1 个必需模式",
      "found_patterns": [
        [
          "_get_trading_adapter|trading_adapter|交易层适配器",
          12
        ]
      ],
      "found_count": 12,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_service.py"
    },
    "fallback_mechanism": {
      "status": "passed",
      "message": "降级服务机制（当策略层适配器不可用时的降级处理）: 找到 11/2 个必需模式",
      "found_patterns": [
        [
          "降级方案|fallback|except.*ImportError|直接实例化|RealTimeStrategyEngine",
          11
        ]
      ],
      "found_count": 11,
      "required_count": 2,
      "file": "src/gateway/web/strategy_execution_service.py"
    },
    "realtime_engine": {
      "status": "passed",
      "message": "实时策略引擎封装: 找到 15/2 个必需模式",
      "found_patterns": [
        [
          "get_realtime_engine|RealTimeStrategyEngine|实时策略引擎",
          15
        ]
      ],
      "found_count": 15,
      "required_count": 2,
      "file": "src/gateway/web/strategy_execution_service.py"
    },
    "persistence_integration": {
      "status": "passed",
      "message": "持久化集成: 找到 11/2 个必需模式",
      "found_patterns": [
        [
          "execution_persistence|save_execution_state|list_execution_states|持久化存储",
          11
        ]
      ],
      "found_count": 11,
      "required_count": 2,
      "file": "src/gateway/web/strategy_execution_service.py"
    }
  },
  "persistence": {
    "file_persistence": {
      "status": "passed",
      "message": "文件系统持久化（JSON格式）: 找到 17/3 个必需模式",
      "found_patterns": [
        [
          "save_execution_state|json\\.dump|文件系统|EXECUTION_STATES_DIR",
          17
        ]
      ],
      "found_count": 17,
      "required_count": 3,
      "file": "src/gateway/web/execution_persistence.py"
    },
    "postgresql_persistence": {
      "status": "passed",
      "message": "PostgreSQL持久化: 找到 9/2 个必需模式",
      "found_patterns": [
        [
          "_save_to_postgresql|_load_from_postgresql|postgresql_persistence|CREATE TABLE.*strategy_execution_states",
          9
        ]
      ],
      "found_count": 9,
      "required_count": 2,
      "file": "src/gateway/web/execution_persistence.py"
    },
    "dual_storage": {
      "status": "passed",
      "message": "双重存储机制（PostgreSQL优先，文件系统降级）: 找到 12/2 个必需模式",
      "found_patterns": [
        [
          "优先.*PostgreSQL|如果.*PostgreSQL|故障转移|fallback|return None|文件系统",
          12
        ]
      ],
      "found_count": 12,
      "required_count": 2,
      "file": "src/gateway/web/execution_persistence.py"
    },
    "crud_operations": {
      "status": "passed",
      "message": "执行状态CRUD操作: 找到 7/4 个必需模式",
      "found_patterns": [
        [
          "save_execution_state|load_execution_state|update_execution_state|delete_execution_state|list_execution_states",
          7
        ]
      ],
      "found_count": 7,
      "required_count": 4,
      "file": "src/gateway/web/execution_persistence.py"
    },
    "unified_logger": {
      "status": "passed",
      "message": "统一日志系统使用: 找到 5/1 个必需模式",
      "found_patterns": [
        [
          "get_unified_logger|统一日志",
          5
        ]
      ],
      "found_count": 5,
      "required_count": 1,
      "file": "src/gateway/web/execution_persistence.py"
    }
  },
  "architecture_compliance": {
    "infrastructure_logger": {
      "status": "passed",
      "message": "基础设施层统一日志系统集成: 找到 2/1 个必需模式",
      "found_patterns": [
        [
          "get_unified_logger",
          2
        ]
      ],
      "found_count": 2,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_routes.py"
    },
    "event_bus_usage": {
      "status": "passed",
      "message": "EventBus事件驱动通信: 找到 30/2 个必需模式",
      "found_patterns": [
        [
          "EventBus|event_bus|\\.publish\\(|EventType\\.",
          30
        ]
      ],
      "found_count": 30,
      "required_count": 2,
      "file": "src/gateway/web/strategy_execution_routes.py"
    },
    "service_container": {
      "status": "passed",
      "message": "ServiceContainer依赖注入: 找到 8/1 个必需模式",
      "found_patterns": [
        [
          "ServiceContainer|DependencyContainer|container\\.resolve|_get_container",
          8
        ]
      ],
      "found_count": 8,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_routes.py"
    },
    "business_orchestrator": {
      "status": "passed",
      "message": "BusinessProcessOrchestrator业务流程编排: 找到 29/1 个必需模式",
      "found_patterns": [
        [
          "BusinessProcessOrchestrator|orchestrator|业务流程|_get_orchestrator",
          29
        ]
      ],
      "found_count": 29,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_routes.py"
    },
    "adapter_factory_usage": {
      "status": "passed",
      "message": "统一适配器工厂使用（策略层和交易层）: 找到 4/2 个必需模式",
      "found_patterns": [
        [
          "get_unified_adapter_factory|BusinessLayerType\\.STRATEGY|BusinessLayerType\\.TRADING",
          4
        ]
      ],
      "found_count": 4,
      "required_count": 2,
      "file": "src/gateway/web/strategy_execution_service.py"
    },
    "strategy_layer_access": {
      "status": "passed",
      "message": "策略层组件访问（通过适配器）: 找到 25/1 个必需模式",
      "found_patterns": [
        [
          "策略层适配器|strategy_adapter|RealTimeStrategyEngine|策略执行服务",
          25
        ]
      ],
      "found_count": 25,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_service.py"
    },
    "trading_layer_access": {
      "status": "passed",
      "message": "交易层组件访问（通过适配器）: 找到 13/1 个必需模式",
      "found_patterns": [
        [
          "交易层适配器|trading_adapter|交易信号服务|交易层服务",
          13
        ]
      ],
      "found_count": 13,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_service.py"
    }
  },
  "strategy_layer_integration": {
    "adapter_factory_usage": {
      "status": "passed",
      "message": "通过统一适配器工厂访问策略层: 找到 12/1 个必需模式",
      "found_patterns": [
        [
          "get_unified_adapter_factory|BusinessLayerType\\.STRATEGY|统一适配器工厂",
          12
        ]
      ],
      "found_count": 12,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_service.py"
    },
    "strategy_adapter_access": {
      "status": "passed",
      "message": "策略层适配器获取: 找到 21/1 个必需模式",
      "found_patterns": [
        [
          "_get_strategy_adapter|strategy_adapter|策略层适配器",
          21
        ]
      ],
      "found_count": 21,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_service.py"
    },
    "realtime_engine_usage": {
      "status": "passed",
      "message": "实时策略引擎使用（通过适配器或降级方案）: 找到 15/1 个必需模式",
      "found_patterns": [
        [
          "RealTimeStrategyEngine|get_realtime_engine|实时策略引擎",
          15
        ]
      ],
      "found_count": 15,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_service.py"
    },
    "execution_state_integration": {
      "status": "passed",
      "message": "策略执行状态集成（从实时引擎获取策略执行状态）: 找到 3/1 个必需模式",
      "found_patterns": [
        [
          "get_strategy_execution_status|策略执行状态|执行状态集成",
          3
        ]
      ],
      "found_count": 3,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_service.py"
    },
    "performance_metrics_integration": {
      "status": "passed",
      "message": "策略性能指标集成（延迟、吞吐量、信号数等）: 找到 2/1 个必需模式",
      "found_patterns": [
        [
          "get_execution_metrics|延迟|吞吐量|信号数|性能指标",
          2
        ]
      ],
      "found_count": 2,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_service.py"
    }
  },
  "trading_layer_integration": {
    "adapter_factory_usage": {
      "status": "passed",
      "message": "通过统一适配器工厂访问交易层: 找到 12/1 个必需模式",
      "found_patterns": [
        [
          "get_unified_adapter_factory|BusinessLayerType\\.TRADING|统一适配器工厂",
          12
        ]
      ],
      "found_count": 12,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_service.py"
    },
    "trading_adapter_access": {
      "status": "passed",
      "message": "交易层适配器获取: 找到 12/1 个必需模式",
      "found_patterns": [
        [
          "_get_trading_adapter|trading_adapter|交易层适配器",
          12
        ]
      ],
      "found_count": 12,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_service.py"
    },
    "realtime_signals_integration": {
      "status": "passed",
      "message": "实时交易信号集成（从交易信号服务获取最近信号）: 找到 5/1 个必需模式",
      "found_patterns": [
        [
          "get_realtime_signals|trading_signal_service|交易信号|最近信号",
          5
        ]
      ],
      "found_count": 5,
      "required_count": 1,
      "file": "src/gateway/web/strategy_execution_routes.py"
    }
  },
  "websocket_integration": {
    "websocket_endpoint": {
      "status": "passed",
      "message": "WebSocket端点注册（/ws/execution-status）: 找到 2/1 个必需模式",
      "found_patterns": [
        [
          "/ws/execution-status|websocket_execution_status",
          2
        ]
      ],
      "found_count": 2,
      "required_count": 1,
      "file": "src/gateway/web/websocket_routes.py"
    },
    "websocket_manager": {
      "status": "passed",
      "message": "执行状态WebSocket广播实现: 找到 8/2 个必需模式",
      "found_patterns": [
        [
          "_broadcast_execution_status|execution_status|get_strategy_execution_status",
          8
        ]
      ],
      "found_count": 8,
      "required_count": 2,
      "file": "src/gateway/web/websocket_manager.py"
    },
    "frontend_websocket": {
      "status": "passed",
      "message": "前端WebSocket消息处理: 找到 5/3 个必需模式",
      "found_patterns": [
        [
          "/ws/execution-status|connectWebSocket|onmessage|execution_status",
          5
        ]
      ],
      "found_count": 5,
      "required_count": 3,
      "file": "web-static/strategy-execution-monitor.html"
    }
  },
  "business_orchestration": {
    "orchestrator_usage": {
      "status": "passed",
      "message": "BusinessProcessOrchestrator使用: 找到 29/2 个必需模式",
      "found_patterns": [
        [
          "BusinessProcessOrchestrator|orchestrator|业务流程|_get_orchestrator|业务流程编排器",
          29
        ]
      ],
      "found_count": 29,
      "required_count": 2,
      "file": "src/gateway/web/strategy_execution_routes.py"
    },
    "process_management": {
      "status": "passed",
      "message": "流程状态管理（策略执行流程状态管理）: 找到 53/2 个必需模式",
      "found_patterns": [
        [
          "start_process|update_process_state|process.*state|流程状态|业务流程编排|orchestrator|process_id|STRATEGY_EXECUTION",
          53
        ]
      ],
      "found_count": 53,
      "required_count": 2,
      "file": "src/gateway/web/strategy_execution_routes.py"
    },
    "event_publishing": {
      "status": "passed",
      "message": "执行流程事件发布（EXECUTION_STARTED, EXECUTION_COMPLETED, SIGNAL_GENERATED）: 找到 8/2 个必需模式",
      "found_patterns": [
        [
          "EventBus\\.publish|event_bus\\.publish|EXECUTION_STARTED|EXECUTION_COMPLETED|SIGNAL_GENERATED|执行.*事件",
          8
        ]
      ],
      "found_count": 8,
      "required_count": 2,
      "file": "src/gateway/web/strategy_execution_routes.py"
    }
  },
  "summary": {
    "total_items": 46,
    "passed": 46,
    "failed": 0,
    "warnings": 0,
    "not_implemented": 0
  }
}
```
