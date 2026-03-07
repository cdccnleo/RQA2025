# 风险控制流程架构符合性检查报告

**检查时间**: 2026-01-10T22:07:07.879099

## 检查摘要

- **总检查项**: 46
- **通过**: 43 ✅
- **失败**: 0 ❌
- **警告**: 3 ⚠️
- **未实现**: 0 📋
- **通过率**: 93.48%

## 1. 前端功能模块检查

### dashboard_exists ✅

- **文件**: web-static/risk-control-monitor.html
- **状态**: passed

### statistics_cards ✅

- **文件**: web-static/risk-control-monitor.html
- **状态**: passed
- **消息**: 统计卡片模块（实时监测覆盖、平均监测延迟、活跃风险告警、当前VaR）: 找到 12/4 个必需模式
- **匹配情况**: 12/4

### workflow_steps ✅

- **文件**: web-static/risk-control-monitor.html
- **状态**: passed
- **消息**: 6个业务流程步骤展示（实时监测、风险评估、风险拦截、合规检查、风险报告、告警通知）: 找到 16/6 个必需模式
- **匹配情况**: 16/6

### api_integration ✅

- **文件**: web-static/risk-control-monitor.html
- **状态**: passed
- **消息**: API集成（/api/v1/risk/control/*）: 找到 16/2 个必需模式
- **匹配情况**: 16/2

### chart_rendering ✅

- **文件**: web-static/risk-control-monitor.html
- **状态**: passed
- **消息**: 图表和可视化渲染（VaR趋势图、风险分布图、风险热力图、风险时间线）: 找到 21/4 个必需模式
- **匹配情况**: 21/4

### step_status_display ✅

- **文件**: web-static/risk-control-monitor.html
- **状态**: passed
- **消息**: 流程步骤状态显示（6个步骤的状态和性能指标）: 找到 41/6 个必需模式
- **匹配情况**: 41/6

## 2. 后端API端点检查

### routes_file_exists ✅

- **文件**: src/gateway/web/risk_control_routes.py
- **状态**: passed

### api_endpoints ✅

- **文件**: src/gateway/web/risk_control_routes.py
- **状态**: passed
- **消息**: API端点实现（GET /api/v1/risk/control/overview, /heatmap, /timeline, /alerts, /stages/{stageId}）: 找到 5/3 个必需模式
- **匹配情况**: 5/3

### service_layer_usage ✅

- **文件**: src/gateway/web/risk_control_routes.py
- **状态**: passed
- **消息**: 服务层封装使用: 找到 7/1 个必需模式
- **匹配情况**: 7/1

### unified_logger ✅

- **文件**: src/gateway/web/risk_control_routes.py
- **状态**: passed
- **消息**: 统一日志系统使用: 找到 4/1 个必需模式
- **匹配情况**: 4/1

### event_bus ✅

- **文件**: src/gateway/web/risk_control_routes.py
- **状态**: passed
- **消息**: 事件总线集成（发布风险控制事件）: 找到 28/2 个必需模式
- **匹配情况**: 28/2

### business_orchestrator ✅

- **文件**: src/gateway/web/risk_control_routes.py
- **状态**: passed
- **消息**: BusinessProcessOrchestrator业务流程编排（start_process, RISK_CONTROL）: 找到 72/2 个必需模式
- **匹配情况**: 72/2

### websocket_broadcast ✅

- **文件**: src/gateway/web/risk_control_routes.py
- **状态**: passed
- **消息**: WebSocket实时广播（manager.broadcast）: 找到 11/1 个必需模式
- **匹配情况**: 11/1

### service_container ✅

- **文件**: src/gateway/web/risk_control_routes.py
- **状态**: passed
- **消息**: ServiceContainer依赖注入: 找到 8/1 个必需模式
- **匹配情况**: 8/1

## 3. 服务层实现检查

### service_file_exists ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed

### unified_logger ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 统一日志系统使用: 找到 4/1 个必需模式
- **匹配情况**: 4/1

### adapter_factory_usage ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 统一适配器工厂使用（风险控制层）: 找到 13/2 个必需模式
- **匹配情况**: 13/2

### risk_adapter ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 风险控制层适配器获取（通过统一适配器工厂）: 找到 21/1 个必需模式
- **匹配情况**: 21/1

### fallback_mechanism ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 降级服务机制（当风险控制层适配器不可用时的降级处理）: 找到 21/2 个必需模式
- **匹配情况**: 21/2

### workflow_steps_collection ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 6个业务流程步骤数据收集（实时监测、风险评估、风险拦截、合规检查、风险报告、告警通知）: 找到 44/6 个必需模式
- **匹配情况**: 44/6

### process_state_mapping ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 流程状态映射（6个步骤与BusinessProcessState的映射关系）: 找到 14/3 个必需模式
- **匹配情况**: 14/3

## 4. 持久化实现检查

### persistence_file_exists ✅

- **文件**: src/gateway/web/risk_control_persistence.py
- **状态**: passed

### file_persistence ✅

- **文件**: src/gateway/web/risk_control_persistence.py
- **状态**: passed
- **消息**: 文件系统持久化（JSON格式）: 找到 19/3 个必需模式
- **匹配情况**: 19/3

### postgresql_persistence ✅

- **文件**: src/gateway/web/risk_control_persistence.py
- **状态**: passed
- **消息**: PostgreSQL持久化: 找到 8/2 个必需模式
- **匹配情况**: 8/2

### workflow_steps_fields ✅

- **文件**: src/gateway/web/risk_control_persistence.py
- **状态**: passed
- **消息**: 6个步骤数据字段（realtime_monitoring, risk_assessment, risk_intercept, compliance_check, risk_report, alert_notify）: 找到 54/6 个必需模式
- **匹配情况**: 54/6

### unified_logger ✅

- **文件**: src/gateway/web/risk_control_persistence.py
- **状态**: passed
- **消息**: 统一日志系统使用: 找到 5/1 个必需模式
- **匹配情况**: 5/1

## 5. 架构设计符合性检查

### infrastructure_logger ✅

- **文件**: src/gateway/web/risk_control_routes.py
- **状态**: passed
- **消息**: 基础设施层统一日志系统集成: 找到 2/1 个必需模式
- **匹配情况**: 2/1

### event_bus_usage ✅

- **文件**: src/gateway/web/risk_control_routes.py
- **状态**: passed
- **消息**: EventBus事件驱动通信: 找到 30/2 个必需模式
- **匹配情况**: 30/2

### service_container ✅

- **文件**: src/gateway/web/risk_control_routes.py
- **状态**: passed
- **消息**: ServiceContainer依赖注入: 找到 8/1 个必需模式
- **匹配情况**: 8/1

### business_orchestrator ✅

- **文件**: src/gateway/web/risk_control_routes.py
- **状态**: passed
- **消息**: BusinessProcessOrchestrator业务流程编排: 找到 40/1 个必需模式
- **匹配情况**: 40/1

### adapter_factory_usage ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 统一适配器工厂使用（风险控制层）: 找到 13/2 个必需模式
- **匹配情况**: 13/2

### risk_layer_access ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 风险控制层组件访问（通过适配器）: 找到 28/2 个必需模式
- **匹配情况**: 28/2

## 6. 风险控制层集成检查

### adapter_factory_usage ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 通过统一适配器工厂访问风险控制层: 找到 13/1 个必需模式
- **匹配情况**: 13/1

### risk_adapter_access ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 风险控制层适配器获取（通过统一适配器工厂）: 找到 21/1 个必需模式
- **匹配情况**: 21/1

### risk_components_usage ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 风险控制层组件使用（RiskManager, RealTimeRiskMonitor, RiskCalculationEngine, AlertSystem）: 找到 5/2 个必需模式
- **匹配情况**: 5/2

## 7. WebSocket实时更新检查

### websocket_endpoint ⚠️

- **文件**: src/gateway/web/websocket_routes.py
- **状态**: warning
- **消息**: WebSocket端点注册（/ws/risk-control）: 仅找到 0/1 个必需模式
- **匹配情况**: 0/1

### websocket_manager ⚠️

- **文件**: src/gateway/web/websocket_manager.py
- **状态**: warning
- **消息**: 风险控制WebSocket广播实现: 仅找到 0/2 个必需模式
- **匹配情况**: 0/2

### frontend_websocket ⚠️

- **文件**: web-static/risk-control-monitor.html
- **状态**: warning
- **消息**: 前端WebSocket消息处理（/ws/risk-control）: 仅找到 0/2 个必需模式
- **匹配情况**: 0/2

## 8. 6个业务流程步骤检查

### step1_realtime_monitoring ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 步骤1: 实时监测（Real-time Monitoring）: 找到 16/2 个必需模式
- **匹配情况**: 16/2

### step2_risk_assessment ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 步骤2: 风险评估（Risk Assessment）: 找到 13/2 个必需模式
- **匹配情况**: 13/2

### step3_risk_intercept ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 步骤3: 风险拦截（Risk Intercept）: 找到 12/2 个必需模式
- **匹配情况**: 12/2

### step4_compliance_check ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 步骤4: 合规检查（Compliance Check）: 找到 13/2 个必需模式
- **匹配情况**: 13/2

### step5_risk_report ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 步骤5: 风险报告（Risk Report）: 找到 11/2 个必需模式
- **匹配情况**: 11/2

### step6_alert_notify ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 步骤6: 告警通知（Alert Notify）: 找到 15/3 个必需模式
- **匹配情况**: 15/3

### step_state_mapping ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 6个步骤与流程状态的映射关系: 找到 14/4 个必需模式
- **匹配情况**: 14/4

## 9. 业务流程编排检查

### orchestrator_usage ✅

- **文件**: src/gateway/web/risk_control_routes.py
- **状态**: passed
- **消息**: BusinessProcessOrchestrator使用: 找到 40/2 个必需模式
- **匹配情况**: 40/2

### process_management ✅

- **文件**: src/gateway/web/risk_control_routes.py
- **状态**: passed
- **消息**: 流程状态管理（风险控制流程状态管理）: 找到 89/2 个必需模式
- **匹配情况**: 89/2

### event_publishing ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 风险控制流程事件发布（6个步骤的事件）: 找到 11/4 个必需模式
- **匹配情况**: 11/4

### state_machine_integration ✅

- **文件**: src/gateway/web/risk_control_service.py
- **状态**: passed
- **消息**: 流程状态机集成（获取当前状态、状态历史）: 找到 8/2 个必需模式
- **匹配情况**: 8/2

## 详细检查结果

```json
{
  "timestamp": "2026-01-10T22:07:07.879099",
  "frontend_modules": {
    "dashboard_exists": {
      "file": "web-static/risk-control-monitor.html",
      "exists": true,
      "status": "passed"
    },
    "statistics_cards": {
      "status": "passed",
      "message": "统计卡片模块（实时监测覆盖、平均监测延迟、活跃风险告警、当前VaR）: 找到 12/4 个必需模式",
      "found_patterns": [
        [
          "monitoring-coverage|avg-monitoring-latency|active-risk-alerts|current-var",
          12
        ]
      ],
      "found_count": 12,
      "required_count": 4,
      "file": "web-static/risk-control-monitor.html"
    },
    "workflow_steps": {
      "status": "passed",
      "message": "6个业务流程步骤展示（实时监测、风险评估、风险拦截、合规检查、风险报告、告警通知）: 找到 16/6 个必需模式",
      "found_patterns": [
        [
          "实时监测|风险评估|风险拦截|合规检查|风险报告|告警通知",
          16
        ]
      ],
      "found_count": 16,
      "required_count": 6,
      "file": "web-static/risk-control-monitor.html"
    },
    "api_integration": {
      "status": "passed",
      "message": "API集成（/api/v1/risk/control/*）: 找到 16/2 个必需模式",
      "found_patterns": [
        [
          "/risk/control/overview|/risk/control/heatmap|/risk/control/timeline|/risk/control/alerts|/risk/control/stages",
          5
        ],
        [
          "fetch\\(|getApiBaseUrl",
          11
        ]
      ],
      "found_count": 16,
      "required_count": 2,
      "file": "web-static/risk-control-monitor.html"
    },
    "chart_rendering": {
      "status": "passed",
      "message": "图表和可视化渲染（VaR趋势图、风险分布图、风险热力图、风险时间线）: 找到 21/4 个必需模式",
      "found_patterns": [
        [
          "varTrendChart|riskDistributionChart|Chart\\.js|new Chart|risk-heatmap|risk-timeline",
          21
        ]
      ],
      "found_count": 21,
      "required_count": 4,
      "file": "web-static/risk-control-monitor.html"
    },
    "step_status_display": {
      "status": "passed",
      "message": "流程步骤状态显示（6个步骤的状态和性能指标）: 找到 41/6 个必需模式",
      "found_patterns": [
        [
          "status-indicator|showStageDetails|monitoring|assessment|interception|compliance|report|notification",
          41
        ]
      ],
      "found_count": 41,
      "required_count": 6,
      "file": "web-static/risk-control-monitor.html"
    }
  },
  "backend_apis": {
    "routes_file_exists": {
      "file": "src/gateway/web/risk_control_routes.py",
      "exists": true,
      "status": "passed"
    },
    "api_endpoints": {
      "status": "passed",
      "message": "API端点实现（GET /api/v1/risk/control/overview, /heatmap, /timeline, /alerts, /stages/{stageId}）: 找到 5/3 个必需模式",
      "found_patterns": [
        [
          "@router\\.get\\(.*/risk/control/overview|@router\\.get\\(.*/risk/control/heatmap|@router\\.get\\(.*/risk/control/timeline|@router\\.get\\(.*/risk/control/alerts|@router\\.get\\(.*/risk/control/stages",
          5
        ]
      ],
      "found_count": 5,
      "required_count": 3,
      "file": "src/gateway/web/risk_control_routes.py"
    },
    "service_layer_usage": {
      "status": "passed",
      "message": "服务层封装使用: 找到 7/1 个必需模式",
      "found_patterns": [
        [
          "from \\.risk_control_service import|get_risk_control_data|risk_control_persistence",
          7
        ]
      ],
      "found_count": 7,
      "required_count": 1,
      "file": "src/gateway/web/risk_control_routes.py"
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
      "file": "src/gateway/web/risk_control_routes.py"
    },
    "event_bus": {
      "status": "passed",
      "message": "事件总线集成（发布风险控制事件）: 找到 28/2 个必需模式",
      "found_patterns": [
        [
          "EventBus|event_bus|\\.publish\\(|EventType\\.RISK_CHECK_COMPLETED",
          28
        ]
      ],
      "found_count": 28,
      "required_count": 2,
      "file": "src/gateway/web/risk_control_routes.py"
    },
    "business_orchestrator": {
      "status": "passed",
      "message": "BusinessProcessOrchestrator业务流程编排（start_process, RISK_CONTROL）: 找到 72/2 个必需模式",
      "found_patterns": [
        [
          "BusinessProcessOrchestrator|orchestrator|业务流程|_get_orchestrator|start_process|RISK_CONTROL",
          72
        ]
      ],
      "found_count": 72,
      "required_count": 2,
      "file": "src/gateway/web/risk_control_routes.py"
    },
    "websocket_broadcast": {
      "status": "passed",
      "message": "WebSocket实时广播（manager.broadcast）: 找到 11/1 个必需模式",
      "found_patterns": [
        [
          "websocket_manager|_get_websocket_manager|manager\\.broadcast|broadcast.*risk_control",
          11
        ]
      ],
      "found_count": 11,
      "required_count": 1,
      "file": "src/gateway/web/risk_control_routes.py"
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
      "file": "src/gateway/web/risk_control_routes.py"
    }
  },
  "service_layer": {
    "service_file_exists": {
      "file": "src/gateway/web/risk_control_service.py",
      "exists": true,
      "status": "passed"
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
      "file": "src/gateway/web/risk_control_service.py"
    },
    "adapter_factory_usage": {
      "status": "passed",
      "message": "统一适配器工厂使用（风险控制层）: 找到 13/2 个必需模式",
      "found_patterns": [
        [
          "get_unified_adapter_factory|BusinessLayerType\\.RISK|统一适配器工厂",
          13
        ]
      ],
      "found_count": 13,
      "required_count": 2,
      "file": "src/gateway/web/risk_control_service.py"
    },
    "risk_adapter": {
      "status": "passed",
      "message": "风险控制层适配器获取（通过统一适配器工厂）: 找到 21/1 个必需模式",
      "found_patterns": [
        [
          "_get_risk_adapter|risk_adapter|风险控制层适配器",
          21
        ]
      ],
      "found_count": 21,
      "required_count": 1,
      "file": "src/gateway/web/risk_control_service.py"
    },
    "fallback_mechanism": {
      "status": "passed",
      "message": "降级服务机制（当风险控制层适配器不可用时的降级处理）: 找到 21/2 个必需模式",
      "found_patterns": [
        [
          "降级方案|fallback|except.*ImportError|直接实例化|最终降级方案",
          21
        ]
      ],
      "found_count": 21,
      "required_count": 2,
      "file": "src/gateway/web/risk_control_service.py"
    },
    "workflow_steps_collection": {
      "status": "passed",
      "message": "6个业务流程步骤数据收集（实时监测、风险评估、风险拦截、合规检查、风险报告、告警通知）: 找到 44/6 个必需模式",
      "found_patterns": [
        [
          "realtime_monitoring|risk_assessment|risk_intercept|compliance_check|risk_report|alert_notify",
          44
        ]
      ],
      "found_count": 44,
      "required_count": 6,
      "file": "src/gateway/web/risk_control_service.py"
    },
    "process_state_mapping": {
      "status": "passed",
      "message": "流程状态映射（6个步骤与BusinessProcessState的映射关系）: 找到 14/3 个必需模式",
      "found_patterns": [
        [
          "step_state_mapping|MONITORING|RISK_ASSESSING|RISK_INTERCEPTING|COMPLIANCE_CHECKING|REPORT_GENERATING|ALERT_NOTIFYING",
          14
        ]
      ],
      "found_count": 14,
      "required_count": 3,
      "file": "src/gateway/web/risk_control_service.py"
    }
  },
  "persistence": {
    "persistence_file_exists": {
      "file": "src/gateway/web/risk_control_persistence.py",
      "exists": true,
      "status": "passed"
    },
    "file_persistence": {
      "status": "passed",
      "message": "文件系统持久化（JSON格式）: 找到 19/3 个必需模式",
      "found_patterns": [
        [
          "save_risk_control_record|json\\.dump|文件系统|RISK_CONTROL_DIR",
          19
        ]
      ],
      "found_count": 19,
      "required_count": 3,
      "file": "src/gateway/web/risk_control_persistence.py"
    },
    "postgresql_persistence": {
      "status": "passed",
      "message": "PostgreSQL持久化: 找到 8/2 个必需模式",
      "found_patterns": [
        [
          "_save_to_postgresql|_load_from_postgresql|postgresql_persistence|CREATE TABLE.*risk_control_records",
          8
        ]
      ],
      "found_count": 8,
      "required_count": 2,
      "file": "src/gateway/web/risk_control_persistence.py"
    },
    "workflow_steps_fields": {
      "status": "passed",
      "message": "6个步骤数据字段（realtime_monitoring, risk_assessment, risk_intercept, compliance_check, risk_report, alert_notify）: 找到 54/6 个必需模式",
      "found_patterns": [
        [
          "realtime_monitoring|risk_assessment|risk_intercept|compliance_check|risk_report|alert_notify",
          54
        ]
      ],
      "found_count": 54,
      "required_count": 6,
      "file": "src/gateway/web/risk_control_persistence.py"
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
      "file": "src/gateway/web/risk_control_persistence.py"
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
      "file": "src/gateway/web/risk_control_routes.py"
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
      "file": "src/gateway/web/risk_control_routes.py"
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
      "file": "src/gateway/web/risk_control_routes.py"
    },
    "business_orchestrator": {
      "status": "passed",
      "message": "BusinessProcessOrchestrator业务流程编排: 找到 40/1 个必需模式",
      "found_patterns": [
        [
          "BusinessProcessOrchestrator|orchestrator|业务流程|_get_orchestrator",
          40
        ]
      ],
      "found_count": 40,
      "required_count": 1,
      "file": "src/gateway/web/risk_control_routes.py"
    },
    "adapter_factory_usage": {
      "status": "passed",
      "message": "统一适配器工厂使用（风险控制层）: 找到 13/2 个必需模式",
      "found_patterns": [
        [
          "get_unified_adapter_factory|BusinessLayerType\\.RISK|统一适配器工厂",
          13
        ]
      ],
      "found_count": 13,
      "required_count": 2,
      "file": "src/gateway/web/risk_control_service.py"
    },
    "risk_layer_access": {
      "status": "passed",
      "message": "风险控制层组件访问（通过适配器）: 找到 28/2 个必需模式",
      "found_patterns": [
        [
          "风险控制层适配器|risk_adapter|get_risk_manager|get_risk_monitor|get_risk_calculator|get_alert_system",
          28
        ]
      ],
      "found_count": 28,
      "required_count": 2,
      "file": "src/gateway/web/risk_control_service.py"
    }
  },
  "risk_layer_integration": {
    "adapter_factory_usage": {
      "status": "passed",
      "message": "通过统一适配器工厂访问风险控制层: 找到 13/1 个必需模式",
      "found_patterns": [
        [
          "get_unified_adapter_factory|BusinessLayerType\\.RISK|统一适配器工厂",
          13
        ]
      ],
      "found_count": 13,
      "required_count": 1,
      "file": "src/gateway/web/risk_control_service.py"
    },
    "risk_adapter_access": {
      "status": "passed",
      "message": "风险控制层适配器获取（通过统一适配器工厂）: 找到 21/1 个必需模式",
      "found_patterns": [
        [
          "_get_risk_adapter|risk_adapter|风险控制层适配器",
          21
        ]
      ],
      "found_count": 21,
      "required_count": 1,
      "file": "src/gateway/web/risk_control_service.py"
    },
    "risk_components_usage": {
      "status": "passed",
      "message": "风险控制层组件使用（RiskManager, RealTimeRiskMonitor, RiskCalculationEngine, AlertSystem）: 找到 5/2 个必需模式",
      "found_patterns": [
        [
          "adapter\\.get_risk_manager|adapter\\.get_risk_monitor|adapter\\.get_risk_calculator|adapter\\.get_alert_system",
          5
        ]
      ],
      "found_count": 5,
      "required_count": 2,
      "file": "src/gateway/web/risk_control_service.py"
    }
  },
  "websocket_integration": {
    "websocket_endpoint": {
      "status": "warning",
      "message": "WebSocket端点注册（/ws/risk-control）: 仅找到 0/1 个必需模式",
      "found_patterns": [],
      "missing_patterns": [
        "/ws/risk-control|websocket_risk_control"
      ],
      "found_count": 0,
      "required_count": 1,
      "file": "src/gateway/web/websocket_routes.py"
    },
    "websocket_manager": {
      "status": "warning",
      "message": "风险控制WebSocket广播实现: 仅找到 0/2 个必需模式",
      "found_patterns": [],
      "missing_patterns": [
        "_broadcast_risk_control|risk_control|get_risk_control_data"
      ],
      "found_count": 0,
      "required_count": 2,
      "file": "src/gateway/web/websocket_manager.py"
    },
    "frontend_websocket": {
      "status": "warning",
      "message": "前端WebSocket消息处理（/ws/risk-control）: 仅找到 0/2 个必需模式",
      "found_patterns": [],
      "missing_patterns": [
        "/ws/risk-control|connectRiskWebSocket|riskWebSocket|onmessage|risk_control_event"
      ],
      "found_count": 0,
      "required_count": 2,
      "file": "web-static/risk-control-monitor.html"
    }
  },
  "business_orchestration": {
    "orchestrator_usage": {
      "status": "passed",
      "message": "BusinessProcessOrchestrator使用: 找到 40/2 个必需模式",
      "found_patterns": [
        [
          "BusinessProcessOrchestrator|orchestrator|业务流程|_get_orchestrator|业务流程编排器",
          40
        ]
      ],
      "found_count": 40,
      "required_count": 2,
      "file": "src/gateway/web/risk_control_routes.py"
    },
    "process_management": {
      "status": "passed",
      "message": "流程状态管理（风险控制流程状态管理）: 找到 89/2 个必需模式",
      "found_patterns": [
        [
          "start_process|update_process_state|process.*state|流程状态|业务流程编排|orchestrator|process_id|RISK_CONTROL",
          89
        ]
      ],
      "found_count": 89,
      "required_count": 2,
      "file": "src/gateway/web/risk_control_routes.py"
    },
    "event_publishing": {
      "status": "passed",
      "message": "风险控制流程事件发布（6个步骤的事件）: 找到 11/4 个必需模式",
      "found_patterns": [
        [
          "EventBus\\.publish|event_bus\\.publish|RISK_ASSESSMENT_COMPLETED|RISK_INTERCEPTED|COMPLIANCE_CHECK_COMPLETED|RISK_REPORT_GENERATED|ALERT_TRIGGERED|ALERT_RESOLVED",
          11
        ]
      ],
      "found_count": 11,
      "required_count": 4,
      "file": "src/gateway/web/risk_control_service.py"
    },
    "state_machine_integration": {
      "status": "passed",
      "message": "流程状态机集成（获取当前状态、状态历史）: 找到 8/2 个必需模式",
      "found_patterns": [
        [
          "get_current_state|state_machine|process_state|state_history|流程状态机",
          8
        ]
      ],
      "found_count": 8,
      "required_count": 2,
      "file": "src/gateway/web/risk_control_service.py"
    }
  },
  "workflow_steps": {
    "step1_realtime_monitoring": {
      "status": "passed",
      "message": "步骤1: 实时监测（Real-time Monitoring）: 找到 16/2 个必需模式",
      "found_patterns": [
        [
          "realtime_monitoring|get_risk_monitor|实时监测|MONITORING",
          16
        ]
      ],
      "found_count": 16,
      "required_count": 2,
      "file": "src/gateway/web/risk_control_service.py"
    },
    "step2_risk_assessment": {
      "status": "passed",
      "message": "步骤2: 风险评估（Risk Assessment）: 找到 13/2 个必需模式",
      "found_patterns": [
        [
          "risk_assessment|RISK_ASSESSMENT_COMPLETED|EventType\\.RISK_ASSESSMENT_COMPLETED|get_risk_calculator|风险评估",
          13
        ]
      ],
      "found_count": 13,
      "required_count": 2,
      "file": "src/gateway/web/risk_control_service.py"
    },
    "step3_risk_intercept": {
      "status": "passed",
      "message": "步骤3: 风险拦截（Risk Intercept）: 找到 12/2 个必需模式",
      "found_patterns": [
        [
          "risk_intercept|RISK_INTERCEPTED|EventType\\.RISK_INTERCEPTED|风险拦截",
          12
        ]
      ],
      "found_count": 12,
      "required_count": 2,
      "file": "src/gateway/web/risk_control_service.py"
    },
    "step4_compliance_check": {
      "status": "passed",
      "message": "步骤4: 合规检查（Compliance Check）: 找到 13/2 个必需模式",
      "found_patterns": [
        [
          "compliance_check|COMPLIANCE_CHECK_COMPLETED|EventType\\.COMPLIANCE_CHECK_COMPLETED|合规检查",
          13
        ]
      ],
      "found_count": 13,
      "required_count": 2,
      "file": "src/gateway/web/risk_control_service.py"
    },
    "step5_risk_report": {
      "status": "passed",
      "message": "步骤5: 风险报告（Risk Report）: 找到 11/2 个必需模式",
      "found_patterns": [
        [
          "risk_report|RISK_REPORT_GENERATED|EventType\\.RISK_REPORT_GENERATED|风险报告",
          11
        ]
      ],
      "found_count": 11,
      "required_count": 2,
      "file": "src/gateway/web/risk_control_service.py"
    },
    "step6_alert_notify": {
      "status": "passed",
      "message": "步骤6: 告警通知（Alert Notify）: 找到 15/3 个必需模式",
      "found_patterns": [
        [
          "alert_notify|ALERT_TRIGGERED|EventType\\.ALERT_TRIGGERED|get_alert_system|告警通知",
          15
        ]
      ],
      "found_count": 15,
      "required_count": 3,
      "file": "src/gateway/web/risk_control_service.py"
    },
    "step_state_mapping": {
      "status": "passed",
      "message": "6个步骤与流程状态的映射关系: 找到 14/4 个必需模式",
      "found_patterns": [
        [
          "step_state_mapping|MONITORING|RISK_ASSESSING|RISK_INTERCEPTING|COMPLIANCE_CHECKING|REPORT_GENERATING|ALERT_NOTIFYING",
          14
        ]
      ],
      "found_count": 14,
      "required_count": 4,
      "file": "src/gateway/web/risk_control_service.py"
    }
  },
  "summary": {
    "total_items": 46,
    "passed": 43,
    "failed": 0,
    "warnings": 3,
    "not_implemented": 0
  }
}
```
