# 量化策略开发流程数据收集功能架构符合性最终复核报告

**复核时间**: 2026-01-10T08:12:12.084702

## 📊 复核摘要

- **总检查项**: 57
- **通过**: 52 ✅
- **失败**: 0 ❌
- **警告**: 5 ⚠️
- **未实现**: 0 📋
- **通过率**: 91.23%

## 1. 前端功能模块检查

### data_sources_config_dashboard ✅

- **文件**: web-static/data-sources-config.html
- **状态**: passed

### crud_operations ✅

- **文件**: N/A
- **状态**: passed
- **消息**: CRUD操作实现: 找到 26/3 个必需模式
- **匹配情况**: 26/3

### websocket_integration ✅

- **文件**: N/A
- **状态**: passed
- **消息**: WebSocket实时更新集成: 找到 30/2 个必需模式
- **匹配情况**: 30/2

### status_monitoring ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 状态监控事件处理: 找到 5/2 个必需模式
- **匹配情况**: 5/2

### data_quality_monitor ✅

- **文件**: web-static/data-quality-monitor.html
- **状态**: passed

### data_performance_monitor ✅

- **文件**: web-static/data-performance-monitor.html
- **状态**: passed

## 2. 后端API端点检查

### data_source_endpoints ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 数据源管理API端点: 找到 6/3 个必需模式
- **匹配情况**: 6/3

### config_manager_usage ✅

- **文件**: N/A
- **状态**: passed
- **消息**: DataSourceConfigManager使用: 找到 8/1 个必需模式
- **匹配情况**: 8/1

### unified_config_integration ✅

- **文件**: N/A
- **状态**: passed
- **消息**: UnifiedConfigManager集成（通过DataSourceConfigManager）: 找到 5/1 个必需模式
- **匹配情况**: 5/1

### event_bus_publish ✅

- **文件**: N/A
- **状态**: passed
- **消息**: EventBus事件发布: 找到 33/1 个必需模式
- **匹配情况**: 33/1

### adapter_pattern_usage ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 适配器模式使用: 找到 34/1 个必需模式
- **匹配情况**: 34/1

### collection_events ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 数据采集事件发布: 找到 7/2 个必需模式
- **匹配情况**: 7/2

### data_layer_access ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 数据层组件访问: 找到 11/1 个必需模式
- **匹配情况**: 11/1

### quality_monitor_api ✅

- **文件**: src/gateway/web/data_management_routes.py
- **状态**: passed

### quality_monitor_integration ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 数据质量监控API集成（通过服务层间接使用UnifiedQualityMonitor）: 找到 16/1 个必需模式
- **匹配情况**: 16/1

### performance_monitor_api ✅

- **文件**: src/gateway/web/data_management_routes.py
- **状态**: passed

### performance_monitor_integration ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 数据性能监控API集成（通过服务层间接使用PerformanceMonitor）: 找到 15/1 个必需模式
- **匹配情况**: 15/1

## 3. 架构符合性检查

### unified_config_usage ✅

- **文件**: N/A
- **状态**: passed
- **消息**: UnifiedConfigManager使用: 找到 5/1 个必需模式
- **匹配情况**: 5/1

### unified_logger_usage ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 统一日志系统使用: 找到 3/1 个必需模式
- **匹配情况**: 3/1

### environment_isolation ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 环境隔离支持: 找到 23/1 个必需模式
- **匹配情况**: 23/1

### config_hot_reload ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 配置热更新: 找到 20/1 个必需模式
- **匹配情况**: 20/1

### event_bus_config_changes ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 数据源配置变更事件发布（使用CONFIG_UPDATED事件类型）: 找到 18/1 个必需模式
- **匹配情况**: 18/1

### event_bus_collection ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 数据采集事件发布: 找到 3/2 个必需模式
- **匹配情况**: 3/2

### service_container ✅

- **文件**: N/A
- **状态**: passed
- **消息**: ServiceContainer依赖注入: 找到 8/1 个必需模式
- **匹配情况**: 8/1

### business_orchestrator ✅

- **文件**: N/A
- **状态**: passed
- **消息**: BusinessProcessOrchestrator使用: 找到 7/1 个必需模式
- **匹配情况**: 7/1

### unified_adapter_access ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 统一适配器访问: 找到 8/1 个必需模式
- **匹配情况**: 8/1

### adapter_pattern ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 数据适配器模式实现: 找到 88/1 个必需模式
- **匹配情况**: 88/1

### adapter_registry ⚠️

- **文件**: N/A
- **状态**: warning
- **消息**: AdapterRegistry使用: 仅找到 0/1 个必需模式
- **匹配情况**: 0/1

### quality_monitor_usage ✅

- **文件**: src/data/quality/unified_quality_monitor.py
- **状态**: passed

### performance_monitor_usage ✅

- **文件**: src/data/monitoring/performance_monitor.py
- **状态**: passed

### data_lake_usage ✅

- **文件**: src/data/lake/data_lake_manager.py
- **状态**: passed

## 4. WebSocket实时更新检查

### config_change_websocket ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 数据源配置变更WebSocket推送: 找到 10/1 个必需模式
- **匹配情况**: 10/1

### websocket_endpoint ✅

- **文件**: N/A
- **状态**: passed
- **消息**: WebSocket端点实现: 找到 2/1 个必需模式
- **匹配情况**: 2/1

### collection_status_websocket ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 数据采集状态WebSocket推送: 找到 1/1 个必需模式
- **匹配情况**: 1/1

### quality_websocket ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 数据质量监控WebSocket端点: 找到 2/1 个必需模式
- **匹配情况**: 2/1

### performance_websocket ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 数据性能监控WebSocket端点: 找到 2/1 个必需模式
- **匹配情况**: 2/1

### frontend_websocket_handling ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 前端WebSocket消息处理: 找到 6/2 个必需模式
- **匹配情况**: 6/2

## 5. 持久化实现检查

### file_persistence ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 文件系统持久化（JSON格式）: 找到 22/2 个必需模式
- **匹配情况**: 22/2

### postgresql_persistence ✅

- **文件**: N/A
- **状态**: passed
- **消息**: PostgreSQL持久化: 找到 22/1 个必需模式
- **匹配情况**: 22/1

### dual_storage ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 双重存储机制: 找到 13/1 个必需模式
- **匹配情况**: 13/1

### collection_record_persistence ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 数据采集记录持久化: 找到 2/1 个必需模式
- **匹配情况**: 2/1

### quality_metrics_persistence ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 数据质量指标持久化: 找到 4/1 个必需模式
- **匹配情况**: 4/1

## 6. 适配器模式使用检查

### adapter_usage ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 适配器工厂使用: 找到 26/1 个必需模式
- **匹配情况**: 26/1

### data_layer_adapter ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 数据层适配器使用: 找到 11/1 个必需模式
- **匹配情况**: 11/1

### adapter_registration ⚠️

- **文件**: N/A
- **状态**: warning
- **消息**: 适配器注册: 仅找到 0/1 个必需模式
- **匹配情况**: 0/1

### adapter_selection ⚠️

- **文件**: N/A
- **状态**: warning
- **消息**: 适配器选择逻辑: 仅找到 0/1 个必需模式
- **匹配情况**: 0/1

### unified_adapter_factory ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 统一适配器工厂获取: 找到 4/1 个必需模式
- **匹配情况**: 4/1

### business_layer_type ✅

- **文件**: N/A
- **状态**: passed
- **消息**: BusinessLayerType.DATA使用: 找到 2/1 个必需模式
- **匹配情况**: 2/1

### fallback_mechanism ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 降级服务机制: 找到 38/1 个必需模式
- **匹配情况**: 38/1

## 7. 业务流程编排检查

### orchestrator_usage ✅

- **文件**: N/A
- **状态**: passed
- **消息**: BusinessProcessOrchestrator使用: 找到 6/1 个必需模式
- **匹配情况**: 6/1

### process_management ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 流程状态管理: 找到 4/1 个必需模式
- **匹配情况**: 4/1

### process_state_machine ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 流程状态机实现: 找到 8/1 个必需模式
- **匹配情况**: 8/1

### process_metrics ⚠️

- **文件**: N/A
- **状态**: warning
- **消息**: 流程指标收集: 仅找到 0/1 个必需模式
- **匹配情况**: 0/1

### process_exception ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 流程异常处理: 找到 50/1 个必需模式
- **匹配情况**: 50/1

### data_collection_workflow ✅

- **文件**: src/core/orchestration/business_process/data_collection_orchestrator.py
- **状态**: passed

### config_validation ✅

- **文件**: N/A
- **状态**: passed
- **消息**: 配置验证流程: 找到 16/1 个必需模式
- **匹配情况**: 16/1

### config_notification ⚠️

- **文件**: N/A
- **状态**: warning
- **消息**: 配置变更通知: 仅找到 0/1 个必需模式
- **匹配情况**: 0/1

## 📝 详细检查结果

```json
{
  "timestamp": "2026-01-10T08:12:12.084702",
  "frontend_modules": {
    "data_sources_config_dashboard": {
      "file": "web-static/data-sources-config.html",
      "exists": true,
      "status": "passed"
    },
    "crud_operations": {
      "status": "passed",
      "message": "CRUD操作实现: 找到 26/3 个必需模式",
      "found_patterns": [
        [
          "loadDataSources|fetchDataSources",
          18
        ],
        [
          "createDataSource|addDataSource",
          3
        ],
        [
          "updateDataSource|editDataSource",
          2
        ],
        [
          "deleteDataSource|removeDataSource",
          3
        ]
      ],
      "found_count": 26,
      "required_count": 3
    },
    "websocket_integration": {
      "status": "passed",
      "message": "WebSocket实时更新集成: 找到 30/2 个必需模式",
      "found_patterns": [
        [
          "WebSocket|websocket",
          27
        ],
        [
          "handleWebSocketMessage|onmessage",
          3
        ]
      ],
      "found_count": 30,
      "required_count": 2
    },
    "status_monitoring": {
      "status": "passed",
      "message": "状态监控事件处理: 找到 5/2 个必需模式",
      "found_patterns": [
        [
          "data_source_created|data_source_updated|data_source_deleted",
          3
        ],
        [
          "data_collection_started|data_collection_completed",
          2
        ]
      ],
      "found_count": 5,
      "required_count": 2
    },
    "data_quality_monitor": {
      "file": "web-static/data-quality-monitor.html",
      "exists": true,
      "status": "passed"
    },
    "data_performance_monitor": {
      "file": "web-static/data-performance-monitor.html",
      "exists": true,
      "status": "passed"
    }
  },
  "backend_apis": {
    "data_source_endpoints": {
      "status": "passed",
      "message": "数据源管理API端点: 找到 6/3 个必需模式",
      "found_patterns": [
        [
          "@router\\.get\\(.*/api/v1/data/sources|@app\\.get\\(.*/api/v1/data/sources",
          3
        ],
        [
          "@router\\.post\\(.*/api/v1/data/sources|@app\\.post\\(.*/api/v1/data/sources",
          1
        ],
        [
          "@router\\.put\\(.*/api/v1/data/sources|@app\\.put\\(.*/api/v1/data/sources",
          1
        ],
        [
          "@router\\.delete\\(.*/api/v1/data/sources|@app\\.delete\\(.*/api/v1/data/sources",
          1
        ]
      ],
      "found_count": 6,
      "required_count": 3
    },
    "config_manager_usage": {
      "status": "passed",
      "message": "DataSourceConfigManager使用: 找到 8/1 个必需模式",
      "found_patterns": [
        [
          "config_manager|configManager",
          8
        ]
      ],
      "found_count": 8,
      "required_count": 1
    },
    "unified_config_integration": {
      "status": "passed",
      "message": "UnifiedConfigManager集成（通过DataSourceConfigManager）: 找到 5/1 个必需模式",
      "found_patterns": [
        [
          "UnifiedConfigManager",
          5
        ]
      ],
      "found_count": 5,
      "required_count": 1
    },
    "event_bus_publish": {
      "status": "passed",
      "message": "EventBus事件发布: 找到 33/1 个必需模式",
      "found_patterns": [
        [
          "EventBus|event_bus",
          30
        ],
        [
          "\\.publish\\(|publish_event\\(",
          3
        ]
      ],
      "found_count": 33,
      "required_count": 1
    },
    "adapter_pattern_usage": {
      "status": "passed",
      "message": "适配器模式使用: 找到 34/1 个必需模式",
      "found_patterns": [
        [
          "get_unified_adapter_factory|get_adapter|BusinessLayerType\\.DATA",
          10
        ],
        [
          "DataLayerAdapter|adapter_factory",
          24
        ]
      ],
      "found_count": 34,
      "required_count": 1
    },
    "collection_events": {
      "status": "passed",
      "message": "数据采集事件发布: 找到 7/2 个必需模式",
      "found_patterns": [
        [
          "DATA_COLLECTION_STARTED|DATA_COLLECTED|DATA_COLLECTION_PROGRESS",
          7
        ]
      ],
      "found_count": 7,
      "required_count": 2
    },
    "data_layer_access": {
      "status": "passed",
      "message": "数据层组件访问: 找到 11/1 个必需模式",
      "found_patterns": [
        [
          "data_adapter|data_layer|EnhancedDataIntegrationManager",
          11
        ]
      ],
      "found_count": 11,
      "required_count": 1
    },
    "quality_monitor_api": {
      "file": "src/gateway/web/data_management_routes.py",
      "exists": true,
      "status": "passed"
    },
    "quality_monitor_integration": {
      "status": "passed",
      "message": "数据质量监控API集成（通过服务层间接使用UnifiedQualityMonitor）: 找到 16/1 个必需模式",
      "found_patterns": [
        [
          "UnifiedQualityMonitor|get_quality_monitor|quality_monitor",
          16
        ]
      ],
      "found_count": 16,
      "required_count": 1
    },
    "performance_monitor_api": {
      "file": "src/gateway/web/data_management_routes.py",
      "exists": true,
      "status": "passed"
    },
    "performance_monitor_integration": {
      "status": "passed",
      "message": "数据性能监控API集成（通过服务层间接使用PerformanceMonitor）: 找到 15/1 个必需模式",
      "found_patterns": [
        [
          "PerformanceMonitor|get_performance_monitor|performance_monitor",
          15
        ]
      ],
      "found_count": 15,
      "required_count": 1
    }
  },
  "architecture_compliance": {
    "unified_config_usage": {
      "status": "passed",
      "message": "UnifiedConfigManager使用: 找到 5/1 个必需模式",
      "found_patterns": [
        [
          "UnifiedConfigManager",
          5
        ]
      ],
      "found_count": 5,
      "required_count": 1
    },
    "unified_logger_usage": {
      "status": "passed",
      "message": "统一日志系统使用: 找到 3/1 个必需模式",
      "found_patterns": [
        [
          "get_unified_logger",
          3
        ]
      ],
      "found_count": 3,
      "required_count": 1
    },
    "environment_isolation": {
      "status": "passed",
      "message": "环境隔离支持: 找到 23/1 个必需模式",
      "found_patterns": [
        [
          "RQA_ENV|production|development|testing",
          12
        ],
        [
          "env\\s*==|environment",
          11
        ]
      ],
      "found_count": 23,
      "required_count": 1
    },
    "config_hot_reload": {
      "status": "passed",
      "message": "配置热更新: 找到 20/1 个必需模式",
      "found_patterns": [
        [
          "load_config|reload_config|auto_save|hot.*reload",
          20
        ]
      ],
      "found_count": 20,
      "required_count": 1
    },
    "event_bus_config_changes": {
      "status": "passed",
      "message": "数据源配置变更事件发布（使用CONFIG_UPDATED事件类型）: 找到 18/1 个必需模式",
      "found_patterns": [
        [
          "CONFIG_UPDATED|config.*updated|data_source.*created|data_source.*updated|data_source.*deleted",
          18
        ]
      ],
      "found_count": 18,
      "required_count": 1
    },
    "event_bus_collection": {
      "status": "passed",
      "message": "数据采集事件发布: 找到 3/2 个必需模式",
      "found_patterns": [
        [
          "DATA_COLLECTION_STARTED|DATA_COLLECTED",
          3
        ]
      ],
      "found_count": 3,
      "required_count": 2
    },
    "service_container": {
      "status": "passed",
      "message": "ServiceContainer依赖注入: 找到 8/1 个必需模式",
      "found_patterns": [
        [
          "DependencyContainer|ServiceContainer|container\\.resolve",
          8
        ]
      ],
      "found_count": 8,
      "required_count": 1
    },
    "business_orchestrator": {
      "status": "passed",
      "message": "BusinessProcessOrchestrator使用: 找到 7/1 个必需模式",
      "found_patterns": [
        [
          "BusinessProcessOrchestrator|orchestrator\\.start_process",
          7
        ]
      ],
      "found_count": 7,
      "required_count": 1
    },
    "unified_adapter_access": {
      "status": "passed",
      "message": "统一适配器访问: 找到 8/1 个必需模式",
      "found_patterns": [
        [
          "get_unified_adapter_factory|BusinessLayerType\\.DATA|adapter_factory\\.get_adapter",
          8
        ]
      ],
      "found_count": 8,
      "required_count": 1
    },
    "adapter_pattern": {
      "status": "passed",
      "message": "数据适配器模式实现: 找到 88/1 个必需模式",
      "found_patterns": [
        [
          "adapter|Adapter|get_adapter",
          88
        ]
      ],
      "found_count": 88,
      "required_count": 1
    },
    "adapter_registry": {
      "status": "warning",
      "message": "AdapterRegistry使用: 仅找到 0/1 个必需模式",
      "found_patterns": [],
      "missing_patterns": [
        "AdapterRegistry|register.*adapter|adapter.*registry"
      ],
      "found_count": 0,
      "required_count": 1
    },
    "quality_monitor_usage": {
      "file": "src/data/quality/unified_quality_monitor.py",
      "exists": true,
      "status": "passed"
    },
    "performance_monitor_usage": {
      "file": "src/data/monitoring/performance_monitor.py",
      "exists": true,
      "status": "passed"
    },
    "data_lake_usage": {
      "file": "src/data/lake/data_lake_manager.py",
      "exists": true,
      "status": "passed"
    }
  },
  "websocket_integration": {
    "config_change_websocket": {
      "status": "passed",
      "message": "数据源配置变更WebSocket推送: 找到 10/1 个必需模式",
      "found_patterns": [
        [
          "websocket_manager\\.broadcast|WebSocket.*broadcast",
          1
        ],
        [
          "data_source_created|data_source_updated|data_source_deleted",
          9
        ]
      ],
      "found_count": 10,
      "required_count": 1
    },
    "websocket_endpoint": {
      "status": "passed",
      "message": "WebSocket端点实现: 找到 2/1 个必需模式",
      "found_patterns": [
        [
          "@app\\.websocket\\(.*/ws/data-sources|websocket_data_sources",
          2
        ]
      ],
      "found_count": 2,
      "required_count": 1
    },
    "collection_status_websocket": {
      "status": "passed",
      "message": "数据采集状态WebSocket推送: 找到 1/1 个必需模式",
      "found_patterns": [
        [
          "websocket_manager\\.broadcast|WebSocket.*broadcast",
          1
        ]
      ],
      "found_count": 1,
      "required_count": 1
    },
    "quality_websocket": {
      "status": "passed",
      "message": "数据质量监控WebSocket端点: 找到 2/1 个必需模式",
      "found_patterns": [
        [
          "/ws/data-quality|websocket_data_quality",
          2
        ]
      ],
      "found_count": 2,
      "required_count": 1
    },
    "performance_websocket": {
      "status": "passed",
      "message": "数据性能监控WebSocket端点: 找到 2/1 个必需模式",
      "found_patterns": [
        [
          "/ws/data-performance|websocket_data_performance",
          2
        ]
      ],
      "found_count": 2,
      "required_count": 1
    },
    "frontend_websocket_handling": {
      "status": "passed",
      "message": "前端WebSocket消息处理: 找到 6/2 个必需模式",
      "found_patterns": [
        [
          "handleWebSocketMessage|onmessage",
          3
        ],
        [
          "data_source_created|data_source_updated|data_collection_started",
          3
        ]
      ],
      "found_count": 6,
      "required_count": 2
    }
  },
  "persistence": {
    "file_persistence": {
      "status": "passed",
      "message": "文件系统持久化（JSON格式）: 找到 22/2 个必需模式",
      "found_patterns": [
        [
          "save_config|save.*json|json\\.dump",
          9
        ],
        [
          "load_config|load.*json|json\\.load",
          13
        ]
      ],
      "found_count": 22,
      "required_count": 2
    },
    "postgresql_persistence": {
      "status": "passed",
      "message": "PostgreSQL持久化: 找到 22/1 个必需模式",
      "found_patterns": [
        [
          "_load_from_postgresql|_save_to_postgresql|PostgreSQL|postgresql",
          22
        ]
      ],
      "found_count": 22,
      "required_count": 1
    },
    "dual_storage": {
      "status": "passed",
      "message": "双重存储机制: 找到 13/1 个必需模式",
      "found_patterns": [
        [
          "_load_from_postgresql|_load_from_file|fallback|backup",
          13
        ]
      ],
      "found_count": 13,
      "required_count": 1
    },
    "collection_record_persistence": {
      "status": "passed",
      "message": "数据采集记录持久化: 找到 2/1 个必需模式",
      "found_patterns": [
        [
          "store.*data|persist.*data|save.*collection|postgresql_persistence",
          2
        ]
      ],
      "found_count": 2,
      "required_count": 1
    },
    "quality_metrics_persistence": {
      "status": "passed",
      "message": "数据质量指标持久化: 找到 4/1 个必需模式",
      "found_patterns": [
        [
          "save.*metrics|persist.*quality|store.*history|history_data",
          4
        ]
      ],
      "found_count": 4,
      "required_count": 1
    }
  },
  "adapter_pattern": {
    "adapter_usage": {
      "status": "passed",
      "message": "适配器工厂使用: 找到 26/1 个必需模式",
      "found_patterns": [
        [
          "get_unified_adapter_factory|adapter_factory|get_adapter",
          26
        ]
      ],
      "found_count": 26,
      "required_count": 1
    },
    "data_layer_adapter": {
      "status": "passed",
      "message": "数据层适配器使用: 找到 11/1 个必需模式",
      "found_patterns": [
        [
          "BusinessLayerType\\.DATA|data_adapter|DataLayerAdapter",
          11
        ]
      ],
      "found_count": 11,
      "required_count": 1
    },
    "adapter_registration": {
      "status": "warning",
      "message": "适配器注册: 仅找到 0/1 个必需模式",
      "found_patterns": [],
      "missing_patterns": [
        "register.*adapter|AdapterRegistry|register_source"
      ],
      "found_count": 0,
      "required_count": 1
    },
    "adapter_selection": {
      "status": "warning",
      "message": "适配器选择逻辑: 仅找到 0/1 个必需模式",
      "found_patterns": [],
      "missing_patterns": [
        "select.*adapter|choose.*adapter|get_adapter.*source"
      ],
      "found_count": 0,
      "required_count": 1
    },
    "unified_adapter_factory": {
      "status": "passed",
      "message": "统一适配器工厂获取: 找到 4/1 个必需模式",
      "found_patterns": [
        [
          "get_unified_adapter_factory",
          4
        ]
      ],
      "found_count": 4,
      "required_count": 1
    },
    "business_layer_type": {
      "status": "passed",
      "message": "BusinessLayerType.DATA使用: 找到 2/1 个必需模式",
      "found_patterns": [
        [
          "BusinessLayerType\\.DATA",
          2
        ]
      ],
      "found_count": 2,
      "required_count": 1
    },
    "fallback_mechanism": {
      "status": "passed",
      "message": "降级服务机制: 找到 38/1 个必需模式",
      "found_patterns": [
        [
          "fallback|except.*ImportError|降级|可选|optional",
          38
        ]
      ],
      "found_count": 38,
      "required_count": 1
    }
  },
  "business_orchestration": {
    "orchestrator_usage": {
      "status": "passed",
      "message": "BusinessProcessOrchestrator使用: 找到 6/1 个必需模式",
      "found_patterns": [
        [
          "BusinessProcessOrchestrator",
          6
        ]
      ],
      "found_count": 6,
      "required_count": 1
    },
    "process_management": {
      "status": "passed",
      "message": "流程状态管理: 找到 4/1 个必需模式",
      "found_patterns": [
        [
          "start_process|update_process_state|orchestrator\\.start",
          4
        ]
      ],
      "found_count": 4,
      "required_count": 1
    },
    "process_state_machine": {
      "status": "passed",
      "message": "流程状态机实现: 找到 8/1 个必需模式",
      "found_patterns": [
        [
          "BusinessProcessState|ProcessState|StateMachine",
          8
        ]
      ],
      "found_count": 8,
      "required_count": 1
    },
    "process_metrics": {
      "status": "warning",
      "message": "流程指标收集: 仅找到 0/1 个必需模式",
      "found_patterns": [],
      "missing_patterns": [
        "process.*metrics|orchestrator.*metrics|collect.*metrics"
      ],
      "found_count": 0,
      "required_count": 1
    },
    "process_exception": {
      "status": "passed",
      "message": "流程异常处理: 找到 50/1 个必需模式",
      "found_patterns": [
        [
          "except.*Exception|try.*except|error.*handling|异常处理",
          50
        ]
      ],
      "found_count": 50,
      "required_count": 1
    },
    "data_collection_workflow": {
      "file": "src/core/orchestration/business_process/data_collection_orchestrator.py",
      "exists": true,
      "status": "passed"
    },
    "config_validation": {
      "status": "passed",
      "message": "配置验证流程: 找到 16/1 个必需模式",
      "found_patterns": [
        [
          "_validate|validate.*config|validation",
          16
        ]
      ],
      "found_count": 16,
      "required_count": 1
    },
    "config_notification": {
      "status": "warning",
      "message": "配置变更通知: 仅找到 0/1 个必需模式",
      "found_patterns": [],
      "missing_patterns": [
        "notify|broadcast|event.*publish|通知"
      ],
      "found_count": 0,
      "required_count": 1
    }
  },
  "summary": {
    "total_items": 57,
    "passed": 52,
    "failed": 0,
    "warnings": 5,
    "not_implemented": 0
  }
}
```
