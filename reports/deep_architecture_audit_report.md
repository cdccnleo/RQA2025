# 深度架构审计报告

## 📊 审计概览

**审计时间**: 2025-08-23T21:31:58.652179
**审计范围**: src目录深度分析
**发现问题**: 646 个
**质量评分**: 0.0/100

### 审计维度
- **层级职责审计**: 检查各层级文件内容是否符合架构职责
- **内容匹配分析**: 分析文件职责匹配度
- **接口规范检查**: 验证接口设计标准性
- **依赖关系分析**: 检查导入依赖的合理性

---

## 🏗️ 层级职责审计结果

### CORE 层级
**文件数量**: 23 个
**职责匹配度**: 60 个关键词匹配
**职责违规数**: 57 个违规项

**发现问题**:
- 🔴 core\architecture_demo.py: 包含违规内容: trading, risk, ml, feature, data
- 🔴 core\architecture_layers.py: 包含违规内容: trading, risk, ml, feature, data
- 🟡 core\base.py: 包含违规内容: data
- 🔴 core\business_process_demo.py: 包含违规内容: trading, risk, feature, data
- 🔴 core\business_process_integration.py: 包含违规内容: trading, risk, feature, data
- 🔴 core\business_process_orchestrator.py: 包含违规内容: trading, risk, feature, data
- 🟡 core\container.py: 包含违规内容: data
- 🔴 core\event_bus.py: 包含违规内容: trading, risk, feature, data
- 🔴 core\layer_interfaces.py: 包含违规内容: trading, risk, feature, data
- 🟡 core\process_config_loader.py: 包含违规内容: ml, data
- 🟡 core\service_container.py: 包含违规内容: data
- 🟡 core\integration\data.py: 包含违规内容: data
- 🟡 core\integration\deployment.py: 包含违规内容: feature, data
- 🔴 core\integration\interface.py: 包含违规内容: trading, risk, feature, data
- 🟡 core\integration\layer_interface.py: 包含违规内容: data
- 🟡 core\integration\system_integration_manager.py: 包含违规内容: data
- 🔴 core\integration\testing.py: 包含违规内容: trading, risk, feature, data
- 🔴 core\optimizations\long_term_optimizations.py: 包含违规内容: trading, risk, ml, feature, data
- 🟡 core\optimizations\medium_term_optimizations.py: 包含违规内容: data
- 🟡 core\optimizations\optimization_implementer.py: 包含违规内容: data
- 🟡 core\optimizations\short_term_optimizations.py: 包含违规内容: trading, data

### INFRASTRUCTURE 层级
**文件数量**: 344 个
**职责匹配度**: 980 个关键词匹配
**职责违规数**: 136 个违规项

**发现问题**:
- 🟡 infrastructure\degradation_manager.py: 包含违规内容: trading
- 🟡 infrastructure\inference_engine.py: 包含违规内容: model
- 🟡 infrastructure\service_launcher.py: 包含违规内容: trading, feature
- 🟡 infrastructure\cache\cache_factory.py: 包含违规内容: strategy
- 🟡 infrastructure\cache\cache_performance_tester.py: 包含违规内容: strategy
- 🟡 infrastructure\cache\cache_utils.py: 包含违规内容: model, feature
- 🟡 infrastructure\cache\caching.py: 包含违规内容: strategy
- 🟡 infrastructure\cache\multi_level_cache.py: 包含违规内容: strategy
- 🟡 infrastructure\cache\quota_manager.py: 包含违规内容: strategy
- 🟡 infrastructure\cache\smart_cache_strategy.py: 包含违规内容: strategy
- 🟡 infrastructure\cache\unified_cache.py: 包含违规内容: strategy
- 🟡 infrastructure\cache\unified_cache_factory.py: 包含违规内容: strategy
- 🔴 infrastructure\config\ai_optimization_enhanced.py: 包含违规内容: strategy, model, feature
- 🔴 infrastructure\config\ai_test_optimizer.py: 包含违规内容: strategy, model, feature
- 🟡 infrastructure\config\app.py: 包含违规内容: strategy, model
- 🟡 infrastructure\config\async_optimizer.py: 包含违规内容: strategy
- 🟡 infrastructure\config\circuit_breaker_manager.py: 包含违规内容: trading
- 🟡 infrastructure\config\configuration.py: 包含违规内容: strategy
- 🟡 infrastructure\config\config_example.py: 包含违规内容: trading
- 🟡 infrastructure\config\config_exceptions.py: 包含违规内容: trading
- 🟡 infrastructure\config\config_loader_service.py: 包含违规内容: strategy
- 🟡 infrastructure\config\config_service.py: 包含违规内容: strategy
- 🟡 infrastructure\config\config_strategy.py: 包含违规内容: strategy
- 🟡 infrastructure\config\config_sync_service.py: 包含违规内容: strategy
- 🟡 infrastructure\config\core.py: 包含违规内容: trading
- 🟡 infrastructure\config\data_api.py: 包含违规内容: model
- 🟡 infrastructure\config\dependency.py: 包含违规内容: trading
- 🟡 infrastructure\config\deployment.py: 包含违规内容: trading
- 🟡 infrastructure\config\deployment_validator.py: 包含违规内容: trading, feature
- 🟡 infrastructure\config\distributed_test_runner.py: 包含违规内容: strategy
- 🟡 infrastructure\config\edge_computing_test_platform.py: 包含违规内容: strategy
- 🟡 infrastructure\config\env_loader.py: 包含违规内容: strategy
- 🟡 infrastructure\config\hybrid_loader.py: 包含违规内容: strategy
- 🟡 infrastructure\config\infrastructure_index.py: 包含违规内容: model
- 🟡 infrastructure\config\integration.py: 包含违规内容: trading
- 🟡 infrastructure\config\json_loader.py: 包含违规内容: strategy
- 🟡 infrastructure\config\microservice_manager.py: 包含违规内容: strategy
- 🟡 infrastructure\config\optimization_strategies.py: 包含违规内容: strategy
- 🟡 infrastructure\config\optimized_components.py: 包含违规内容: trading
- 🟡 infrastructure\config\paths.py: 包含违规内容: model
- 🟡 infrastructure\config\performance_config.py: 包含违规内容: strategy, feature
- 🟡 infrastructure\config\regulatory_tester.py: 包含违规内容: trading
- 🟡 infrastructure\config\report_generator.py: 包含违规内容: trading
- 🔴 infrastructure\config\standard_interfaces.py: 包含违规内容: trading, strategy, model, feature
- 🟡 infrastructure\config\strategy.py: 包含违规内容: strategy
- 🟡 infrastructure\config\sync_conflict_manager.py: 包含违规内容: strategy
- 🟡 infrastructure\config\unified_interface.py: 包含违规内容: model
- 🟡 infrastructure\config\unified_loaders.py: 包含违规内容: strategy
- 🟡 infrastructure\config\unified_manager.py: 包含违规内容: strategy
- 🟡 infrastructure\config\unified_service.py: 包含违规内容: strategy
- 🟡 infrastructure\config\unified_strategy.py: 包含违规内容: strategy
- 🟡 infrastructure\config\unified_sync.py: 包含违规内容: strategy
- 🟡 infrastructure\config\unified_sync_service.py: 包含违规内容: strategy
- 🟡 infrastructure\config\validators.py: 包含违规内容: trading
- 🟡 infrastructure\config\web_management_service.py: 包含违规内容: strategy
- 🟡 infrastructure\config\yaml_loader.py: 包含违规内容: strategy
- 🟡 infrastructure\error\archive_failure_handler.py: 包含违规内容: strategy
- 🔴 infrastructure\error\error_codes_utils.py: 包含违规内容: trading, strategy, model
- 🟡 infrastructure\error\error_exceptions.py: 包含违规内容: trading
- 🟡 infrastructure\error\error_handler.py: 包含违规内容: trading, strategy
- 🟡 infrastructure\error\market_aware_retry.py: 包含违规内容: trading
- 🟡 infrastructure\error\retry_handler.py: 包含违规内容: strategy
- 🟡 infrastructure\error\retry_policy.py: 包含违规内容: strategy
- 🟡 infrastructure\error\trading_error_handler.py: 包含违规内容: trading, strategy
- 🟡 infrastructure\health\basic_health_checker.py: 包含违规内容: model
- 🟡 infrastructure\health\enhanced_health_checker.py: 包含违规内容: model
- 🟡 infrastructure\health\fastapi_health_checker.py: 包含违规内容: model
- 🟡 infrastructure\logging\circuit_breaker.py: 包含违规内容: trading
- 🟡 infrastructure\logging\load_balancer.py: 包含违规内容: strategy
- 🟡 infrastructure\logging\logging_strategy.py: 包含违规内容: trading, strategy
- 🟡 infrastructure\logging\log_aggregator_plugin.py: 包含违规内容: trading, feature
- 🟡 infrastructure\logging\log_backpressure_plugin.py: 包含违规内容: trading
- 🟡 infrastructure\logging\log_compressor_plugin.py: 包含违规内容: trading, strategy
- 🟡 infrastructure\logging\log_sampler.py: 包含违规内容: trading, strategy
- 🟡 infrastructure\logging\log_sampler_plugin.py: 包含违规内容: trading, strategy
- 🟡 infrastructure\logging\market_data_logger.py: 包含违规内容: trading
- 🟡 infrastructure\logging\network_manager.py: 包含违规内容: strategy
- 🟡 infrastructure\logging\trading_logger.py: 包含违规内容: trading
- 🟡 infrastructure\logging\unified_logger.py: 包含违规内容: trading
- 🟡 infrastructure\resource\backtest_monitor_plugin.py: 包含违规内容: strategy
- 🟡 infrastructure\resource\behavior_monitor_plugin.py: 包含违规内容: trading
- 🔴 infrastructure\resource\business_metrics_monitor.py: 包含违规内容: trading, strategy, model
- 🟡 infrastructure\resource\business_metrics_plugin.py: 包含违规内容: strategy
- 🟡 infrastructure\resource\model_monitor_plugin.py: 包含违规内容: model, feature
- 🟡 infrastructure\resource\performance.py: 包含违规内容: trading, strategy
- 🟡 infrastructure\resource\performance_monitor.py: 包含违规内容: trading, model
- 🟡 infrastructure\resource\performance_optimizer.py: 包含违规内容: strategy
- 🟡 infrastructure\resource\performance_optimizer_manager.py: 包含违规内容: strategy
- 🟡 infrastructure\resource\resource_api.py: 包含违规内容: strategy
- 🟡 infrastructure\resource\resource_dashboard.py: 包含违规内容: strategy
- 🔴 infrastructure\services\api_service.py: 包含违规内容: trading, strategy, model
- 🔴 infrastructure\services\business_service.py: 包含违规内容: trading, strategy, model, feature
- 🟡 infrastructure\services\cache_service.py: 包含违规内容: strategy
- 🟡 infrastructure\services\micro_service.py: 包含违规内容: trading, model
- 🟡 infrastructure\services\model_service.py: 包含违规内容: model, feature
- 🔴 infrastructure\services\trading_service.py: 包含违规内容: trading, strategy, feature
- 🟡 infrastructure\utils\date_utils.py: 包含违规内容: trading
- 🟡 infrastructure\utils.backup_20250823_212847\date_utils.py: 包含违规内容: trading
- 🟡 infrastructure\utils.backup_20250823_212847\enhanced_database_manager.py: 包含违规内容: strategy
- 🟡 infrastructure\utils.backup_20250823_212847\tools.py: 包含违规内容: model, feature

### DATA 层级
**文件数量**: 123 个
**职责匹配度**: 148 个关键词匹配
**职责违规数**: 73 个违规项

**发现问题**:
- 🟡 data\api.py: 包含违规内容: model
- 🟡 data\base_adapter.py: 包含违规内容: model
- 🟡 data\data_manager.py: 包含违规内容: model
- 🟡 data\interfaces.py: 包含违规内容: model
- 🟡 data\models.py: 包含违规内容: model
- 🟡 data\validator.py: 包含违规内容: model
- 🟡 data\adapters\miniqmt\adapter.py: 包含违规内容: strategy
- 🟡 data\adapters\miniqmt\local_cache.py: 包含违规内容: strategy
- 🟡 data\adapters\miniqmt\miniqmt_trade_adapter.py: 包含违规内容: strategy
- 🟡 data\adapters\miniqmt\rate_limiter.py: 包含违规内容: strategy
- 🟡 data\adapters.backup_20250823_212847\generic_china_data_adapter.py: 包含违规内容: trading
- 🟡 data\adapters.backup_20250823_212847\china\adapter.py: 包含违规内容: trading, model
- 🟡 data\adapters.backup_20250823_212847\china\dragon_board.py: 包含违规内容: model
- 🟡 data\adapters.backup_20250823_212847\china\financial_adapter.py: 包含违规内容: model
- 🟡 data\adapters.backup_20250823_212847\china\index_adapter.py: 包含违规内容: model
- 🟡 data\adapters.backup_20250823_212847\china\margin_trading.py: 包含违规内容: trading, model
- 🟡 data\adapters.backup_20250823_212847\china\news_adapter.py: 包含违规内容: model
- 🟡 data\adapters.backup_20250823_212847\china\sentiment_adapter.py: 包含违规内容: model
- 🟡 data\adapters.backup_20250823_212847\china\stock_adapter.py: 包含违规内容: trading, model
- 🟡 data\adapters.backup_20250823_212847\crypto\ccxt_mock_adapter.py: 包含违规内容: trading
- 🟡 data\cache\cache_manager.py: 包含违规内容: strategy
- 🟡 data\cache\enhanced_cache_strategy.py: 包含违规内容: strategy
- 🟡 data\cache\lfu_strategy.py: 包含违规内容: strategy
- 🟡 data\china\adapter.py: 包含违规内容: trading
- 🟡 data\china\adapters.py: 包含违规内容: trading
- 🟡 data\china\dragon_board_updater.py: 包含违规内容: feature
- 🟡 data\china\market.py: 包含违规内容: trading
- 🟡 data\core\data_model.py: 包含违规内容: model
- 🟡 data\core\models.py: 包含违规内容: model
- 🟡 data\distributed\distributed_data_loader.py: 包含违规内容: strategy, model
- 🟡 data\distributed\load_balancer.py: 包含违规内容: strategy
- 🟡 data\distributed\sharding_manager.py: 包含违规内容: strategy
- 🟡 data\edge\edge_node.py: 包含违规内容: model
- 🟡 data\export\data_exporter.py: 包含违规内容: model
- 🟡 data\integration\enhanced_data_integration.py: 包含违规内容: strategy, feature
- 🟡 data\interfaces\IDataModel.py: 包含违规内容: model
- 🟡 data\lake\data_lake_manager.py: 包含违规内容: strategy
- 🟡 data\lake\partition_manager.py: 包含违规内容: strategy
- 🟡 data\loader\enhanced_data_loader.py: 包含违规内容: model
- 🟡 data\loader\stock_loader.py: 包含违规内容: trading, feature
- 🟡 data\ml\quality_assessor.py: 包含违规内容: feature
- 🟡 data\monitoring\quality_monitor.py: 包含违规内容: model
- 🟡 data\optimization\advanced_optimizer.py: 包含违规内容: strategy, model
- 🟡 data\optimization\data_optimizer.py: 包含违规内容: model
- 🟡 data\optimization\data_preloader.py: 包含违规内容: strategy, model
- 🟡 data\optimization\performance_optimizer.py: 包含违规内容: strategy
- 🟡 data\parallel\enhanced_parallel_loader.py: 包含违规内容: model
- 🟡 data\parallel\parallel_loader.py: 包含违规内容: model
- 🟡 data\preload\preloader.py: 包含违规内容: model
- 🟡 data\processing\data_processor.py: 包含违规内容: model
- 🟡 data\processing\unified_processor.py: 包含违规内容: model
- 🟡 data\quality\advanced_quality_monitor.py: 包含违规内容: model
- 🟡 data\quality\data_quality_monitor.py: 包含违规内容: model
- 🟡 data\quality\enhanced_quality_monitor_v2.py: 包含违规内容: strategy
- 🔴 data\quantum\quantum_circuit.py: 包含违规内容: trading, strategy, feature
- 🟡 data\repair\data_repairer.py: 包含违规内容: strategy, model
- 🟡 data\sources\intelligent_source_manager.py: 包含违规内容: model
- 🟡 data\sync\multi_market_sync.py: 包含违规内容: trading
- 🟡 data\transformers\data_transformer.py: 包含违规内容: feature
- 🟡 data\validation\china_stock_validator.py: 包含违规内容: model
- 🟡 data\version_control\test_version_manager.py: 包含违规内容: model
- 🟡 data\version_control\version_manager.py: 包含违规内容: model

### GATEWAY 层级
**文件数量**: 1 个
**职责匹配度**: 7 个关键词匹配
**职责违规数**: 3 个违规项

**发现问题**:
- 🔴 gateway\api_gateway.py: 包含违规内容: trading, model, feature

### FEATURES 层级
**文件数量**: 90 个
**职责匹配度**: 107 个关键词匹配
**职责违规数**: 33 个违规项

**发现问题**:
- 🟡 features\api.py: 包含违规内容: model
- 🟡 features\config_classes.py: 包含违规内容: model
- 🟡 features\exceptions.py: 包含违规内容: model
- 🟡 features\feature_importance.py: 包含违规内容: model
- 🟡 features\feature_manager.py: 包含违规内容: model
- 🟡 features\sentiment_analyzer.py: 包含违规内容: model
- 🟡 features\acceleration\scalability_enhancer.py: 包含违规内容: strategy
- 🟡 features\acceleration\fpga\fpga_accelerator.py: 包含违规内容: strategy
- 🟡 features\acceleration\fpga\fpga_optimizer.py: 包含违规内容: trading
- 🟡 features\acceleration\fpga\fpga_orderbook_optimizer.py: 包含违规内容: strategy
- 🟡 features\acceleration\fpga\fpga_order_optimizer.py: 包含违规内容: trading, strategy
- 🟡 features\acceleration\gpu\gpu_accelerator.py: 包含违规内容: model
- 🟡 features\acceleration\gpu\gpu_scheduler.py: 包含违规内容: strategy, model
- 🟡 features\core\engine.py: 包含违规内容: model
- 🟡 features\distributed\distributed_processor.py: 包含违规内容: strategy
- 🟡 features\intelligent\auto_feature_selector.py: 包含违规内容: strategy, model
- 🟡 features\intelligent\intelligent_enhancement_manager.py: 包含违规内容: strategy, model
- 🟡 features\intelligent\ml_model_integration.py: 包含违规内容: model
- 🟡 features\monitoring\performance_analyzer.py: 包含违规内容: model
- 🟡 features\performance\performance_optimizer.py: 包含违规内容: strategy
- 🟡 features\performance\scalability_manager.py: 包含违规内容: strategy
- 🟡 features\processors\advanced_feature_selector.py: 包含违规内容: model
- 🟡 features\processors\feature_correlation.py: 包含违规内容: model
- 🟡 features\processors\feature_importance.py: 包含违规内容: model
- 🟡 features\processors\feature_selector.py: 包含违规内容: strategy, model
- 🟡 features\processors\feature_standardizer.py: 包含违规内容: model
- 🟡 features\processors\advanced\advanced_feature_processor.py: 包含违规内容: model
- 🟡 features\sentiment\models\sentiment_model.py: 包含违规内容: model

### ML 层级
**文件数量**: 28 个
**职责匹配度**: 69 个关键词匹配
**职责违规数**: 0 个违规项

### BACKTEST 层级
**文件数量**: 22 个
**职责匹配度**: 46 个关键词匹配
**职责违规数**: 2 个违规项

**发现问题**:
- 🟡 backtest\config_manager.py: 包含违规内容: production
- 🟡 backtest\microservice_architecture.py: 包含违规内容: production

### RISK 层级
**文件数量**: 10 个
**职责匹配度**: 28 个关键词匹配
**职责违规数**: 6 个违规项

**发现问题**:
- 🟡 risk\api.py: 包含违规内容: order
- 🔴 risk\compliance_checker.py: 包含违规内容: trading, order, execution
- 🟡 risk\real_time_monitor.py: 包含违规内容: order
- 🟡 risk\risk_manager.py: 包含违规内容: order

### TRADING 层级
**文件数量**: 99 个
**职责匹配度**: 164 个关键词匹配
**职责违规数**: 15 个违规项

**发现问题**:
- 🟡 trading\backtester.py: 包含违规内容: backtest
- 🟡 trading\backtest_analyzer.py: 包含违规内容: backtest
- 🟡 trading\live_trading.py: 包含违规内容: backtest
- 🟡 trading\strategy_optimizer.py: 包含违规内容: backtest
- 🟡 trading\portfolio\portfolio_manager.py: 包含违规内容: backtest
- 🟡 trading\strategies\base_strategy.py: 包含违规内容: backtest
- 🟡 trading\strategies\enhanced.py: 包含违规内容: backtest
- 🟡 trading\strategies\factory.py: 包含违规内容: backtest
- 🟡 trading\strategies\reinforcement_learning.py: 包含违规内容: backtest
- 🟡 trading\strategies\strategy_auto_optimizer.py: 包含违规内容: backtest
- 🟡 trading\strategies\optimization\genetic_optimizer.py: 包含违规内容: backtest
- 🟡 trading\strategy_workspace\analyzer.py: 包含违规内容: simulation
- 🟡 trading\strategy_workspace\simulator.py: 包含违规内容: backtest, simulation
- 🟡 trading\strategy_workspace\store.py: 包含违规内容: simulation

### ENGINE 层级
**文件数量**: 49 个
**职责匹配度**: 172 个关键词匹配
**职责违规数**: 34 个违规项

**发现问题**:
- 🟡 engine\dispatcher.py: 包含违规内容: strategy
- 🟡 engine\level2.py: 包含违规内容: order
- 🟡 engine\realtime.py: 包含违规内容: order
- 🟡 engine\realtime_engine.py: 包含违规内容: order
- 🟡 engine\stress_test.py: 包含违规内容: order
- 🟡 engine\config\config_schema.py: 包含违规内容: order, model
- 🟡 engine\config\config_validator.py: 包含违规内容: order
- 🟡 engine\config\engine_config_manager.py: 包含违规内容: order, model
- 🟡 engine\inference\optimized_inference_engine.py: 包含违规内容: model
- 🟡 engine\level2\level2_adapter.py: 包含违规内容: trading, order
- 🟡 engine\logging\business_logger.py: 包含违规内容: order
- 🟡 engine\monitoring.backup_20250823_212847\metrics_collector.py: 包含违规内容: order
- 🟡 engine\optimization\dispatcher_optimizer.py: 包含违规内容: order, strategy
- 🟡 engine\optimization\level2_optimizer.py: 包含违规内容: order
- 🟡 engine\production\model_serving.py: 包含违规内容: model
- 🟡 engine\testing\test_data_generator.py: 包含违规内容: order
- 🟡 engine\testing\test_data_manager.py: 包含违规内容: order
- 🟡 engine\testing\test_data_validator.py: 包含违规内容: order
- 🟡 engine\web\data_api.py: 包含违规内容: model
- 🟡 engine\web\unified_dashboard.py: 包含违规内容: strategy, model
- 🟡 engine\web\websocket_api.py: 包含违规内容: order
- 🟡 engine\web\modules\base_module.py: 包含违规内容: model
- 🔴 engine\web\modules\config_module.py: 包含违规内容: trading, order, model
- 🟡 engine\web\modules\features_module.py: 包含违规内容: model
- 🟡 engine\web\modules\fpga_module.py: 包含违规内容: model
- 🟡 engine\web\modules\module_registry.py: 包含违规内容: order
- 🟡 engine\web\modules\resource_module.py: 包含违规内容: model

## 📋 内容职责匹配分析

### CORE 层级匹配统计
- **总文件数**: 23 个
- **高匹配文件**: 4 个 (≥70%)
- **中等匹配文件**: 6 个 (40-70%)
- **低匹配文件**: 10 个 (>0-40%)
- **无匹配文件**: 3 个 (0%)

### INFRASTRUCTURE 层级匹配统计
- **总文件数**: 344 个
- **高匹配文件**: 10 个 (≥70%)
- **中等匹配文件**: 97 个 (40-70%)
- **低匹配文件**: 221 个 (>0-40%)
- **无匹配文件**: 16 个 (0%)

### DATA 层级匹配统计
- **总文件数**: 123 个
- **高匹配文件**: 1 个 (≥70%)
- **中等匹配文件**: 12 个 (40-70%)
- **低匹配文件**: 72 个 (>0-40%)
- **无匹配文件**: 38 个 (0%)

### GATEWAY 层级匹配统计
- **总文件数**: 1 个
- **高匹配文件**: 1 个 (≥70%)
- **中等匹配文件**: 0 个 (40-70%)
- **低匹配文件**: 0 个 (>0-40%)
- **无匹配文件**: 0 个 (0%)

### FEATURES 层级匹配统计
- **总文件数**: 90 个
- **高匹配文件**: 0 个 (≥70%)
- **中等匹配文件**: 5 个 (40-70%)
- **低匹配文件**: 68 个 (>0-40%)
- **无匹配文件**: 17 个 (0%)

### ML 层级匹配统计
- **总文件数**: 28 个
- **高匹配文件**: 1 个 (≥70%)
- **中等匹配文件**: 16 个 (40-70%)
- **低匹配文件**: 11 个 (>0-40%)
- **无匹配文件**: 0 个 (0%)

### BACKTEST 层级匹配统计
- **总文件数**: 22 个
- **高匹配文件**: 0 个 (≥70%)
- **中等匹配文件**: 10 个 (40-70%)
- **低匹配文件**: 10 个 (>0-40%)
- **无匹配文件**: 2 个 (0%)

### RISK 层级匹配统计
- **总文件数**: 10 个
- **高匹配文件**: 1 个 (≥70%)
- **中等匹配文件**: 6 个 (40-70%)
- **低匹配文件**: 1 个 (>0-40%)
- **无匹配文件**: 2 个 (0%)

### TRADING 层级匹配统计
- **总文件数**: 99 个
- **高匹配文件**: 3 个 (≥70%)
- **中等匹配文件**: 26 个 (40-70%)
- **低匹配文件**: 46 个 (>0-40%)
- **无匹配文件**: 24 个 (0%)

### ENGINE 层级匹配统计
- **总文件数**: 49 个
- **高匹配文件**: 15 个 (≥70%)
- **中等匹配文件**: 14 个 (40-70%)
- **低匹配文件**: 20 个 (>0-40%)
- **无匹配文件**: 0 个 (0%)

## 🔗 接口规范检查

### CORE 层级接口
- **接口文件**: 1 个
- **基础实现文件**: 1 个
- **标准接口**: 0 个
- **非标准接口**: 1 个

**接口问题**:
- ⚠️ core\base.py: 基础实现类不符合标准模式
- ⚠️ core\layer_interfaces.py: 接口命名不符合标准规范

### INFRASTRUCTURE 层级接口
- **接口文件**: 10 个
- **基础实现文件**: 11 个
- **标准接口**: 8 个
- **非标准接口**: 2 个

**接口问题**:
- ⚠️ infrastructure\config\standard_interfaces.py: 接口命名不符合标准规范
- ⚠️ infrastructure\config\unified_interfaces.py: 接口命名不符合标准规范
- ⚠️ infrastructure\utils.backup_20250823_212847\base_database.py: 基础实现类不符合标准模式
- ⚠️ infrastructure\utils.backup_20250823_212847\database.py: 基础实现类不符合标准模式
- ⚠️ infrastructure\utils.backup_20250823_212847\unified_database.py: 基础实现类不符合标准模式

### DATA 层级接口
- **接口文件**: 1 个
- **基础实现文件**: 1 个
- **标准接口**: 0 个
- **非标准接口**: 1 个

**接口问题**:
- ⚠️ data\interfaces.py: 接口命名不符合标准规范
- ⚠️ data\adapters.backup_20250823_212847\base.py: 基础实现类不符合标准模式

### ML 层级接口
- **接口文件**: 0 个
- **基础实现文件**: 1 个
- **标准接口**: 0 个
- **非标准接口**: 0 个

**接口问题**:
- ⚠️ ml\tuning\optimizers\base.py: 基础实现类不符合标准模式

## ⚡ 依赖关系分析

### CORE 层级依赖
- **内部导入**: 11 个
- **外部导入**: 213 个
- **跨层级导入**: 1 个

**依赖问题**:
- ⚠️ core\architecture_layers.py: 跨层级导入

### INFRASTRUCTURE 层级依赖
- **内部导入**: 75 个
- **外部导入**: 2397 个
- **跨层级导入**: 40 个

**依赖问题**:
- ⚠️ infrastructure\data_sync.py: 跨层级导入
- ⚠️ infrastructure\degradation_manager.py: 跨层级导入
- ⚠️ infrastructure\disaster_recovery.py: 跨层级导入
- ⚠️ infrastructure\final_deployment_check.py: 跨层级导入
- ⚠️ infrastructure\service_launcher.py: 跨层级导入
- ⚠️ infrastructure\config\alert_manager.py: 跨层级导入
- ⚠️ infrastructure\config\data_api.py: 跨层级导入
- ⚠️ infrastructure\config\data_api.py: 跨层级导入
- ⚠️ infrastructure\config\data_api.py: 跨层级导入
- ⚠️ infrastructure\config\data_api.py: 跨层级导入
- ⚠️ infrastructure\config\deployment.py: 跨层级导入
- ⚠️ infrastructure\config\deployment_validator.py: 跨层级导入
- ⚠️ infrastructure\config\disaster_tester.py: 跨层级导入
- ⚠️ infrastructure\config\paths.py: 跨层级导入
- ⚠️ infrastructure\config\regulatory_tester.py: 跨层级导入
- ⚠️ infrastructure\config\regulatory_tester.py: 跨层级导入
- ⚠️ infrastructure\config\regulatory_tester.py: 跨层级导入
- ⚠️ infrastructure\config\report_generator.py: 跨层级导入
- ⚠️ infrastructure\config\report_generator.py: 跨层级导入
- ⚠️ infrastructure\config\report_generator.py: 跨层级导入
- ⚠️ infrastructure\config\report_generator.py: 跨层级导入
- ⚠️ infrastructure\config\unified_core.py: 跨层级导入
- ⚠️ infrastructure\config\unified_query.py: 跨层级导入
- ⚠️ infrastructure\config\websocket_api.py: 跨层级导入
- ⚠️ infrastructure\config\websocket_api.py: 跨层级导入
- ⚠️ infrastructure\config\websocket_api.py: 跨层级导入
- ⚠️ infrastructure\config\websocket_api.py: 跨层级导入
- ⚠️ infrastructure\config\websocket_api.py: 跨层级导入
- ⚠️ infrastructure\resource\behavior_monitor_plugin.py: 跨层级导入
- ⚠️ infrastructure\resource\disaster_monitor_plugin.py: 跨层级导入
- ⚠️ infrastructure\resource\performance_monitor.py: 跨层级导入
- ⚠️ infrastructure\services\api_service.py: 跨层级导入
- ⚠️ infrastructure\services\api_service.py: 跨层级导入
- ⚠️ infrastructure\services\business_service.py: 跨层级导入
- ⚠️ infrastructure\services\cache_service.py: 跨层级导入
- ⚠️ infrastructure\services\cache_service.py: 跨层级导入
- ⚠️ infrastructure\services\data_validation_service.py: 跨层级导入
- ⚠️ infrastructure\services\micro_service.py: 跨层级导入
- ⚠️ infrastructure\services\micro_service.py: 跨层级导入
- ⚠️ infrastructure\services\trading_service.py: 跨层级导入

### DATA 层级依赖
- **内部导入**: 70 个
- **外部导入**: 872 个
- **跨层级导入**: 97 个

**依赖问题**:
- ⚠️ data\api.py: 跨层级导入
- ⚠️ data\backup_recovery.py: 跨层级导入
- ⚠️ data\base_adapter.py: 跨层级导入
- ⚠️ data\data_manager.py: 跨层级导入
- ⚠️ data\data_manager.py: 跨层级导入
- ⚠️ data\data_manager.py: 跨层级导入
- ⚠️ data\data_manager.py: 跨层级导入
- ⚠️ data\enhanced_integration_manager.py: 跨层级导入
- ⚠️ data\market_data.py: 跨层级导入
- ⚠️ data\registry.py: 跨层级导入
- ⚠️ data\adapters\miniqmt\adapter.py: 跨层级导入
- ⚠️ data\adapters\miniqmt\adapter.py: 跨层级导入
- ⚠️ data\adapters\miniqmt\adapter.py: 跨层级导入
- ⚠️ data\adapters\miniqmt\adapter.py: 跨层级导入
- ⚠️ data\adapters\miniqmt\adapter.py: 跨层级导入
- ⚠️ data\adapters\miniqmt\miniqmt_data_adapter.py: 跨层级导入
- ⚠️ data\adapters\miniqmt\miniqmt_trade_adapter.py: 跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\adapter_registry.py: 跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\base_adapter.py: 跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\generic_china_data_adapter.py: 跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\china\adapter.py: 跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\china\financial_adapter.py: 跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\china\index_adapter.py: 跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\china\news_adapter.py: 跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\china\sentiment_adapter.py: 跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\crypto\crypto_adapter.py: 跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\crypto\crypto_adapter.py: 跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\international\international_stock_adapter.py: 跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\macro\macro_economic_adapter.py: 跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\news\news_sentiment_adapter.py: 跨层级导入
- ⚠️ data\alignment\data_aligner.py: 跨层级导入
- ⚠️ data\alignment\data_aligner.py: 跨层级导入
- ⚠️ data\cache\cache_manager.py: 跨层级导入
- ⚠️ data\cache\cache_manager.py: 跨层级导入
- ⚠️ data\cache\disk_cache.py: 跨层级导入
- ⚠️ data\cache\disk_cache.py: 跨层级导入
- ⚠️ data\cache\enhanced_cache_manager.py: 跨层级导入
- ⚠️ data\cache\enhanced_cache_strategy.py: 跨层级导入
- ⚠️ data\cache\multi_level_cache.py: 跨层级导入
- ⚠️ data\cache\redis_cache_adapter.py: 跨层级导入
- ⚠️ data\china\adapter.py: 跨层级导入
- ⚠️ data\china\dragon_board_updater.py: 跨层级导入
- ⚠️ data\decoders\level2_decoder.py: 跨层级导入
- ⚠️ data\distributed\cluster_manager.py: 跨层级导入
- ⚠️ data\distributed\distributed_data_loader.py: 跨层级导入
- ⚠️ data\distributed\load_balancer.py: 跨层级导入
- ⚠️ data\distributed\sharding_manager.py: 跨层级导入
- ⚠️ data\export\data_exporter.py: 跨层级导入
- ⚠️ data\export\data_exporter.py: 跨层级导入
- ⚠️ data\governance\enterprise_governance.py: 跨层级导入
- ⚠️ data\integration\enhanced_data_integration.py: 跨层级导入
- ⚠️ data\lake\data_lake_manager.py: 跨层级导入
- ⚠️ data\lake\metadata_manager.py: 跨层级导入
- ⚠️ data\loader\bond_loader.py: 跨层级导入
- ⚠️ data\loader\bond_loader.py: 跨层级导入
- ⚠️ data\loader\commodity_loader.py: 跨层级导入
- ⚠️ data\loader\commodity_loader.py: 跨层级导入
- ⚠️ data\loader\crypto_loader.py: 跨层级导入
- ⚠️ data\loader\enhanced_data_loader.py: 跨层级导入
- ⚠️ data\loader\enhanced_data_loader.py: 跨层级导入
- ⚠️ data\loader\financial_loader.py: 跨层级导入
- ⚠️ data\loader\financial_loader.py: 跨层级导入
- ⚠️ data\loader\financial_loader.py: 跨层级导入
- ⚠️ data\loader\forex_loader.py: 跨层级导入
- ⚠️ data\loader\forex_loader.py: 跨层级导入
- ⚠️ data\loader\index_loader.py: 跨层级导入
- ⚠️ data\loader\index_loader.py: 跨层级导入
- ⚠️ data\loader\macro_loader.py: 跨层级导入
- ⚠️ data\loader\news_loader.py: 跨层级导入
- ⚠️ data\loader\options_loader.py: 跨层级导入
- ⚠️ data\loader\options_loader.py: 跨层级导入
- ⚠️ data\loader\parallel_loader.py: 跨层级导入
- ⚠️ data\loader\stock_loader.py: 跨层级导入
- ⚠️ data\loader\stock_loader.py: 跨层级导入
- ⚠️ data\monitoring\dashboard.py: 跨层级导入
- ⚠️ data\monitoring\performance_monitor.py: 跨层级导入
- ⚠️ data\monitoring\performance_monitor.py: 跨层级导入
- ⚠️ data\optimization\advanced_optimizer.py: 跨层级导入
- ⚠️ data\optimization\data_optimizer.py: 跨层级导入
- ⚠️ data\optimization\data_preloader.py: 跨层级导入
- ⚠️ data\optimization\performance_monitor.py: 跨层级导入
- ⚠️ data\optimization\performance_optimizer.py: 跨层级导入
- ⚠️ data\parallel\enhanced_parallel_loader.py: 跨层级导入
- ⚠️ data\parallel\parallel_loader.py: 跨层级导入
- ⚠️ data\preload\preloader.py: 跨层级导入
- ⚠️ data\processing\data_processor.py: 跨层级导入
- ⚠️ data\processing\unified_processor.py: 跨层级导入
- ⚠️ data\quality\advanced_quality_monitor.py: 跨层级导入
- ⚠️ data\quality\advanced_quality_monitor.py: 跨层级导入
- ⚠️ data\quality\enhanced_quality_monitor.py: 跨层级导入
- ⚠️ data\quality\enhanced_quality_monitor_v2.py: 跨层级导入
- ⚠️ data\repair\data_repairer.py: 跨层级导入
- ⚠️ data\sources\intelligent_source_manager.py: 跨层级导入
- ⚠️ data\sync\multi_market_sync.py: 跨层级导入
- ⚠️ data\version_control\test_version_manager.py: 跨层级导入
- ⚠️ data\version_control\version_manager.py: 跨层级导入
- ⚠️ data\version_control\version_manager.py: 跨层级导入

### GATEWAY 层级依赖
- **内部导入**: 0 个
- **外部导入**: 17 个
- **跨层级导入**: 0 个

### FEATURES 层级依赖
- **内部导入**: 35 个
- **外部导入**: 699 个
- **跨层级导入**: 77 个

**依赖问题**:
- ⚠️ features\api.py: 跨层级导入
- ⚠️ features\config_integration.py: 跨层级导入
- ⚠️ features\config_integration.py: 跨层级导入
- ⚠️ features\feature_engineer.py: 跨层级导入
- ⚠️ features\feature_importance.py: 跨层级导入
- ⚠️ features\feature_manager.py: 跨层级导入
- ⚠️ features\feature_metadata.py: 跨层级导入
- ⚠️ features\feature_store.py: 跨层级导入
- ⚠️ features\minimal_feature_main_flow.py: 跨层级导入
- ⚠️ features\optimized_feature_manager.py: 跨层级导入
- ⚠️ features\parallel_feature_processor.py: 跨层级导入
- ⚠️ features\parallel_feature_processor.py: 跨层级导入
- ⚠️ features\quality_assessor.py: 跨层级导入
- ⚠️ features\sentiment_analyzer.py: 跨层级导入
- ⚠️ features\signal_generator.py: 跨层级导入
- ⚠️ features\signal_generator.py: 跨层级导入
- ⚠️ features\version_management.py: 跨层级导入
- ⚠️ features\acceleration\fpga\fpga_order_optimizer.py: 跨层级导入
- ⚠️ features\acceleration\fpga\fpga_order_optimizer.py: 跨层级导入
- ⚠️ features\acceleration\fpga\fpga_risk_engine.py: 跨层级导入
- ⚠️ features\acceleration\fpga\fpga_sentiment_analyzer.py: 跨层级导入
- ⚠️ features\core\config.py: 跨层级导入
- ⚠️ features\core\engine.py: 跨层级导入
- ⚠️ features\core\engine.py: 跨层级导入
- ⚠️ features\core\engine.py: 跨层级导入
- ⚠️ features\core\engine.py: 跨层级导入
- ⚠️ features\core\engine.py: 跨层级导入
- ⚠️ features\core\engine.py: 跨层级导入
- ⚠️ features\core\manager.py: 跨层级导入
- ⚠️ features\core\manager.py: 跨层级导入
- ⚠️ features\distributed\distributed_processor.py: 跨层级导入
- ⚠️ features\distributed\task_scheduler.py: 跨层级导入
- ⚠️ features\distributed\worker_manager.py: 跨层级导入
- ⚠️ features\intelligent\auto_feature_selector.py: 跨层级导入
- ⚠️ features\intelligent\intelligent_enhancement_manager.py: 跨层级导入
- ⚠️ features\intelligent\ml_model_integration.py: 跨层级导入
- ⚠️ features\intelligent\smart_alert_system.py: 跨层级导入
- ⚠️ features\monitoring\alert_manager.py: 跨层级导入
- ⚠️ features\monitoring\benchmark_runner.py: 跨层级导入
- ⚠️ features\monitoring\benchmark_runner.py: 跨层级导入
- ⚠️ features\monitoring\features_monitor.py: 跨层级导入
- ⚠️ features\monitoring\metrics_collector.py: 跨层级导入
- ⚠️ features\monitoring\performance_analyzer.py: 跨层级导入
- ⚠️ features\performance\performance_optimizer.py: 跨层级导入
- ⚠️ features\performance\scalability_manager.py: 跨层级导入
- ⚠️ features\plugins\base_plugin.py: 跨层级导入
- ⚠️ features\plugins\base_plugin.py: 跨层级导入
- ⚠️ features\plugins\plugin_loader.py: 跨层级导入
- ⚠️ features\plugins\plugin_loader.py: 跨层级导入
- ⚠️ features\plugins\plugin_manager.py: 跨层级导入
- ⚠️ features\plugins\plugin_manager.py: 跨层级导入
- ⚠️ features\plugins\plugin_registry.py: 跨层级导入
- ⚠️ features\plugins\plugin_registry.py: 跨层级导入
- ⚠️ features\plugins\plugin_validator.py: 跨层级导入
- ⚠️ features\plugins\plugin_validator.py: 跨层级导入
- ⚠️ features\processors\base_processor.py: 跨层级导入
- ⚠️ features\processors\distributed_processor.py: 跨层级导入
- ⚠️ features\processors\distributed_processor.py: 跨层级导入
- ⚠️ features\processors\feature_correlation.py: 跨层级导入
- ⚠️ features\processors\feature_importance.py: 跨层级导入
- ⚠️ features\processors\feature_processor.py: 跨层级导入
- ⚠️ features\processors\feature_quality_assessor.py: 跨层级导入
- ⚠️ features\processors\feature_selector.py: 跨层级导入
- ⚠️ features\processors\feature_stability.py: 跨层级导入
- ⚠️ features\processors\feature_standardizer.py: 跨层级导入
- ⚠️ features\processors\general_processor.py: 跨层级导入
- ⚠️ features\processors\general_processor.py: 跨层级导入
- ⚠️ features\processors\advanced\advanced_feature_processor.py: 跨层级导入
- ⚠️ features\processors\distributed\distributed_feature_processor.py: 跨层级导入
- ⚠️ features\processors\gpu\gpu_technical_processor.py: 跨层级导入
- ⚠️ features\processors\gpu\multi_gpu_processor.py: 跨层级导入
- ⚠️ features\processors\technical\technical_processor.py: 跨层级导入
- ⚠️ features\processors\technical\technical_processor.py: 跨层级导入
- ⚠️ features\sentiment\sentiment_analyzer.py: 跨层级导入
- ⚠️ features\sentiment\sentiment_analyzer.py: 跨层级导入
- ⚠️ features\utils\feature_metadata.py: 跨层级导入
- ⚠️ features\utils\selector.py: 跨层级导入

### ML 层级依赖
- **内部导入**: 0 个
- **外部导入**: 265 个
- **跨层级导入**: 12 个

**依赖问题**:
- ⚠️ ml\models\ab_testing.py: 跨层级导入
- ⚠️ ml\models\api.py: 跨层级导入
- ⚠️ ml\models\api.py: 跨层级导入
- ⚠️ ml\models\api.py: 跨层级导入
- ⚠️ ml\models\api.py: 跨层级导入
- ⚠️ ml\models\deep_learning_models.py: 跨层级导入
- ⚠️ ml\models\deep_learning_models.py: 跨层级导入
- ⚠️ ml\models\inference\batch_inference_processor.py: 跨层级导入
- ⚠️ ml\models\inference\gpu_inference_engine.py: 跨层级导入
- ⚠️ ml\models\inference\inference_cache.py: 跨层级导入
- ⚠️ ml\models\inference\inference_manager.py: 跨层级导入
- ⚠️ ml\models\inference\model_loader.py: 跨层级导入

### BACKTEST 层级依赖
- **内部导入**: 1 个
- **外部导入**: 237 个
- **跨层级导入**: 6 个

**依赖问题**:
- ⚠️ backtest\analyzer.py: 跨层级导入
- ⚠️ backtest\parameter_optimizer.py: 跨层级导入
- ⚠️ backtest\visualizer.py: 跨层级导入
- ⚠️ backtest\evaluation\model_evaluator.py: 跨层级导入
- ⚠️ backtest\evaluation\model_evaluator.py: 跨层级导入
- ⚠️ backtest\utils\backtest_utils.py: 跨层级导入

### RISK 层级依赖
- **内部导入**: 0 个
- **外部导入**: 72 个
- **跨层级导入**: 0 个

### TRADING 层级依赖
- **内部导入**: 28 个
- **外部导入**: 570 个
- **跨层级导入**: 110 个

**依赖问题**:
- ⚠️ trading\backtester.py: 跨层级导入
- ⚠️ trading\backtest_analyzer.py: 跨层级导入
- ⚠️ trading\broker_adapter.py: 跨层级导入
- ⚠️ trading\gateway.py: 跨层级导入
- ⚠️ trading\intelligent_rebalancing.py: 跨层级导入
- ⚠️ trading\live_trader.py: 跨层级导入
- ⚠️ trading\order_manager.py: 跨层级导入
- ⚠️ trading\performance_analyzer.py: 跨层级导入
- ⚠️ trading\strategy_optimizer.py: 跨层级导入
- ⚠️ trading\strategy_optimizer.py: 跨层级导入
- ⚠️ trading\trading_engine.py: 跨层级导入
- ⚠️ trading\trading_engine.py: 跨层级导入
- ⚠️ trading\trading_engine.py: 跨层级导入
- ⚠️ trading\trading_engine_with_distributed.py: 跨层级导入
- ⚠️ trading\trading_engine_with_distributed.py: 跨层级导入
- ⚠️ trading\trading_engine_with_distributed.py: 跨层级导入
- ⚠️ trading\trading_engine_with_distributed.py: 跨层级导入
- ⚠️ trading\advanced_analysis\clustering_engine.py: 跨层级导入
- ⚠️ trading\advanced_analysis\clustering_engine.py: 跨层级导入
- ⚠️ trading\advanced_analysis\portfolio_optimizer.py: 跨层级导入
- ⚠️ trading\advanced_analysis\portfolio_optimizer.py: 跨层级导入
- ⚠️ trading\advanced_analysis\relationship_network.py: 跨层级导入
- ⚠️ trading\advanced_analysis\relationship_network.py: 跨层级导入
- ⚠️ trading\advanced_analysis\similarity_analyzer.py: 跨层级导入
- ⚠️ trading\advanced_analysis\similarity_analyzer.py: 跨层级导入
- ⚠️ trading\distributed\distributed_trading_node.py: 跨层级导入
- ⚠️ trading\distributed\distributed_trading_node.py: 跨层级导入
- ⚠️ trading\distributed\distributed_trading_node.py: 跨层级导入
- ⚠️ trading\distributed\distributed_trading_node.py: 跨层级导入
- ⚠️ trading\distributed\distributed_trading_node.py: 跨层级导入
- ⚠️ trading\distributed\intelligent_order_router.py: 跨层级导入
- ⚠️ trading\distributed\intelligent_order_router.py: 跨层级导入
- ⚠️ trading\distributed\intelligent_order_router.py: 跨层级导入
- ⚠️ trading\distributed\intelligent_order_router.py: 跨层级导入
- ⚠️ trading\execution\execution_engine.py: 跨层级导入
- ⚠️ trading\execution\execution_engine.py: 跨层级导入
- ⚠️ trading\execution\multi_market_adapter.py: 跨层级导入
- ⚠️ trading\execution\order_router.py: 跨层级导入
- ⚠️ trading\execution\order_router.py: 跨层级导入
- ⚠️ trading\execution\reporting.py: 跨层级导入
- ⚠️ trading\ml_integration\auto_optimizer.py: 跨层级导入
- ⚠️ trading\ml_integration\auto_optimizer.py: 跨层级导入
- ⚠️ trading\ml_integration\hyperparameter_tuner.py: 跨层级导入
- ⚠️ trading\ml_integration\hyperparameter_tuner.py: 跨层级导入
- ⚠️ trading\ml_integration\multi_objective_optimizer.py: 跨层级导入
- ⚠️ trading\ml_integration\multi_objective_optimizer.py: 跨层级导入
- ⚠️ trading\ml_integration\optimization_engine.py: 跨层级导入
- ⚠️ trading\ml_integration\performance_predictor.py: 跨层级导入
- ⚠️ trading\ml_integration\recommendation_engine.py: 跨层级导入
- ⚠️ trading\ml_integration\similarity_analyzer.py: 跨层级导入
- ⚠️ trading\ml_integration\similarity_analyzer.py: 跨层级导入
- ⚠️ trading\ml_integration\strategy_recommender.py: 跨层级导入
- ⚠️ trading\ml_integration\strategy_recommender.py: 跨层级导入
- ⚠️ trading\portfolio\portfolio_manager.py: 跨层级导入
- ⚠️ trading\portfolio\portfolio_optimizer.py: 跨层级导入
- ⚠️ trading\portfolio\strategy_portfolio.py: 跨层级导入
- ⚠️ trading\realtime\realtime_trading_system.py: 跨层级导入
- ⚠️ trading\risk\risk_compliance_engine.py: 跨层级导入
- ⚠️ trading\risk\risk_compliance_engine.py: 跨层级导入
- ⚠️ trading\risk\risk_controller.py: 跨层级导入
- ⚠️ trading\risk\risk_controller.py: 跨层级导入
- ⚠️ trading\risk\risk_controller.py: 跨层级导入
- ⚠️ trading\risk\china\circuit_breaker.py: 跨层级导入
- ⚠️ trading\risk\china\market_rule_checker.py: 跨层级导入
- ⚠️ trading\risk\china\position_limits.py: 跨层级导入
- ⚠️ trading\risk\china\star_market.py: 跨层级导入
- ⚠️ trading\risk\china\star_market_adapter.py: 跨层级导入
- ⚠️ trading\risk\china\t1_restriction.py: 跨层级导入
- ⚠️ trading\settlement\settlement_engine.py: 跨层级导入
- ⚠️ trading\strategies\base_strategy.py: 跨层级导入
- ⚠️ trading\strategies\enhanced.py: 跨层级导入
- ⚠️ trading\strategies\factory.py: 跨层级导入
- ⚠️ trading\strategies\multi_strategy_integration.py: 跨层级导入
- ⚠️ trading\strategies\performance_evaluation.py: 跨层级导入
- ⚠️ trading\strategies\reinforcement_learning.py: 跨层级导入
- ⚠️ trading\strategies\china\base_strategy.py: 跨层级导入
- ⚠️ trading\strategies\china\basic_strategy.py: 跨层级导入
- ⚠️ trading\strategies\china\dragon_tiger.py: 跨层级导入
- ⚠️ trading\strategies\china\limit_up.py: 跨层级导入
- ⚠️ trading\strategies\china\margin.py: 跨层级导入
- ⚠️ trading\strategies\china\ml_strategy.py: 跨层级导入
- ⚠️ trading\strategies\china\st.py: 跨层级导入
- ⚠️ trading\strategies\china\star_market_strategy.py: 跨层级导入
- ⚠️ trading\strategies\optimization\advanced_optimizer.py: 跨层级导入
- ⚠️ trading\strategies\optimization\genetic_optimizer.py: 跨层级导入
- ⚠️ trading\strategies\optimization\performance_tuner.py: 跨层级导入
- ⚠️ trading\strategies\optimization\performance_tuner.py: 跨层级导入
- ⚠️ trading\strategies\optimization\performance_tuner.py: 跨层级导入
- ⚠️ trading\strategies\optimization\performance_tuner.py: 跨层级导入
- ⚠️ trading\strategy\high_freq_optimizer.py: 跨层级导入
- ⚠️ trading\strategy_workspace\analyzer.py: 跨层级导入
- ⚠️ trading\strategy_workspace\analyzer.py: 跨层级导入
- ⚠️ trading\strategy_workspace\optimizer.py: 跨层级导入
- ⚠️ trading\strategy_workspace\optimizer.py: 跨层级导入
- ⚠️ trading\strategy_workspace\simulator.py: 跨层级导入
- ⚠️ trading\strategy_workspace\simulator.py: 跨层级导入
- ⚠️ trading\strategy_workspace\store.py: 跨层级导入
- ⚠️ trading\strategy_workspace\store.py: 跨层级导入
- ⚠️ trading\strategy_workspace\strategy_generator.py: 跨层级导入
- ⚠️ trading\strategy_workspace\strategy_generator.py: 跨层级导入
- ⚠️ trading\strategy_workspace\visual_editor.py: 跨层级导入
- ⚠️ trading\strategy_workspace\web_interface.py: 跨层级导入
- ⚠️ trading\strategy_workspace\web_interface.py: 跨层级导入
- ⚠️ trading\strategy_workspace\web_interface_demo.py: 跨层级导入
- ⚠️ trading\universe\adaptive_factor_model.py: 跨层级导入
- ⚠️ trading\universe\comprehensive_scoring.py: 跨层级导入
- ⚠️ trading\universe\dynamic_universe_manager.py: 跨层级导入
- ⚠️ trading\universe\dynamic_weight_adjuster.py: 跨层级导入
- ⚠️ trading\universe\filters.py: 跨层级导入
- ⚠️ trading\universe\intelligent_updater.py: 跨层级导入

### ENGINE 层级依赖
- **内部导入**: 34 个
- **外部导入**: 431 个
- **跨层级导入**: 34 个

**依赖问题**:
- ⚠️ engine\stress_test.py: 跨层级导入
- ⚠️ engine\stress_test.py: 跨层级导入
- ⚠️ engine\optimization\buffer_optimizer.py: 跨层级导入
- ⚠️ engine\optimization\buffer_optimizer.py: 跨层级导入
- ⚠️ engine\optimization\dispatcher_optimizer.py: 跨层级导入
- ⚠️ engine\optimization\dispatcher_optimizer.py: 跨层级导入
- ⚠️ engine\optimization\level2_optimizer.py: 跨层级导入
- ⚠️ engine\optimization\level2_optimizer.py: 跨层级导入
- ⚠️ engine\web\app_factory.py: 跨层级导入
- ⚠️ engine\web\app_factory.py: 跨层级导入
- ⚠️ engine\web\app_factory.py: 跨层级导入
- ⚠️ engine\web\app_factory.py: 跨层级导入
- ⚠️ engine\web\app_factory.py: 跨层级导入
- ⚠️ engine\web\app_factory.py: 跨层级导入
- ⚠️ engine\web\app_factory.py: 跨层级导入
- ⚠️ engine\web\data_api.py: 跨层级导入
- ⚠️ engine\web\data_api.py: 跨层级导入
- ⚠️ engine\web\data_api.py: 跨层级导入
- ⚠️ engine\web\data_api.py: 跨层级导入
- ⚠️ engine\web\unified_dashboard.py: 跨层级导入
- ⚠️ engine\web\unified_dashboard.py: 跨层级导入
- ⚠️ engine\web\unified_dashboard.py: 跨层级导入
- ⚠️ engine\web\unified_dashboard.py: 跨层级导入
- ⚠️ engine\web\websocket_api.py: 跨层级导入
- ⚠️ engine\web\websocket_api.py: 跨层级导入
- ⚠️ engine\web\websocket_api.py: 跨层级导入
- ⚠️ engine\web\websocket_api.py: 跨层级导入
- ⚠️ engine\web\websocket_api.py: 跨层级导入
- ⚠️ engine\web\modules\config_module.py: 跨层级导入
- ⚠️ engine\web\modules\fpga_module.py: 跨层级导入
- ⚠️ engine\web\modules\fpga_module.py: 跨层级导入
- ⚠️ engine\web\modules\resource_module.py: 跨层级导入
- ⚠️ engine\web\modules\resource_module.py: 跨层级导入
- ⚠️ engine\web\modules\resource_module.py: 跨层级导入

## 🔍 详细问题列表

### 🔴 Layer Responsibility
**层级**: core
**文件**: `core\architecture_demo.py`
**严重程度**: high
**违规内容**: trading, risk, ml, feature, data

### 🔴 Layer Responsibility
**层级**: core
**文件**: `core\architecture_layers.py`
**严重程度**: high
**违规内容**: trading, risk, ml, feature, data

### 🟡 Layer Responsibility
**层级**: core
**文件**: `core\base.py`
**严重程度**: medium
**违规内容**: data

### 🔴 Layer Responsibility
**层级**: core
**文件**: `core\business_process_demo.py`
**严重程度**: high
**违规内容**: trading, risk, feature, data

### 🔴 Layer Responsibility
**层级**: core
**文件**: `core\business_process_integration.py`
**严重程度**: high
**违规内容**: trading, risk, feature, data

### 🔴 Layer Responsibility
**层级**: core
**文件**: `core\business_process_orchestrator.py`
**严重程度**: high
**违规内容**: trading, risk, feature, data

### 🟡 Layer Responsibility
**层级**: core
**文件**: `core\container.py`
**严重程度**: medium
**违规内容**: data

### 🔴 Layer Responsibility
**层级**: core
**文件**: `core\event_bus.py`
**严重程度**: high
**违规内容**: trading, risk, feature, data

### 🔴 Layer Responsibility
**层级**: core
**文件**: `core\layer_interfaces.py`
**严重程度**: high
**违规内容**: trading, risk, feature, data

### 🟡 Layer Responsibility
**层级**: core
**文件**: `core\process_config_loader.py`
**严重程度**: medium
**违规内容**: ml, data

### 🟡 Layer Responsibility
**层级**: core
**文件**: `core\service_container.py`
**严重程度**: medium
**违规内容**: data

### 🟡 Layer Responsibility
**层级**: core
**文件**: `core\integration\data.py`
**严重程度**: medium
**违规内容**: data

### 🟡 Layer Responsibility
**层级**: core
**文件**: `core\integration\deployment.py`
**严重程度**: medium
**违规内容**: feature, data

### 🔴 Layer Responsibility
**层级**: core
**文件**: `core\integration\interface.py`
**严重程度**: high
**违规内容**: trading, risk, feature, data

### 🟡 Layer Responsibility
**层级**: core
**文件**: `core\integration\layer_interface.py`
**严重程度**: medium
**违规内容**: data

### 🟡 Layer Responsibility
**层级**: core
**文件**: `core\integration\system_integration_manager.py`
**严重程度**: medium
**违规内容**: data

### 🔴 Layer Responsibility
**层级**: core
**文件**: `core\integration\testing.py`
**严重程度**: high
**违规内容**: trading, risk, feature, data

### 🔴 Layer Responsibility
**层级**: core
**文件**: `core\optimizations\long_term_optimizations.py`
**严重程度**: high
**违规内容**: trading, risk, ml, feature, data

### 🟡 Layer Responsibility
**层级**: core
**文件**: `core\optimizations\medium_term_optimizations.py`
**严重程度**: medium
**违规内容**: data

### 🟡 Layer Responsibility
**层级**: core
**文件**: `core\optimizations\optimization_implementer.py`
**严重程度**: medium
**违规内容**: data

### 🟡 Layer Responsibility
**层级**: core
**文件**: `core\optimizations\short_term_optimizations.py`
**严重程度**: medium
**违规内容**: trading, data

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\degradation_manager.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\inference_engine.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\service_launcher.py`
**严重程度**: medium
**违规内容**: trading, feature

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\cache\cache_factory.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\cache\cache_performance_tester.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\cache\cache_utils.py`
**严重程度**: medium
**违规内容**: model, feature

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\cache\caching.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\cache\multi_level_cache.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\cache\quota_manager.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\cache\smart_cache_strategy.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\cache\unified_cache.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\cache\unified_cache_factory.py`
**严重程度**: medium
**违规内容**: strategy

### 🔴 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\ai_optimization_enhanced.py`
**严重程度**: high
**违规内容**: strategy, model, feature

### 🔴 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\ai_test_optimizer.py`
**严重程度**: high
**违规内容**: strategy, model, feature

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\app.py`
**严重程度**: medium
**违规内容**: strategy, model

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\async_optimizer.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\circuit_breaker_manager.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\configuration.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\config_example.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\config_exceptions.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\config_loader_service.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\config_service.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\config_strategy.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\config_sync_service.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\core.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\data_api.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\dependency.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\deployment.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\deployment_validator.py`
**严重程度**: medium
**违规内容**: trading, feature

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\distributed_test_runner.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\edge_computing_test_platform.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\env_loader.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\hybrid_loader.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\infrastructure_index.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\integration.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\json_loader.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\microservice_manager.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\optimization_strategies.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\optimized_components.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\paths.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\performance_config.py`
**严重程度**: medium
**违规内容**: strategy, feature

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\regulatory_tester.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\report_generator.py`
**严重程度**: medium
**违规内容**: trading

### 🔴 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\standard_interfaces.py`
**严重程度**: high
**违规内容**: trading, strategy, model, feature

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\strategy.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\sync_conflict_manager.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\unified_interface.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\unified_loaders.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\unified_manager.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\unified_service.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\unified_strategy.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\unified_sync.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\unified_sync_service.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\validators.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\web_management_service.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\config\yaml_loader.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\error\archive_failure_handler.py`
**严重程度**: medium
**违规内容**: strategy

### 🔴 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\error\error_codes_utils.py`
**严重程度**: high
**违规内容**: trading, strategy, model

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\error\error_exceptions.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\error\error_handler.py`
**严重程度**: medium
**违规内容**: trading, strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\error\market_aware_retry.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\error\retry_handler.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\error\retry_policy.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\error\trading_error_handler.py`
**严重程度**: medium
**违规内容**: trading, strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\health\basic_health_checker.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\health\enhanced_health_checker.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\health\fastapi_health_checker.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\logging\circuit_breaker.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\logging\load_balancer.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\logging\logging_strategy.py`
**严重程度**: medium
**违规内容**: trading, strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\logging\log_aggregator_plugin.py`
**严重程度**: medium
**违规内容**: trading, feature

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\logging\log_backpressure_plugin.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\logging\log_compressor_plugin.py`
**严重程度**: medium
**违规内容**: trading, strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\logging\log_sampler.py`
**严重程度**: medium
**违规内容**: trading, strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\logging\log_sampler_plugin.py`
**严重程度**: medium
**违规内容**: trading, strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\logging\market_data_logger.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\logging\network_manager.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\logging\trading_logger.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\logging\unified_logger.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\resource\backtest_monitor_plugin.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\resource\behavior_monitor_plugin.py`
**严重程度**: medium
**违规内容**: trading

### 🔴 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\resource\business_metrics_monitor.py`
**严重程度**: high
**违规内容**: trading, strategy, model

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\resource\business_metrics_plugin.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\resource\model_monitor_plugin.py`
**严重程度**: medium
**违规内容**: model, feature

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\resource\performance.py`
**严重程度**: medium
**违规内容**: trading, strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\resource\performance_monitor.py`
**严重程度**: medium
**违规内容**: trading, model

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\resource\performance_optimizer.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\resource\performance_optimizer_manager.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\resource\resource_api.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\resource\resource_dashboard.py`
**严重程度**: medium
**违规内容**: strategy

### 🔴 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\services\api_service.py`
**严重程度**: high
**违规内容**: trading, strategy, model

### 🔴 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\services\business_service.py`
**严重程度**: high
**违规内容**: trading, strategy, model, feature

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\services\cache_service.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\services\micro_service.py`
**严重程度**: medium
**违规内容**: trading, model

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\services\model_service.py`
**严重程度**: medium
**违规内容**: model, feature

### 🔴 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\services\trading_service.py`
**严重程度**: high
**违规内容**: trading, strategy, feature

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\utils\date_utils.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\utils.backup_20250823_212847\date_utils.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\utils.backup_20250823_212847\enhanced_database_manager.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: infrastructure
**文件**: `infrastructure\utils.backup_20250823_212847\tools.py`
**严重程度**: medium
**违规内容**: model, feature

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\api.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\base_adapter.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\data_manager.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\interfaces.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\models.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\validator.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\adapters\miniqmt\adapter.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\adapters\miniqmt\local_cache.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\adapters\miniqmt\miniqmt_trade_adapter.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\adapters\miniqmt\rate_limiter.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\generic_china_data_adapter.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\china\adapter.py`
**严重程度**: medium
**违规内容**: trading, model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\china\dragon_board.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\china\financial_adapter.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\china\index_adapter.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\china\margin_trading.py`
**严重程度**: medium
**违规内容**: trading, model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\china\news_adapter.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\china\sentiment_adapter.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\china\stock_adapter.py`
**严重程度**: medium
**违规内容**: trading, model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\crypto\ccxt_mock_adapter.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\cache\cache_manager.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\cache\enhanced_cache_strategy.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\cache\lfu_strategy.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\china\adapter.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\china\adapters.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\china\dragon_board_updater.py`
**严重程度**: medium
**违规内容**: feature

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\china\market.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\core\data_model.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\core\models.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\distributed\distributed_data_loader.py`
**严重程度**: medium
**违规内容**: strategy, model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\distributed\load_balancer.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\distributed\sharding_manager.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\edge\edge_node.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\export\data_exporter.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\integration\enhanced_data_integration.py`
**严重程度**: medium
**违规内容**: strategy, feature

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\interfaces\IDataModel.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\lake\data_lake_manager.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\lake\partition_manager.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\loader\enhanced_data_loader.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\loader\stock_loader.py`
**严重程度**: medium
**违规内容**: trading, feature

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\ml\quality_assessor.py`
**严重程度**: medium
**违规内容**: feature

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\monitoring\quality_monitor.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\optimization\advanced_optimizer.py`
**严重程度**: medium
**违规内容**: strategy, model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\optimization\data_optimizer.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\optimization\data_preloader.py`
**严重程度**: medium
**违规内容**: strategy, model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\optimization\performance_optimizer.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\parallel\enhanced_parallel_loader.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\parallel\parallel_loader.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\preload\preloader.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\processing\data_processor.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\processing\unified_processor.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\quality\advanced_quality_monitor.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\quality\data_quality_monitor.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\quality\enhanced_quality_monitor_v2.py`
**严重程度**: medium
**违规内容**: strategy

### 🔴 Layer Responsibility
**层级**: data
**文件**: `data\quantum\quantum_circuit.py`
**严重程度**: high
**违规内容**: trading, strategy, feature

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\repair\data_repairer.py`
**严重程度**: medium
**违规内容**: strategy, model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\sources\intelligent_source_manager.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\sync\multi_market_sync.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\transformers\data_transformer.py`
**严重程度**: medium
**违规内容**: feature

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\validation\china_stock_validator.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\version_control\test_version_manager.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: data
**文件**: `data\version_control\version_manager.py`
**严重程度**: medium
**违规内容**: model

### 🔴 Layer Responsibility
**层级**: gateway
**文件**: `gateway\api_gateway.py`
**严重程度**: high
**违规内容**: trading, model, feature

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\api.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\config_classes.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\exceptions.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\feature_importance.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\feature_manager.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\sentiment_analyzer.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\acceleration\scalability_enhancer.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\acceleration\fpga\fpga_accelerator.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\acceleration\fpga\fpga_optimizer.py`
**严重程度**: medium
**违规内容**: trading

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\acceleration\fpga\fpga_orderbook_optimizer.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\acceleration\fpga\fpga_order_optimizer.py`
**严重程度**: medium
**违规内容**: trading, strategy

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\acceleration\gpu\gpu_accelerator.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\acceleration\gpu\gpu_scheduler.py`
**严重程度**: medium
**违规内容**: strategy, model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\core\engine.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\distributed\distributed_processor.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\intelligent\auto_feature_selector.py`
**严重程度**: medium
**违规内容**: strategy, model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\intelligent\intelligent_enhancement_manager.py`
**严重程度**: medium
**违规内容**: strategy, model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\intelligent\ml_model_integration.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\monitoring\performance_analyzer.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\performance\performance_optimizer.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\performance\scalability_manager.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\processors\advanced_feature_selector.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\processors\feature_correlation.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\processors\feature_importance.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\processors\feature_selector.py`
**严重程度**: medium
**违规内容**: strategy, model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\processors\feature_standardizer.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\processors\advanced\advanced_feature_processor.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: features
**文件**: `features\sentiment\models\sentiment_model.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: backtest
**文件**: `backtest\config_manager.py`
**严重程度**: medium
**违规内容**: production

### 🟡 Layer Responsibility
**层级**: backtest
**文件**: `backtest\microservice_architecture.py`
**严重程度**: medium
**违规内容**: production

### 🟡 Layer Responsibility
**层级**: risk
**文件**: `risk\api.py`
**严重程度**: medium
**违规内容**: order

### 🔴 Layer Responsibility
**层级**: risk
**文件**: `risk\compliance_checker.py`
**严重程度**: high
**违规内容**: trading, order, execution

### 🟡 Layer Responsibility
**层级**: risk
**文件**: `risk\real_time_monitor.py`
**严重程度**: medium
**违规内容**: order

### 🟡 Layer Responsibility
**层级**: risk
**文件**: `risk\risk_manager.py`
**严重程度**: medium
**违规内容**: order

### 🟡 Layer Responsibility
**层级**: trading
**文件**: `trading\backtester.py`
**严重程度**: medium
**违规内容**: backtest

### 🟡 Layer Responsibility
**层级**: trading
**文件**: `trading\backtest_analyzer.py`
**严重程度**: medium
**违规内容**: backtest

### 🟡 Layer Responsibility
**层级**: trading
**文件**: `trading\live_trading.py`
**严重程度**: medium
**违规内容**: backtest

### 🟡 Layer Responsibility
**层级**: trading
**文件**: `trading\strategy_optimizer.py`
**严重程度**: medium
**违规内容**: backtest

### 🟡 Layer Responsibility
**层级**: trading
**文件**: `trading\portfolio\portfolio_manager.py`
**严重程度**: medium
**违规内容**: backtest

### 🟡 Layer Responsibility
**层级**: trading
**文件**: `trading\strategies\base_strategy.py`
**严重程度**: medium
**违规内容**: backtest

### 🟡 Layer Responsibility
**层级**: trading
**文件**: `trading\strategies\enhanced.py`
**严重程度**: medium
**违规内容**: backtest

### 🟡 Layer Responsibility
**层级**: trading
**文件**: `trading\strategies\factory.py`
**严重程度**: medium
**违规内容**: backtest

### 🟡 Layer Responsibility
**层级**: trading
**文件**: `trading\strategies\reinforcement_learning.py`
**严重程度**: medium
**违规内容**: backtest

### 🟡 Layer Responsibility
**层级**: trading
**文件**: `trading\strategies\strategy_auto_optimizer.py`
**严重程度**: medium
**违规内容**: backtest

### 🟡 Layer Responsibility
**层级**: trading
**文件**: `trading\strategies\optimization\genetic_optimizer.py`
**严重程度**: medium
**违规内容**: backtest

### 🟡 Layer Responsibility
**层级**: trading
**文件**: `trading\strategy_workspace\analyzer.py`
**严重程度**: medium
**违规内容**: simulation

### 🟡 Layer Responsibility
**层级**: trading
**文件**: `trading\strategy_workspace\simulator.py`
**严重程度**: medium
**违规内容**: backtest, simulation

### 🟡 Layer Responsibility
**层级**: trading
**文件**: `trading\strategy_workspace\store.py`
**严重程度**: medium
**违规内容**: simulation

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\dispatcher.py`
**严重程度**: medium
**违规内容**: strategy

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\level2.py`
**严重程度**: medium
**违规内容**: order

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\realtime.py`
**严重程度**: medium
**违规内容**: order

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\realtime_engine.py`
**严重程度**: medium
**违规内容**: order

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\stress_test.py`
**严重程度**: medium
**违规内容**: order

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\config\config_schema.py`
**严重程度**: medium
**违规内容**: order, model

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\config\config_validator.py`
**严重程度**: medium
**违规内容**: order

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\config\engine_config_manager.py`
**严重程度**: medium
**违规内容**: order, model

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\inference\optimized_inference_engine.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\level2\level2_adapter.py`
**严重程度**: medium
**违规内容**: trading, order

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\logging\business_logger.py`
**严重程度**: medium
**违规内容**: order

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\monitoring.backup_20250823_212847\metrics_collector.py`
**严重程度**: medium
**违规内容**: order

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\optimization\dispatcher_optimizer.py`
**严重程度**: medium
**违规内容**: order, strategy

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\optimization\level2_optimizer.py`
**严重程度**: medium
**违规内容**: order

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\production\model_serving.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\testing\test_data_generator.py`
**严重程度**: medium
**违规内容**: order

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\testing\test_data_manager.py`
**严重程度**: medium
**违规内容**: order

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\testing\test_data_validator.py`
**严重程度**: medium
**违规内容**: order

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\web\data_api.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\web\unified_dashboard.py`
**严重程度**: medium
**违规内容**: strategy, model

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\web\websocket_api.py`
**严重程度**: medium
**违规内容**: order

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\web\modules\base_module.py`
**严重程度**: medium
**违规内容**: model

### 🔴 Layer Responsibility
**层级**: engine
**文件**: `engine\web\modules\config_module.py`
**严重程度**: high
**违规内容**: trading, order, model

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\web\modules\features_module.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\web\modules\fpga_module.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\web\modules\module_registry.py`
**严重程度**: medium
**违规内容**: order

### 🟡 Layer Responsibility
**层级**: engine
**文件**: `engine\web\modules\resource_module.py`
**严重程度**: medium
**违规内容**: model

### 🟡 Interface Compliance
**层级**: core
**文件**: `core\base.py`
**严重程度**: medium
**问题描述**: 基础实现类不符合标准模式

### 🟡 Interface Compliance
**层级**: core
**文件**: `core\layer_interfaces.py`
**严重程度**: medium
**问题描述**: 接口命名不符合标准规范

### 🟡 Interface Compliance
**层级**: infrastructure
**文件**: `infrastructure\config\standard_interfaces.py`
**严重程度**: medium
**问题描述**: 接口命名不符合标准规范

### 🟡 Interface Compliance
**层级**: infrastructure
**文件**: `infrastructure\config\unified_interfaces.py`
**严重程度**: medium
**问题描述**: 接口命名不符合标准规范

### 🟡 Interface Compliance
**层级**: infrastructure
**文件**: `infrastructure\utils.backup_20250823_212847\base_database.py`
**严重程度**: medium
**问题描述**: 基础实现类不符合标准模式

### 🟡 Interface Compliance
**层级**: infrastructure
**文件**: `infrastructure\utils.backup_20250823_212847\database.py`
**严重程度**: medium
**问题描述**: 基础实现类不符合标准模式

### 🟡 Interface Compliance
**层级**: infrastructure
**文件**: `infrastructure\utils.backup_20250823_212847\unified_database.py`
**严重程度**: medium
**问题描述**: 基础实现类不符合标准模式

### 🟡 Interface Compliance
**层级**: data
**文件**: `data\interfaces.py`
**严重程度**: medium
**问题描述**: 接口命名不符合标准规范

### 🟡 Interface Compliance
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\base.py`
**严重程度**: medium
**问题描述**: 基础实现类不符合标准模式

### 🟡 Interface Compliance
**层级**: ml
**文件**: `ml\tuning\optimizers\base.py`
**严重程度**: medium
**问题描述**: 基础实现类不符合标准模式

### 🔴 Dependency
**层级**: core
**文件**: `core\architecture_layers.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.core.config.unified_config_manager import UnifiedConfigManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\data_sync.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\degradation_manager.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\disaster_recovery.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\final_deployment_check.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\service_launcher.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\alert_manager.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\data_api.py`
**严重程度**: high
**导入语句**: `from src.data.data_manager import DataManagerSingleton`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\data_api.py`
**严重程度**: high
**导入语句**: `from src.data.monitoring import PerformanceMonitor`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\data_api.py`
**严重程度**: high
**导入语句**: `from src.data.quality import DataQualityMonitor, AdvancedQualityMonitor`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\data_api.py`
**严重程度**: high
**导入语句**: `from src.data.loader import (`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\deployment.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\deployment_validator.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\disaster_tester.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\paths.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\regulatory_tester.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\regulatory_tester.py`
**严重程度**: high
**导入语句**: `from src.trading.execution.order_manager import OrderManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\regulatory_tester.py`
**严重程度**: high
**导入语句**: `from src.trading.risk.china.risk_controller import ChinaRiskController`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\report_generator.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\report_generator.py`
**严重程度**: high
**导入语句**: `from src.data.china.stock import ChinaDataAdapter`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\report_generator.py`
**严重程度**: high
**导入语句**: `from src.trading.execution.execution_engine import ExecutionEngine`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\report_generator.py`
**严重程度**: high
**导入语句**: `from src.trading.risk.risk_controller import RiskController`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\unified_core.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\unified_query.py`
**严重程度**: high
**导入语句**: `from src.adapters.miniqmt.data_cache import ParquetStorage`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\websocket_api.py`
**严重程度**: high
**导入语句**: `from src.data.data_manager import DataManagerSingleton`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\websocket_api.py`
**严重程度**: high
**导入语句**: `from src.data.monitoring import PerformanceMonitor`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\websocket_api.py`
**严重程度**: high
**导入语句**: `from src.data.quality import DataQualityMonitor, AdvancedQualityMonitor`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\websocket_api.py`
**严重程度**: high
**导入语句**: `from src.data.loader import (`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\config\websocket_api.py`
**严重程度**: high
**导入语句**: `from src.data.data_manager import DataManagerSingleton`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\resource\behavior_monitor_plugin.py`
**严重程度**: high
**导入语句**: `from src.trading.risk import RiskController`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\resource\disaster_monitor_plugin.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\resource\performance_monitor.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\services\api_service.py`
**严重程度**: high
**导入语句**: `from src.core import EventBus, Event, EventType, ServiceContainer`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\services\api_service.py`
**严重程度**: high
**导入语句**: `from src.services.base_service import BaseService, ServiceStatus`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\services\business_service.py`
**严重程度**: high
**导入语句**: `from src.core import EventBus, Event, EventType, ServiceContainer`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\services\cache_service.py`
**严重程度**: high
**导入语句**: `from src.core import EventBus, Event, EventType, ServiceContainer`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\services\cache_service.py`
**严重程度**: high
**导入语句**: `from src.services.base_service import BaseService, ServiceStatus`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\services\data_validation_service.py`
**严重程度**: high
**导入语句**: `from src.data.adapters.base_data_adapter import BaseDataAdapter`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\services\micro_service.py`
**严重程度**: high
**导入语句**: `from src.core import EventBus, Event, EventType, ServiceContainer`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\services\micro_service.py`
**严重程度**: high
**导入语句**: `from src.services.base_service import BaseService, ServiceStatus as BaseServiceStatus`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: infrastructure
**文件**: `infrastructure\services\trading_service.py`
**严重程度**: high
**导入语句**: `from src.core import EventBus, Event, EventType, ServiceContainer`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\api.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\backup_recovery.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\base_adapter.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\data_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\data_manager.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.utils.exceptions import DataLoaderError`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\data_manager.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\data_manager.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.core.resource_manager import global_resource_manager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\enhanced_integration_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\market_data.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\registry.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters\miniqmt\adapter.py`
**严重程度**: high
**导入语句**: `from src.adapters.miniqmt.miniqmt_data_adapter import MiniQMTDataAdapter`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters\miniqmt\adapter.py`
**严重程度**: high
**导入语句**: `from src.adapters.miniqmt.miniqmt_trade_adapter import MiniQMTTradeAdapter`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters\miniqmt\adapter.py`
**严重程度**: high
**导入语句**: `from src.adapters.miniqmt.data_cache import MiniQMTDataCache`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters\miniqmt\adapter.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.config.unified_manager import UnifiedConfigManager as ConfigManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters\miniqmt\adapter.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.monitoring.metrics import MetricsCollector`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters\miniqmt\miniqmt_data_adapter.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.error.exceptions import DataFetchError`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters\miniqmt\miniqmt_trade_adapter.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.error.exceptions import TradeError`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\adapter_registry.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.interfaces.standard_interfaces import DataRequest`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\base_adapter.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.interfaces.standard_interfaces import DataLoader, DataRequest`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\generic_china_data_adapter.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\china\adapter.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\china\financial_adapter.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\china\index_adapter.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\china\news_adapter.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\china\sentiment_adapter.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\crypto\crypto_adapter.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.interfaces.standard_interfaces import DataRequest`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\crypto\crypto_adapter.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.utils.exceptions import DataLoaderError`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\international\international_stock_adapter.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.interfaces.standard_interfaces import DataRequest`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\macro\macro_economic_adapter.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.interfaces.standard_interfaces import DataRequest`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\adapters.backup_20250823_212847\news\news_sentiment_adapter.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.interfaces.standard_interfaces import DataRequest`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\alignment\data_aligner.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\alignment\data_aligner.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.utils.exceptions import DataProcessingError`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\cache\cache_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\cache\cache_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\cache\disk_cache.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\cache\disk_cache.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\cache\enhanced_cache_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\cache\enhanced_cache_strategy.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\cache\multi_level_cache.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\cache\redis_cache_adapter.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\china\adapter.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\china\dragon_board_updater.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\decoders\level2_decoder.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\distributed\cluster_manager.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.utils.helpers.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\distributed\distributed_data_loader.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.utils.helpers.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\distributed\load_balancer.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.utils.helpers.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\distributed\sharding_manager.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.utils.helpers.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\export\data_exporter.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\export\data_exporter.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.utils.exceptions import DataLoaderError`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\governance\enterprise_governance.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\integration\enhanced_data_integration.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\lake\data_lake_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\lake\metadata_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\bond_loader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\bond_loader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\commodity_loader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\commodity_loader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\crypto_loader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\enhanced_data_loader.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.interfaces.standard_interfaces import DataRequest`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\enhanced_data_loader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\financial_loader.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.utils.helpers.environment import is_production`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\financial_loader.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\financial_loader.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.utils.exceptions import DataLoaderError`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\forex_loader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\forex_loader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\index_loader.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.utils.exceptions import DataLoaderError`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\index_loader.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\macro_loader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\news_loader.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.utils.exceptions import DataLoaderError`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\options_loader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\options_loader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\parallel_loader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\stock_loader.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.utils.exceptions import DataLoaderError`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\loader\stock_loader.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\monitoring\dashboard.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\monitoring\performance_monitor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\monitoring\performance_monitor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\optimization\advanced_optimizer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\optimization\data_optimizer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\optimization\data_preloader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\optimization\performance_monitor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\optimization\performance_optimizer.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.utils.helpers.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\parallel\enhanced_parallel_loader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\parallel\parallel_loader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\preload\preloader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\processing\data_processor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\processing\unified_processor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\quality\advanced_quality_monitor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\quality\advanced_quality_monitor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\quality\enhanced_quality_monitor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\quality\enhanced_quality_monitor_v2.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.utils.helpers.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\repair\data_repairer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\sources\intelligent_source_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\sync\multi_market_sync.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\version_control\test_version_manager.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.utils.exceptions import DataVersionError`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\version_control\version_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: data
**文件**: `data\version_control\version_manager.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.utils.exceptions import DataVersionError`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\api.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\config_integration.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.config.factory import ConfigFactory`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\config_integration.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.core.config.interfaces.unified_interface import IConfigManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\feature_engineer.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\feature_importance.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\feature_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\feature_metadata.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\feature_store.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\minimal_feature_main_flow.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\optimized_feature_manager.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\parallel_feature_processor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\parallel_feature_processor.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\quality_assessor.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\sentiment_analyzer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\signal_generator.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\signal_generator.py`
**严重程度**: high
**导入语句**: `from src.acceleration.fpga import FpgaManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\version_management.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\acceleration\fpga\fpga_order_optimizer.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\acceleration\fpga\fpga_order_optimizer.py`
**严重程度**: high
**导入语句**: `from src.trading.execution.optimizer import BaseOrderOptimizer`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\acceleration\fpga\fpga_risk_engine.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\acceleration\fpga\fpga_sentiment_analyzer.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\core\config.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\core\engine.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\core\engine.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\core\engine.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.interfaces.standard_interfaces import FeatureRequest`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\core\engine.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.interfaces.standard_interfaces import FeatureRequest`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\core\engine.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.interfaces.standard_interfaces import FeatureRequest`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\core\engine.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.interfaces.standard_interfaces import FeatureRequest`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\core\manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\core\manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\distributed\distributed_processor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\distributed\task_scheduler.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\distributed\worker_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\intelligent\auto_feature_selector.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\intelligent\intelligent_enhancement_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\intelligent\ml_model_integration.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\intelligent\smart_alert_system.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\monitoring\alert_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\monitoring\benchmark_runner.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\monitoring\benchmark_runner.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\monitoring\features_monitor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\monitoring\metrics_collector.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\monitoring\performance_analyzer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\performance\performance_optimizer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\performance\scalability_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\plugins\base_plugin.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\plugins\base_plugin.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\plugins\plugin_loader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\plugins\plugin_loader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\plugins\plugin_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\plugins\plugin_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\plugins\plugin_registry.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\plugins\plugin_registry.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\plugins\plugin_validator.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\plugins\plugin_validator.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\processors\base_processor.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.interfaces.standard_interfaces import FeatureProcessor, FeatureRequest`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\processors\distributed_processor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\processors\distributed_processor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\processors\feature_correlation.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\processors\feature_importance.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\processors\feature_processor.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\processors\feature_quality_assessor.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\processors\feature_selector.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\processors\feature_stability.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\processors\feature_standardizer.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\processors\general_processor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\processors\general_processor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\processors\advanced\advanced_feature_processor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\processors\distributed\distributed_feature_processor.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\processors\gpu\gpu_technical_processor.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\processors\gpu\multi_gpu_processor.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\processors\technical\technical_processor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\processors\technical\technical_processor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\sentiment\sentiment_analyzer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\sentiment\sentiment_analyzer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\utils\feature_metadata.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: features
**文件**: `features\utils\selector.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: ml
**文件**: `ml\models\ab_testing.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: ml
**文件**: `ml\models\api.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: ml
**文件**: `ml\models\api.py`
**严重程度**: high
**导入语句**: `from src.models.model_manager import ModelManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: ml
**文件**: `ml\models\api.py`
**严重程度**: high
**导入语句**: `from src.models.trainer import SimpleModelTrainer`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: ml
**文件**: `ml\models\api.py`
**严重程度**: high
**导入语句**: `from src.models.predictor import SimpleModelPredictor`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: ml
**文件**: `ml\models\deep_learning_models.py`
**严重程度**: high
**导入语句**: `from src.models.base_model import BaseModel`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: ml
**文件**: `ml\models\deep_learning_models.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: ml
**文件**: `ml\models\inference\batch_inference_processor.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: ml
**文件**: `ml\models\inference\gpu_inference_engine.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: ml
**文件**: `ml\models\inference\inference_cache.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: ml
**文件**: `ml\models\inference\inference_manager.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: ml
**文件**: `ml\models\inference\model_loader.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: backtest
**文件**: `backtest\analyzer.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: backtest
**文件**: `backtest\parameter_optimizer.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: backtest
**文件**: `backtest\visualizer.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: backtest
**文件**: `backtest\evaluation\model_evaluator.py`
**严重程度**: high
**导入语句**: `from src.trading.risk.china import ChinaMarketRuleChecker`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: backtest
**文件**: `backtest\evaluation\model_evaluator.py`
**严重程度**: high
**导入语句**: `from src.trading.risk.china import ChinaMarketRuleChecker`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: backtest
**文件**: `backtest\utils\backtest_utils.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\backtester.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\backtest_analyzer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\broker_adapter.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\gateway.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\intelligent_rebalancing.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\live_trader.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\order_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\performance_analyzer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategy_optimizer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategy_optimizer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\trading_engine.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\trading_engine.py`
**严重程度**: high
**导入语句**: `from src.infrastructure import SystemMonitor`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\trading_engine.py`
**严重程度**: high
**导入语句**: `from src.infrastructure import get_default_monitor`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\trading_engine_with_distributed.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\trading_engine_with_distributed.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.distributed.distributed_lock import DistributedLockManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\trading_engine_with_distributed.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.distributed.config_center import ConfigCenterManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\trading_engine_with_distributed.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.distributed.distributed_monitoring import DistributedMonitoringManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\advanced_analysis\clustering_engine.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\advanced_analysis\clustering_engine.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\advanced_analysis\portfolio_optimizer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\advanced_analysis\portfolio_optimizer.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\advanced_analysis\relationship_network.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\advanced_analysis\relationship_network.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\advanced_analysis\similarity_analyzer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\advanced_analysis\similarity_analyzer.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\distributed\distributed_trading_node.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\distributed\distributed_trading_node.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.distributed.distributed_lock import DistributedLockManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\distributed\distributed_trading_node.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.distributed.config_center import ConfigCenterManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\distributed\distributed_trading_node.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.distributed.distributed_monitoring import DistributedMonitoringManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\distributed\distributed_trading_node.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\distributed\intelligent_order_router.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\distributed\intelligent_order_router.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.distributed.config_center import ConfigCenterManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\distributed\intelligent_order_router.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.distributed.distributed_monitoring import DistributedMonitoringManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\distributed\intelligent_order_router.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\execution\execution_engine.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.core.config.unified_manager import UnifiedConfigManager as ConfigManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\execution\execution_engine.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.core.monitoring.metrics import MetricsCollector`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\execution\multi_market_adapter.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\execution\order_router.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.core.monitoring.performance_monitor import PerformanceMonitor, PerformanceMetric`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\execution\order_router.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.core.config.unified_manager import UnifiedConfigManager as ConfigManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\execution\reporting.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\ml_integration\auto_optimizer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\ml_integration\auto_optimizer.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\ml_integration\hyperparameter_tuner.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\ml_integration\hyperparameter_tuner.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\ml_integration\multi_objective_optimizer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\ml_integration\multi_objective_optimizer.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\ml_integration\optimization_engine.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\ml_integration\performance_predictor.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\ml_integration\recommendation_engine.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\ml_integration\similarity_analyzer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\ml_integration\similarity_analyzer.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\ml_integration\strategy_recommender.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\ml_integration\strategy_recommender.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\portfolio\portfolio_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\portfolio\portfolio_optimizer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\portfolio\strategy_portfolio.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\realtime\realtime_trading_system.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\risk\risk_compliance_engine.py`
**严重程度**: high
**导入语句**: `from src.core.event_bus import EventBus, Event, EventPriority`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\risk\risk_compliance_engine.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.di.container import DependencyContainer`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\risk\risk_controller.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\risk\risk_controller.py`
**严重程度**: high
**导入语句**: `from src.features.feature_engineer import FeatureEngineer`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\risk\risk_controller.py`
**严重程度**: high
**导入语句**: `from src.acceleration.fpga import FpgaManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\risk\china\circuit_breaker.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\risk\china\market_rule_checker.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\risk\china\position_limits.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\risk\china\star_market.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\risk\china\star_market_adapter.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\risk\china\t1_restriction.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\settlement\settlement_engine.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\base_strategy.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.interfaces.standard_interfaces import TradingStrategy`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\enhanced.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\factory.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\multi_strategy_integration.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\performance_evaluation.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\reinforcement_learning.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\china\base_strategy.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\china\basic_strategy.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\china\dragon_tiger.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\china\limit_up.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\china\margin.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\china\ml_strategy.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\china\st.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\china\star_market_strategy.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\optimization\advanced_optimizer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\optimization\genetic_optimizer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\optimization\performance_tuner.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\optimization\performance_tuner.py`
**严重程度**: high
**导入语句**: `from src.data.data_manager import DataManagerSingleton`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\optimization\performance_tuner.py`
**严重程度**: high
**导入语句**: `from src.features.feature_manager import FeatureManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategies\optimization\performance_tuner.py`
**严重程度**: high
**导入语句**: `from src.models.model_manager import ModelManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategy\high_freq_optimizer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategy_workspace\analyzer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategy_workspace\analyzer.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategy_workspace\optimizer.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategy_workspace\optimizer.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategy_workspace\simulator.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategy_workspace\simulator.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategy_workspace\store.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategy_workspace\store.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategy_workspace\strategy_generator.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategy_workspace\strategy_generator.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategy_workspace\visual_editor.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategy_workspace\web_interface.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategy_workspace\web_interface.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\strategy_workspace\web_interface_demo.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\universe\adaptive_factor_model.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\universe\comprehensive_scoring.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\universe\dynamic_universe_manager.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\universe\dynamic_weight_adjuster.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\universe\filters.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: trading
**文件**: `trading\universe\intelligent_updater.py`
**严重程度**: high
**导入语句**: `from src.engine.logging.unified_logger import get_unified_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\stress_test.py`
**严重程度**: high
**导入语句**: `from src.utils.logger import get_logger`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\stress_test.py`
**严重程度**: high
**导入语句**: `from src.data.china.china_data_adapter import ChinaDataAdapter`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\optimization\buffer_optimizer.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.monitoring import MetricsCollector`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\optimization\buffer_optimizer.py`
**严重程度**: high
**导入语句**: `from src.data.market_data import MarketData`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\optimization\dispatcher_optimizer.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.monitoring import MetricsCollector`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\optimization\dispatcher_optimizer.py`
**严重程度**: high
**导入语句**: `from src.data.market_data import MarketData`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\optimization\level2_optimizer.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.monitoring import MetricsCollector`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\optimization\level2_optimizer.py`
**严重程度**: high
**导入语句**: `from src.data.market_data import MarketData`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\app_factory.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.health.health_check import HealthCheck`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\app_factory.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.monitoring.application_monitor import ApplicationMonitor`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\app_factory.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.config.unified_manager import UnifiedConfigManager as ConfigManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\app_factory.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.error.error_handler import ErrorHandler`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\app_factory.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.monitoring.resource_api import ResourceAPI`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\app_factory.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.resource import GPUManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\app_factory.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.resource.resource_manager import ResourceManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\data_api.py`
**严重程度**: high
**导入语句**: `from src.data import DataManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\data_api.py`
**严重程度**: high
**导入语句**: `from src.data.monitoring import PerformanceMonitor`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\data_api.py`
**严重程度**: high
**导入语句**: `from src.data.quality import DataQualityMonitor, AdvancedQualityMonitor`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\data_api.py`
**严重程度**: high
**导入语句**: `from src.data.loader import (`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\unified_dashboard.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.config.unified_manager import UnifiedConfigManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\unified_dashboard.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.monitoring.application_monitor import ApplicationMonitor`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\unified_dashboard.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.resource.resource_manager import ResourceManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\unified_dashboard.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.health.health_check import HealthCheck`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\websocket_api.py`
**严重程度**: high
**导入语句**: `from src.data import DataManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\websocket_api.py`
**严重程度**: high
**导入语句**: `from src.data.monitoring import PerformanceMonitor`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\websocket_api.py`
**严重程度**: high
**导入语句**: `from src.data.quality import DataQualityMonitor, AdvancedQualityMonitor`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\websocket_api.py`
**严重程度**: high
**导入语句**: `from src.data.loader import (`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\websocket_api.py`
**严重程度**: high
**导入语句**: `from src.data.data_manager import DataManagerSingleton`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\modules\config_module.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.config.unified_manager import UnifiedConfigManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\modules\fpga_module.py`
**严重程度**: high
**导入语句**: `from src.acceleration.fpga.fpga_performance_monitor import FPGAPerformanceMonitor`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\modules\fpga_module.py`
**严重程度**: high
**导入语句**: `from src.acceleration.fpga.fpga_manager import FPGAManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\modules\resource_module.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.resource.resource_manager import ResourceManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\modules\resource_module.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.resource.gpu_manager import GPUManager`
**问题描述**: 跨层级导入

### 🔴 Dependency
**层级**: engine
**文件**: `engine\web\modules\resource_module.py`
**严重程度**: high
**导入语句**: `from src.infrastructure.resource.quota_manager import QuotaManager`
**问题描述**: 跨层级导入

## 💡 改进建议

- 🔴 紧急修复: 处理 399 个高严重度架构问题
- 🟡 重点关注: 处理 247 个中等严重度问题
- 📋 检查 core 层级: 3 个文件职责匹配度低
- 📋 检查 infrastructure 层级: 16 个文件职责匹配度低
- 📋 检查 data 层级: 38 个文件职责匹配度低
- 📋 检查 features 层级: 17 个文件职责匹配度低
- 📋 检查 backtest 层级: 2 个文件职责匹配度低
- 📋 检查 risk 层级: 2 个文件职责匹配度低
- 📋 检查 trading 层级: 24 个文件职责匹配度低
- ⚡ 优化 core 层级: 减少 1 个跨层级导入
- ⚡ 优化 infrastructure 层级: 减少 40 个跨层级导入
- ⚡ 优化 data 层级: 减少 97 个跨层级导入
- ⚡ 优化 features 层级: 减少 77 个跨层级导入
- ⚡ 优化 ml 层级: 减少 12 个跨层级导入
- ⚡ 优化 backtest 层级: 减少 6 个跨层级导入
- ⚡ 优化 trading 层级: 减少 110 个跨层级导入
- ⚡ 优化 engine 层级: 减少 34 个跨层级导入
- 🔴 架构质量需紧急改进

## 📈 审计质量评分

### 综合指标
- **总审计层级**: 10 个
- **分析文件数**: 789 个
- **职责违规数**: 359 个
- **接口符合率**: 66.7%
- **依赖问题数**: 377 个
- **综合质量分**: 0.0/100

### 评分标准
- **90-100**: 优秀架构质量
- **70-89**: 良好架构质量
- **50-69**: 一般架构质量
- **0-49**: 需要改进架构质量

---

**审计工具**: scripts/deep_architecture_audit.py
**审计标准**: 基于架构设计文档 v5.0
**建议处理**: 按严重程度从高到低修复问题
