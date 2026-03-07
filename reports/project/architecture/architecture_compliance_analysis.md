# 测试文件架构合规性分析报告

## features层分析

### ⚠️ 废弃 (9个文件)

**tests\unit\features\auto_test_feature_engine.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\features\auto_test_feature_engineer.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\features\auto_test_feature_manager.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\features\auto_test_sentiment_analyzer.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\features\auto_test_signal_generator.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\features\test_feature_engineer_isolated.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\features\test_feature_importance_isolated.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\features\test_feature_manager_offline.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\features\test_feature_metadata_isolated.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

### 🗑️ 删除 (1个文件)

**tests\unit\features\conftest.py**
问题:
- 文件符合删除模式
建议:
- 建议删除此文件

### 🔧 需更新 (30个文件)

**tests\unit\features\test_enums.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\test_feature_config.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\test_feature_engine.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\test_feature_engineer.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\test_feature_importance.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\test_feature_manager.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\test_feature_metadata.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\test_feature_processor.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\features\test_feature_saver.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\features\test_feature_selector.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureStandardizer', 'FeatureSaver']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\features\test_feature_standardizer.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\test_high_freq_optimizer.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
- 测试了废弃模块: ['HighFreqOptimizer']
建议:
- 建议添加对核心组件的测试
- 建议移除对废弃模块的测试

**tests\unit\features\test_integration.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\features\test_level2_analyzer.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\test_minimal_feature_main_flow.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\test_orderbook_config.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\features\test_orderbook_metrics.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\test_order_book_analyzer.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\features\test_sentiment_analyzer.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\test_signal_generator.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\test_technical_processor.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\features\optimizer\test_optimizers.py**
问题:
- 缺少对核心组件的测试: ['FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
- 测试了废弃模块: ['HighFreqOptimizer']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议移除对废弃模块的测试
- 建议增加测试用例

**tests\unit\features\orderbook\test_analyzer.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\orderbook\test_order_book_analyzer.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\features\processors\test_sentiment_processor.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\processors\test_technical.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\processors\test_technical_processor.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\sentiment\test_sentiment_analyzer.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\sentiment\test_sentiment_analyzer_full.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

**tests\unit\features\technical\test_processor.py**
问题:
- 缺少对核心组件的测试: ['FeatureEngineer', 'FeatureProcessor', 'FeatureSelector', 'FeatureStandardizer', 'FeatureSaver']
建议:
- 建议添加对核心组件的测试

## infrastructure层分析

### ⚠️ 废弃 (52个文件)

**tests\unit\infrastructure\auto_test_circuit_breaker.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\auto_test_service_launcher.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\auto_test_visual_monitor.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\test_async_inference_engine_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\test_async_inference_engine_coverage.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\test_config_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\test_config_manager_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\test_config_manager_coverage.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\test_config_manager_simple.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\test_core_modules_simple.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\test_core_modules_standalone.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\test_error_handling_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\test_integration_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\test_logging_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\test_service_launcher_coverage.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\config\test_config_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\config\test_config_exceptions_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\config\test_config_result_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\config\test_config_schema_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\config\test_database_storage_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\config\test_database_storage_standalone.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\config\test_file_storage_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\config\test_redis_storage_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\config\test_storage_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\config\test_strategies_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\config\test_typed_config_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\config\test_unified_cache_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\config\test_unified_config_manager_simple.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\database\test_database_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\database\test_database_manager_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\database\test_database_manager_simple.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\database\test_unified_database_manager_simple.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\deployment\test_deployment_manager_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\error\test_circuit_breaker_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\error\test_circuit_breaker_simple.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\error\test_error_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\error\test_error_handler_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\error\test_error_handling_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\logging\test_logging_system_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\monitoring\test_monitoring_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\monitoring\test_monitoring_system_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\m_logging\test_logging_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\m_logging\test_logging_manager_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\m_logging\test_log_manager_simple.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\network\test_network_manager_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\scheduler\test_task_scheduler_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\security\test_security_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\security\test_security_system_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\storage\test_kafka_storage_coverage.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\storage\test_storage_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\storage\test_storage_system_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

**tests\unit\infrastructure\utils\test_utils_comprehensive.py**
问题:
- 文件符合废弃模式
建议:
- 建议废弃此文件，保留核心功能测试

### 🔧 需更新 (258个文件)

**tests\unit\infrastructure\test_alert_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_application_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_app_factory.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_async_inference_engine.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_async_inference_engine_top20.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_auto_recovery.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_backtest_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_behavior_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_cache_manager.py**
问题:
- 缺少对核心组件的测试: ['DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_circuit_breaker.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_circuit_breaker_fixed.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_circuit_breaker_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_circuit_breaker_tester.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_config_event.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_config_exceptions.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_config_loader_service.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_config_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_config_version.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_connection_pool.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_coverage_boost.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_database_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_database_storage.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_data_sync.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_db.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_degradation_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_deployment_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_deployment_validator.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_disaster_recovery.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_env_loader.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_error_handler.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_event.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_event_service.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_factory.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_file_storage.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_final_deployment_check.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_gpu_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_infrastructure.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_infrastructure_core.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_init_infrastructure.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_json_loader.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_load_balancer.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_lock.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_market_aware_retry_test.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_metrics.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_metrics_collector.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_minimal_infra_main_flow.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_model_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_monitoring.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_monitoring_system.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_monitor_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_network_exceptions.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_network_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_network_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_notification.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_performance_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_persistent_error_handler_test.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_prometheus_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_quota_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_redis_storage.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_regulatory_compliance.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_regulatory_reporter.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_resource_api.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_resource_dashboard.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_resource_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_retry_policy.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_schema_validator.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_service_launcher.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_standard_interfaces.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_storage_exceptions.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_storage_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\test_system_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_thread_management.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_unified_config_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_validators.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_version.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_visual_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_yaml_loader.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\auto_test_config\config_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\auto_test_database\database_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\auto_test_monitoring\system_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\auto_test_m_logging\logger.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\cache\test_icache_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\cache\test_thread_safe_cache.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\compliance\test_regulatory_compliance.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\compliance\test_regulatory_reporter.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\compliance\test_report_generator.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_abstract_classes_fixed.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_ashare_features.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_cacheservice.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_config_encryption.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_config_interfaces.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\config\test_config_loader.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_config_schema_fixed.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_config_service_refactored.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_config_storage.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_diff_service.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_encryption_service_fixed.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_error_handling.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_interfaces_fixed.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_jsonloader.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_lock_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_migration.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_optimized_cache_service.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_performance.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_performance_basic.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_security.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_typed_config_fixed.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_unified_config_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_unified_strategies.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_user_session_managers.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_version_service.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_version_storage.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\config\test_yamlloader.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\config\services\test_cache_service.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\core\test_core_modules.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\dashboard\test_resource_dashboard.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\database\test_connection_pool.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\database\test_database_health_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\database\test_database_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\database\test_database_manager_fixed.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\database\test_database_manager_focused.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\database\test_database_refactor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\database\test_data_consistency_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\database\test_influxdb_adapter.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\database\test_influxdb_error_handler.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\database\test_influxdb_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\database\test_migrator.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\database\test_monitoring_components.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\database\test_performance_optimization.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\database\test_postgresql_adapter.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\database\test_redis_adapter.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\database\test_sqlite_adapter.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\database\test_unified_database_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\database\test_unified_data_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\degradation\test_degradation_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\di\test_container.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\disaster\disaster_recovery_test.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\distributed\test_config_center.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\distributed\test_distributed_lock.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\distributed\test_distributed_monitoring.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\error\error_handler_test.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\error\test_circuit_breaker_focused.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\error\test_error_handler.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\error\test_error_handler_advanced.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\error\test_error_handler_fixed.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\error\test_error_handler_focused.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\error\test_error_handling.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\error\test_exceptions.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\error\test_retry_handler.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\error\test_trading_error_handler.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\health\test_health_check.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\health\test_health_checker.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\logging\test_logging.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\message_queue\test_message_queue.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\application_monitor_test.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\monitoring_test.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\monitoring\monitor_test.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\test_alert_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\test_application_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\test_automation_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\test_backtest_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\test_behavior_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\test_business_metrics_collector.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\test_decorators.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\test_disaster_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\test_influxdb_store.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\test_metrics.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\monitoring\test_metrics_collector.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\test_model_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\test_monitoring.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\test_monitoringservice.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\monitoring\test_monitoring_extreme.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\test_monitoring_system.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\monitoring\test_performance_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\test_prometheus_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\test_resource_api.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\test_storage_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\monitoring\test_system_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\m_logging\test_advanced_logger.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\m_logging\test_boundary_conditions.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\m_logging\test_boundary_conditions_simplified.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\m_logging\test_config_validator.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\m_logging\test_enhanced_log_sampler.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\m_logging\test_integration.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\m_logging\test_logger.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\m_logging\test_logging_basic.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\m_logging\test_log_aggregator.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\m_logging\test_log_compressor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\m_logging\test_log_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\m_logging\test_log_manager_integration.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\m_logging\test_log_manager_working.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\m_logging\test_log_metrics.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\m_logging\test_log_sampler.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\m_logging\test_optimized_components.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\m_logging\test_quant_filter.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\m_logging\test_resource_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\m_logging\test_security_filter.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\network\test_network_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\notification\test_notification_service.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\resource\resource_manager_test.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\resource\test_gpu_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\resource\test_quota_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\resource\test_resource_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\scheduler\test_exceptions.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\scheduler\test_job_scheduler.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\scheduler\test_priority_queue.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\scheduler\test_scheduler_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\scheduler\test_task_scheduler.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\security\test_data_sanitizer.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\security\test_security.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\security\test_security_service.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\storage\test_archive_failure_handler.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\storage\test_core.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\storage\test_data_consistency.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\storage\test_file_storage.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\storage\test_kafka_storage.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\storage\test_redis.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\storage\test_redis_cluster.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\storage\test_redis_enhanced.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\storage\test_storage.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\storage\test_storage_core.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\storage\test_unified_query.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\storage\adapters\test_database_adapter.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\storage\adapters\test_database_adapter_fixed.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\storage\adapters\test_file_system_adapter.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\storage\adapters\test_redis_adapter.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\testing\test_chaos_engine.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\testing\test_chaos_integration.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\testing\test_chaos_orchestrator.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\testing\test_config_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\testing\test_disaster_recovery.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\testing\test_disaster_tester.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\testing\test_error_handler.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\testing\test_regulatory_tester.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\testing\test_regulatory_tester_new.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_config\config_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_database\database_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_monitoring\system_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\test_m_logging\logger.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\third_party\test_third_party_integration.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\utils\test_data_generator.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\utils\test_data_visualizer.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\utils\test_date_utils.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\utils\test_environment_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\utils\test_exception_utils_enhanced.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\infrastructure\utils\test_logger_focused.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\utils\test_monitor.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\utils\test_performance.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\utils\test_report_generator.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\utils\test_tools.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\utils\test_utils.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
- 测试文件过大，可能过于复杂
建议:
- 建议添加对核心组件的测试
- 建议拆分为多个测试文件

**tests\unit\infrastructure\versioning\test_data_version_manager.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\versioning\test_data_version_manager_debug.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\versioning\test_minimal_debug.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\versioning\test_storage_adapter.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\versioning\test_timestamp_conflict_debug.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\web\app_factory_test.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\infrastructure\web\test_app_factory.py**
问题:
- 缺少对核心组件的测试: ['CacheManager', 'DatabaseManager', 'MonitorManager', 'ConfigManager']
建议:
- 建议添加对核心组件的测试

### 🗑️ 删除 (2个文件)

**tests\unit\infrastructure\config\conftest.py**
问题:
- 文件符合删除模式
建议:
- 建议删除此文件

**tests\unit\infrastructure\error\conftest.py**
问题:
- 文件符合删除模式
建议:
- 建议删除此文件

## integration层分析

### 🔧 需更新 (11个文件)

**tests\unit\integration\test_backtest_engine.py**
问题:
- 缺少对核心组件的测试: ['SystemIntegrationManager', 'LayerInterface', 'UnifiedConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\integration\test_china_integration.py**
问题:
- 缺少对核心组件的测试: ['SystemIntegrationManager', 'LayerInterface', 'UnifiedConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\integration\test_data_layer.py**
问题:
- 缺少对核心组件的测试: ['SystemIntegrationManager', 'LayerInterface', 'UnifiedConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\integration\test_feature_engine.py**
问题:
- 缺少对核心组件的测试: ['SystemIntegrationManager', 'LayerInterface', 'UnifiedConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\integration\test_fpga_optimizer.py**
问题:
- 缺少对核心组件的测试: ['SystemIntegrationManager', 'LayerInterface', 'UnifiedConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\integration\test_infrastructure_integration.py**
问题:
- 缺少对核心组件的测试: ['SystemIntegrationManager', 'LayerInterface', 'UnifiedConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\integration\test_integration.py**
问题:
- 缺少对核心组件的测试: ['SystemIntegrationManager', 'LayerInterface', 'UnifiedConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\integration\test_layer_interface.py**
问题:
- 缺少对核心组件的测试: ['SystemIntegrationManager', 'UnifiedConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\integration\test_sentiment_analyzer.py**
问题:
- 缺少对核心组件的测试: ['SystemIntegrationManager', 'LayerInterface', 'UnifiedConfigManager']
建议:
- 建议添加对核心组件的测试

**tests\unit\integration\test_system_integration_manager.py**
问题:
- 缺少对核心组件的测试: ['LayerInterface', 'UnifiedConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

**tests\unit\integration\test_minimal_data_main_flow\test_test_minimal_data_main_flow.py**
问题:
- 缺少对核心组件的测试: ['SystemIntegrationManager', 'LayerInterface', 'UnifiedConfigManager']
- 测试文件过小，可能测试覆盖不足
建议:
- 建议添加对核心组件的测试
- 建议增加测试用例

## 总体统计

- 总文件数: 363
- 合规文件: 0 (0.0%)
- 废弃文件: 61 (16.8%)
- 需删除文件: 3 (0.8%)
- 需更新文件: 299 (82.4%)
