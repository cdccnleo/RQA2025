# 基础设施层日志依赖修复报告

## 修复统计
- 修复文件数: 109
- 跳过文件数: 509
- 错误文件数: 0

## 修复的文件
### src\infrastructure\error_handler.py
- 备份: backup\logging_dependency_fix\src\infrastructure\error_handler.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\inference_engine.py
- 备份: backup\logging_dependency_fix\src\infrastructure\inference_engine.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\init_infrastructure.py
- 备份: backup\logging_dependency_fix\src\infrastructure\init_infrastructure.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\compliance\regulatory_compliance.py
- 备份: backup\logging_dependency_fix\src\infrastructure\compliance\regulatory_compliance.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\compliance\regulatory_reporter.py
- 备份: backup\logging_dependency_fix\src\infrastructure\compliance\regulatory_reporter.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\core\config_storage.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\core\config_storage.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\core\config_version_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\core\config_version_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\core\performance.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\core\performance.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\core\provider.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\core\provider.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\core\unified_validator.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\core\unified_validator.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\error\error_handler.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\error\error_handler.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\error\retry_handler.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\error\retry_handler.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\error\unified_error_exceptions.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\error\unified_error_exceptions.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\monitoring\audit_logger.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\monitoring\audit_logger.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\monitoring\config_monitor.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\monitoring\config_monitor.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\monitoring\health_checker.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\monitoring\health_checker.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\performance\cache_optimizer.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\performance\cache_optimizer.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\performance\concurrency_controller.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\performance\concurrency_controller.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\performance\performance_monitor.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\performance\performance_monitor.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\services\config_encryption_service.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\services\config_encryption_service.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\services\config_sync_service.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\services\config_sync_service.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\services\hot_reload_service.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\services\hot_reload_service.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\services\session_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\services\session_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\services\sync_conflict_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\services\sync_conflict_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\services\sync_node_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\services\sync_node_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\services\unified_hot_reload.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\services\unified_hot_reload.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\services\unified_hot_reload_service.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\services\unified_hot_reload_service.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\services\unified_service.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\services\unified_service.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\services\unified_sync.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\services\unified_sync.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\services\unified_sync_service.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\services\unified_sync_service.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\services\user_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\services\user_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\services\version_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\services\version_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\services\web_auth_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\services\web_auth_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\services\web_config_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\services\web_config_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\services\web_management_service.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\services\web_management_service.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\storage\database_storage.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\storage\database_storage.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\strategies\unified_loaders.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\strategies\unified_loaders.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\strategies\unified_strategy.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\strategies\unified_strategy.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\utils\dependency.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\utils\dependency.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\utils\migration.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\utils\migration.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\config\web\app.py
- 备份: backup\logging_dependency_fix\src\infrastructure\config\web\app.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\dashboard\resource_dashboard.py
- 备份: backup\logging_dependency_fix\src\infrastructure\dashboard\resource_dashboard.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\dashboard\strategy_analyzer_dashboard.py
- 备份: backup\logging_dependency_fix\src\infrastructure\dashboard\strategy_analyzer_dashboard.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\database\audit_logger.py
- 备份: backup\logging_dependency_fix\src\infrastructure\database\audit_logger.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\database\config_validator.py
- 备份: backup\logging_dependency_fix\src\infrastructure\database\config_validator.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\database\database_health_monitor.py
- 备份: backup\logging_dependency_fix\src\infrastructure\database\database_health_monitor.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\database\data_consistency_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\database\data_consistency_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\database\enhanced_database_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\database\enhanced_database_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\database\health_check_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\database\health_check_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\database\influxdb_error_handler.py
- 备份: backup\logging_dependency_fix\src\infrastructure\database\influxdb_error_handler.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\database\optimized_connection_pool.py
- 备份: backup\logging_dependency_fix\src\infrastructure\database\optimized_connection_pool.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\database\slow_query_monitor.py
- 备份: backup\logging_dependency_fix\src\infrastructure\database\slow_query_monitor.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\database\unified_database_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\database\unified_database_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\di\enhanced_container.py
- 备份: backup\logging_dependency_fix\src\infrastructure\di\enhanced_container.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\distributed\config_center.py
- 备份: backup\logging_dependency_fix\src\infrastructure\distributed\config_center.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\distributed\distributed_lock.py
- 备份: backup\logging_dependency_fix\src\infrastructure\distributed\distributed_lock.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\distributed\distributed_monitoring.py
- 备份: backup\logging_dependency_fix\src\infrastructure\distributed\distributed_monitoring.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\docs\document_generator.py
- 备份: backup\logging_dependency_fix\src\infrastructure\docs\document_generator.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\docs\document_quality_checker.py
- 备份: backup\logging_dependency_fix\src\infrastructure\docs\document_quality_checker.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\docs\document_sync_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\docs\document_sync_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\docs\document_version_controller.py
- 备份: backup\logging_dependency_fix\src\infrastructure\docs\document_version_controller.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\email\secure_config.py
- 备份: backup\logging_dependency_fix\src\infrastructure\email\secure_config.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\error\circuit_breaker.py
- 备份: backup\logging_dependency_fix\src\infrastructure\error\circuit_breaker.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\error\comprehensive_error_plugin.py
- 备份: backup\logging_dependency_fix\src\infrastructure\error\comprehensive_error_plugin.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\error\enhanced_error_handler.py
- 备份: backup\logging_dependency_fix\src\infrastructure\error\enhanced_error_handler.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\error\error_handler.py
- 备份: backup\logging_dependency_fix\src\infrastructure\error\error_handler.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\error\retry_handler.py
- 备份: backup\logging_dependency_fix\src\infrastructure\error\retry_handler.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\error\trading_error_handler.py
- 备份: backup\logging_dependency_fix\src\infrastructure\error\trading_error_handler.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\error\unified_error_handler.py
- 备份: backup\logging_dependency_fix\src\infrastructure\error\unified_error_handler.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\health\health_check.py
- 备份: backup\logging_dependency_fix\src\infrastructure\health\health_check.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\interfaces\base.py
- 备份: backup\logging_dependency_fix\src\infrastructure\interfaces\base.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\interfaces\compatibility.py
- 备份: backup\logging_dependency_fix\src\infrastructure\interfaces\compatibility.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\interfaces\unified_interface_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\interfaces\unified_interface_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\logging\business_log_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\logging\business_log_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\logging\enhanced_log_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\logging\enhanced_log_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger
- 替换: from src\.engine\.logging\.unified_context import UnifiedLogContext -> from ..logging.infrastructure_logger import InfrastructureLogContext

### src\infrastructure\logging\logging_strategy.py
- 备份: backup\logging_dependency_fix\src\infrastructure\logging\logging_strategy.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\logging\log_aggregator_plugin.py
- 备份: backup\logging_dependency_fix\src\infrastructure\logging\log_aggregator_plugin.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\logging\quant_filter.py
- 备份: backup\logging_dependency_fix\src\infrastructure\logging\quant_filter.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\logging\security_filter.py
- 备份: backup\logging_dependency_fix\src\infrastructure\logging\security_filter.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\logging\trading_logger.py
- 备份: backup\logging_dependency_fix\src\infrastructure\logging\trading_logger.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\logging\unified_logging_interface.py
- 备份: backup\logging_dependency_fix\src\infrastructure\logging\unified_logging_interface.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger
- 替换: from src\.engine\.logging\.unified_context import UnifiedLogContext -> from ..logging.infrastructure_logger import InfrastructureLogContext

### src\infrastructure\logging\logger\logger.py
- 备份: backup\logging_dependency_fix\src\infrastructure\logging\logger\logger.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\monitoring\application_monitor.py
- 备份: backup\logging_dependency_fix\src\infrastructure\monitoring\application_monitor.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\monitoring\automation_monitor.py
- 备份: backup\logging_dependency_fix\src\infrastructure\monitoring\automation_monitor.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\monitoring\decorators.py
- 备份: backup\logging_dependency_fix\src\infrastructure\monitoring\decorators.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\monitoring\health_checker.py
- 备份: backup\logging_dependency_fix\src\infrastructure\monitoring\health_checker.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\monitoring\influxdb_store.py
- 备份: backup\logging_dependency_fix\src\infrastructure\monitoring\influxdb_store.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\monitoring\metrics_collector.py
- 备份: backup\logging_dependency_fix\src\infrastructure\monitoring\metrics_collector.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\monitoring\model_monitor_plugin.py
- 备份: backup\logging_dependency_fix\src\infrastructure\monitoring\model_monitor_plugin.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\monitoring\prometheus_monitor.py
- 备份: backup\logging_dependency_fix\src\infrastructure\monitoring\prometheus_monitor.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\monitoring\resource_api.py
- 备份: backup\logging_dependency_fix\src\infrastructure\monitoring\resource_api.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\monitoring\system_monitor.py
- 备份: backup\logging_dependency_fix\src\infrastructure\monitoring\system_monitor.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\ops\deployment_plugin.py
- 备份: backup\logging_dependency_fix\src\infrastructure\ops\deployment_plugin.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\ops\monitoring_dashboard.py
- 备份: backup\logging_dependency_fix\src\infrastructure\ops\monitoring_dashboard.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\performance\performance_optimizer_plugin.py
- 备份: backup\logging_dependency_fix\src\infrastructure\performance\performance_optimizer_plugin.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\resource\gpu_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\resource\gpu_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\resource\quota_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\resource\quota_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\resource\resource_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\resource\resource_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\security\auth_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\security\auth_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\security\enhanced_security_manager.py
- 备份: backup\logging_dependency_fix\src\infrastructure\security\enhanced_security_manager.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\security\security.py
- 备份: backup\logging_dependency_fix\src\infrastructure\security\security.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\security\security_auditor.py
- 备份: backup\logging_dependency_fix\src\infrastructure\security\security_auditor.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\storage\archive_failure_handler.py
- 备份: backup\logging_dependency_fix\src\infrastructure\storage\archive_failure_handler.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\storage\data_consistency.py
- 备份: backup\logging_dependency_fix\src\infrastructure\storage\data_consistency.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\storage\unified_query.py
- 备份: backup\logging_dependency_fix\src\infrastructure\storage\unified_query.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\testing\chaos_engine.py
- 备份: backup\logging_dependency_fix\src\infrastructure\testing\chaos_engine.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\testing\chaos_orchestrator.py
- 备份: backup\logging_dependency_fix\src\infrastructure\testing\chaos_orchestrator.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### src\infrastructure\trading\persistent_error_handler.py
- 备份: backup\logging_dependency_fix\src\infrastructure\trading\persistent_error_handler.py
- 替换: from src\.engine\.logging\.unified_logger import get_unified_logger -> from ..logging.infrastructure_logger import get_unified_logger

### tests\unit\infrastructure\logging\test_enhanced_log_manager.py
- 备份: backup\logging_dependency_fix\tests\unit\infrastructure\logging\test_enhanced_log_manager.py
- 替换: from src\.engine\.logging\.unified_context import UnifiedLogContext -> from ..logging.infrastructure_logger import InfrastructureLogContext

## 后续建议
1. 运行测试验证修复效果
2. 检查内存使用情况
3. 验证日志功能正常