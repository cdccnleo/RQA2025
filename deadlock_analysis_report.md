# 死锁检测报告

## 概览
- 扫描文件数: 126
- 有问题文件数: 74
- 锁定义总数: 718
- 嵌套锁总数: 152

## 🚨 有问题的文件

### src\infrastructure\base.py
- ⚠️ 过多锁定义 (6) - 可能增加死锁风险
  锁定义:
    - threading.Lock()
    - threading.Lock()
    - Lock()
    - Lock()
    - ._lock =

### src\infrastructure\visual_monitor.py
- ⚠️ 发现嵌套锁使用 (1) - 高死锁风险
  锁定义:
    - threading.Lock()
    - Lock()
    - .lock =
    - self.lock =
  嵌套锁:
    - 第334行: 3层嵌套

### src\infrastructure\cache\advanced_cache_manager.py
- ⚠️ 发现嵌套锁使用 (1) - 高死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
    - ._lock =
    - self._lock =
  嵌套锁:
    - 第305行: 2层嵌套

### src\infrastructure\cache\cache_strategy_manager.py
- ⚠️ 过多锁定义 (10) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (1) - 高死锁风险
  锁定义:
    - threading.RLock()
    - threading.RLock()
    - Lock()
    - Lock()
    - RLock()
  嵌套锁:
    - 第570行: 8层嵌套

### src\infrastructure\cache\cache_utils.py
- ⚠️ 过多锁定义 (20) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (2) - 高死锁风险
  锁定义:
    - threading.RLock()
    - threading.RLock()
    - threading.RLock()
    - threading.RLock()
    - Lock()
  嵌套锁:
    - 第418行: 19层嵌套
    - 第634行: 19层嵌套

### src\infrastructure\cache\distributed_cache_manager.py
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
    - .lock =
    - self.lock =

### src\infrastructure\cache\distributed_consistency_manager.py
- ⚠️ 发现嵌套锁使用 (3) - 高死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
    - .lock =
    - self.lock =
  嵌套锁:
    - 第201行: 2层嵌套
    - 第243行: 3层嵌套
    - 第252行: 2层嵌套

### src\infrastructure\cache\memory_cache.py
- ⚠️ 过多锁定义 (10) - 可能增加死锁风险
  锁定义:
    - threading.RLock()
    - threading.RLock()
    - Lock()
    - Lock()
    - RLock()

### src\infrastructure\cache\multi_level_cache.py
- ⚠️ 过多锁定义 (15) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (12) - 高死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.RLock()
    - threading.RLock()
    - threading.RLock()
    - Lock()
    - Lock()
  嵌套锁:
    - 第351行: 6层嵌套
    - 第367行: 5层嵌套
    - 第853行: 14层嵌套

### src\infrastructure\cache\optimized_cache_service.py
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
    - ._lock =
    - self._lock =

### src\infrastructure\cache\smart_cache_strategies.py
- ⚠️ 过多锁定义 (25) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (3) - 高死锁风险
  锁定义:
    - threading.RLock()
    - threading.RLock()
    - threading.RLock()
    - threading.RLock()
    - threading.RLock()
  嵌套锁:
    - 第421行: 10层嵌套
    - 第475行: 9层嵌套
    - 第803行: 16层嵌套

### src\infrastructure\cache\smart_cache_strategy.py
- ⚠️ 过多锁定义 (15) - 可能增加死锁风险
  锁定义:
    - threading.RLock()
    - threading.RLock()
    - threading.RLock()
    - Lock()
    - Lock()

### src\infrastructure\cache\unified_cache.py
- ⚠️ 过多锁定义 (15) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (1) - 高死锁风险
  锁定义:
    - threading.RLock()
    - threading.RLock()
    - threading.RLock()
    - Lock()
    - Lock()
  嵌套锁:
    - 第265行: 6层嵌套

### src\infrastructure\cache\unified_cache_manager_refactored.py
- ⚠️ 发现嵌套锁使用 (5) - 高死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
    - .lock =
    - self.lock =
  嵌套锁:
    - 第696行: 6层嵌套
    - 第943行: 5层嵌套
    - 第1137行: 6层嵌套

### src\infrastructure\config\core\config_storage.py
- ⚠️ 过多锁定义 (15) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (8) - 高死锁风险
  锁定义:
    - threading.RLock()
    - threading.RLock()
    - threading.RLock()
    - Lock()
    - Lock()
  嵌套锁:
    - 第164行: 2层嵌套
    - 第263行: 4层嵌套
    - 第333行: 5层嵌套

### src\infrastructure\config\environment\cloud_native_enhanced.py
- ⚠️ 过多锁定义 (16) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (5) - 高死锁风险
  锁定义:
    - threading.Lock()
    - threading.Lock()
    - threading.Lock()
    - threading.Lock()
    - Lock()
  嵌套锁:
    - 第671行: 6层嵌套
    - 第807行: 5层嵌套
    - 第891行: 4层嵌套

### src\infrastructure\config\monitoring\performance_monitor_dashboard.py
- ⚠️ 过多锁定义 (15) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (2) - 高死锁风险
  锁定义:
    - threading.RLock()
    - threading.RLock()
    - threading.RLock()
    - Lock()
    - Lock()
  嵌套锁:
    - 第1013行: 14层嵌套
    - 第1028行: 13层嵌套

### src\infrastructure\config\security\enhanced_secure_config.py
- ⚠️ 过多锁定义 (23) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (1) - 高死锁风险
  锁定义:
    - threading.Lock()
    - threading.RLock()
    - threading.RLock()
    - threading.RLock()
    - threading.RLock()
  嵌套锁:
    - 第343行: 3层嵌套

### src\infrastructure\config\services\event.py
- ⚠️ 过多锁定义 (7) - 可能增加死锁风险
  锁定义:
    - threading.Lock()
    - threading.RLock()
    - Lock()
    - Lock()
    - RLock()

### src\infrastructure\config\tools\optimization_strategies.py
- ⚠️ 发现嵌套锁使用 (3) - 高死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
    - ._lock =
    - self._lock =
  嵌套锁:
    - 第435行: 2层嵌套
    - 第461行: 3层嵌套
    - 第487行: 4层嵌套

### src\infrastructure\error\archive_failure_handler.py
- ⚠️ 过多锁定义 (6) - 可能增加死锁风险
  锁定义:
    - threading.RLock()
    - threading.RLock()
    - Lock()
    - Lock()
    - RLock()

### src\infrastructure\error\comprehensive_error_plugin.py
- ⚠️ 过多锁定义 (12) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (1) - 高死锁风险
  锁定义:
    - threading.Lock()
    - threading.RLock()
    - threading.RLock()
    - Lock()
    - Lock()
  嵌套锁:
    - 第291行: 3层嵌套

### src\infrastructure\error\container.py
- ⚠️ 过多锁定义 (10) - 可能增加死锁风险
  锁定义:
    - threading.RLock()
    - threading.RLock()
    - Lock()
    - Lock()
    - RLock()

### src\infrastructure\error\enhanced_global_exception_handler.py
- ⚠️ 过多锁定义 (7) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (3) - 高死锁风险
  锁定义:
    - threading.Lock()
    - threading.RLock()
    - Lock()
    - Lock()
    - RLock()
  嵌套锁:
    - 第210行: 3层嵌套
    - 第289行: 3层嵌套
    - 第307行: 2层嵌套

### src\infrastructure\error\error_handler.py
- ⚠️ 过多锁定义 (9) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (4) - 高死锁风险
  锁定义:
    - threading.Lock()
    - threading.RLock()
    - Lock()
    - Lock()
    - RLock()
  嵌套锁:
    - 第193行: 5层嵌套
    - 第218行: 4层嵌套
    - 第228行: 3层嵌套

### src\infrastructure\error\global_exception_handler.py
- ⚠️ 发现嵌套锁使用 (1) - 高死锁风险
  锁定义:
    - threading.Lock()
    - Lock()
    - ._lock =
    - self._lock =
  嵌套锁:
    - 第172行: 2层嵌套

### src\infrastructure\error\handler.py
- ⚠️ 过多锁定义 (8) - 可能增加死锁风险
  锁定义:
    - threading.Lock()
    - Lock()
    - Lock()
    - Lock()
    - ._lock =

### src\infrastructure\error\lock.py
- ⚠️ 过多锁定义 (12) - 可能增加死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
- ⚠️ 锁获取(0)和释放(4)次数不匹配
  锁定义:
    - threading.Lock()
    - threading.Lock()
    - threading.Lock()
    - Lock()
    - Lock()

### src\infrastructure\error\retry_handler.py
- ⚠️ 过多锁定义 (8) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (2) - 高死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.Lock()
    - threading.Lock()
    - Lock()
    - Lock()
    - ._lock =
  嵌套锁:
    - 第393行: 5层嵌套
    - 第399行: 4层嵌套

### src\infrastructure\error\trading_error_handler.py
- ⚠️ 过多锁定义 (6) - 可能增加死锁风险
  锁定义:
    - threading.RLock()
    - threading.RLock()
    - Lock()
    - Lock()
    - RLock()

### src\infrastructure\error\unified_error_handler.py
- ⚠️ 发现嵌套锁使用 (2) - 高死锁风险
  锁定义:
    - threading.Lock()
    - Lock()
    - ._lock =
    - self._lock =
  嵌套锁:
    - 第121行: 4层嵌套
    - 第280行: 9层嵌套

### src\infrastructure\health\enhanced_health_checker.py
- ⚠️ 发现嵌套锁使用 (5) - 高死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
    - ._lock =
    - self._lock =
  嵌套锁:
    - 第243行: 3层嵌套
    - 第288行: 3层嵌套
    - 第307行: 2层嵌套

### src\infrastructure\health\inference_engine.py
- ⚠️ 过多锁定义 (8) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (2) - 高死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.Lock()
    - threading.Lock()
    - threading.Lock()
    - threading.Lock()
    - Lock()
  嵌套锁:
    - 第424行: 5层嵌套
    - 第575行: 10层嵌套

### src\infrastructure\health\monitoring_dashboard.py
- ⚠️ 过多锁定义 (8) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (4) - 高死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.Lock()
    - threading.Lock()
    - Lock()
    - Lock()
    - ._lock =
  嵌套锁:
    - 第362行: 10层嵌套
    - 第521行: 12层嵌套
    - 第531行: 11层嵌套

### src\infrastructure\health\prometheus_exporter.py
- ⚠️ 过多锁定义 (7) - 可能增加死锁风险
  锁定义:
    - threading.Lock()
    - threading.RLock()
    - Lock()
    - Lock()
    - RLock()

### src\infrastructure\logging\async_log_processor.py
- ⚠️ 过多锁定义 (7) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (4) - 高死锁风险
  锁定义:
    - threading.Lock()
    - threading.RLock()
    - Lock()
    - Lock()
    - RLock()
  嵌套锁:
    - 第219行: 5层嵌套
    - 第223行: 4层嵌套
    - 第443行: 4层嵌套

### src\infrastructure\logging\audit.py
- ⚠️ 过多锁定义 (6) - 可能增加死锁风险
  锁定义:
    - threading.Lock()
    - threading.Lock()
    - Lock()
    - Lock()
    - ._lock =

### src\infrastructure\logging\audit_logger.py
- ⚠️ 过多锁定义 (6) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (1) - 高死锁风险
  锁定义:
    - Lock()
    - Lock()
    - ._lock =
    - ._lock =
    - self._lock =
  嵌套锁:
    - 第342行: 2层嵌套

### src\infrastructure\logging\connection_pool.py
- ⚠️ 发现嵌套锁使用 (1) - 高死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
    - .lock =
    - self.lock =
  嵌套锁:
    - 第210行: 5层嵌套

### src\infrastructure\logging\distributed_lock.py
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.Lock()
    - Lock()

### src\infrastructure\logging\distributed_monitoring.py
- ⚠️ 过多锁定义 (10) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (1) - 高死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.Lock()
    - threading.Lock()
    - threading.Lock()
    - Lock()
    - Lock()
  嵌套锁:
    - 第566行: 6层嵌套

### src\infrastructure\logging\enhanced_container.py
- ⚠️ 过多锁定义 (12) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (1) - 高死锁风险
  锁定义:
    - threading.Lock()
    - threading.RLock()
    - threading.RLock()
    - Lock()
    - Lock()
  嵌套锁:
    - 第357行: 3层嵌套

### src\infrastructure\logging\enhanced_logger.py
- ⚠️ 过多锁定义 (9) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (2) - 高死锁风险
  锁定义:
    - threading.Lock()
    - threading.Lock()
    - threading.RLock()
    - Lock()
    - Lock()
  嵌套锁:
    - 第420行: 3层嵌套
    - 第570行: 3层嵌套

### src\infrastructure\logging\hot_reload_service.py
- ⚠️ 过多锁定义 (7) - 可能增加死锁风险
  锁定义:
    - threading.Lock()
    - threading.RLock()
    - Lock()
    - Lock()
    - RLock()

### src\infrastructure\logging\log_archiver.py
- ⚠️ 发现嵌套锁使用 (2) - 高死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.Lock()
    - threading.RLock()
    - Lock()
    - Lock()
    - RLock()
  嵌套锁:
    - 第300行: 4层嵌套
    - 第463行: 4层嵌套

### src\infrastructure\logging\log_correlation_plugin.py
- ⚠️ 发现嵌套锁使用 (3) - 高死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
    - ._lock =
    - self._lock =
  嵌套锁:
    - 第285行: 3层嵌套
    - 第382行: 2层嵌套
    - 第402行: 2层嵌套

### src\infrastructure\logging\log_level_optimizer.py
- ⚠️ 过多锁定义 (7) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (4) - 高死锁风险
  锁定义:
    - threading.Lock()
    - threading.RLock()
    - Lock()
    - Lock()
    - RLock()
  嵌套锁:
    - 第250行: 5层嵌套
    - 第325行: 5层嵌套
    - 第330行: 4层嵌套

### src\infrastructure\logging\log_metrics_plugin.py
- ⚠️ 发现嵌套锁使用 (1) - 高死锁风险
  锁定义:
    - Lock()
    - Lock()
    - ._lock =
    - self._lock =
  嵌套锁:
    - 第228行: 2层嵌套

### src\infrastructure\logging\log_sampler.py
- ⚠️ 发现嵌套锁使用 (1) - 高死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
    - ._lock =
    - self._lock =
  嵌套锁:
    - 第333行: 9层嵌套

### src\infrastructure\logging\metrics_aggregator.py
- ⚠️ 发现嵌套锁使用 (4) - 高死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
    - ._lock =
    - self._lock =
  嵌套锁:
    - 第239行: 2层嵌套
    - 第323行: 3层嵌套
    - 第596行: 7层嵌套

### src\infrastructure\logging\microservice_manager.py
- ⚠️ 过多锁定义 (25) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (5) - 高死锁风险
  锁定义:
    - threading.RLock()
    - threading.RLock()
    - threading.RLock()
    - threading.RLock()
    - threading.RLock()
  嵌套锁:
    - 第397行: 12层嵌套
    - 第463行: 12层嵌套
    - 第939行: 25层嵌套

### src\infrastructure\logging\micro_service.py
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()

### src\infrastructure\logging\service_launcher.py
- ⚠️ 发现嵌套锁使用 (2) - 高死锁风险
  锁定义:
    - threading.Lock()
    - Lock()
    - .lock =
    - self.lock =
  嵌套锁:
    - 第284行: 4层嵌套
    - 第323行: 4层嵌套

### src\infrastructure\logging\slow_query_monitor.py
- ⚠️ 发现嵌套锁使用 (4) - 高死锁风险
  锁定义:
    - Lock()
    - ._lock =
    - self._lock =
  嵌套锁:
    - 第345行: 6层嵌套
    - 第367行: 5层嵌套
    - 第424行: 4层嵌套

### src\infrastructure\logging\smart_log_filter.py
- ⚠️ 过多锁定义 (20) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (6) - 高死锁风险
  锁定义:
    - threading.Lock()
    - threading.RLock()
    - threading.RLock()
    - threading.RLock()
    - threading.RLock()
  嵌套锁:
    - 第429行: 10层嵌套
    - 第436行: 9层嵌套
    - 第472行: 9层嵌套

### src\infrastructure\logging\storage_adapter.py
- ⚠️ 发现嵌套锁使用 (4) - 高死锁风险
  锁定义:
    - Lock()
    - RLock()
    - ._lock =
    - self._lock =
  嵌套锁:
    - 第68行: 2层嵌套
    - 第89行: 2层嵌套
    - 第169行: 5层嵌套

### src\infrastructure\logging\structured_logger.py
- ⚠️ 过多锁定义 (9) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (2) - 高死锁风险
  锁定义:
    - threading.Lock()
    - threading.Lock()
    - threading.RLock()
    - Lock()
    - Lock()
  嵌套锁:
    - 第415行: 3层嵌套
    - 第565行: 3层嵌套

### src\infrastructure\logging\unified_hot_reload_service.py
- ⚠️ 过多锁定义 (7) - 可能增加死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.Lock()
    - threading.RLock()
    - Lock()
    - Lock()
    - RLock()

### src\infrastructure\logging\unified_logger.py
- ⚠️ 过多锁定义 (6) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (5) - 高死锁风险
  锁定义:
    - threading.Lock()
    - threading.Lock()
    - Lock()
    - Lock()
    - ._lock =
  嵌套锁:
    - 第368行: 6层嵌套
    - 第383行: 5层嵌套
    - 第393行: 4层嵌套

### src\infrastructure\logging\unified_sync_service.py
- ⚠️ 过多锁定义 (7) - 可能增加死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.Lock()
    - threading.RLock()
    - Lock()
    - Lock()
    - RLock()

### src\infrastructure\logging\engine\business_logger.py
- ⚠️ 发现嵌套锁使用 (1) - 高死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
  嵌套锁:
    - 第264行: 2层嵌套

### src\infrastructure\logging\engine\correlation_tracker.py
- ⚠️ 发现嵌套锁使用 (2) - 高死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
  嵌套锁:
    - 第249行: 3层嵌套
    - 第299行: 2层嵌套

### src\infrastructure\monitoring\exception_monitoring_alert.py
- ⚠️ 发现嵌套锁使用 (4) - 高死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
    - ._lock =
    - self._lock =
  嵌套锁:
    - 第301行: 4层嵌套
    - 第323行: 3层嵌套
    - 第394行: 3层嵌套

### src\infrastructure\resource\business_metrics_monitor.py
- ⚠️ 发现嵌套锁使用 (5) - 高死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
    - ._lock =
    - self._lock =
  嵌套锁:
    - 第640行: 13层嵌套
    - 第647行: 12层嵌套
    - 第666行: 11层嵌套

### src\infrastructure\resource\monitoring_alert_system.py
- ⚠️ 过多锁定义 (12) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (5) - 高死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.Lock()
    - threading.Lock()
    - threading.Lock()
    - Lock()
    - Lock()
  嵌套锁:
    - 第324行: 5层嵌套
    - 第479行: 8层嵌套
    - 第569行: 8层嵌套

### src\infrastructure\resource\resource_manager.py
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.Lock()
    - Lock()
    - ._lock =
    - self._lock =

### src\infrastructure\resource\task_scheduler.py
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.Lock()
    - Lock()
    - ._lock =
    - self._lock =

### src\infrastructure\utils\advanced_connection_pool.py
- ⚠️ 发现嵌套锁使用 (1) - 高死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
  嵌套锁:
    - 第403行: 4层嵌套

### src\infrastructure\utils\concurrency_controller.py
- ⚠️ 过多锁定义 (7) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (1) - 高死锁风险
  锁定义:
    - threading.Lock()
    - threading.Lock()
    - threading.RLock()
    - Lock()
    - Lock()
  嵌套锁:
    - 第256行: 9层嵌套

### src\infrastructure\utils\connection_pool.py
- ⚠️ 过多锁定义 (6) - 可能增加死锁风险
- ⚠️ 发现嵌套锁使用 (2) - 高死锁风险
  锁定义:
    - threading.Lock()
    - threading.Lock()
    - Lock()
    - Lock()
    - ._lock =
  嵌套锁:
    - 第178行: 5层嵌套
    - 第203行: 5层嵌套

### src\infrastructure\utils\memory_object_pool.py
- ⚠️ 发现嵌套锁使用 (2) - 高死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
    - ._lock =
    - self._lock =
  嵌套锁:
    - 第268行: 5层嵌套
    - 第303行: 4层嵌套

### src\infrastructure\utils\optimized_connection_pool.py
- ⚠️ 发现嵌套锁使用 (1) - 高死锁风险
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
    - ._lock =
    - self._lock =
  嵌套锁:
    - 第416行: 5层嵌套

### src\infrastructure\utils\smart_cache_optimizer.py
- ⚠️ 发现嵌套锁使用 (3) - 高死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()
    - ._lock =
    - self._lock =
  嵌套锁:
    - 第354行: 6层嵌套
    - 第361行: 5层嵌套
    - 第368行: 4层嵌套

### src\infrastructure\utils\unified_query.py
- ⚠️ 并发操作与锁同时使用 - 需要检查死锁风险
  锁定义:
    - threading.RLock()
    - Lock()
    - RLock()

## 📊 详细统计

- src\infrastructure\base.py
  - 锁定义: 6
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\services_cache_service.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\version.py
  - 锁定义: 4
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\visual_monitor.py
  - 锁定义: 4
  - 嵌套锁: 1
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\cache\advanced_cache_manager.py
  - 锁定义: 5
  - 嵌套锁: 1
  - 并发操作: 1
  - 问题数: 2
- src\infrastructure\cache\base_cache_manager.py
  - 锁定义: 5
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\cache\business_metrics_plugin.py
  - 锁定义: 4
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\cache\cache_strategy.py
  - 锁定义: 5
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\cache\cache_strategy_manager.py
  - 锁定义: 10
  - 嵌套锁: 1
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\cache\cache_utils.py
  - 锁定义: 20
  - 嵌套锁: 2
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\cache\cache_warmup_optimizer.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 3
  - 问题数: 0
- src\infrastructure\cache\distributed_cache_manager.py
  - 锁定义: 5
  - 嵌套锁: 0
  - 并发操作: 3
  - 问题数: 1
- src\infrastructure\cache\distributed_consistency_manager.py
  - 锁定义: 5
  - 嵌套锁: 3
  - 并发操作: 2
  - 问题数: 2
- src\infrastructure\cache\enhanced_health_checker.py
  - 锁定义: 5
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\cache\memory_cache.py
  - 锁定义: 10
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\cache\multi_level_cache.py
  - 锁定义: 15
  - 嵌套锁: 12
  - 并发操作: 2
  - 问题数: 3
- src\infrastructure\cache\optimized_cache_service.py
  - 锁定义: 5
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 1
- src\infrastructure\cache\redis_adapter_unified.py
  - 锁定义: 3
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\cache\simple_memory_cache.py
  - 锁定义: 5
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\cache\smart_cache_strategies.py
  - 锁定义: 25
  - 嵌套锁: 3
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\cache\smart_cache_strategy.py
  - 锁定义: 15
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\cache\smart_performance_monitor.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\cache\unified_cache.py
  - 锁定义: 15
  - 嵌套锁: 1
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\cache\unified_cache_manager.py
  - 锁定义: 5
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\cache\unified_cache_manager_refactored.py
  - 锁定义: 5
  - 嵌套锁: 5
  - 并发操作: 2
  - 问题数: 2
- src\infrastructure\config\core\config_service.py
  - 锁定义: 5
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\config\core\config_storage.py
  - 锁定义: 15
  - 嵌套锁: 8
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\config\core\config_strategy.py
  - 锁定义: 5
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\config\environment\cloud_native_enhanced.py
  - 锁定义: 16
  - 嵌套锁: 5
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\config\monitoring\performance_monitor_dashboard.py
  - 锁定义: 15
  - 嵌套锁: 2
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\config\security\enhanced_secure_config.py
  - 锁定义: 23
  - 嵌套锁: 1
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\config\services\cache_service.py
  - 锁定义: 5
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\config\services\event.py
  - 锁定义: 7
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\config\tools\optimization_strategies.py
  - 锁定义: 5
  - 嵌套锁: 3
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\config\utils\enhanced_config_validator.py
  - 锁定义: 2
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\distributed\distributed_lock.py
  - 锁定义: 5
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\distributed\distributed_monitoring.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\error\archive_failure_handler.py
  - 锁定义: 6
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\error\async_exception_handler.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\error\circuit_breaker.py
  - 锁定义: 4
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\error\comprehensive_error_plugin.py
  - 锁定义: 12
  - 嵌套锁: 1
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\error\container.py
  - 锁定义: 10
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\error\database_exception_handler.py
  - 锁定义: 5
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\error\enhanced_global_exception_handler.py
  - 锁定义: 7
  - 嵌套锁: 3
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\error\error_handler.py
  - 锁定义: 9
  - 嵌套锁: 4
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\error\global_exception_handler.py
  - 锁定义: 4
  - 嵌套锁: 1
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\error\handler.py
  - 锁定义: 8
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\error\integration.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\error\lock.py
  - 锁定义: 12
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 3
- src\infrastructure\error\retry_handler.py
  - 锁定义: 8
  - 嵌套锁: 2
  - 并发操作: 1
  - 问题数: 3
- src\infrastructure\error\trading_error_handler.py
  - 锁定义: 6
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\error\unified_error_handler.py
  - 锁定义: 4
  - 嵌套锁: 2
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\health\database_health_monitor.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\health\enhanced_health_checker.py
  - 锁定义: 5
  - 嵌套锁: 5
  - 并发操作: 1
  - 问题数: 2
- src\infrastructure\health\final_deployment_check.py
  - 锁定义: 4
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\health\inference_engine.py
  - 锁定义: 8
  - 嵌套锁: 2
  - 并发操作: 1
  - 问题数: 3
- src\infrastructure\health\load_balancer.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\health\monitoring_dashboard.py
  - 锁定义: 8
  - 嵌套锁: 4
  - 并发操作: 2
  - 问题数: 3
- src\infrastructure\health\network_monitor.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\health\prometheus_exporter.py
  - 锁定义: 7
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\health\prometheus_integration.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\health\web_management_interface.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\logging\advanced_logger.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\logging\alert_rule_engine.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\logging\async_log_processor.py
  - 锁定义: 7
  - 嵌套锁: 4
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\logging\audit.py
  - 锁定义: 6
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\logging\audit_logger.py
  - 锁定义: 6
  - 嵌套锁: 1
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\logging\base_monitor.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\logging\circuit_breaker.py
  - 锁定义: 5
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\logging\connection_pool.py
  - 锁定义: 5
  - 嵌套锁: 1
  - 并发操作: 1
  - 问题数: 2
- src\infrastructure\logging\data_consistency.py
  - 锁定义: 3
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\logging\data_sanitizer.py
  - 锁定义: 5
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\logging\deployment_validator.py
  - 锁定义: 4
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\logging\distributed_lock.py
  - 锁定义: 2
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 1
- src\infrastructure\logging\distributed_monitoring.py
  - 锁定义: 10
  - 嵌套锁: 1
  - 并发操作: 1
  - 问题数: 3
- src\infrastructure\logging\enhanced_container.py
  - 锁定义: 12
  - 嵌套锁: 1
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\logging\enhanced_logger.py
  - 锁定义: 9
  - 嵌套锁: 2
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\logging\hot_reload_service.py
  - 锁定义: 7
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\logging\logger.py
  - 锁定义: 2
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\logging\log_archiver.py
  - 锁定义: 5
  - 嵌套锁: 2
  - 并发操作: 1
  - 问题数: 2
- src\infrastructure\logging\log_correlation_plugin.py
  - 锁定义: 5
  - 嵌套锁: 3
  - 并发操作: 1
  - 问题数: 2
- src\infrastructure\logging\log_level_optimizer.py
  - 锁定义: 7
  - 嵌套锁: 4
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\logging\log_metrics_plugin.py
  - 锁定义: 4
  - 嵌套锁: 1
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\logging\log_sampler.py
  - 锁定义: 5
  - 嵌套锁: 1
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\logging\log_sampler_plugin.py
  - 锁定义: 5
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\logging\metrics_aggregator.py
  - 锁定义: 5
  - 嵌套锁: 4
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\logging\microservice_manager.py
  - 锁定义: 25
  - 嵌套锁: 5
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\logging\micro_service.py
  - 锁定义: 3
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 1
- src\infrastructure\logging\service_launcher.py
  - 锁定义: 4
  - 嵌套锁: 2
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\logging\slow_query_monitor.py
  - 锁定义: 3
  - 嵌套锁: 4
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\logging\smart_log_filter.py
  - 锁定义: 20
  - 嵌套锁: 6
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\logging\storage_adapter.py
  - 锁定义: 4
  - 嵌套锁: 4
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\logging\structured_logger.py
  - 锁定义: 9
  - 嵌套锁: 2
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\logging\trading_logger.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\logging\unified_hot_reload_service.py
  - 锁定义: 7
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 2
- src\infrastructure\logging\unified_logger.py
  - 锁定义: 6
  - 嵌套锁: 5
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\logging\unified_sync_service.py
  - 锁定义: 7
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 2
- src\infrastructure\logging\engine\business_logger.py
  - 锁定义: 3
  - 嵌套锁: 1
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\logging\engine\correlation_tracker.py
  - 锁定义: 3
  - 嵌套锁: 2
  - 并发操作: 1
  - 问题数: 2
- src\infrastructure\logging\engine\engine_logger.py
  - 锁定义: 5
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\logging\engine\performance_logger.py
  - 锁定义: 3
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\logging\engine\unified_logger.py
  - 锁定义: 5
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\monitoring\alert_system.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\monitoring\application_monitor.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\monitoring\continuous_monitoring_system.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\monitoring\exception_monitoring_alert.py
  - 锁定义: 5
  - 嵌套锁: 4
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\monitoring\production_monitor.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\ops\monitoring_dashboard.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\resource\business_metrics_monitor.py
  - 锁定义: 5
  - 嵌套锁: 5
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\resource\monitoring_alert_system.py
  - 锁定义: 12
  - 嵌套锁: 5
  - 并发操作: 4
  - 问题数: 3
- src\infrastructure\resource\resource_manager.py
  - 锁定义: 4
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 1
- src\infrastructure\resource\system_monitor.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 2
  - 问题数: 0
- src\infrastructure\resource\task_scheduler.py
  - 锁定义: 4
  - 嵌套锁: 0
  - 并发操作: 3
  - 问题数: 1
- src\infrastructure\utils\advanced_connection_pool.py
  - 锁定义: 3
  - 嵌套锁: 1
  - 并发操作: 1
  - 问题数: 2
- src\infrastructure\utils\async_io_optimizer.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 6
  - 问题数: 0
- src\infrastructure\utils\benchmark_framework.py
  - 锁定义: 0
  - 嵌套锁: 0
  - 并发操作: 1
  - 问题数: 0
- src\infrastructure\utils\concurrency_controller.py
  - 锁定义: 7
  - 嵌套锁: 1
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\utils\connection_pool.py
  - 锁定义: 6
  - 嵌套锁: 2
  - 并发操作: 0
  - 问题数: 2
- src\infrastructure\utils\core.py
  - 锁定义: 5
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\utils\database_adapter.py
  - 锁定义: 4
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\utils\log_compressor_plugin.py
  - 锁定义: 4
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\utils\memory_object_pool.py
  - 锁定义: 5
  - 嵌套锁: 2
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\utils\optimized_connection_pool.py
  - 锁定义: 5
  - 嵌套锁: 1
  - 并发操作: 1
  - 问题数: 2
- src\infrastructure\utils\smart_cache_optimizer.py
  - 锁定义: 5
  - 嵌套锁: 3
  - 并发操作: 0
  - 问题数: 1
- src\infrastructure\utils\sqlite_adapter.py
  - 锁定义: 4
  - 嵌套锁: 0
  - 并发操作: 0
  - 问题数: 0
- src\infrastructure\utils\unified_query.py
  - 锁定义: 3
  - 嵌套锁: 0
  - 并发操作: 2
  - 问题数: 1