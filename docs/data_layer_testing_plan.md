# 数据层功能增强测试计划

## 概述

本文档提供了RQA2025项目数据层功能增强的详细测试计划，包括测试策略、测试用例设计和测试覆盖率目标。该计划旨在确保所有新增功能的质量和稳定性，符合项目的测试覆盖要求。

## 测试原则

1. **测试驱动开发**：先编写测试用例，再实现功能，确保代码质量
2. **全面覆盖**：确保所有功能、边界条件和异常情况都有对应的测试用例
3. **自动化优先**：尽可能使用自动化测试，减少人工测试的工作量
4. **持续集成**：将测试集成到CI/CD流程中，确保每次代码提交都经过测试验证

## 测试覆盖要求

根据项目的测试覆盖要求，我们需要确保以下几点：

1. **参数化测试**：使用`@pytest.mark.parametrize`覆盖多输入组合，特别是边界值和异常值
2. **异常断言**：对所有`raise`语句添加`pytest.raises`验证，确保异常处理逻辑正确
3. **Fixtures复用**：减少重复代码，提高测试效率
4. **Mock外部依赖**：使用`unittest.mock`模拟文件系统、网络请求等外部资源，确保测试环境稳定
5. **重点补充未覆盖代码**：针对覆盖率报告中显示的未覆盖代码块，设计专门的测试用例
6. **正则表达式匹配**：测试用例使用正则表达式匹配，避免描述不一致导致断言失败

## 测试环境

- Python版本：3.8+
- 测试框架：pytest
- 覆盖率工具：pytest-cov
- Mock工具：unittest.mock
- 测试环境：conda rqa环境

## 详细测试计划

### 1. 并行数据加载测试

#### 1.1 单元测试

**测试文件**：`src/data/parallel/test_parallel_loader.py`

**测试用例**：

1. **基本功能测试**
   - 测试并行加载单个任务
   - 测试并行加载多个任务
   - 测试结果收集和排序

2. **参数化测试**
   ```python
   @pytest.mark.parametrize("max_workers,task_count", [
       (None, 5),  # 默认线程数
       (1, 5),     # 单线程（等同于串行）
       (2, 5),     # 少量线程
       (10, 5),    # 线程数大于任务数
       (5, 20),    # 任务数大于线程数
   ])
   def test_parallel_execution(max_workers, task_count):
       # 测试不同线程数和任务数组合下的并行执行
   ```

3. **异常处理测试**
   ```python
   def test_exception_handling():
       # 创建一个会抛出异常的任务
       def failing_task():
           raise ValueError("Task failed")
       
       # 测试异常是否被正确捕获和处理
       with pytest.raises(ValueError, match="Task failed"):
           # 执行测试代码
   ```

4. **边界条件测试**
   - 测试空任务列表
   - 测试任务执行时间为0
   - 测试任务执行时间很长

5. **性能测试**
   - 测试并行加载与串行加载的性能对比
   - 测试不同线程数下的性能变化

#### 1.2 集成测试

**测试文件**：`src/data/test_data_manager.py`

**测试用例**：

1. **DataManager并行加载测试**
   - 测试`load_data_parallel`方法
   - 测试不同数据类型的并行加载
   - 测试并行加载结果的正确性

2. **异常处理测试**
   - 测试部分数据加载失败的情况
   - 测试全部数据加载失败的情况

3. **Mock测试**
   ```python
   @pytest.fixture
   def mock_loaders():
       # 创建模拟的数据加载器
       with patch('src.data.loader.stock_loader.StockDataLoader') as mock_stock_loader, \
            patch('src.data.loader.index_loader.IndexDataLoader') as mock_index_loader:
           # 配置模拟对象的行为
           yield {
               'stock': mock_stock_loader,
               'index': mock_index_loader
           }
   
   def test_load_data_parallel_with_mock(mock_loaders):
       # 使用模拟的加载器测试并行加载功能
   ```

### 2. 缓存策略测试

#### 2.1 单元测试

**测试文件**：`src/data/cache/test_data_cache.py`

**测试用例**：

1. **内存缓存测试**
   - 测试缓存命中
   - 测试缓存未命中
   - 测试缓存容量限制
   - 测试LRU淘汰策略

2. **磁盘缓存测试**
   - 测试缓存写入
   - 测试缓存读取
   - 测试缓存文件格式
   - 测试缓存目录管理

3. **参数化测试**
   ```python
   @pytest.mark.parametrize("memory_size,data_size", [
       (10, 5),    # 缓存足够大
       (5, 10),    # 缓存不足
       (0, 5),     # 禁用内存缓存
       (100, 100), # 边界情况
   ])
   def test_cache_capacity(memory_size, data_size):
       # 测试不同缓存容量和数据大小下的缓存行为
   ```

4. **异常处理测试**
   - 测试缓存目录不存在
   - 测试缓存文件损坏
   - 测试缓存键生成异常

5. **Mock文件系统测试**
   ```python
   @pytest.fixture
   def mock_filesystem():
       with patch('os.path.exists') as mock_exists, \
            patch('os.makedirs') as mock_makedirs, \
            patch('builtins.open', mock_open()) as mock_file:
           # 配置模拟文件系统
           yield {
               'exists': mock_exists,
               'makedirs': mock_makedirs,
               'file': mock_file
           }
   
   def test_disk_cache_with_mock_filesystem(mock_filesystem):
       # 使用模拟文件系统测试磁盘缓存功能
   ```

#### 2.2 集成测试

**测试文件**：`src/data/test_data_manager.py`

**测试用例**：

1. **DataManager缓存功能测试**
   - 测试使用缓存加载数据
   - 测试禁用缓存加载数据
   - 测试缓存命中率统计

2. **缓存一致性测试**
   - 测试数据更新后缓存是否正确更新
   - 测试缓存过期策略

3. **性能测试**
   - 测试使用缓存前后的性能对比
   - 测试不同缓存配置下的性能变化

### 3. 数据质量监控测试

#### 3.1 单元测试

**测试文件**：`src/data/quality/test_data_quality.py`

**测试用例**：

1. **缺失值检查测试**
   - 测试无缺失值的数据
   - 测试部分缺失值的数据
   - 测试全部缺失值的数据

2. **重复值检查测试**
   - 测试无重复值的数据
   - 测试部分重复值的数据
   - 测试全部重复值的数据

3. **异常值检查测试**
   - 测试使用IQR方法检测异常值
   - 测试使用Z-Score方法检测异常值
   - 测试不同阈值下的异常值检测

4. **参数化测试**
   ```python
   @pytest.mark.parametrize("method,threshold,expected_outliers", [
       ('iqr', 1.5, [1, 100]),    # IQR方法，标准阈值
       ('iqr', 3.0, []),          # IQR方法，宽松阈值
       ('zscore', 2.0, [1, 100]), # Z-Score方法，标准阈值
       ('zscore', 5.0, []),       # Z-Score方法，宽松阈值
   ])
   def test_outlier_detection(method, threshold, expected_outliers):
       # 测试不同方法和阈值下的异常值检测
   ```

5. **日期范围检查测试**
   - 测试连续日期数据
   - 测试不连续日期数据
   - 测试日期格式异常数据

6. **股票代码覆盖率检查测试**
   - 测试完全覆盖的数据
   - 测试部分覆盖的数据
   - 测试无覆盖的数据

#### 3.2 集成测试

**测试文件**：`src/data/test_data_manager.py`

**测试用例**：

1. **DataManager数据质量检查测试**
   - 测试`check_data_quality`方法
   - 测试不同数据类型的质量检查
   - 测试质量检查结果的正确性

2. **Mock测试**
   ```python
   @pytest.fixture
   def mock_data_model():
       # 创建模拟的数据模型
       mock_model = MagicMock()
       mock_model.data = pd.DataFrame({
           'date': pd.date_range('2020-01-01', periods=10),
           'symbol': ['000001.SZ'] * 10,
           'close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
           'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
           'missing_col': [None, 1, 2, None, 4, 5, None, 7, 8, 9]
       })
       mock_model.get_metadata.return_value = {'type': 'stock', 'source': 'test'}
       return mock_model
   
   def test_check_data_quality_with_mock(mock_data_model):
       # 使用模拟数据模型测试数据质量检查功能
   ```

### 4. 异常告警测试

#### 4.1 单元测试

**测试文件**：`src/data/monitoring/test_alert_manager.py`

**测试用例**：

1. **阈值检查测试**
   - 测试值低于最小阈值
   - 测试值高于最大阈值
   - 测试值在阈值范围内

2. **告警级别测试**
   - 测试不同级别的告警
   - 测试告警级别过滤

3. **告警通知测试**
   - 测试邮件通知
   - 测试Webhook通知
   - 测试日志通知

4. **参数化测试**
   ```python
   @pytest.mark.parametrize("level,min_level,should_notify", [
       ('info', 'info', True),      # 级别相同
       ('warning', 'info', True),   # 级别高于最小级别
       ('info', 'warning', False),  # 级别低于最小级别
       ('critical', 'error', True), # 级别高于最小级别
   ])
   def test_alert_level_filtering(level, min_level, should_notify):
       # 测试不同告警级别和最小通知级别下的通知行为
   ```

5. **告警历史测试**
   - 测试添加告警记录
   - 测试查询告警历史
   - 测试按级别筛选告警
   - 测试按时间筛选告警

6. **Mock测试**
   ```python
   @pytest.fixture
   def mock_notification_services():
       with patch('smtplib.SMTP') as mock_smtp, \
            patch('requests.post') as mock_post, \
            patch('builtins.open', mock_open()) as mock_file:
           # 配置模拟服务
           yield {
               'smtp': mock_smtp,
               'post': mock_post,
               'file': mock_file
           }
   
   def test_email_notification_with_mock(mock_notification_services):
       # 使用模拟SMTP服务测试邮件通知功能
   ```

#### 4.2 集成测试

**测试文件**：`src/data/test_data_manager.py`

**测试用例**：

1. **DataManager告警功能测试**
   - 测试`check_data_thresholds`方法
   - 测试`alert`方法
   - 测试`get_alerts`方法

2. **数据质量告警测试**
   - 测试缺失值超过阈值时的告警
   - 测试重复值超过阈值时的告警
   - 测试异常值超过阈值时的告警

### 5. 数据导出测试

#### 5.1 单元测试

**测试文件**：`src/data/export/test_data_exporter.py`

**测试用例**：

1. **CSV导出测试**
   - 测试基本CSV导出
   - 测试带索引的CSV导出
   - 测试自定义分隔符的