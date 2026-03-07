# pytest-mock迁移报告

发现 11 个使用pytest-mock的文件

## tests\unit\data\test_cache_disk_cache.py
- mocker参数使用: 2 处
- mocker调用: 2 处
- patch调用: 2 处
### 迁移建议:
文件 test_cache_disk_cache.py 使用了mocker参数，建议替换为:
  from unittest.mock import Mock, patch, MagicMock
  # 移除mocker参数，使用patch装饰器或上下文管理器
发现patch调用，建议替换为:
  with patch('module.function') as mock_func:
      mock_func.return_value = expected_value
      # 执行测试
      mock_func.assert_called_once()

## tests\unit\data\test_data_exporter.py
- mocker参数使用: 1 处
- mocker调用: 1 处
- patch调用: 1 处
### 迁移建议:
文件 test_data_exporter.py 使用了mocker参数，建议替换为:
  from unittest.mock import Mock, patch, MagicMock
  # 移除mocker参数，使用patch装饰器或上下文管理器
发现patch调用，建议替换为:
  with patch('module.function') as mock_func:
      mock_func.return_value = expected_value
      # 执行测试
      mock_func.assert_called_once()

## tests\unit\data\test_financial_loader.py
- mocker参数使用: 15 处
- mocker调用: 0 处
- patch调用: 0 处
### 迁移建议:
文件 test_financial_loader.py 使用了mocker参数，建议替换为:
  from unittest.mock import Mock, patch, MagicMock
  # 移除mocker参数，使用patch装饰器或上下文管理器

## tests\unit\data\test_index_loader.py
- mocker参数使用: 5 处
- mocker调用: 2 处
- patch调用: 2 处
### 迁移建议:
文件 test_index_loader.py 使用了mocker参数，建议替换为:
  from unittest.mock import Mock, patch, MagicMock
  # 移除mocker参数，使用patch装饰器或上下文管理器
发现patch调用，建议替换为:
  with patch('module.function') as mock_func:
      mock_func.return_value = expected_value
      # 执行测试
      mock_func.assert_called_once()

## tests\unit\data\test_news_loader.py
- mocker参数使用: 10 处
- mocker调用: 8 处
- patch调用: 8 处
### 迁移建议:
文件 test_news_loader.py 使用了mocker参数，建议替换为:
  from unittest.mock import Mock, patch, MagicMock
  # 移除mocker参数，使用patch装饰器或上下文管理器
发现patch调用，建议替换为:
  with patch('module.function') as mock_func:
      mock_func.return_value = expected_value
      # 执行测试
      mock_func.assert_called_once()

## tests\unit\data\test_stock_loader.py
- mocker参数使用: 21 处
- mocker调用: 3 处
- patch调用: 3 处
### 迁移建议:
文件 test_stock_loader.py 使用了mocker参数，建议替换为:
  from unittest.mock import Mock, patch, MagicMock
  # 移除mocker参数，使用patch装饰器或上下文管理器
发现patch调用，建议替换为:
  with patch('module.function') as mock_func:
      mock_func.return_value = expected_value
      # 执行测试
      mock_func.assert_called_once()

## tests\unit\data\loader\test_sentiment_news_loader.py
- mocker参数使用: 6 处
- mocker调用: 0 处
- patch调用: 0 处
### 迁移建议:
文件 test_sentiment_news_loader.py 使用了mocker参数，建议替换为:
  from unittest.mock import Mock, patch, MagicMock
  # 移除mocker参数，使用patch装饰器或上下文管理器

## tests\unit\features\test_level2_analyzer.py
- mocker参数使用: 1 处
- mocker调用: 1 处
- patch调用: 1 处
### 迁移建议:
文件 test_level2_analyzer.py 使用了mocker参数，建议替换为:
  from unittest.mock import Mock, patch, MagicMock
  # 移除mocker参数，使用patch装饰器或上下文管理器
发现patch调用，建议替换为:
  with patch('module.function') as mock_func:
      mock_func.return_value = expected_value
      # 执行测试
      mock_func.assert_called_once()

## tests\unit\infrastructure\security\test_security_service.py
- mocker参数使用: 3 处
- mocker调用: 0 处
- patch调用: 0 处
### 迁移建议:
文件 test_security_service.py 使用了mocker参数，建议替换为:
  from unittest.mock import Mock, patch, MagicMock
  # 移除mocker参数，使用patch装饰器或上下文管理器

## tests\unit\infrastructure\utils\test_tools.py
- mocker参数使用: 2 处
- mocker调用: 2 处
- patch调用: 2 处
### 迁移建议:
文件 test_tools.py 使用了mocker参数，建议替换为:
  from unittest.mock import Mock, patch, MagicMock
  # 移除mocker参数，使用patch装饰器或上下文管理器
发现patch调用，建议替换为:
  with patch('module.function') as mock_func:
      mock_func.return_value = expected_value
      # 执行测试
      mock_func.assert_called_once()

## tests\unit\models\test_lstm.py
- mocker参数使用: 1 处
- mocker调用: 3 处
- patch调用: 2 处
### 迁移建议:
文件 test_lstm.py 使用了mocker参数，建议替换为:
  from unittest.mock import Mock, patch, MagicMock
  # 移除mocker参数，使用patch装饰器或上下文管理器
发现patch调用，建议替换为:
  with patch('module.function') as mock_func:
      mock_func.return_value = expected_value
      # 执行测试
      mock_func.assert_called_once()
