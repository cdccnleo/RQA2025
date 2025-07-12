import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.data.loaders import (
    StockDataLoader,
    IndexDataLoader,
    NewsDataLoader
)

class TestDataLoaders(unittest.TestCase):
    """统一后的数据加载器测试"""

    def setUp(self):
        # 创建临时测试文件
        self.temp_dir = tempfile.mkdtemp()
        self.stock_file = os.path.join(self.temp_dir, "stock.csv")
        with open(self.stock_file, 'w') as f:
            f.write("symbol,price,volume\n600000.SH,42.50,100000")

        # 初始化各加载器
        self.stock_loader = StockDataLoader()
        self.index_loader = IndexDataLoader()
        self.news_loader = NewsDataLoader()

    def tearDown(self):
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_stock_loading(self):
        """测试股票数据加载"""
        data = self.stock_loader.load(self.stock_file)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['symbol'], '600000.SH')

    # 从test_stock_loader.py合并的测试
    @patch('src.data.loaders.StockDataLoader._fetch_from_api')
    def test_stock_api_fallback(self, mock_fetch):
        """测试股票数据API回退逻辑"""
        mock_fetch.return_value = [{'symbol': '600000.SH', 'price': 42.50}]

        # 模拟文件不存在时回退到API
        data = self.stock_loader.load("nonexistent.csv")
        self.assertEqual(data[0]['symbol'], '600000.SH')

    # 从test_index_loader.py合并的测试
    @patch('src.data.loaders.IndexDataLoader._validate_index_data')
    def test_index_validation(self, mock_validate):
        """测试指数数据验证"""
        mock_validate.return_value = True
        test_data = [{'index': 'SH000001', 'value': 3000.0}]

        result = self.index_loader.validate(test_data)
        self.assertTrue(result)

    # 新增批量加载测试
    def test_batch_loading(self):
        """测试批量数据加载性能"""
        files = [self.stock_file] * 10
        results = self.stock_loader.batch_load(files)
        self.assertEqual(len(results), 10)
        self.assertTrue(all(len(r) == 1 for r in results))

    # 从test_news_loader.py合并的异常测试
    def test_news_loader_exceptions(self):
        """测试新闻加载器异常处理"""
        with self.assertRaises(ValueError):
            self.news_loader.load("invalid_format.json")

if __name__ == '__main__':
    unittest.main()
