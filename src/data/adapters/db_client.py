"""
数据库客户端模块
提供数据库连接和操作接口
"""

from typing import Dict, List
from unittest.mock import MagicMock


class MockDatabase:

    """模拟数据库类"""

    def __init__(self, name: str):

        self.name = name
        self.dragon_board = MockCollection("dragon_board")

    def get_collection(self, name: str):
        """获取集合"""
        return MockCollection(name)


class MockCursor:

    """模拟游标类"""

    def __init__(self, data: List[Dict]):

        self._data = data
        self._sort_key = None
        self._sort_direction = 1
        self._limit_count = None

    def sort(self, key: str, direction: int = 1):
        """排序"""
        self._sort_key = key
        self._sort_direction = direction
        return self

    def limit(self, count: int):
        """限制结果数量"""
        self._limit_count = count
        return self

    def __iter__(self):
        """迭代器"""
        data = self._data.copy()

        # 应用排序
        if self._sort_key:
            reverse = self._sort_direction == -1
            data.sort(key=lambda x: x.get(self._sort_key, 0), reverse=reverse)

        # 应用限制
        if self._limit_count:
            data = data[:self._limit_count]

        return iter(data)


class MockCollection:

    """模拟集合类"""

    def __init__(self, name: str):

        self.name = name
        self._data = []

    def find(self, query: Dict = None, projection: Dict = None):
        """查找文档"""
        return MockCursor(self._data)

    def find_one(self, query: Dict = None):
        """查找单个文档"""
        return self._data[0] if self._data else None

    def insert_one(self, document: Dict):
        """插入单个文档"""
        self._data.append(document)
        return MagicMock(inserted_id="mock_id")

    def insert_many(self, documents: List[Dict]):
        """插入多个文档"""
        self._data.extend(documents)
        return MagicMock(inserted_ids=["mock_id"] * len(documents))

    def update_one(self, query: Dict, update: Dict):
        """更新单个文档"""
        return MagicMock(matched_count=1, modified_count=1)

    def bulk_write(self, operations: List):
        """批量写入操作"""
        return MagicMock(
            matched_count=len(operations),
            modified_count=len(operations),
            upserted_count=0
        )


class MockClient:

    """模拟数据库客户端"""

    def __init__(self):

        self._databases = {}

    def get_database(self, name: str):
        """获取数据库"""
        if name not in self._databases:
            self._databases[name] = MockDatabase(name)
        return self._databases[name]

    def close(self):
        """关闭连接"""


def get_db_client():
    """获取数据库客户端

    Returns:
        MockClient: 模拟数据库客户端实例
    """
    return MockClient()
