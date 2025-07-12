"""
版本管理器使用示例
"""
import os
import pandas as pd
import logging
from pathlib import Path

from src.data.version_control.version_manager import DataVersionManager
from src.data.data_manager import DataModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 创建版本目录
version_dir = Path("./data/versions")
version_dir.mkdir(parents=True, exist_ok=True)

# 创建版本管理器
version_manager = DataVersionManager(str(version_dir))

def create_sample_data():
    """创建示例数据"""
    # 创建第一个版本的数据
    data1 = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35]
    })
    metadata1 = {
        'source': 'sample_data',
        'created_at': '2023-01-01',
        'description': '初始数据集'
    }
    model1 = DataModel(data1, metadata1)

    # 创建第一个版本
    version1 = version_manager.create_version(
        model1,
        description="初始版本",
        tags=["sample", "v1"],
        creator="system"
    )
    print(f"创建初始版本: {version1}")

    # 创建第二个版本的数据（修改数据）
    data2 = data1.copy()
    data2.loc[1, 'age'] = 31  # 修改Bob的年龄
    data2.loc[2, 'name'] = 'Charles'  # 修改Charlie的名字
    metadata2 = metadata1.copy()
    metadata2['updated_at'] = '2023-01-02'
    metadata2['description'] = '更新了Bob的年龄和Charlie的名字'
    model2 = DataModel(data2, metadata2)

    # 创建第二个版本
    version2 = version_manager.create_version(
        model2,
        description="更新了用户信息",
        tags=["sample", "v2"],
        creator="user1"
    )
    print(f"创建第二个版本: {version2}")

    # 创建第三个版本的数据（添加新行和列）
    data3 = data2.copy()
    data3.loc[3] = [4, 'David', 40]  # 添加新行
    data3['gender'] = ['F', 'M', 'M', 'M']  # 添加新列
    metadata3 = metadata2.copy()
    metadata3['updated_at'] = '2023-01-03'
    metadata3['description'] = '添加了新用户David和性别列'
    model3 = DataModel(data3, metadata3)

    # 创建第三个版本
    version3 = version_manager.create_version(
        model3,
        description="添加了新用户和性别信息",
        tags=["sample", "v3"],
        creator="user2"
    )
    print(f"创建第三个版本: {version3}")

    return version1, version2, version3

def demonstrate_version_operations(version1, version2, version3):
    """演示版本操作"""
    # 列出所有版本
    print("\n所有版本:")
    versions = version_manager.list_versions()
    for v in versions:
        print(f"- {v['version_id']}: {v['description']} (创建者: {v['creator']})")

    # 按标签筛选版本
    print("\n标签为'v2'的版本:")
    v2_versions = version_manager.list_versions(tags=["v2"])
    for v in v2_versions:
        print(f"- {v['version_id']}: {v['description']}")

    # 比较版本
    print("\n比较版本1和版本3:")
    diff = version_manager.compare_versions(version1, version3)

    print("元数据差异:")
    if diff['metadata_diff']['added']:
        print(f"- 添加: {diff['metadata_diff']['added']}")
    if diff['metadata_diff']['removed']:
        print(f"- 删除: {diff['metadata_diff']['removed']}")
    if diff['metadata_diff']['changed']:
        print(f"- 修改: {diff['metadata_diff']['changed']}")

    print("数据差异:")
    print(f"- 行数变化: {diff['data_diff']['shape_diff']['rows']}")
    print(f"- 列数变化: {diff['data_diff']['shape_diff']['columns']}")
    print(f"- 添加列: {diff['data_diff']['columns_diff']['added']}")
    print(f"- 删除列: {diff['data_diff']['columns_diff']['removed']}")
    print(f"- 值变化: {diff['data_diff']['value_diff']}")

    # 回滚到版本1
    print("\n回滚到版本1:")
    rollback_version = version_manager.rollback(version1)
    print(f"创建了新版本: {rollback_version}")

    # 获取回滚后的版本
    rollback_model = version_manager.get_version(rollback_version)
    print("回滚后的数据:")
    print(rollback_model.data)

    # 获取版本信息
    rollback_info = version_manager.get_version_info(rollback_version)
    print(f"回滚版本标签: {rollback_info['tags']}")

    # 删除版本2
    print("\n删除版本2:")
    version_manager.delete_version(version2)
    print("删除成功")

    # 列出剩余版本
    print("\n剩余版本:")
    versions = version_manager.list_versions()
    for v in versions:
        print(f"- {v['version_id']}: {v['description']}")

if __name__ == "__main__":
    try:
        # 创建示例数据
        version1, version2, version3 = create_sample_data()

        # 演示版本操作
        demonstrate_version_operations(version1, version2, version3)

    except Exception as e:
        print(f"发生错误: {e}")
"""