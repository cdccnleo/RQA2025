# API模块重构文件

本目录包含重构后的API模块新结构文件。

## 文件说明

- `test_configs.py` - 测试配置对象
- `template_manager.py` - 测试模板管理器
- `test_case_builder.py` - 测试用例构建器基类
- `data_service_test_generator.py` - 数据服务测试生成器示例

## 重构步骤

1. 完善各个模板文件中的TODO部分
2. 从原APITestCaseGenerator迁移具体实现
3. 创建其他服务测试生成器（feature, trading, monitoring）
4. 创建TestSuiteExporter和TestStatisticsCollector
5. 创建APITestSuiteCoordinator协调器
6. 编写单元测试
7. 更新导入引用
8. 删除或标记为deprecated原有的大类

## 迁移注意事项

- 保持向后兼容性
- 充分测试每个迁移步骤
- 更新相关文档
