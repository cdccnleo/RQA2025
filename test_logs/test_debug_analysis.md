# 测试失败分析

## 问题场景

### test_migrate_table_target_failure_final

**测试设置：**
- total_count = 1
- execute_query返回值设置：
  1. count查询返回1
  2. 数据查询返回1条记录
  3. 第二次数据查询返回空（用于检测是否还有数据）
- batch_execute总是抛出异常

**期望行为：**
- 查询到1条数据
- 尝试写入3次，全部失败
- 返回 failed=1, success=False

**实际发生：**
1. 第1次迭代：查询到1条数据，重试3次写入均失败，返回 {processed:0, failed:1}
2. migrated=0, failed=1, total_count=1
3. 循环条件：migrated(0) < total_count(1) 仍然满足
4. 第2次迭代：查询数据，返回空，返回 {processed:0, failed:0}
5. 连续空批次计数增加
6. 第3次迭代：继续查询，但side_effect已经用尽

**核心问题：**
循环退出条件应该考虑：
- `migrated + failed >= total_count` 时应该退出
- 我已经在代码第106-107行添加了这个条件，但可能执行顺序有问题

让我检查实际执行流程...

**实际执行流程：**
```
迭代1：
  - 查询数据 -> 1条记录
  - 写入失败 -> {processed:0, failed:1}
  - migrated += 0  -> migrated = 0
  - failed += 1    -> failed = 1
  - 检查 migrated(0) + failed(1) >= total_count(1)? YES -> 应该break

迭代2：不应该执行
```

所以代码逻辑应该是正确的，但为什么还会继续迭代？

让我重新检查代码...

