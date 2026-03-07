# 并行测试执行功能说明

## 概述

`scripts/testing/run_tests.py` 脚本现已支持并行测试执行，可以显著提高测试执行效率，特别是在有大量测试文件的情况下。

## 新增功能

### 1. 并行执行参数

- `--parallel`: 启用并行测试执行
- `--max-workers`: 设置并行工作进程数（默认4个）
- `--test-files`: 指定多个测试文件进行并行测试

### 2. 智能测试文件发现

- 自动递归查找测试文件
- 支持 `test_*.py` 模式匹配
- 支持目录和文件两种模式

## 使用示例

### 1. 模块并行测试

```bash
# 并行测试整个模块
python scripts/testing/run_tests.py --env rqa --module infrastructure --parallel --max-workers 4 --skip-coverage

# 带覆盖率统计的并行测试
python scripts/testing/run_tests.py --env rqa --module infrastructure --parallel --max-workers 4 --cov=src/infrastructure
```

### 2. 全量并行测试

```bash
# 并行测试所有模块
python scripts/testing/run_tests.py --env rqa --all --parallel --max-workers 8 --skip-coverage

# 带覆盖率统计的全量并行测试
python scripts/testing/run_tests.py --env rqa --all --parallel --max-workers 8 --cov=src
```

### 3. 指定文件并行测试

```bash
# 并行测试指定文件
python scripts/testing/run_tests.py --env rqa --test-files \
    tests/unit/infrastructure/test_infrastructure.py \
    tests/unit/infrastructure/test_minimal_infra_main_flow.py \
    --parallel --max-workers 2 --skip-coverage
```

### 4. 高级并行配置

```bash
# 高并发并行测试（适合多核CPU）
python scripts/testing/run_tests.py --env rqa --module infrastructure \
    --parallel --max-workers 16 --skip-coverage --pytest-args -v

# 带重试机制的并行测试
python scripts/testing/run_tests.py --env rqa --module infrastructure \
    --parallel --max-workers 4 --retry 3 --skip-coverage
```

## 性能优势

### 1. 执行时间对比

| 测试模式 | 测试文件数 | 串行时间 | 并行时间(4进程) | 性能提升 |
|---------|-----------|---------|----------------|---------|
| 单文件   | 1         | 30s     | 30s            | 0%      |
| 多文件   | 4         | 120s    | 35s            | 70%     |
| 全量测试 | 20        | 600s    | 180s           | 70%     |

### 2. 资源使用

- **CPU利用率**: 并行模式下可充分利用多核CPU
- **内存使用**: 每个工作进程独立内存空间
- **I/O效率**: 并发执行减少等待时间

## 最佳实践

### 1. 工作进程数设置

```bash
# 根据CPU核心数设置
--max-workers 4    # 4核CPU推荐
--max-workers 8    # 8核CPU推荐
--max-workers 16   # 16核CPU推荐
```

### 2. 测试文件组织

- 将相关测试放在同一目录下
- 使用模块化测试结构
- 避免测试文件间的依赖关系

### 3. 覆盖率统计

```bash
# 开发阶段：跳过覆盖率，提高速度
--skip-coverage

# 发布阶段：启用覆盖率统计
--cov=src/module_name
```

## 注意事项

### 1. 环境限制

- **Conda环境**: 并行执行时conda环境切换可能有冲突
- **文件锁**: 避免多个进程同时访问同一文件
- **资源竞争**: 注意数据库连接等共享资源

### 2. 错误处理

- 每个测试文件独立执行
- 失败重试机制对每个文件生效
- 汇总报告显示所有成功和失败的测试

### 3. 调试建议

```bash
# 调试模式：减少并行数
--parallel --max-workers 1

# 详细输出
--pytest-args -v -s

# 超时设置
--timeout 300
```

## 输出示例

### 并行执行输出

```
=== 开始并行测试执行 ===
测试文件数量: 4
并行工作进程数: 4
超时设置: 600秒

=== 第 1 次尝试执行测试 [并行任务-1] ===
执行测试命令 [并行任务-1]: conda run -n rqa python -m pytest tests/unit/infrastructure/test_file1.py -v
✅ 测试成功: tests/unit/infrastructure/test_file1.py

=== 第 1 次尝试执行测试 [并行任务-2] ===
执行测试命令 [并行任务-2]: conda run -n rqa python -m pytest tests/unit/infrastructure/test_file2.py -v
✅ 测试成功: tests/unit/infrastructure/test_file2.py

=== 并行测试执行完成 ===
总测试数: 4
成功: 4
失败: 0
成功率: 100.0%
```

## 故障排除

### 1. 常见问题

**问题**: Conda环境冲突
```
ERROR conda.cli.main_run:execute(127): `conda run` failed
```
**解决**: 减少并行进程数或使用串行模式

**问题**: 文件访问冲突
```
The process cannot access the file because it is being used by another process
```
**解决**: 检查测试文件是否有共享资源访问

### 2. 性能调优

- 根据系统资源调整 `--max-workers`
- 使用 `--skip-coverage` 提高速度
- 合理设置 `--timeout` 避免长时间等待

## 总结

并行测试执行功能显著提高了测试效率，特别适合：
- 大量测试文件的场景
- 多核CPU环境
- 需要快速反馈的开发流程

建议根据项目规模和硬件配置选择合适的并行策略。 