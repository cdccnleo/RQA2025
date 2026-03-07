# 增强版 run_tests.py 使用说明

## 概述

`run_tests.py` 脚本已经增强，现在支持更灵活的测试路径指定、自动发现测试文件、并行测试运行等功能。

## 新增功能

### 1. 灵活的路径支持

- **相对路径**: 支持相对于项目根目录的路径
- **绝对路径**: 支持完整的绝对路径
- **多路径**: 支持用逗号分隔的多个路径
- **智能识别**: 自动识别文件和目录，递归查找测试文件

### 2. 自动发现模式

- **自动扫描**: 自动扫描项目中的测试目录
- **智能分类**: 按优先级分类测试文件
- **路径验证**: 验证路径存在性和测试文件有效性

### 3. 并行测试支持

- **并行运行**: 支持并行运行测试以提高效率
- **工作进程**: 可配置并行工作进程数量
- **资源管理**: 智能内存管理和超时控制

### 4. 增强的统计信息

- **详细统计**: 显示测试通过率、内存使用、运行时间等
- **失败分析**: 详细显示失败的测试文件
- **性能监控**: 实时监控内存使用和性能指标

## 命令行参数

### 基本参数

```bash
--module MODULE          指定测试模块（用于轻量级模式）
--path PATH              指定测试路径（支持相对路径和绝对路径，多个路径用逗号分隔）
--auto-discover          自动发现项目中的测试文件
--list-paths             列出可用的测试路径
--timeout SECONDS        超时时间(秒)，默认300
--max-memory MB          最大内存使用(MB)，默认2048
--pytest-args ARGS       pytest参数，默认["-v"]
--lightweight            使用轻量级模式
```

### 新增参数

```bash
--parallel               启用并行测试运行
--workers N              并行工作进程数量，默认2
```

## 使用示例

### 1. 运行特定目录的测试

```bash
# 运行单个目录
python scripts/testing/run_tests.py --path tests/unit/infrastructure

# 运行多个目录
python scripts/testing/run_tests.py --path "tests/unit/infrastructure,tests/unit/data"

# 使用相对路径
python scripts/testing/run_tests.py --path tests/unit
```

### 2. 自动发现并运行测试

```bash
# 自动发现所有测试
python scripts/testing/run_tests.py --auto-discover

# 自动发现并设置超时
python scripts/testing/run_tests.py --auto-discover --timeout 600
```

### 3. 列出可用的测试路径

```bash
# 查看所有可用的测试路径
python scripts/testing/run_tests.py --list-paths
```

### 4. 并行运行测试

```bash
# 启用并行模式，使用4个工作进程
python scripts/testing/run_tests.py --path tests/unit --parallel --workers 4

# 并行运行并设置内存限制
python scripts/testing/run_tests.py --path tests/unit --parallel --workers 4 --max-memory 4096
```

### 5. 轻量级模式

```bash
# 轻量级模式运行特定模块
python scripts/testing/run_tests.py --module infrastructure --lightweight

# 轻量级模式运行并设置超时
python scripts/testing/run_tests.py --module infrastructure --lightweight --timeout 180
```

### 6. 自定义pytest参数

```bash
# 使用自定义pytest参数
python scripts/testing/run_tests.py --path tests/unit --pytest-args -v -s --tb=short

# 启用详细输出和失败继续
python scripts/testing/run_tests.py --path tests/unit --pytest-args -v -s --tb=long --maxfail=5
```

## 路径处理规则

### 1. 路径类型识别

- **文件**: 直接运行指定的测试文件
- **目录**: 递归查找目录中的所有测试文件
- **混合**: 支持文件和目录的混合指定

### 2. 测试文件识别

- **命名规则**: 自动识别 `test_*.py` 和 `*_test.py` 文件
- **优先级**: `test_` 开头的文件优先运行
- **递归搜索**: 支持深层目录结构的递归搜索

### 3. 路径验证

- **存在性检查**: 验证指定的路径是否存在
- **有效性检查**: 检查路径是否包含测试文件
- **错误处理**: 优雅处理无效路径，继续运行有效路径

## 性能优化特性

### 1. 内存管理

- **智能限制**: 根据测试复杂度自动设置内存限制
- **垃圾回收**: 强制垃圾回收以释放内存
- **监控告警**: 实时监控内存使用，超出限制时自动终止

### 2. 超时控制

- **分层超时**: 不同复杂度的测试使用不同的超时设置
- **进程管理**: 超时时自动终止测试进程
- **重试机制**: 支持失败重试，提高测试稳定性

### 3. 并行优化

- **工作进程**: 可配置的并行工作进程数量
- **资源分配**: 智能分配测试文件到不同工作进程
- **负载均衡**: 根据测试复杂度进行负载均衡

## 错误处理和日志

### 1. 错误处理

- **路径错误**: 优雅处理无效路径，继续运行有效测试
- **测试失败**: 详细的失败信息和分析
- **超时处理**: 自动处理超时情况，避免测试卡死

### 2. 日志输出

- **分级日志**: INFO、WARNING、ERROR等不同级别的日志
- **进度显示**: 实时显示测试进度和状态
- **统计信息**: 详细的测试结果统计和性能指标

## 最佳实践

### 1. 路径指定

- 使用相对路径以提高可移植性
- 多个路径用逗号分隔，避免空格问题
- 优先使用目录路径而不是单个文件

### 2. 性能调优

- 根据系统资源调整工作进程数量
- 设置合理的内存限制和超时时间
- 使用轻量级模式运行大型测试套件

### 3. 错误排查

- 使用 `--list-paths` 验证路径有效性
- 检查日志输出了解详细错误信息
- 使用 `--pytest-args` 添加调试参数

## 兼容性说明

- 保持与原有参数的完全兼容
- 新增功能不影响现有功能
- 支持Windows和Unix/Linux平台
- 兼容Python 3.7+版本

## 故障排除

### 常见问题

1. **路径不存在**: 检查路径是否正确，使用 `--list-paths` 验证
2. **内存不足**: 调整 `--max-memory` 参数或使用轻量级模式
3. **超时问题**: 增加 `--timeout` 参数值
4. **并行问题**: 确保安装了 `pytest-xdist` 插件

### 调试技巧

- 使用 `--pytest-args -v -s` 获取详细输出
- 检查日志文件了解详细错误信息
- 使用较小的测试集进行调试
