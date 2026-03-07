# PowerShell环境测试问题解决方案

## 问题描述

在PowerShell环境下运行pytest时存在以下问题：
1. 测试执行完成后进程未正常返回
2. 超时机制失效
3. 进程管理异常
4. 输出流处理问题

## 解决方案

### 1. 改进的Python测试运行脚本

#### 主要改进
- **PowerShell环境检测**：自动识别PowerShell环境并应用特殊处理
- **实时输出处理**：使用`capture_output=True`和`text=True`参数
- **超时处理**：改进的超时异常处理机制
- **重试机制**：自动重试失败的测试
- **进程管理**：更好的进程生命周期管理

#### 使用方法
```bash
# 基本用法
python run_tests.py --env rqa --module infrastructure.logging --cov=src/infrastructure/logging --pytest-args -v

# 指定测试文件
python run_tests.py --env rqa --test-file tests/unit/trading/test_minimal_trading_main_flow.py --cov=src/trading --pytest-args -v

# 运行所有测试
python run_tests.py --env rqa --all --cov=src --pytest-args -v

# 跳过覆盖率检查
python run_tests.py --env rqa --module infrastructure.logging --skip-coverage --pytest-args -v

# 自定义超时和重试
python run_tests.py --env rqa --module infrastructure.logging --cov=src/infrastructure/logging --timeout 300 --retry 3 --pytest-args -v

# 禁用重试机制
python run_tests.py --env rqa --module infrastructure.logging --cov=src/infrastructure/logging --no-retry --pytest-args -v
```

### 2. PowerShell专用测试运行脚本

#### 特性
- **专用PowerShell处理**：针对PowerShell环境的特殊优化
- **线程级超时控制**：使用独立线程监控超时
- **实时输出流处理**：逐行读取和显示输出
- **进程强制终止**：确保超时后进程被正确终止

#### 使用方法
```bash
# 使用PowerShell专用脚本
python run_tests_powershell.py --env rqa --module infrastructure.logging --cov=src/infrastructure/logging --pytest-args -v
```

### 3. CMD批处理脚本

#### 特性
- **避免PowerShell问题**：直接在CMD环境下运行
- **批处理语法**：使用Windows批处理命令
- **参数解析**：支持完整的参数解析
- **重试机制**：内置重试逻辑

#### 使用方法
```cmd
# 在CMD环境下运行
run_tests_cmd.bat --env rqa --module infrastructure.logging --cov=src/infrastructure/logging --pytest-args -v

# 指定测试文件
run_tests_cmd.bat --env rqa --test-file tests/unit/trading/test_minimal_trading_main_flow.py --cov=src/trading --pytest-args -v

# 运行所有测试
run_tests_cmd.bat --env rqa --all --cov=src --pytest-args -v
```

### 4. PowerShell脚本

#### 特性
- **原生PowerShell**：使用PowerShell原生语法
- **进程管理**：使用System.Diagnostics.Process
- **作业管理**：使用PowerShell Jobs进行超时控制
- **实时输出**：逐行处理标准输出和错误输出

#### 使用方法
```powershell
# 使用PowerShell脚本
.\run_tests_powershell.ps1 -Env rqa -Module infrastructure.logging -Cov src/infrastructure/logging -PytestArgs -v

# 指定测试文件
.\run_tests_powershell.ps1 -Env rqa -TestFile tests/unit/trading/test_minimal_trading_main_flow.py -Cov src/trading -PytestArgs -v

# 运行所有测试
.\run_tests_powershell.ps1 -Env rqa -All -Cov src -PytestArgs -v

# 自定义超时和重试
.\run_tests_powershell.ps1 -Env rqa -Module infrastructure.logging -Cov src/infrastructure/logging -Timeout 300 -Retry 3 -PytestArgs -v
```

## 推荐使用方案

### 1. 首选方案：改进的Python脚本
```bash
python run_tests.py --env rqa --module infrastructure.logging --cov=src/infrastructure/logging --pytest-args -v
```

### 2. PowerShell环境专用方案
```bash
python run_tests_powershell.py --env rqa --module infrastructure.logging --cov=src/infrastructure/logging --pytest-args -v
```

### 3. CMD环境方案
```cmd
run_tests_cmd.bat --env rqa --module infrastructure.logging --cov=src/infrastructure/logging --pytest-args -v
```

### 4. PowerShell脚本方案
```powershell
.\run_tests_powershell.ps1 -Env rqa -Module infrastructure.logging -Cov src/infrastructure/logging -PytestArgs -v
```

## 参数说明

### 通用参数
- `--env` / `-Env`：测试环境名称（必需）
- `--module` / `-Module`：指定测试模块
- `--test-file` / `-TestFile`：指定测试文件路径
- `--all` / `-All`：运行所有测试
- `--timeout` / `-Timeout`：超时时间（秒，默认600）
- `--cov` / `-Cov`：覆盖率统计模块路径
- `--skip-coverage` / `-SkipCoverage`：跳过覆盖率检查
- `--pytest-args` / `-PytestArgs`：传递给pytest的额外参数
- `--retry` / `-Retry`：失败重试次数（默认2）
- `--no-retry` / `-NoRetry`：禁用重试机制

## 故障排除

### 1. 进程未返回问题
- 使用改进的Python脚本，自动检测PowerShell环境
- 使用CMD批处理脚本避免PowerShell问题
- 使用PowerShell专用脚本进行优化处理

### 2. 超时问题
- 所有脚本都包含超时机制
- 可以自定义超时时间
- 超时后自动终止进程

### 3. 输出问题
- 实时输出处理
- 错误输出分离显示
- 支持详细模式输出

### 4. 重试机制
- 自动重试失败的测试
- 可配置重试次数
- 可禁用重试机制

## 环境要求

- Python 3.7+
- conda环境
- pytest
- Windows 10/11
- PowerShell 5.1+ 或 CMD

## 注意事项

1. **环境变量**：确保`CONDA_PATH`环境变量正确设置
2. **权限问题**：确保有足够的权限运行conda命令
3. **路径问题**：确保测试路径正确
4. **编码问题**：使用UTF-8编码处理输出
5. **进程清理**：脚本会自动清理残留进程 