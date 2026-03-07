# RQA2025Conda环境使用指南

## 概述

本文档介绍如何在conda test环境下运行RQA225项目的分层测试和自动化推进脚本。run_tests.py脚本现已默认在test环境下运行，用户可通过--env参数切换至rqa、base等其它环境。

## 环境准备

### 1. 检查conda环境

首先确保您已安装Anaconda或Miniconda，然后检查rqa或base环境：

```bash
# 查看所有环境
conda info --envs

# 如果rqa环境不存在，可手动创建
conda create -n rqa python=3.9 -y
```

### 2. 激活test或rqa环境

```bash
# 激活test环境（推荐，run_tests.py默认）
conda activate test

# 如需rqa环境
conda activate rqa

# 验证环境
python --version
```

### 3. 安装必需的包

```bash
# 安装测试相关包
pip install pytest pytest-cov

# 安装项目依赖
pip install pandas numpy requests pyyaml jinja2
```

## 运行测试脚本

### 方法1: 使用run_tests.py（推荐）

```bash
# 在项目根目录下运行，默认test环境
python scripts/testing/run_tests.py --all --cov=src
# 如需指定其它环境
python scripts/testing/run_tests.py --env rqa --all --cov=src
# 传递pytest参数时，注意每个参数单独空格分隔，不要用引号包裹多个参数！
# 正确：
python scripts/testing/run_tests.py --all --cov=src --pytest-args --cov-report=html --maxfail=2
# 错误：
python scripts/testing/run_tests.py --all --cov=src --pytest-args "--cov-report=html --maxfail=2"
```

> **注意：--pytest-args 后的每个pytest参数必须用空格分隔，不要用引号包裹多个参数，否则pytest-cov等插件会解析失败。**

### 方法2: 使用批处理脚本

```bash
scripts\run_conda_tests.bat
```

### 方法3: 使用PowerShell脚本

```powershell
.\scripts\run_conda_tests.ps1
```

### 方法4: Python测试运行器

```bash
python scripts/conda_test_runner.py
```

## 脚本功能说明

### 1 run_conda_tests.bat
- 自动检查conda环境
- 激活test环境
- 检查Python和pytest
- 按层执行测试
- 生成覆盖率报告

### 2. run_conda_tests.ps1
- PowerShell版本的测试运行器
- 支持参数化运行
- 更详细的错误处理
- 彩色输出

### 3. conda_test_runner.py
- Python版本的测试运行器
- 自动环境检查
- 生成详细报告
- 支持自定义配置

## 测试层说明

项目包含以下测试层：

1. **infrastructure** - 基础设施层
   - 目标覆盖率: 90%
   - 优先级: 最高

2 **data** - 数据层
   - 目标覆盖率: 80   - 优先级: 高

3 **features** - 特征层
   - 目标覆盖率: 80   - 优先级: 高

4. **models** - 模型层
   - 目标覆盖率: 80   - 优先级: 中
5. **trading** - 交易层
   - 目标覆盖率: 80   - 优先级: 中

6. **backtest** - 回测层
   - 目标覆盖率:80%
   - 优先级: 低

## 报告文件

运行测试后，会生成以下报告文件：

- `reports/` - 覆盖率HTML报告
- `docs/conda_test_report.md` - 测试报告
- `docs/conda_test_results.json` - 测试数据
- `docs/progress_report_conda.md` - 进度报告

## 故障排除

### 常见问题

1 **conda命令未找到**
   - 确保已安装Anaconda或Miniconda
   - 将conda添加到系统PATH

2 **rqa环境不存在**
   - 运行: `conda create -n rqa python=3.9-y`
   - 然后: `conda activate rqa`
3. **pytest未安装**
   - 在rqa环境中运行: `pip install pytest pytest-cov`

4 **Python版本问题**
   - 确保使用Python30.8本
   - 检查: `python --version`
5. **测试失败**
   - 检查测试文件是否存在
   - 查看错误输出
   - 确保所有依赖已安装

### 调试模式

启用详细输出：

```bash
# 批处理脚本
set DEBUG=1
scripts\run_conda_tests.bat

# PowerShell脚本
$env:DEBUG = 1.\scripts\run_conda_tests.ps1

# Python脚本
python scripts/conda_test_runner.py --debug
```

## 自动化推进

### 完整自动化流程

1*环境检查**
   ```bash
   conda activate rqa
   python scripts/conda_test_runner.py
   ```

2*分层测试**
   ```bash
   # 运行所有层
   scripts\run_conda_tests.bat
   
   # 或运行特定层
   python -m pytest tests/unit/infrastructure/ --cov=src/infrastructure
   ```

3*生成报告**
   ```bash
   # 自动生成报告
   python scripts/conda_test_runner.py
   ```

### 持续集成

可以将这些脚本集成到CI/CD流程中：

```yaml
# .github/workflows/test.yml 示例
name: RQA2025ts
on: [push, pull_request]
jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v2
      - uses: conda-incubator/setup-miniconda@v2
        with:
          environment-file: environment.yml
      - name: Run tests
        run: |
          conda activate rqa
          scripts\run_conda_tests.bat
```

## 最佳实践
1. **环境隔离**
   - 始终在rqa环境中运行测试
   - 避免在全局环境中安装项目依赖

2
   - 将environment.yml文件纳入版本控制
   - 记录依赖版本
3. **定期更新**
   - 定期更新conda和pip
   - 更新测试依赖
4. **报告管理**
   - 定期清理旧的报告文件
   - 备份重要的测试结果

## 联系支持

如果遇到问题，请：

1. 检查本文档的故障排除部分2看项目日志文件
3 提交issue并附上详细的错误信息

---

*最后更新: 2025年1月* 