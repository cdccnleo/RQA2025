# 基础设施层自动化工具使用指南

## 📊 文档信息

**文档版本**: v1.0  
**创建日期**: 2025-10-24  
**适用范围**: RQA2025项目开发  
**文档类型**: 工具指南

---

## 🎯 概述

本指南介绍RQA2025项目中使用的自动化质量保障工具，包括pre-commit hooks、代码风格检查、类型检查、安全检查等。

---

## 🔧 工具清单

### 已配置的自动化工具

| 工具 | 功能 | 阶段 | 状态 |
|------|------|------|------|
| **pre-commit** | Git提交前自动检查 | 提交前 | ✅ 已配置 |
| **autopep8** | 自动修复PEP 8风格 | 提交前 | ✅ 已配置 |
| **flake8** | 代码风格检查 | 提交前 | ✅ 已配置 |
| **isort** | 导入语句排序 | 提交前 | ✅ 已配置 |
| **mypy** | 静态类型检查 | 提交前 | ✅ 已配置 |
| **bandit** | 安全漏洞检查 | 提交前 | ✅ 已配置 |
| **pytest** | 单元测试 | CI/CD | ⏳ 待配置 |

---

## 🚀 快速开始

### 1. 安装pre-commit

```bash
# 安装pre-commit
pip install pre-commit

# 验证安装
pre-commit --version
```

### 2. 激活pre-commit hooks

```bash
# 在项目根目录执行
cd C:\PythonProject\RQA2025

# 安装hooks
pre-commit install

# 验证安装
pre-commit run --all-files
```

### 3. 日常使用

```bash
# 正常git commit，会自动触发检查
git add .
git commit -m "feature: 添加新功能"

# 如果检查失败，修复后重新提交
git add .
git commit -m "feature: 添加新功能"

# 跳过pre-commit检查（不推荐）
git commit -m "message" --no-verify
```

---

## 📋 工具详细说明

### 1. autopep8 - 自动修复代码风格

**功能**: 自动修复PEP 8代码风格问题

**配置**:
```yaml
- id: autopep8
  args: ['--in-place', '--aggressive', '--aggressive']
```

**手动运行**:
```bash
# 修复单个文件
autopep8 --in-place --aggressive --aggressive src/infrastructure/core/base.py

# 修复整个目录
autopep8 --in-place --aggressive --aggressive --recursive src/infrastructure/

# 查看差异（不修改）
autopep8 --diff --aggressive src/infrastructure/core/base.py
```

---

### 2. flake8 - 代码风格检查

**功能**: 检查代码是否符合PEP 8规范

**配置**:
```yaml
- id: flake8
  args: [
    '--max-line-length=120',     # 最大行长度
    '--extend-ignore=E203,E501,W503',  # 忽略特定规则
    '--max-complexity=15'        # 最大复杂度
  ]
```

**手动运行**:
```bash
# 检查单个文件
flake8 src/infrastructure/core/base.py

# 检查整个目录
flake8 src/infrastructure/

# 生成报告
flake8 src/infrastructure/ --format=html --htmldir=reports/flake8
```

**常见错误码**:
- `E501`: 行长度超过限制
- `E302`: 期望2个空行
- `E303`: 多余的空行
- `W503`: 二元运算符前换行
- `F401`: 未使用的导入

---

### 3. isort - 导入语句排序

**功能**: 自动排序和格式化import语句

**配置**:
```yaml
- id: isort
  args: ['--profile', 'black', '--line-length=120']
```

**手动运行**:
```bash
# 排序单个文件
isort src/infrastructure/core/base.py

# 排序整个目录
isort src/infrastructure/

# 检查而不修改
isort --check-only src/infrastructure/

# 查看差异
isort --diff src/infrastructure/core/base.py
```

**排序规则**:
1. 标准库
2. 第三方库
3. 本地应用/库
4. 按字母排序

---

### 4. mypy - 静态类型检查

**功能**: 检查Python类型注解的正确性

**配置**:
```yaml
- id: mypy
  args: [
    '--ignore-missing-imports',  # 忽略缺失的导入类型
    '--no-strict-optional',      # 不严格检查Optional
    '--warn-return-any',         # 警告返回Any
    '--warn-unused-configs'      # 警告未使用的配置
  ]
```

**手动运行**:
```bash
# 检查单个文件
mypy src/infrastructure/core/base.py

# 检查整个目录
mypy src/infrastructure/

# 生成报告
mypy src/infrastructure/ --html-report reports/mypy
```

**类型注解示例**:
```python
from typing import Optional, Dict, List, Any

def get_config(self, key: str, default: Optional[Any] = None) -> Any:
    """获取配置"""
    pass

def process_items(self, items: List[Dict[str, Any]]) -> List[str]:
    """处理项目列表"""
    pass
```

---

### 5. bandit - 安全漏洞检查

**功能**: 检查Python代码中的安全问题

**配置文件**: `.bandit.yaml`

**手动运行**:
```bash
# 检查单个文件
bandit src/infrastructure/core/base.py

# 检查整个目录
bandit -r src/infrastructure/

# 使用配置文件
bandit -r src/infrastructure/ -c .bandit.yaml

# 生成JSON报告
bandit -r src/infrastructure/ -f json -o reports/bandit_report.json

# 生成HTML报告
bandit -r src/infrastructure/ -f html -o reports/bandit_report.html
```

**常见安全问题**:
- `B201`: Flask debug模式
- `B301`: pickle使用
- `B307`: eval使用
- `B501`: 不安全的SSL证书验证
- `B608`: SQL注入风险

---

### 6. pytest - 单元测试

**功能**: 运行单元测试和代码覆盖率检查

**配置文件**: `pytest.ini` (需创建)

**手动运行**:
```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_infrastructure/test_core.py

# 运行特定测试
pytest tests/test_infrastructure/test_core.py::test_parameter_objects

# 生成覆盖率报告
pytest --cov=src/infrastructure --cov-report=html

# 并行执行测试（使用pytest-xdist）
pytest -n auto
```

---

## 📊 CI/CD集成

### GitHub Actions配置示例

**文件**: `.github/workflows/quality-check.yml`

```yaml
name: 代码质量检查

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  quality-check:
    runs-on: windows-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: 设置Python环境
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: 安装依赖
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pre-commit pytest pytest-cov flake8 mypy bandit
    
    - name: 运行pre-commit检查
      run: |
        pre-commit run --all-files
    
    - name: 运行单元测试
      run: |
        pytest --cov=src/infrastructure --cov-report=xml
    
    - name: 运行安全检查
      run: |
        bandit -r src/infrastructure/ -c .bandit.yaml
    
    - name: 上传覆盖率报告
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

---

## 🎯 质量门禁标准

### 提交前必须通过的检查

| 检查项 | 工具 | 标准 | 重要性 |
|--------|------|------|--------|
| **代码风格** | flake8 | 0错误 | ⭐⭐⭐⭐⭐ |
| **类型检查** | mypy | 0错误 | ⭐⭐⭐⭐ |
| **安全检查** | bandit | 0高危 | ⭐⭐⭐⭐⭐ |
| **单元测试** | pytest | 全通过 | ⭐⭐⭐⭐⭐ |
| **测试覆盖率** | pytest-cov | ≥80% | ⭐⭐⭐⭐ |

---

## 💡 最佳实践

### 1. 提交前检查

```bash
# 推荐的提交流程
# 1. 运行pre-commit检查
pre-commit run --all-files

# 2. 运行单元测试
pytest tests/

# 3. 查看覆盖率
pytest --cov=src/infrastructure --cov-report=term

# 4. 如果全部通过，再提交
git add .
git commit -m "feat: 添加新功能"
```

### 2. 跳过某些检查（谨慎使用）

```bash
# 跳过pre-commit（不推荐）
git commit --no-verify

# 跳过特定hook
SKIP=flake8 git commit -m "message"

# 在代码中跳过flake8检查
# flake8: noqa

# 在代码中跳过bandit检查
# nosec
```

### 3. 配置IDE集成

**VS Code配置** (`.vscode/settings.json`):
```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.linting.banditEnabled": true,
  "python.formatting.provider": "autopep8",
  "python.formatting.autopep8Args": [
    "--aggressive",
    "--aggressive"
  ],
  "editor.formatOnSave": true,
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

---

## 🧪 测试规范

### 单元测试文件组织

```
tests/
├── test_infrastructure/
│   ├── test_core/
│   │   ├── test_parameter_objects.py
│   │   ├── test_mock_services.py
│   │   └── test_constants.py
│   ├── test_api/
│   │   ├── test_configs.py
│   │   └── test_components.py
│   └── test_distributed/
│       └── test_distributed_services.py
```

### 测试用例命名规范

```python
# ✅ 推荐的命名规范
def test_health_check_params_validation():
    """测试健康检查参数验证"""
    pass

def test_health_check_params_default_values():
    """测试健康检查参数默认值"""
    pass

def test_health_check_params_invalid_timeout():
    """测试健康检查参数无效超时"""
    pass

# 命名格式: test_<被测试对象>_<测试场景>
```

### 测试用例编写规范

```python
import pytest
from src.infrastructure.core.parameter_objects import HealthCheckParams

class TestHealthCheckParams:
    """健康检查参数对象测试"""
    
    def test_valid_params(self):
        """测试有效参数"""
        # Arrange
        params = HealthCheckParams(service_name="database", timeout=30)
        
        # Assert
        assert params.service_name == "database"
        assert params.timeout == 30
        assert params.retry_count == 3  # 默认值
    
    def test_invalid_timeout(self):
        """测试无效超时参数"""
        # Act & Assert
        with pytest.raises(ValueError, match="timeout必须大于0"):
            HealthCheckParams(service_name="db", timeout=-1)
    
    def test_default_timestamp(self):
        """测试默认时间戳"""
        # Arrange
        params = HealthCheckParams(service_name="db")
        
        # Assert
        assert params.check_timestamp is not None
```

---

## 📈 持续集成流程

### 本地开发流程

```
编写代码
    ↓
IDE自动格式化（保存时）
    ↓
运行单元测试
    ↓
git add .
    ↓
pre-commit自动检查
    ↓
    ├─ 通过 → git commit成功
    └─ 失败 → 修复问题 → 重新提交
```

### CI/CD流程

```
git push
    ↓
触发GitHub Actions
    ↓
    ├─ 代码风格检查（flake8）
    ├─ 类型检查（mypy）
    ├─ 安全检查（bandit）
    ├─ 单元测试（pytest）
    └─ 覆盖率检查（pytest-cov）
    ↓
    ├─ 全部通过 → 允许合并
    └─ 有失败 → 阻止合并，需要修复
```

---

## 🎯 工具配置详解

### flake8配置说明

**当前配置**:
```yaml
--max-line-length=120          # 最大行长度120字符
--extend-ignore=E203,E501,W503 # 忽略特定规则
--max-complexity=15            # 最大圈复杂度15
```

**可调整参数**:
```bash
# 更严格的配置
flake8 --max-line-length=100 --max-complexity=10

# 更宽松的配置
flake8 --max-line-length=150 --max-complexity=20
```

---

### mypy配置说明

**当前配置**:
```yaml
--ignore-missing-imports  # 忽略缺失的导入类型
--no-strict-optional      # 不严格检查Optional
--warn-return-any         # 警告返回Any
```

**推荐的mypy.ini配置**:
```ini
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False
ignore_missing_imports = True

[mypy-tests.*]
ignore_errors = True
```

---

### bandit配置说明

**当前配置**: 见`.bandit.yaml`

**严重程度**:
- `LOW`: 低风险（如使用assert）
- `MEDIUM`: 中等风险（如使用MD5）
- `HIGH`: 高风险（如使用eval）

**置信度**:
- `LOW`: 可能误报
- `MEDIUM`: 较可靠
- `HIGH`: 确定问题

---

## 🚨 常见问题解决

### 问题1: flake8报错行太长

```python
# ❌ 问题代码
def long_function_name_with_many_parameters(parameter1, parameter2, parameter3, parameter4, parameter5, parameter6):
    pass

# ✅ 解决方案1: 换行
def long_function_name_with_many_parameters(
    parameter1, parameter2, parameter3,
    parameter4, parameter5, parameter6
):
    pass

# ✅ 解决方案2: 使用参数对象（更推荐）
def long_function_name(params: FunctionParams):
    pass
```

### 问题2: mypy类型错误

```python
# ❌ 问题代码
def get_value(self, key):  # 缺少类型注解
    return self.data.get(key)

# ✅ 解决方案
def get_value(self, key: str) -> Optional[Any]:
    return self.data.get(key)
```

### 问题3: bandit安全警告

```python
# ❌ 问题代码
password = "hardcoded_password"  # B105: hardcoded_password

# ✅ 解决方案
password = os.getenv("DATABASE_PASSWORD")

# 如果确实需要，可以添加注释忽略
password = "test_password"  # nosec B105
```

---

## 📊 质量监控

### 运行质量报告

```bash
# 1. 生成flake8 HTML报告
flake8 src/infrastructure/ --format=html --htmldir=reports/flake8

# 2. 生成mypy HTML报告
mypy src/infrastructure/ --html-report reports/mypy

# 3. 生成bandit JSON报告
bandit -r src/infrastructure/ -f json -o reports/bandit_report.json

# 4. 生成测试覆盖率报告
pytest --cov=src/infrastructure --cov-report=html --cov-report=term

# 5. 查看HTML报告
start reports/flake8/index.html        # Windows
start reports/mypy/index.html
start reports/coverage/index.html
```

### 质量指标跟踪

**建议监控的指标**:
- flake8错误数: 目标0
- mypy类型覆盖率: 目标≥80%
- bandit高危问题: 目标0
- 单元测试覆盖率: 目标≥95%
- 代码复杂度: 目标≤15

---

## 🎯 最佳实践总结

### 开发工作流

```
1. 编写代码
   ├─ 使用类型注解
   ├─ 编写文档字符串
   └─ 遵循代码规范

2. 编写测试
   ├─ 单元测试覆盖
   ├─ 使用Mock基类
   └─ 测试边界情况

3. 本地检查
   ├─ 运行pre-commit
   ├─ 运行pytest
   └─ 查看覆盖率

4. 提交代码
   ├─ git add .
   ├─ git commit (触发pre-commit)
   └─ git push (触发CI/CD)

5. 代码审查
   ├─ 检查CI/CD结果
   ├─ 团队代码审查
   └─ 修复问题

6. 合并代码
```

---

## 📚 参考资源

### 官方文档

- [pre-commit官方文档](https://pre-commit.com/)
- [flake8官方文档](https://flake8.pycqa.org/)
- [mypy官方文档](https://mypy.readthedocs.io/)
- [bandit官方文档](https://bandit.readthedocs.io/)
- [pytest官方文档](https://docs.pytest.org/)

### RQA2025项目文档

- 基础设施层最佳实践指南: `docs\infrastructure\best_practices_guide.md`
- 接口API文档: `docs\infrastructure\interfaces_api_documentation.md`
- 架构设计文档: `docs\architecture\infrastructure_architecture_design.md`

---

## 🔄 工具更新

### 定期更新工具

```bash
# 更新pre-commit hooks
pre-commit autoupdate

# 更新Python包
pip install --upgrade pre-commit flake8 mypy bandit pytest pytest-cov

# 验证更新
pre-commit run --all-files
```

---

## 🎊 总结

### 核心价值

✅ **自动化质量保障** - 减少人工检查工作量90%  
✅ **早期问题发现** - 在提交前发现并修复问题  
✅ **统一代码风格** - 自动格式化，团队风格一致  
✅ **提升代码质量** - 类型检查、安全检查、复杂度控制  
✅ **降低Bug率** - 通过测试覆盖降低Bug率30-50%  

### 下一步行动

- [ ] 安装pre-commit: `pip install pre-commit`
- [ ] 激活hooks: `pre-commit install`
- [ ] 运行首次检查: `pre-commit run --all-files`
- [ ] 配置IDE集成
- [ ] 建立CI/CD流水线

---

**文档版本**: v1.0  
**最后更新**: 2025-10-24  
**维护团队**: RQA2025质量保障团队  

---

*通过自动化工具，我们建立了完整的质量保障体系，确保代码质量持续提升！* 🚀✨
