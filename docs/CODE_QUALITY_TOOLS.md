# RQA2025 代码质量工具使用指南

## 概述

本项目配置了完整的代码质量工具链，包括代码格式化、导入排序、静态检查和提交前自动化检查。

## 工具清单

| 工具 | 用途 | 配置文件 |
|------|------|----------|
| **Black** | 代码格式化 | `pyproject.toml` |
| **isort** | 导入排序 | `pyproject.toml` |
| **Flake8** | 代码检查 | `pyproject.toml` |
| **pre-commit** | 提交前检查 | `.pre-commit-config.yaml` |
| **mypy** | 类型检查 | `pyproject.toml` |

## 快速开始

### 1. 安装工具

```bash
# 安装所有开发依赖
pip install -e ".[dev]"

# 或单独安装
pip install black flake8 isort pre-commit mypy
```

### 2. 配置pre-commit钩子

```bash
# 安装pre-commit钩子
pre-commit install

# 安装commit-msg钩子（可选）
pre-commit install --hook-type commit-msg
```

### 3. 验证安装

```bash
# 检查各工具版本
black --version
flake8 --version
isort --version
pre-commit --version
```

## 使用指南

### Black - 代码格式化

```bash
# 检查代码格式（不修改文件）
black --check src/

# 格式化代码
black src/

# 格式化特定文件
black src/core/business_process/manager_components.py

# 显示差异
black --diff src/
```

**配置说明**:
- 行长度: 100字符
- Python版本: 3.9+
- 排除目录: backups/, production_simulation/, docs/, reports/

### isort - 导入排序

```bash
# 检查导入排序（不修改文件）
isort --check-only src/

# 排序导入
isort src/

# 显示差异
isort --diff src/

# 排序特定文件
isort src/core/business_process/manager_components.py
```

**配置说明**:
- 使用Black兼容配置
- 行长度: 100字符
- 识别src为first-party模块

### Flake8 - 代码检查

```bash
# 检查整个项目
flake8 src/

# 检查特定错误类型
flake8 src/ --select=E,W,F

# 排除特定错误
flake8 src/ --ignore=E501,W503

# 显示统计信息
flake8 src/ --count --statistics

# 输出到文件
flake8 src/ --output-file=flake8_report.txt
```

**配置说明**:
- 最大行长度: 100
- 忽略: E203, W503 (与Black兼容)
- 最大复杂度: 15
- 排除目录: backups/, production_simulation/, docs/, reports/

### pre-commit - 提交前检查

```bash
# 手动运行所有钩子
pre-commit run --all-files

# 运行特定钩子
pre-commit run black --all-files
pre-commit run isort --all-files
pre-commit run flake8 --all-files

# 更新钩子版本
pre-commit autoupdate

# 跳过钩子（不推荐）
git commit -m "message" --no-verify
```

**钩子列表**:
1. **基础检查**: 合并冲突、YAML/JSON/TOML语法、行尾空格、大文件、私钥
2. **Black**: 代码格式化
3. **isort**: 导入排序
4. **Flake8**: 代码检查
5. **Commitizen**: 提交信息格式检查

### mypy - 类型检查

```bash
# 类型检查
mypy src/

# 显示错误代码
mypy src/ --show-error-codes

# 忽略缺少类型的第三方库
mypy src/ --ignore-missing-imports

# 检查特定模块
mypy src/core/business_process/
```

## 集成到开发流程

### 推荐的开发流程

1. **编写代码** → 正常开发
2. **本地检查** → 运行 `pre-commit run --all-files`
3. **修复问题** → 根据检查结果修改
4. **提交代码** → pre-commit自动运行检查
5. **推送代码** → CI/CD流水线再次检查

### IDE集成

#### VS Code

安装扩展:
- Python (Microsoft)
- Black Formatter
- Flake8

配置 `.vscode/settings.json`:
```json
{
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "editor.formatOnSave": true,
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.sortImports.args": ["--profile", "black"],
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm

1. **Black集成**:
   - Settings → Tools → External Tools
   - 添加Black作为外部工具
   - 配置快捷键

2. **Flake8集成**:
   - Settings → Editor → Inspections
   - 启用Flake8检查

3. **isort集成**:
   - Settings → Editor → Code Style
   - 配置导入排序规则

## 故障排除

### 常见问题

#### 1. pre-commit安装失败

```bash
# 清除缓存重试
pre-commit clean
pre-commit install
```

#### 2. Black和Flake8配置冲突

确保 `pyproject.toml` 中:
- Black: `line-length = 100`
- Flake8: `max-line-length = 100`, `extend-ignore = ["E203", "W503"]`

#### 3. isort和Black冲突

确保 `pyproject.toml` 中:
- isort: `profile = "black"`

#### 4. 忽略特定文件

在 `pyproject.toml` 中添加排除模式:
```toml
[tool.black]
extend-exclude = '''
/(
  | specific_file\.py
  | specific_directory/
)/
'''
```

### 性能优化

对于大型项目:

```bash
# 只检查修改的文件
pre-commit run

# 并行运行
pre-commit run --all-files -j 4

# 跳过慢速检查器
SKIP=flake8 git commit -m "message"
```

## 配置详解

### pyproject.toml 完整配置

详见项目根目录的 `pyproject.toml` 文件，包含:
- 项目元数据
- 构建系统配置
- Black配置
- isort配置
- Flake8配置
- mypy配置
- pytest配置
- coverage配置

### .pre-commit-config.yaml 完整配置

详见项目根目录的 `.pre-commit-config.yaml` 文件，包含:
- 基础文件检查钩子
- Black格式化钩子
- isort排序钩子
- Flake8检查钩子
- 提交信息检查钩子

## 最佳实践

1. **始终使用pre-commit**: 在团队开发中强制使用
2. **配置IDE集成**: 在保存时自动格式化
3. **定期检查**: 每周运行一次完整检查
4. **逐步修复**: 不要试图一次性修复所有问题
5. **文档化例外**: 对于必须忽略的检查，添加注释说明原因

## 参考链接

- [Black文档](https://black.readthedocs.io/)
- [isort文档](https://pycqa.github.io/isort/)
- [Flake8文档](https://flake8.pycqa.org/)
- [pre-commit文档](https://pre-commit.com/)
- [mypy文档](https://mypy.readthedocs.io/)

---

**维护者**: RQA2025 Development Team  
**最后更新**: 2026-03-08
