# Phase 3 完整执行指南

## 概述

本指南提供Phase 3深度优化阶段的完整执行步骤，包括所有命令和脚本。

---

## 环境准备

### 1. 确保工具已安装

```bash
# 检查工具版本
black --version
isort --version
flake8 --version
pre-commit --version
```

如果未安装：
```bash
pip install black flake8 isort pre-commit mypy
```

### 2. 配置pre-commit钩子

```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

---

## Week 1: 代码格式化和基础修复

### Day 1-2: 代码格式化

#### 步骤1: 运行Black格式化

```bash
# 检查需要格式化的文件数量
black --check src/ --line-length 100 --target-version py39 2>&1 | grep -c "would be reformatted"

# 执行格式化（这会修改文件）
black src/ --line-length 100 --target-version py39

# 验证结果
black --check src/ --line-length 100
```

**预期输出**:
```
All done! ✨ 🍰 ✨
XXX files would be left unchanged.
```

#### 步骤2: 运行isort排序导入

```bash
# 检查需要排序的文件
isort --check-only src/ --profile black --line-length 100

# 执行排序
isort src/ --profile black --line-length 100

# 验证结果
isort --check-only src/ --profile black
```

**预期输出**:
```
Skipped X files
```

#### 步骤3: 提交格式化变更

```bash
# 查看变更
 git status

# 添加所有变更
git add .

# 提交
git commit -m "style: format code with Black and isort

- Format all Python files with Black (line length 100)
- Sort imports with isort (black profile)
- Fix whitespace and blank line issues

Phase 3 Week 1 Day 1-2"
```

---

### Day 3-4: 批量修复简单错误

#### 步骤1: 运行自动修复脚本

```bash
# 运行批量修复脚本
python scripts/batch_fix_simple_issues.py
```

**脚本功能**:
- 修复W291: 行尾空格
- 修复W293: 空行空格
- 修复W391: 文件末尾空行
- 修复E501: 行过长（简单情况）

#### 步骤2: 手动修复剩余E501错误

对于无法自动修复的长行，手动处理：

```bash
# 查看所有E501错误
flake8 src/ --select=E501 --show-source
```

常见修复方法：

**情况1: 长字符串**
```python
# 修复前
long_string = "这是一个非常长的字符串，超过了100个字符的限制，需要换行处理"

# 修复后
long_string = (
    "这是一个非常长的字符串，超过了100个字符的限制，"
    "需要换行处理"
)
```

**情况2: 长函数调用**
```python
# 修复前
result = some_function(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)

# 修复后
result = some_function(
    arg1, arg2, arg3, arg4, arg5,
    arg6, arg7, arg8, arg9, arg10
)
```

**情况3: 长列表/字典**
```python
# 修复前
my_list = [item1, item2, item3, item4, item5, item6, item7, item8, item9, item10]

# 修复后
my_list = [
    item1, item2, item3, item4, item5,
    item6, item7, item8, item9, item10
]
```

#### 步骤3: 修复空行错误 (E302, E305)

```bash
# 查看空行错误
flake8 src/ --select=E302,E305 --show-source
```

修复规则：
- 类定义前应该有2个空行
- 函数定义前应该有1个空行
- 类/函数结束后应该有2个空行

#### 步骤4: 提交修复

```bash
git add .
git commit -m "style: fix simple flake8 errors

- Fix W291, W293, W391 whitespace errors
- Fix E501 line too long errors
- Fix E302, E305 blank line errors

Phase 3 Week 1 Day 3-4"
```

---

### Day 5: 验证和测试

#### 步骤1: 运行完整Flake8检查

```bash
# 生成详细报告
flake8 src/ \
    --max-line-length=100 \
    --extend-ignore=E203,W503 \
    --exclude=backups,production_simulation,docs,reports,__pycache__ \
    --count \
    --statistics \
    --output-file=week1_flake8_report.txt

# 查看统计
cat week1_flake8_report.txt | tail -20
```

#### 步骤2: 运行单元测试

```bash
# 运行所有测试
pytest tests/ -v --tb=short

# 或者只运行快速测试
pytest tests/ -v --tb=short -m "not slow"
```

#### 步骤3: 生成质量报告

```bash
python scripts/generate_quality_report.py
```

#### 步骤4: 提交报告

```bash
git add week1_flake8_report.txt code_quality_report.json
git commit -m "docs: add week 1 quality reports

- Add flake8 report
- Add quality score report

Phase 3 Week 1 Day 5"
```

---

## Week 2: 复杂错误修复

### Day 6-7: 修复未使用导入 (F401)

#### 步骤1: 识别未使用导入

```bash
flake8 src/ --select=F401 --output-file=f401_errors.txt
```

#### 步骤2: 批量修复

创建脚本 `fix_f401.py`:

```python
#!/usr/bin/env python3
import ast
import re
from pathlib import Path

def fix_unused_imports(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False
    
    # 收集导入和使用情况
    imports = {}
    used_names = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports[name] = {'type': 'import', 'node': node}
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                if name != '*':
                    imports[name] = {
                        'type': 'from_import',
                        'module': module,
                        'node': node
                    }
        elif isinstance(node, ast.Name):
            used_names.add(node.id)
    
    # 找出未使用的导入
    unused = set(imports.keys()) - used_names
    
    if not unused:
        return False
    
    # 移除未使用的导入
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        should_keep = True
        line_stripped = line.strip()
        
        for unused_name in unused:
            if line_stripped.startswith('import ') and unused_name in line_stripped:
                should_keep = False
                break
            elif line_stripped.startswith('from ') and ' import ' in line_stripped:
                # 处理 from x import a, b, c 的情况
                match = re.match(r'from\s+(\S+)\s+import\s+(.+)', line_stripped)
                if match:
                    module, names = match.groups()
                    name_list = [n.strip() for n in names.split(',')]
                    if unused_name in name_list:
                        if len(name_list) == 1:
                            should_keep = False
                        else:
                            # 移除单个名称，保留其他
                            new_names = [n for n in name_list if n != unused_name]
                            line = line.replace(names, ', '.join(new_names))
        
        if should_keep:
            new_lines.append(line)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    
    return True

# 处理所有文件
src_dir = Path('src')
for py_file in src_dir.rglob('*.py'):
    if fix_unused_imports(py_file):
        print(f"Fixed: {py_file}")
```

运行脚本：
```bash
python fix_f401.py
```

#### 步骤3: 提交

```bash
git add .
git commit -m "refactor: remove unused imports

- Fix F401 unused import errors
- Clean up import statements

Phase 3 Week 2 Day 6-7"
```

---

### Day 8-9: 修复未定义变量 (F821, F822)

#### 步骤1: 识别未定义变量

```bash
flake8 src/ --select=F821,F822 --output-file=f821_errors.txt
```

#### 步骤2: 分类处理

**类型A: 缺失导入**
```python
# 错误
logger.info("message")  # F821 undefined name 'logger'

# 修复
import logging
logger = logging.getLogger(__name__)
logger.info("message")
```

**类型B: 缺失typing导入**
```python
# 错误
def func(data: Dict[str, Any]) -> List[int]:  # F821 undefined name 'Dict', 'Any', 'List'
    pass

# 修复
from typing import Dict, Any, List

def func(data: Dict[str, Any]) -> List[int]:
    pass
```

**类型C: __all__导出错误**
```python
# 错误
__all__ = ['NonExistentClass']  # F822 undefined name 'NonExistentClass'

# 修复
__all__ = ['ExistingClass']  # 修正为实际存在的名称
```

#### 步骤3: 批量修复脚本

创建脚本 `fix_f821.py`:

```python
#!/usr/bin/env python3
import re
from pathlib import Path

# 常见缺失导入映射
MISSING_IMPORTS = {
    'logger': 'import logging\nlogger = logging.getLogger(__name__)',
    'np': 'import numpy as np',
    'pd': 'import pandas as pd',
    'datetime': 'from datetime import datetime',
    'timedelta': 'from datetime import timedelta',
    'Any': 'from typing import Any',
    'Dict': 'from typing import Dict',
    'List': 'from typing import List',
    'Optional': 'from typing import Optional',
    'Union': 'from typing import Union',
    'Tuple': 'from typing import Tuple',
    'Callable': 'from typing import Callable',
    'Protocol': 'from typing import Protocol',
    'asyncio': 'import asyncio',
    'threading': 'import threading',
    'json': 'import json',
    'time': 'import time',
}

def add_missing_import(file_path, missing_name):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if missing_name not in MISSING_IMPORTS:
        return False
    
    import_line = MISSING_IMPORTS[missing_name]
    
    # 检查是否已存在
    if import_line.split('\n')[0] in content:
        return False
    
    # 在文件开头添加导入（在docstring之后）
    lines = content.split('\n')
    import_idx = 0
    in_docstring = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if in_docstring:
                in_docstring = False
                import_idx = i + 1
            else:
                in_docstring = True
        elif not in_docstring and (line.startswith('import ') or line.startswith('from ')):
            import_idx = i + 1
    
    lines.insert(import_idx, import_line)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return True

# 读取错误文件并修复
with open('f821_errors.txt', 'r') as f:
    for line in f:
        # 解析错误行
        match = re.match(r'([^:]+):(\d+):(\d+): F821 undefined name \'(\w+)\'', line)
        if match:
            file_path, line_no, col, name = match.groups()
            if add_missing_import(file_path, name):
                print(f"Fixed {name} in {file_path}")
```

#### 步骤4: 提交

```bash
git add .
git commit -m "fix: add missing imports and fix undefined names

- Fix F821 undefined name errors
- Fix F822 undefined name in __all__ errors
- Add missing typing imports
- Add missing standard library imports

Phase 3 Week 2 Day 8-9"
```

---

### Day 10: 修复变量名错误

#### 步骤1: 识别拼写错误

常见拼写错误：
```python
# 错误
strategy_ids_ids_tuple  # 重复
avaliable  # 应为 available
recieve    # 应为 receive
occured    # 应为 occurred
seperate   # 应为 separate
```

#### 步骤2: 批量修复

```bash
# 使用sed批量替换（谨慎操作）
sed -i 's/avaliable/available/g' src/**/*.py
sed -i 's/recieve/receive/g' src/**/*.py
sed -i 's/occured/occurred/g' src/**/*.py
sed -i 's/seperate/separate/g' src/**/*.py
```

#### 步骤3: 提交

```bash
git add .
git commit -m "fix: correct spelling errors

- Fix common spelling mistakes
- Correct variable names

Phase 3 Week 2 Day 10"
```

---

## Week 3: 类型注解添加

### Day 11-13: 为核心模块添加类型注解

#### 步骤1: 识别需要注解的函数

```bash
# 使用mypy找出缺失类型注解的函数
mypy src/ --ignore-missing-imports --disallow-untyped-defs 2>&1 | grep "Function is missing a type annotation"
```

#### 步骤2: 渐进式添加类型注解

优先为核心模块添加：
- `src/core/business_process/`
- `src/core/services/`
- `src/infrastructure/automation/`

示例：

```python
# 添加前
def process_data(data, config):
    pass

# 添加后
from typing import Dict, Any

def process_data(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    pass
```

#### 步骤3: 验证类型注解

```bash
mypy src/core/business_process/ --ignore-missing-imports
mypy src/core/services/ --ignore-missing-imports
```

#### 步骤4: 提交

```bash
git add .
git commit -m "type: add type annotations to core modules

- Add type hints to business_process module
- Add type hints to services module
- Add type hints to automation module

Phase 3 Week 3 Day 11-13"
```

---

## Week 4: 代码重构和最终验证

### Day 16-18: 代码重构

#### 步骤1: 识别重复代码

```bash
# 使用pylint
pylint src/ --disable=all --enable=duplicate-code

# 或使用jscpd
jscpd src/ --min-lines 10 --min-tokens 50
```

#### 步骤2: 提取公共函数

示例：提取日志装饰器

```python
# 创建 src/utils/decorators.py
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def log_execution(func):
    """记录函数执行的装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Enter {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Exit {func.__name__} - Success")
            return result
        except Exception as e:
            logger.error(f"Exit {func.__name__} - Error: {e}")
            raise
    return wrapper
```

#### 步骤3: 提交重构

```bash
git add .
git commit -m "refactor: extract common code and reduce duplication

- Extract logging decorator
- Extract common utility functions
- Reduce code duplication

Phase 3 Week 4 Day 16-18"
```

---

### Day 19-20: 最终验证

#### 步骤1: 完整质量检查

```bash
# Flake8完整检查
flake8 src/ \
    --max-line-length=100 \
    --extend-ignore=E203,W503 \
    --exclude=backups,production_simulation,docs,reports,__pycache__ \
    --count \
    --statistics \
    --output-file=final_flake8_report.txt

# 类型检查
mypy src/ --ignore-missing-imports --show-error-codes

# 测试覆盖率
pytest tests/ --cov=src --cov-report=html --cov-report=term
```

#### 步骤2: 生成最终报告

```bash
python scripts/generate_quality_report.py
```

#### 步骤3: 对比改进效果

```bash
# 对比Phase 1和Phase 3结束时的错误数量
echo "Phase 1错误数: $(wc -l < phase1_flake8_report.txt)"
echo "Phase 3错误数: $(wc -l < final_flake8_report.txt)"
```

#### 步骤4: 提交最终报告

```bash
git add final_flake8_report.txt code_quality_report.json
git commit -m "docs: add final quality reports

- Add final flake8 report
- Add final quality score report
- Phase 3 complete

Phase 3 Week 4 Day 19-20"
```

---

## 质量门禁检查

### 提交前检查清单

```bash
#!/bin/bash
# pre_commit_check.sh

echo "Running pre-commit checks..."

# 1. Black检查
echo "1. Checking Black formatting..."
black --check src/ --line-length 100
if [ $? -ne 0 ]; then
    echo "❌ Black formatting check failed"
    exit 1
fi

# 2. isort检查
echo "2. Checking import sorting..."
isort --check-only src/ --profile black
if [ $? -ne 0 ]; then
    echo "❌ Import sorting check failed"
    exit 1
fi

# 3. Flake8检查
echo "3. Running Flake8..."
flake8 src/ --max-line-length=100 --extend-ignore=E203,W503
if [ $? -ne 0 ]; then
    echo "❌ Flake8 check failed"
    exit 1
fi

# 4. 类型检查
echo "4. Running mypy..."
mypy src/ --ignore-missing-imports
if [ $? -ne 0 ]; then
    echo "❌ Type check failed"
    exit 1
fi

# 5. 测试
echo "5. Running tests..."
pytest tests/ -x -q
if [ $? -ne 0 ]; then
    echo "❌ Tests failed"
    exit 1
fi

echo "✅ All checks passed!"
```

---

## 预期成果

### Week 1 完成后
- ✅ 所有代码符合Black格式
- ✅ 所有导入已排序
- ✅ E501, W291, W293, W391, E302, E305 错误清零

### Week 2 完成后
- ✅ F401 未使用导入清零
- ✅ F821, F822 未定义变量清零
- ✅ 拼写错误修复完成

### Week 3 完成后
- ✅ 核心模块类型注解覆盖率 > 60%
- ✅ mypy检查通过

### Week 4 完成后
- ✅ 重复代码减少 > 30%
- ✅ 代码质量评分 > 8.0
- ✅ 所有测试通过

---

## 故障排除

### 问题1: Black格式化后测试失败

**原因**: 格式化改变了字符串格式或导入顺序

**解决**:
```bash
# 回滚并检查
 git diff HEAD~1

# 手动修复问题
# 重新提交
```

### 问题2: 类型注解导致循环导入

**解决**:
```python
# 使用TYPE_CHECKING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .other_module import SomeClass
```

### 问题3: 修复后错误更多

**解决**:
```bash
# 分步提交，便于回滚
git add -p  # 选择性添加

# 或者使用git bisect定位问题
```

---

## 附录：快速命令参考

```bash
# 格式化
black src/ --line-length 100
isort src/ --profile black

# 检查
flake8 src/ --count --statistics
mypy src/ --ignore-missing-imports
pytest tests/ -v

# 修复
python scripts/batch_fix_simple_issues.py
python fix_f401.py
python fix_f821.py

# 报告
python scripts/generate_quality_report.py
```

---

**执行者**: RQA2025 Development Team  
**创建时间**: 2026-03-08  
**版本**: 1.0
