# 特征层命名规范检查报告

## 执行摘要
本报告对RQA2025项目特征层（src/features）的命名规范进行了全面检查，识别了不符合Python命名规范的问题，并提出了修复建议。

## 命名规范检查结果

### 1. 类命名检查 ✅
- **PascalCase规范**: 所有类名都符合PascalCase规范
- **示例**: `TechnicalProcessor`, `FeatureEngineer`, `FeaturesMonitor`

### 2. 函数命名检查 ⚠️
发现了一些不符合snake_case规范的函数名：

#### 需要修复的函数名
1. **`src/features/technical/technical_processor.py`**:
   - `calc_ma` → `calculate_ma`
   - `calc_rsi` → `calculate_rsi`
   - `calc_macd` → `calculate_macd`
   - `calc_bollinger` → `calculate_bollinger`
   - `calc_obv` → `calculate_obv`
   - `calc_atr` → `calculate_atr`
   - `calc_indicators` → `calculate_indicators`

2. **`src/features/signal_generator.py`**:
   - `_init_a_share_rules` → `_init_ashare_rules`
   - `_check_a_share_restrictions` → `_check_ashare_restrictions`
   - `_generate_with_fpga` → `_generate_with_fpga` (已正确)
   - `_init_a_share_features` → `_init_ashare_features`
   - `_adjust_position_for_a_share` → `_adjust_position_for_ashare`

### 3. 变量命名检查 ✅
- **snake_case规范**: 大部分变量名符合snake_case规范
- **常量命名**: 常量使用UPPER_CASE规范

### 4. 模块命名检查 ✅
- **snake_case规范**: 所有模块文件名符合snake_case规范
- **示例**: `feature_engineer.py`, `technical_processor.py`

## 修复计划

### 阶段1：函数命名修复（本周）

#### 1.1 修复TechnicalProcessor中的函数名
```python
# 修复前
def calc_ma(self, prices=None, window=5, price_col='close'):
def calc_rsi(self, prices=None, window=14, price_col='close'):
def calc_macd(self, prices=None, fast_window=None, ...):
def calc_bollinger(self, prices=None, window=20, num_std=2, price_col='close'):
def calc_obv(self, df, price_col='close', volume_col='volume'):
def calc_atr(self, df, window=14, high_col='high', low_col='low', close_col='close'):
def calc_indicators(self, df, indicators, params=None):

# 修复后
def calculate_ma(self, prices=None, window=5, price_col='close'):
def calculate_rsi(self, prices=None, window=14, price_col='close'):
def calculate_macd(self, prices=None, fast_window=None, ...):
def calculate_bollinger(self, prices=None, window=20, num_std=2, price_col='close'):
def calculate_obv(self, df, price_col='close', volume_col='volume'):
def calculate_atr(self, df, window=14, high_col='high', low_col='low', close_col='close'):
def calculate_indicators(self, df, indicators, params=None):
```

#### 1.2 修复SignalGenerator中的函数名
```python
# 修复前
def _init_a_share_rules(self):
def _check_a_share_restrictions(self, symbol: str, signal_type: str) -> bool:
def _init_a_share_features(self):
def _adjust_position_for_a_share(self, position: float, features: Dict[str, float]) -> float:

# 修复后
def _init_ashare_rules(self):
def _check_ashare_restrictions(self, symbol: str, signal_type: str) -> bool:
def _init_ashare_features(self):
def _adjust_position_for_ashare(self, position: float, features: Dict[str, float]) -> float:
```

### 阶段2：向后兼容性处理（本周）

#### 2.1 添加别名方法
为了保持向后兼容性，在修复函数名的同时添加别名：

```python
# 在TechnicalProcessor中添加别名
calculate_ma = calc_ma  # 保持向后兼容
calculate_rsi = calc_rsi  # 保持向后兼容
calculate_macd = calc_macd  # 保持向后兼容
calculate_bollinger = calc_bollinger  # 保持向后兼容
calculate_obv = calc_obv  # 保持向后兼容
calculate_atr = calc_atr  # 保持向后兼容
calculate_indicators = calc_indicators  # 保持向后兼容
```

#### 2.2 更新文档和测试
- 更新所有相关文档
- 更新测试用例中的函数调用
- 添加弃用警告

### 阶段3：全面检查（下周）

#### 3.1 自动化检查
- 建立命名规范检查脚本
- 集成到CI/CD流程
- 定期运行检查

#### 3.2 代码风格统一
- 统一缩进风格
- 统一注释风格
- 统一导入顺序

## 实施步骤

### 步骤1：修复TechnicalProcessor函数名
1. 重命名函数
2. 添加别名方法
3. 更新内部调用
4. 更新测试用例

### 步骤2：修复SignalGenerator函数名
1. 重命名函数
2. 更新内部调用
3. 更新相关文档

### 步骤3：验证修复
1. 运行所有测试
2. 检查导入是否正常
3. 验证向后兼容性

## 风险评估

### 低风险
- **函数重命名**: 影响范围可控，有别名保护
- **文档更新**: 风险低，可并行进行

### 中风险
- **测试用例更新**: 需要全面测试
- **向后兼容性**: 需要仔细验证

### 高风险
- **无**: 所有修改都有保护机制

## 结论

特征层命名规范整体良好，主要问题集中在函数命名上。通过系统性的修复工作，可以显著提升代码的一致性和可读性。

**关键建议**:
1. 优先修复函数命名问题
2. 保持向后兼容性
3. 建立自动化检查机制
4. 统一代码风格

---

**报告生成时间**: 2025-01-27  
**报告维护**: 开发团队  
**版本**: 1.0 