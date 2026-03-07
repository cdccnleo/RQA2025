#!/usr/bin/env python3
"""
RQA2025 团队培训计划脚本
提供自动化工具使用培训、预提交钩子使用说明、覆盖率仪表板查看方法
"""

import sys
from pathlib import Path
from typing import Dict


class TeamTrainingPlan:
    """团队培训计划"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.training_materials = {
            "自动化工具使用培训": self._generate_automation_training(),
            "预提交钩子使用说明": self._generate_pre_commit_guide(),
            "覆盖率仪表板查看方法": self._generate_dashboard_guide(),
            "CI/CD流水线使用指南": self._generate_cicd_guide(),
            "质量门禁使用说明": self._generate_quality_gate_guide()
        }

    def _generate_automation_training(self) -> str:
        """生成自动化工具使用培训内容"""
        return """
# 🚀 自动化工具使用培训

## 📋 培训目标
- 掌握自动化测试工具的使用方法
- 了解覆盖率监控和报告系统
- 学会使用CI/CD流水线
- 掌握质量门禁的使用

## 🛠️ 核心工具介绍

### 1. 自动化覆盖率流水线
**文件**: `scripts/testing/automated_coverage_pipeline.py`
**功能**: 自动运行测试并生成覆盖率报告
**使用方法**:
```bash
# 运行完整流水线
python scripts/testing/automated_coverage_pipeline.py

# 运行特定模块测试
python scripts/testing/run_tests.py --module infrastructure --env test
```

### 2. 覆盖率阈值检查器
**文件**: `scripts/testing/check_coverage_threshold.py`
**功能**: 检查覆盖率是否达到阈值要求
**使用方法**:
```bash
python scripts/testing/check_coverage_threshold.py
```

### 3. 覆盖率仪表板生成器
**文件**: `scripts/testing/generate_coverage_dashboard.py`
**功能**: 生成可视化的覆盖率仪表板
**使用方法**:
```bash
python scripts/testing/generate_coverage_dashboard.py
```

## 📊 覆盖率报告解读

### 报告文件位置
- **Markdown报告**: `reports/testing/coverage_report_*.md`
- **JSON数据**: `reports/testing/coverage_results_*.json`
- **HTML仪表板**: `reports/testing/dashboard/coverage_dashboard.html`

### 关键指标说明
- **平均覆盖率**: 所有模块的覆盖率平均值
- **模块覆盖率**: 每个模块的独立覆盖率
- **未覆盖代码**: 尚未被测试覆盖的代码行数
- **覆盖率趋势**: 历史覆盖率变化趋势

## 🎯 最佳实践

### 1. 测试编写规范
- 每个函数至少有一个测试用例
- 测试用例要覆盖正常流程和异常情况
- 使用描述性的测试名称
- 保持测试的独立性和可重复性

### 2. 覆盖率提升策略
- 优先测试核心业务逻辑
- 关注边界条件和异常处理
- 定期运行覆盖率检查
- 设置合理的覆盖率目标

### 3. 自动化流程使用
- 提交代码前运行预提交钩子
- 定期查看覆盖率仪表板
- 关注CI/CD流水线状态
- 及时修复失败的测试

## 📚 学习资源
- 项目文档: `docs/testing/`
- 测试框架文档: pytest官方文档
- 覆盖率工具: pytest-cov文档
- CI/CD文档: GitHub Actions文档
"""

    def _generate_pre_commit_guide(self) -> str:
        """生成预提交钩子使用说明"""
        return """
# 🔍 预提交钩子使用说明

## 📋 概述
预提交钩子是在代码提交前自动运行的检查工具，确保代码质量和测试覆盖率。

## 🛠️ 安装和配置

### 1. 自动安装
```bash
python scripts/testing/deploy_automation.py
```

### 2. 手动安装
```bash
# 复制钩子脚本
cp scripts/testing/pre_commit_hook.py .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## 🔄 工作流程

### 1. 正常提交流程
```bash
git add .
git commit -m "feat: 新功能"
# 预提交钩子自动运行检查
```

### 2. 检查内容
- **代码质量检查**: 语法错误检查
- **变更模块识别**: 自动识别修改的模块
- **测试覆盖率检查**: 运行相关模块的测试
- **覆盖率阈值验证**: 检查是否达到覆盖率要求

### 3. 检查结果
- ✅ **通过**: 所有检查通过，可以提交
- ❌ **失败**: 发现问题，需要修复后重新提交

## 📊 覆盖率阈值

### 模块特定阈值
- **infrastructure**: 80%
- **data**: 80%
- **features**: 80%
- **models**: 80%
- **ensemble**: 80%
- **trading**: 80%
- **backtest**: 80%
- **其他模块**: 75%

### 检查逻辑
- 只检查变更的模块
- 如果模块覆盖率未达标，提交会被阻止
- 提供详细的错误信息和改进建议

## 🚨 常见问题解决

### 1. 编码问题
**问题**: UnicodeDecodeError
**解决**: 确保文件使用UTF-8编码

### 2. 测试超时
**问题**: 测试运行时间过长
**解决**: 优化测试用例，减少不必要的等待

### 3. 覆盖率不足
**问题**: 覆盖率低于阈值
**解决**: 添加更多测试用例

### 4. 跳过检查
**紧急情况**: 使用 `--no-verify` 参数
```bash
git commit --no-verify -m "紧急修复"
```

## 💡 使用技巧

### 1. 增量测试
- 只运行变更模块的测试
- 提高检查效率
- 减少等待时间

### 2. 快速反馈
- 提交前即时检查
- 详细的错误信息
- 明确的改进建议

### 3. 团队协作
- 统一的代码质量标准
- 自动化的质量检查
- 减少人工审查工作量

## 📈 效果监控

### 1. 质量指标
- 代码提交成功率
- 覆盖率提升趋势
- 测试失败率

### 2. 效率指标
- 检查运行时间
- 问题修复时间
- 团队接受度

## 🔧 自定义配置

### 1. 修改阈值
编辑 `scripts/testing/pre_commit_hook.py` 中的阈值设置

### 2. 添加检查规则
在 `check_code_quality()` 方法中添加新的检查逻辑

### 3. 调整超时时间
修改测试运行的超时设置
"""

    def _generate_dashboard_guide(self) -> str:
        """生成覆盖率仪表板查看方法"""
        return """
# 📊 覆盖率仪表板查看方法

## 📋 概述
覆盖率仪表板提供了项目测试覆盖率的可视化展示，帮助团队了解测试状态和趋势。

## 🛠️ 访问方式

### 1. 本地查看
```bash
# 生成仪表板
python scripts/testing/generate_coverage_dashboard.py

# 打开仪表板
start reports/testing/dashboard/coverage_dashboard.html
```

### 2. GitHub Pages查看
- 访问: https://[username].github.io/RQA2025/
- 自动更新: 每次推送代码后自动更新

## 📊 仪表板功能

### 1. 总体概览
- **平均覆盖率**: 所有模块的覆盖率平均值
- **模块数量**: 参与统计的模块总数
- **趋势图表**: 历史覆盖率变化趋势
- **状态指示**: 成功/失败状态显示

### 2. 模块详情
- **模块列表**: 所有模块的覆盖率列表
- **覆盖率排序**: 按覆盖率高低排序
- **详细统计**: 每个模块的详细覆盖率信息
- **改进建议**: 针对低覆盖率模块的建议

### 3. 历史趋势
- **时间轴**: 覆盖率变化的时间轴
- **趋势线**: 覆盖率变化趋势线
- **关键节点**: 重要的覆盖率变化点
- **预测分析**: 基于历史数据的趋势预测

## 📈 数据解读

### 1. 覆盖率指标
- **行覆盖率**: 代码行被测试覆盖的比例
- **分支覆盖率**: 代码分支被测试覆盖的比例
- **函数覆盖率**: 函数被测试调用的比例
- **类覆盖率**: 类被测试实例化的比例

### 2. 质量评估
- **优秀**: 覆盖率 ≥ 90%
- **良好**: 覆盖率 80-89%
- **一般**: 覆盖率 70-79%
- **需要改进**: 覆盖率 < 70%

### 3. 趋势分析
- **上升趋势**: 覆盖率持续提升
- **稳定状态**: 覆盖率保持稳定
- **下降趋势**: 覆盖率出现下降
- **波动状态**: 覆盖率出现波动

## 🎯 使用策略

### 1. 定期查看
- **每日**: 查看最新的覆盖率状态
- **每周**: 分析覆盖率趋势
- **每月**: 评估测试策略效果

### 2. 重点关注
- **低覆盖率模块**: 优先提升覆盖率
- **核心业务模块**: 确保高覆盖率
- **新增代码**: 及时添加测试用例

### 3. 团队协作
- **分享报告**: 定期分享覆盖率报告
- **讨论改进**: 讨论覆盖率提升策略
- **设定目标**: 设定合理的覆盖率目标

## 🔧 自定义配置

### 1. 修改阈值
编辑仪表板生成脚本中的阈值设置

### 2. 添加模块
在模块列表中添加新的模块

### 3. 调整样式
修改HTML模板中的样式设置

## 📱 移动端查看

### 1. 响应式设计
- 适配不同屏幕尺寸
- 触摸友好的交互
- 快速加载的页面

### 2. 关键信息
- 总体覆盖率状态
- 重要模块的覆盖率
- 趋势变化提醒

## 🔄 自动更新

### 1. CI/CD集成
- 每次代码推送后自动更新
- 定时生成最新报告
- 自动部署到GitHub Pages

### 2. 通知机制
- 覆盖率下降时发送通知
- 重要变化时提醒团队
- 定期发送覆盖率报告

## 📚 最佳实践

### 1. 定期维护
- 清理过期的历史数据
- 更新仪表板样式
- 优化加载性能

### 2. 团队培训
- 培训团队成员使用仪表板
- 解释各项指标的含义
- 指导如何解读数据

### 3. 持续改进
- 根据使用反馈优化功能
- 添加新的可视化图表
- 改进用户体验
"""

    def _generate_cicd_guide(self) -> str:
        """生成CI/CD流水线使用指南"""
        return """
# 🔄 CI/CD流水线使用指南

## 📋 概述
CI/CD流水线自动化了代码的构建、测试和部署过程，确保代码质量和项目稳定性。

## 🛠️ 流水线配置

### 1. GitHub Actions工作流
**文件**: `.github/workflows/test_coverage.yml`
**功能**: 自动化测试和覆盖率检查

### 2. 触发条件
- **Push**: 推送到main或develop分支
- **Pull Request**: 创建或更新PR
- **定时**: 每天凌晨2点自动运行

### 3. 执行环境
- **Python版本**: 3.9, 3.10
- **操作系统**: Ubuntu Latest
- **缓存**: 依赖包缓存

## 🔄 工作流程

### 1. 代码检出
```yaml
- name: Checkout code
  uses: actions/checkout@v3
```

### 2. Python环境设置
```yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: ${{ matrix.python-version }}
```

### 3. 依赖安装
```yaml
- name: Install dependencies
  run: |
    pip install pytest pytest-cov pytest-mock
```

### 4. 测试执行
```yaml
- name: Run tests
  run: |
    python scripts/testing/automated_coverage_pipeline.py
```

### 5. 覆盖率检查
```yaml
- name: Check coverage threshold
  run: |
    python scripts/testing/check_coverage_threshold.py
```

### 6. 报告上传
```yaml
- name: Upload coverage reports
  uses: actions/upload-artifact@v3
```

## 📊 流水线状态

### 1. 成功状态
- ✅ 所有测试通过
- ✅ 覆盖率达标
- ✅ 代码质量检查通过

### 2. 失败状态
- ❌ 测试失败
- ❌ 覆盖率未达标
- ❌ 代码质量检查失败

### 3. 部分成功
- ⚠️ 部分测试通过
- ⚠️ 覆盖率接近阈值

## 🔍 状态查看

### 1. GitHub界面
- 访问仓库的Actions标签页
- 查看最新的工作流运行状态
- 查看详细的执行日志

### 2. 通知机制
- 失败时发送邮件通知
- PR评论中显示覆盖率结果
- 状态徽章显示在README中

### 3. 报告下载
- 从Actions页面下载测试报告
- 查看覆盖率详细数据
- 分析测试失败原因

## 🚨 问题处理

### 1. 测试失败
**原因**: 代码变更导致测试失败
**解决**: 
- 检查测试用例是否需要更新
- 修复代码中的问题
- 重新运行测试

### 2. 覆盖率下降
**原因**: 新增代码没有测试覆盖
**解决**:
- 为新功能添加测试用例
- 提高现有代码的覆盖率
- 检查测试策略

### 3. 环境问题
**原因**: 依赖包版本冲突或环境配置问题
**解决**:
- 检查依赖包版本
- 更新环境配置
- 清理缓存重新运行

## 📈 性能优化

### 1. 并行执行
- 多个Python版本并行测试
- 不同模块并行运行
- 利用GitHub Actions的并行能力

### 2. 缓存优化
- 缓存依赖包
- 缓存测试结果
- 减少重复下载

### 3. 超时控制
- 设置合理的超时时间
- 避免长时间运行的测试
- 及时终止失败的测试

## 🔧 自定义配置

### 1. 修改触发条件
编辑工作流文件中的触发条件

### 2. 添加新的检查
在工作流中添加新的检查步骤

### 3. 调整超时设置
修改测试和检查的超时时间

## 📚 最佳实践

### 1. 快速反馈
- 保持测试运行时间短
- 提供清晰的错误信息
- 及时通知问题

### 2. 稳定性
- 避免不稳定的测试
- 使用可靠的依赖包
- 定期维护流水线

### 3. 团队协作
- 培训团队成员使用CI/CD
- 建立问题处理流程
- 定期回顾和改进

## 🎯 监控指标

### 1. 执行时间
- 平均执行时间
- 最长执行时间
- 执行时间趋势

### 2. 成功率
- 成功率统计
- 失败原因分析
- 改进措施跟踪

### 3. 覆盖率趋势
- 覆盖率变化趋势
- 模块覆盖率分布
- 覆盖率提升效果
"""

    def _generate_quality_gate_guide(self) -> str:
        """生成质量门禁使用说明"""
        return """
# 🚪 质量门禁使用说明

## 📋 概述
质量门禁是确保代码质量的重要机制，在代码进入生产环境前进行自动检查。

## 🛠️ 门禁类型

### 1. 覆盖率门禁
**目标**: 确保测试覆盖率达标
**阈值**: 模块特定阈值（75%-80%）
**检查**: 自动检查覆盖率是否达标

### 2. 代码质量门禁
**目标**: 确保代码语法正确
**检查**: 语法错误检查
**工具**: py_compile

### 3. 测试稳定性门禁
**目标**: 确保测试稳定可靠
**检查**: 测试是否通过
**工具**: pytest

### 4. 预提交门禁
**目标**: 提交前质量检查
**触发**: git commit
**检查**: 综合质量检查

## 🔄 门禁流程

### 1. 预提交检查
```bash
git commit -m "feat: 新功能"
# 自动触发预提交钩子
# 检查代码质量和覆盖率
# 通过则提交，失败则阻止
```

### 2. CI/CD检查
```yaml
# GitHub Actions工作流
- name: Check coverage threshold
  run: python scripts/testing/check_coverage_threshold.py
```

### 3. 部署前检查
```bash
# 部署前运行检查
python scripts/testing/check_coverage_threshold.py
```

## 📊 门禁配置

### 1. 覆盖率阈值
```python
# 模块特定阈值
thresholds = {
    "infrastructure": 80.0,
    "data": 80.0,
    "features": 80.0,
    "models": 80.0,
    "ensemble": 80.0,
    "trading": 80.0,
    "backtest": 80.0
}
```

### 2. 质量检查规则
- 语法错误检查
- 代码风格检查
- 安全漏洞检查
- 性能问题检查

### 3. 测试要求
- 单元测试覆盖率
- 集成测试覆盖率
- 端到端测试覆盖率

## 🚨 门禁失败处理

### 1. 覆盖率不足
**问题**: 覆盖率低于阈值
**解决**:
- 添加更多测试用例
- 优化现有测试
- 调整覆盖率策略

### 2. 代码质量问题
**问题**: 代码存在语法错误或质量问题
**解决**:
- 修复语法错误
- 改进代码质量
- 使用代码格式化工具

### 3. 测试失败
**问题**: 测试用例失败
**解决**:
- 修复失败的测试
- 更新测试用例
- 检查代码变更

## 📈 门禁效果

### 1. 质量提升
- 代码质量显著提升
- 测试覆盖率稳步增长
- 缺陷率明显下降

### 2. 开发效率
- 早期发现问题
- 减少后期修复成本
- 提高开发信心

### 3. 团队协作
- 统一的质量标准
- 自动化的质量检查
- 减少人工审查

## 🔧 自定义配置

### 1. 修改阈值
编辑配置文件中的阈值设置

### 2. 添加检查规则
在门禁脚本中添加新的检查逻辑

### 3. 调整检查策略
根据项目需求调整检查策略

## 📚 最佳实践

### 1. 渐进式提升
- 从较低的阈值开始
- 逐步提高要求
- 给团队适应时间

### 2. 持续监控
- 定期检查门禁效果
- 分析失败原因
- 优化门禁策略

### 3. 团队培训
- 培训团队成员
- 解释门禁的重要性
- 提供使用指导

## 🎯 成功指标

### 1. 质量指标
- 代码质量评分
- 测试覆盖率
- 缺陷密度

### 2. 效率指标
- 门禁通过率
- 问题修复时间
- 开发周期

### 3. 团队指标
- 团队接受度
- 使用频率
- 反馈评分
"""

    def generate_training_materials(self) -> Dict[str, str]:
        """生成所有培训材料"""
        return self.training_materials

    def save_training_materials(self):
        """保存培训材料到文件"""
        training_dir = self.project_root / "docs" / "training"
        training_dir.mkdir(parents=True, exist_ok=True)

        for title, content in self.training_materials.items():
            filename = title.replace(" ", "_").replace("/", "_") + ".md"
            filepath = training_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✅ 已生成培训材料: {filepath}")

    def generate_training_summary(self) -> str:
        """生成培训计划总结"""
        return f"""
# 📚 RQA2025 团队培训计划总结

## 🎯 培训目标
通过系统化的培训，使团队成员熟练掌握自动化测试工具，建立质量意识，提升项目整体质量。

## 📋 培训内容

### 1. 自动化工具使用培训
- **目标**: 掌握自动化测试工具的使用方法
- **内容**: 覆盖率流水线、阈值检查器、仪表板生成器
- **时长**: 2小时
- **方式**: 理论讲解 + 实操演练

### 2. 预提交钩子使用说明
- **目标**: 学会使用预提交钩子进行代码质量检查
- **内容**: 安装配置、工作流程、问题处理
- **时长**: 1小时
- **方式**: 演示 + 练习

### 3. 覆盖率仪表板查看方法
- **目标**: 学会查看和解读覆盖率仪表板
- **内容**: 仪表板功能、数据解读、使用策略
- **时长**: 1小时
- **方式**: 在线演示 + 互动

### 4. CI/CD流水线使用指南
- **目标**: 了解CI/CD流水线的工作原理和使用方法
- **内容**: 流水线配置、状态查看、问题处理
- **时长**: 1.5小时
- **方式**: 理论 + 实践

### 5. 质量门禁使用说明
- **目标**: 理解质量门禁的重要性和使用方法
- **内容**: 门禁类型、配置方法、效果监控
- **时长**: 1小时
- **方式**: 案例分析 + 讨论

## 📅 培训安排

### 第一周：基础培训
- **Day 1**: 自动化工具使用培训
- **Day 2**: 预提交钩子使用说明
- **Day 3**: 覆盖率仪表板查看方法

### 第二周：进阶培训
- **Day 1**: CI/CD流水线使用指南
- **Day 2**: 质量门禁使用说明
- **Day 3**: 综合练习和答疑

### 第三周：实战演练
- **Day 1-3**: 实际项目中的工具使用
- **Day 4-5**: 问题解决和优化

## 📊 培训效果评估

### 1. 知识掌握
- 工具使用熟练度测试
- 覆盖率报告解读能力
- 问题诊断和解决能力

### 2. 实践应用
- 实际项目中的应用情况
- 工具使用频率统计
- 质量改进效果

### 3. 团队反馈
- 培训满意度调查
- 工具易用性评价
- 改进建议收集

## 🎯 预期成果

### 1. 短期目标（1个月内）
- 所有团队成员熟练掌握工具使用
- 自动化流程100%运行
- 覆盖率监控机制建立

### 2. 中期目标（3个月内）
- 团队自动化文化建立
- 质量意识显著提升
- 项目质量指标改善

### 3. 长期目标（6个月内）
- 自动化测试成为开发习惯
- 质量门禁100%生效
- 项目达到生产标准

## 📚 培训材料位置
所有培训材料已保存到: `docs/training/`

## 🔄 持续改进
- 根据培训反馈优化内容
- 定期更新培训材料
- 跟踪培训效果并调整策略
"""

    def run_training_plan(self):
        """运行培训计划"""
        print("🚀 开始生成团队培训计划...")
        print("=" * 60)

        # 保存培训材料
        self.save_training_materials()

        # 生成培训总结
        summary = self.generate_training_summary()
        summary_file = self.project_root / "docs" / "training" / "training_plan_summary.md"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)

        print(f"✅ 已生成培训计划总结: {summary_file}")
        print("\n" + "=" * 60)
        print("📚 培训计划生成完成！")
        print("=" * 60)
        print("\n📋 培训材料位置:")
        print("- 自动化工具使用培训: docs/training/自动化工具使用培训.md")
        print("- 预提交钩子使用说明: docs/training/预提交钩子使用说明.md")
        print("- 覆盖率仪表板查看方法: docs/training/覆盖率仪表板查看方法.md")
        print("- CI/CD流水线使用指南: docs/training/CI_CD流水线使用指南.md")
        print("- 质量门禁使用说明: docs/training/质量门禁使用说明.md")
        print("- 培训计划总结: docs/training/training_plan_summary.md")

        return True


def main():
    """主函数"""
    trainer = TeamTrainingPlan()
    success = trainer.run_training_plan()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
