#!/usr/bin/env python3
"""
准备Phase 1重构环境

功能:
1. 创建重构分支
2. 备份当前代码
3. 建立测试框架
4. 配置质量检查工具
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent


def print_step(step_num, total, message):
    """打印步骤信息"""
    print(f"\n{'='*70}")
    print(f"步骤 {step_num}/{total}: {message}")
    print('='*70)


def run_command(cmd, description):
    """运行命令"""
    print(f"  执行: {description}")
    print(f"  命令: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=project_root)
        if result.returncode == 0:
            print(f"  ✅ 成功")
            if result.stdout:
                print(f"  输出: {result.stdout.strip()[:200]}")
            return True
        else:
            print(f"  ❌ 失败")
            if result.stderr:
                print(f"  错误: {result.stderr.strip()[:200]}")
            return False
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "="*70)
    print(" "*15 + "🚀 Phase 1重构环境准备")
    print("="*70)
    
    total_steps = 7
    current_step = 0
    
    # 步骤1: 检查Git状态
    current_step += 1
    print_step(current_step, total_steps, "检查Git状态")
    
    if not (project_root / '.git').exists():
        print("  ⚠️ 未检测到Git仓库，跳过Git相关操作")
        skip_git = True
    else:
        skip_git = False
        run_command("git status", "检查当前Git状态")
    
    # 步骤2: 创建备份
    current_step += 1
    print_step(current_step, total_steps, "备份当前代码")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = project_root / 'backups' / f'core_before_phase1_{timestamp}'
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        src_core = project_root / 'src' / 'core'
        if src_core.exists():
            shutil.copytree(src_core, backup_dir / 'core', dirs_exist_ok=True)
            print(f"  ✅ 代码已备份到: {backup_dir}")
        else:
            print(f"  ⚠️ 源目录不存在: {src_core}")
    except Exception as e:
        print(f"  ❌ 备份失败: {e}")
    
    # 步骤3: 创建重构分支
    current_step += 1
    print_step(current_step, total_steps, "创建重构分支")
    
    if not skip_git:
        branch_name = f"refactor/core-layer-phase1-{timestamp}"
        if run_command(f"git checkout -b {branch_name}", "创建新分支"):
            print(f"  ✅ 已切换到分支: {branch_name}")
        else:
            print("  ⚠️ 分支创建失败，可能已存在或有未提交更改")
    
    # 步骤4: 创建测试目录结构
    current_step += 1
    print_step(current_step, total_steps, "创建测试目录结构")
    
    test_dirs = [
        'tests/unit/core/business/optimizer',
        'tests/unit/core/business/orchestrator',
        'tests/unit/core/event_bus',
        'tests/unit/core/infrastructure/security',
        'tests/integration/core',
        'tests/performance/core'
    ]
    
    for test_dir in test_dirs:
        dir_path = project_root / test_dir
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # 创建__init__.py
        init_file = dir_path / '__init__.py'
        if not init_file.exists():
            init_file.write_text('"""测试模块"""\n', encoding='utf-8')
        
        print(f"  ✅ 创建目录: {test_dir}")
    
    # 步骤5: 创建重构工具脚本目录
    current_step += 1
    print_step(current_step, total_steps, "准备重构工具")
    
    tools_dir = project_root / 'scripts' / 'refactoring_tools'
    tools_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建工具脚本模板
    tools = {
        '__init__.py': '"""重构工具集"""\n',
        'class_analyzer.py': '''"""大类分析工具"""

def analyze_class_responsibilities(file_path: str, class_name: str):
    """分析类的职责分布"""
    # TODO: 实现职责分析逻辑
    pass
''',
        'component_generator.py': '''"""组件生成工具"""

def generate_component_template(component_name: str, methods: list):
    """生成组件模板"""
    # TODO: 实现模板生成逻辑
    pass
''',
        'test_generator.py': '''"""测试生成工具"""

def generate_unit_tests(component_path: str):
    """生成单元测试"""
    # TODO: 实现测试生成逻辑
    pass
'''
    }
    
    for tool_name, tool_content in tools.items():
        tool_file = tools_dir / tool_name
        if not tool_file.exists():
            tool_file.write_text(tool_content, encoding='utf-8')
            print(f"  ✅ 创建工具: {tool_name}")
    
    # 步骤6: 创建配置文件
    current_step += 1
    print_step(current_step, total_steps, "创建配置文件")
    
    config_dir = project_root / 'config' / 'refactoring'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Phase 1配置
    phase1_config = {
        'phase': 'Phase 1',
        'start_date': datetime.now().strftime('%Y-%m-%d'),
        'target_quality_score': 0.820,
        'tasks': [
            {'task_id': 'TASK-1', 'class': 'IntelligentBusinessProcessOptimizer', 'components': 5},
            {'task_id': 'TASK-2', 'class': 'BusinessProcessOrchestrator', 'components': 5},
            {'task_id': 'TASK-3', 'class': 'EventBus', 'components': 5},
            {'task_id': 'TASK-4', 'class': 'AccessControlManager', 'components': 4},
            {'task_id': 'TASK-5', 'class': 'DataEncryptionManager', 'components': 4},
            {'task_id': 'TASK-6', 'class': 'AuditLoggingManager', 'components': 4}
        ],
        'quality_gates': {
            'test_coverage': 0.80,
            'max_class_lines': 250,
            'max_function_lines': 30,
            'max_complexity': 10
        }
    }
    
    import json
    config_file = config_dir / 'phase1_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(phase1_config, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 创建配置: phase1_config.json")
    
    # 步骤7: 生成README
    current_step += 1
    print_step(current_step, total_steps, "生成环境说明文档")
    
    readme_content = f"""# Phase 1重构环境说明

## 环境准备完成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 备份信息
- 备份目录: {backup_dir}
- 备份时间: {timestamp}
- 备份内容: src/core完整代码

## Git分支
- 分支名称: refactor/core-layer-phase1-{timestamp}
- 基于: main/master

## 测试框架
已创建以下测试目录:
{chr(10).join(f'- {d}' for d in test_dirs)}

## 重构工具
工具目录: scripts/refactoring_tools/
包含: 类分析器、组件生成器、测试生成器等

## 配置文件
配置文件: config/refactoring/phase1_config.json

## 质量门禁
- 测试覆盖率: ≥ 80%
- 最大类行数: ≤ 250行
- 最大函数行数: ≤ 30行
- 最大复杂度: ≤ 10

## 下一步
1. 阅读执行计划: docs/refactoring/core_layer_phase1_execution_plan.md
2. 开始Week 1准备工作
3. 召开Kick-off会议

## 相关文档
- 执行计划: docs/refactoring/core_layer_phase1_execution_plan.md
- 审查报告: docs/code_review/core_layer_ai_review_report.md
- 架构设计: docs/architecture/core_service_layer_architecture_design.md
"""
    
    readme_file = project_root / 'PHASE1_ENVIRONMENT_README.md'
    readme_file.write_text(readme_content, encoding='utf-8')
    print(f"  ✅ 创建说明: PHASE1_ENVIRONMENT_README.md")
    
    # 完成总结
    print("\n" + "="*70)
    print(" "*15 + "✅ Phase 1环境准备完成！")
    print("="*70)
    
    print("\n📁 创建的目录和文件:")
    print(f"  • 备份目录: {backup_dir}")
    print(f"  • 测试目录: {len(test_dirs)}个")
    print(f"  • 工具脚本: {len(tools)}个")
    print(f"  • 配置文件: 1个")
    print(f"  • 说明文档: 1个")
    
    print("\n🚀 下一步:")
    print("  1. 阅读 PHASE1_ENVIRONMENT_README.md")
    print("  2. 阅读 docs/refactoring/core_layer_phase1_execution_plan.md")
    print("  3. 召开Kick-off会议")
    print("  4. 开始Week 1准备工作")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()

