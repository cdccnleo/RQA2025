#!/usr/bin/env python3
"""
RQA2025 自动化部署脚本
部署CI/CD流水线和相关工具到生产环境
"""

import subprocess
import sys
import os
from pathlib import Path


class AutomationDeployer:
    """自动化部署器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.scripts_dir = self.project_root / "scripts" / "testing"

    def install_git_hooks(self) -> bool:
        """安装Git钩子"""
        print("🔧 安装Git钩子...")

        hooks_dir = self.project_root / ".git" / "hooks"
        if not hooks_dir.exists():
            print("❌ .git/hooks 目录不存在")
            return False

        # 创建预提交钩子
        pre_commit_hook = hooks_dir / "pre-commit"
        hook_content = f"""#!/bin/bash
# RQA2025 预提交钩子
python "{self.scripts_dir / 'pre_commit_hook.py'}"
"""

        try:
            with open(pre_commit_hook, 'w') as f:
                f.write(hook_content)

            # 设置执行权限
            os.chmod(pre_commit_hook, 0o755)

            print("✅ Git预提交钩子安装成功")
            return True

        except Exception as e:
            print(f"❌ 安装Git钩子失败: {e}")
            return False

    def setup_ci_cd(self) -> bool:
        """设置CI/CD环境"""
        print("🔧 设置CI/CD环境...")

        # 检查GitHub Actions目录
        workflows_dir = self.project_root / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)

        # 检查工作流文件是否存在
        workflow_file = workflows_dir / "test_coverage.yml"
        if not workflow_file.exists():
            print("❌ GitHub Actions工作流文件不存在")
            return False

        print("✅ CI/CD环境设置完成")
        return True

    def install_dependencies(self) -> bool:
        """安装依赖包"""
        print("🔧 安装依赖包...")

        try:
            # 安装pytest相关包
            subprocess.run([
                "pip", "install", "pytest", "pytest-cov", "pytest-mock"
            ], check=True)

            print("✅ 依赖包安装成功")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ 依赖包安装失败: {e}")
            return False

    def create_directories(self) -> bool:
        """创建必要的目录"""
        print("🔧 创建目录结构...")

        directories = [
            self.project_root / "reports" / "testing",
            self.project_root / "reports" / "testing" / "dashboard",
            self.project_root / ".github" / "workflows"
        ]

        try:
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)

            print("✅ 目录结构创建完成")
            return True

        except Exception as e:
            print(f"❌ 创建目录失败: {e}")
            return False

    def test_automation_tools(self) -> bool:
        """测试自动化工具"""
        print("🔧 测试自动化工具...")

        tools = [
            ("自动化流水线", "automated_coverage_pipeline.py"),
            ("覆盖率检查工具", "check_coverage_threshold.py"),
            ("预提交钩子", "pre_commit_hook.py"),
            ("仪表板生成器", "generate_coverage_dashboard.py")
        ]

        all_passed = True

        for tool_name, script_name in tools:
            script_path = self.scripts_dir / script_name

            if not script_path.exists():
                print(f"❌ {tool_name}不存在: {script_path}")
                all_passed = False
                continue

            # 测试脚本语法
            try:
                result = subprocess.run([
                    "python", "-m", "py_compile", str(script_path)
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    print(f"✅ {tool_name} 语法检查通过")
                else:
                    print(f"❌ {tool_name} 语法错误:")
                    print(result.stderr)
                    all_passed = False

            except Exception as e:
                print(f"❌ {tool_name} 测试失败: {e}")
                all_passed = False

        return all_passed

    def generate_deployment_report(self) -> str:
        """生成部署报告"""
        report = f"""# RQA2025 自动化部署报告

## 📊 部署摘要

**部署时间**: {self._get_current_time()}
**部署状态**: ✅ 完成
**部署项目**: 自动化测试流水线

## 🔧 部署组件

### 1. Git钩子
- **预提交钩子**: ✅ 已安装
- **位置**: .git/hooks/pre-commit
- **功能**: 代码提交前自动检查覆盖率

### 2. CI/CD流水线
- **GitHub Actions**: ✅ 已配置
- **工作流文件**: .github/workflows/test_coverage.yml
- **触发条件**: push, pull_request, schedule

### 3. 自动化工具
- **自动化流水线**: ✅ 已部署
- **覆盖率检查工具**: ✅ 已部署
- **仪表板生成器**: ✅ 已部署
- **预提交钩子**: ✅ 已部署

### 4. 依赖包
- **pytest**: ✅ 已安装
- **pytest-cov**: ✅ 已安装
- **pytest-mock**: ✅ 已安装

## 📁 目录结构

```
RQA2025/
├── .github/workflows/
│   └── test_coverage.yml
├── scripts/testing/
│   ├── automated_coverage_pipeline.py
│   ├── check_coverage_threshold.py
│   ├── pre_commit_hook.py
│   └── generate_coverage_dashboard.py
├── reports/testing/
│   └── dashboard/
└── .git/hooks/
    └── pre-commit
```

## 🚀 使用方法

### 1. 运行自动化流水线
```bash
python scripts/testing/automated_coverage_pipeline.py
```

### 2. 检查覆盖率阈值
```bash
python scripts/testing/check_coverage_threshold.py --min-coverage 75
```

### 3. 生成覆盖率仪表板
```bash
python scripts/testing/generate_coverage_dashboard.py
```

### 4. 手动运行预提交检查
```bash
python scripts/testing/pre_commit_hook.py
```

## 📋 下一步

1. **配置GitHub Secrets**: 设置必要的环境变量
2. **测试CI/CD流水线**: 推送代码触发自动化测试
3. **监控覆盖率**: 定期检查覆盖率报告
4. **团队培训**: 组织自动化工具使用培训

---
**报告生成时间**: {self._get_current_time()}
"""
        return report

    def _get_current_time(self) -> str:
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def deploy(self) -> bool:
        """执行完整部署"""
        print("🚀 开始部署RQA2025自动化测试流水线...")
        print("=" * 60)

        steps = [
            ("创建目录结构", self.create_directories),
            ("安装依赖包", self.install_dependencies),
            ("设置CI/CD环境", self.setup_ci_cd),
            ("安装Git钩子", self.install_git_hooks),
            ("测试自动化工具", self.test_automation_tools)
        ]

        all_passed = True

        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            if not step_func():
                all_passed = False
                print(f"❌ {step_name}失败")
            else:
                print(f"✅ {step_name}成功")

        # 生成部署报告
        if all_passed:
            report = self.generate_deployment_report()
            report_file = self.project_root / "reports" / "testing" / "deployment_report.md"

            try:
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"\n📄 部署报告已保存: {report_file}")
            except Exception as e:
                print(f"❌ 保存部署报告失败: {e}")

        print("\n" + "=" * 60)
        if all_passed:
            print("🎉 自动化测试流水线部署完成！")
            print("\n💡 下一步:")
            print("1. 推送代码到GitHub触发CI/CD")
            print("2. 运行自动化流水线测试")
            print("3. 查看覆盖率仪表板")
        else:
            print("❌ 部署过程中遇到问题，请检查错误信息")

        return all_passed


def main():
    """主函数"""
    deployer = AutomationDeployer()
    success = deployer.deploy()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
