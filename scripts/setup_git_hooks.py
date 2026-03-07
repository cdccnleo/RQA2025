#!/usr/bin/env python3
"""
设置Git钩子用于代码质量检查

包括重复代码检测、代码格式检查等
"""

import stat
from pathlib import Path


class GitHooksSetup:
    """Git钩子设置器"""

    def __init__(self):
        self.hooks_dir = Path('.git/hooks')
        self.scripts_dir = Path('scripts')

    def setup_hooks(self):
        """设置所有钩子"""
        print('🔧 开始设置Git钩子')
        print('=' * 40)

        if not self.hooks_dir.exists():
            print('❌ .git/hooks 目录不存在，请确保在Git仓库中运行')
            return False

        # 创建预提交钩子
        self._create_pre_commit_hook()

        # 创建提交消息钩子
        self._create_commit_msg_hook()

        # 创建推送前钩子
        self._create_pre_push_hook()

        print('\\n✅ Git钩子设置完成')
        print('📋 已安装的钩子:')
        print('   • pre-commit: 代码质量检查')
        print('   • commit-msg: 提交消息格式检查')
        print('   • pre-push: 推送前最终检查')

        return True

    def _create_pre_commit_hook(self):
        """创建预提交钩子"""
        hook_content = '''#!/bin/bash
"""
Git预提交钩子 - 代码质量检查
"""

echo "🔍 正在进行代码质量检查..."

# 检查是否有Python文件变更
CHANGED_PY_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep "\\.py$" || true)

if [ -n "$CHANGED_PY_FILES" ]; then
    echo "📁 检测到Python文件变更: $CHANGED_PY_FILES"

    # 运行代码重复检测
    echo "🔄 运行代码重复检测..."
    python scripts/automated_duplicate_detection.py --check
    DUPLICATE_EXIT_CODE=$?

    if [ $DUPLICATE_EXIT_CODE -ne 0 ]; then
        echo "❌ 代码重复检测失败！"
        echo "💡 请修复重复代码后再提交，或使用 --no-verify 跳过检查"
        exit 1
    fi

    # 运行代码格式检查
    echo "📏 运行代码格式检查..."
    python -m py_compile $CHANGED_PY_FILES 2>/dev/null
    COMPILE_EXIT_CODE=$?

    if [ $COMPILE_EXIT_CODE -ne 0 ]; then
        echo "❌ 代码语法检查失败！"
        echo "💡 请修复语法错误后再提交"
        python -m py_compile $CHANGED_PY_FILES
        exit 1
    fi

    echo "✅ 代码质量检查通过"
else
    echo "ℹ️ 没有Python文件变更，跳过质量检查"
fi

exit 0
'''

        hook_path = self.hooks_dir / 'pre-commit'
        self._write_hook_file(hook_path, hook_content)
        print(f'✅ 已创建预提交钩子: {hook_path}')

    def _create_commit_msg_hook(self):
        """创建提交消息钩子"""
        hook_content = '''#!/bin/bash
"""
Git提交消息钩子 - 消息格式检查
"""

COMMIT_MSG_FILE=$1

# 读取提交消息
COMMIT_MSG=$(cat $COMMIT_MSG_FILE)

# 检查提交消息格式
if [[ ! $COMMIT_MSG =~ ^(feat|fix|docs|style|refactor|test|chore)(\\(.+\\))?:\\ .+ ]]; then
    echo "❌ 提交消息格式不符合规范!"
    echo ""
    echo "📋 正确的格式示例:"
    echo "  feat: 添加新功能"
    echo "  fix: 修复bug"
    echo "  docs: 更新文档"
    echo "  style: 代码格式调整"
    echo "  refactor: 代码重构"
    echo "  test: 添加测试"
    echo "  chore: 其他修改"
    echo ""
    echo "💡 当前消息: '$COMMIT_MSG'"
    exit 1
fi

echo "✅ 提交消息格式检查通过"
exit 0
'''

        hook_path = self.hooks_dir / 'commit-msg'
        self._write_hook_file(hook_path, hook_content)
        print(f'✅ 已创建提交消息钩子: {hook_path}')

    def _create_pre_push_hook(self):
        """创建推送前钩子"""
        hook_content = '''#!/bin/bash
"""
Git推送前钩子 - 最终质量检查
"""

REMOTE=$1
URL=$2

echo "🚀 正在进行推送前最终检查..."

# 运行完整的代码质量检查
echo "🔍 运行完整代码质量检查..."
python scripts/automated_duplicate_detection.py --strict
QUALITY_EXIT_CODE=$?

if [ $QUALITY_EXIT_CODE -ne 0 ]; then
    echo "❌ 代码质量检查失败！"
    echo "💡 请修复所有问题后再推送"
    exit 1
fi

# 检查测试覆盖率
echo "🧪 检查测试状态..."
if [ -f "pytest.ini" ] || [ -f "setup.cfg" ] || [ -f "pyproject.toml" ]; then
    python -m pytest --collect-only -q > /dev/null 2>&1
    TEST_EXIT_CODE=$?

    if [ $TEST_EXIT_CODE -ne 0 ]; then
        echo "⚠️  测试检查失败，但允许推送"
        echo "💡 建议：确保所有测试都能正常运行"
    else
        echo "✅ 测试检查通过"
    fi
fi

echo "✅ 推送前检查完成"
exit 0
'''

        hook_path = self.hooks_dir / 'pre-push'
        self._write_hook_file(hook_path, hook_content)
        print(f'✅ 已创建推送前钩子: {hook_path}')

    def _write_hook_file(self, hook_path: Path, content: str):
        """写入钩子文件"""
        with open(hook_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # 设置执行权限 (在Windows上这可能不会完全工作，但钩子仍然可以运行)
        try:
            hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except:
            pass  # 在Windows上忽略权限设置


def main():
    """主函数"""
    setup = GitHooksSetup()

    if setup.setup_hooks():
        print('\\n🎉 Git钩子设置成功！')
        print('\\n📚 使用说明:')
        print('• 提交代码时会自动运行质量检查')
        print('• 提交消息必须遵循约定格式')
        print('• 推送前会进行最终质量验证')
        print('\\n💡 如需跳过检查，使用 --no-verify 参数')
        print('   git commit --no-verify -m "your message"')
    else:
        print('\\n❌ Git钩子设置失败')
        exit(1)


if __name__ == "__main__":
    main()
