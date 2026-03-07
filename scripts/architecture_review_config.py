#!/usr/bin/env python3
"""
架构审查配置

定义审查规则和配置，支持灵活的审查策略
"""

from typing import Dict, Any
from pathlib import Path


class ReviewConfig:
    """审查配置类"""

    def __init__(self):
        self.config = self._load_default_config()

    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            "review_rules": {
                "structure_rules": {
                    "max_file_size": 800,  # 调整为800行，适应业务代码
                    "max_class_size": 300,
                    "max_function_size": 80,
                    "max_module_depth": 4,
                    "ignore_files": [
                        "__init__.py",
                        "setup.py",
                        "conftest.py",
                        "*test*.py",
                        "*_test.py"
                    ]
                },
                "dependency_rules": {
                    "allow_circular_deps": False,
                    "max_imports_per_file": 20,
                    "forbidden_imports": ["import *"],
                    "ignore_patterns": [
                        "typing",
                        "dataclasses",
                        "enum"
                    ]
                },
                "design_rules": {
                    "require_interfaces": False,  # 放宽要求
                    "max_inheritance_depth": 4,
                    "require_docstrings": False,  # 放宽要求，允许无文档的项目
                    "naming_conventions": {
                        "classes": "PascalCase",
                        "functions": "snake_case",
                        "variables": "snake_case",
                        "constants": "UPPER_CASE"
                    },
                    "ignore_undocumented": [
                        "test_*",
                        "__init__",
                        "main"
                    ]
                },
                "architecture_rules": {
                    "layer_dependencies": {
                        "infrastructure": ["core"],
                        "data": ["core", "infrastructure"],
                        "features": ["core", "infrastructure", "data"],
                        "ml": ["core", "infrastructure", "features"],
                        "gateway": ["core", "infrastructure"],
                        "backtest": ["core", "infrastructure", "features", "ml"],
                        "trading": ["core", "infrastructure", "features", "ml", "backtest"],
                        "risk": ["core", "infrastructure", "features", "ml"],
                        "engine": ["core", "infrastructure", "features", "ml", "trading", "risk"]
                    },
                    "allowed_patterns": [
                        "Factory",
                        "Builder",
                        "Strategy",
                        "Observer",
                        "Adapter",
                        "Decorator"
                    ]
                },
                "security_rules": {
                    "check_sql_injection": True,
                    "check_hardcoded_credentials": True,
                    "check_path_traversal": True,
                    "suspicious_patterns": [
                        "eval(",
                        "exec(",
                        "pickle.loads(",
                        "yaml.load("
                    ]
                },
                "performance_rules": {
                    "check_large_loops": True,
                    "check_memory_usage": False,  # 暂时关闭
                    "check_concurrent_access": True,
                    "max_loop_iterations": 100000
                }
            },
            "severity_levels": {
                "critical": {
                    "circular_dependencies": True,
                    "architecture_violations": True,
                    "security_vulnerabilities": True
                },
                "high": {
                    "large_files": False,  # 降级文件大小问题
                    "complex_functions": False,
                    "missing_interfaces": False
                },
                "medium": {
                    "undocumented_code": False,  # 降级文档问题
                    "naming_violations": True,
                    "import_issues": True
                },
                "low": {
                    "style_issues": True,
                    "optimization_opportunities": True
                }
            },
            "exclusions": {
                "files": [
                    "*/test/*",
                    "*/tests/*",
                    "*_test.py",
                    "test_*.py",
                    "conftest.py"
                ],
                "directories": [
                    ".git",
                    "__pycache__",
                    "*.egg-info",
                    "node_modules",
                    ".pytest_cache"
                ],
                "patterns": [
                    "# TODO",
                    "# FIXME",
                    "# XXX",
                    "pass",
                    "NotImplemented"
                ]
            },
            "output_config": {
                "max_issues_per_category": 50,  # 限制每类问题数量
                "include_line_numbers": True,
                "include_suggestions": True,
                "group_by_category": True,
                "sort_by_severity": True
            }
        }

    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self.config

    def get_rule(self, category: str, rule: str) -> Any:
        """获取特定规则"""
        return self.config.get("review_rules", {}).get(category, {}).get(rule)

    def is_excluded(self, file_path: str) -> bool:
        """检查文件是否被排除"""
        path = Path(file_path)

        # 检查文件排除模式
        for pattern in self.config["exclusions"]["files"]:
            if path.match(pattern):
                return True

        # 检查目录排除模式
        for pattern in self.config["exclusions"]["directories"]:
            if any(part.match(pattern) for part in path.parts):
                return True

        return False

    def should_check_severity(self, severity: str, issue_type: str) -> bool:
        """检查是否应该检查特定严重程度的特定问题"""
        return self.config.get("severity_levels", {}).get(severity, {}).get(issue_type, True)

    def update_config(self, updates: Dict[str, Any]) -> None:
        """更新配置"""
        def deep_update(original: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in updates.items():
                if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                    original[key] = deep_update(original[key], value)
                else:
                    original[key] = value
            return original

        self.config = deep_update(self.config, updates)

    def load_from_file(self, config_file: str) -> None:
        """从文件加载配置"""
        import json
        import yaml

        path = Path(config_file)
        if not path.exists():
            return

        try:
            if config_file.endswith('.json'):
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
            elif config_file.endswith(('.yaml', '.yml')):
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
            else:
                return

            self.update_config(file_config)
        except Exception as e:
            print(f"加载配置文件失败: {e}")

    def save_to_file(self, config_file: str) -> None:
        """保存配置到文件"""
        import json

        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存配置文件失败: {e}")

    def create_project_config(self, project_root: str) -> None:
        """为项目创建配置文件"""
        config_dir = Path(project_root) / "config" / "architecture"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "review_config.json"

        self.save_to_file(str(config_file))
        print(f"✅ 项目审查配置文件已创建: {config_file}")

    def get_summary(self) -> str:
        """获取配置摘要"""
        return f"""
审查配置摘要:
================

结构规则:
- 最大文件大小: {self.config['review_rules']['structure_rules']['max_file_size']} 行
- 最大类大小: {self.config['review_rules']['structure_rules']['max_class_size']} 行
- 最大函数大小: {self.config['review_rules']['structure_rules']['max_function_size']} 行

依赖规则:
- 最大导入数: {self.config['review_rules']['dependency_rules']['max_imports_per_file']}
- 允许循环依赖: {self.config['review_rules']['dependency_rules']['allow_circular_deps']}

设计规则:
- 需要接口: {self.config['review_rules']['design_rules']['require_interfaces']}
- 需要文档: {self.config['review_rules']['design_rules']['require_docstrings']}
- 最大继承深度: {self.config['review_rules']['design_rules']['max_inheritance_depth']}

安全规则:
- SQL注入检查: {self.config['review_rules']['security_rules']['check_sql_injection']}
- 硬编码凭据检查: {self.config['review_rules']['security_rules']['check_hardcoded_credentials']}

性能规则:
- 大循环检查: {self.config['review_rules']['performance_rules']['check_large_loops']}
- 最大循环次数: {self.config['review_rules']['performance_rules']['max_loop_iterations']}

输出配置:
- 每类最大问题数: {self.config['output_config']['max_issues_per_category']}
- 包含行号: {self.config['output_config']['include_line_numbers']}
- 包含建议: {self.config['output_config']['include_suggestions']}
"""


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='架构审查配置工具')
    parser.add_argument('--create', help='为项目创建配置文件')
    parser.add_argument('--show', action='store_true', help='显示当前配置')
    parser.add_argument('--save', help='保存配置到文件')

    args = parser.parse_args()

    config = ReviewConfig()

    if args.create:
        config.create_project_config(args.create)
    elif args.show:
        print(config.get_summary())
    elif args.save:
        config.save_to_file(args.save)
        print(f"✅ 配置已保存到: {args.save}")
    else:
        print("使用方法:")
        print("  python scripts/architecture_review_config.py --create /path/to/project  # 创建项目配置")
        print("  python scripts/architecture_review_config.py --show                    # 显示配置")
        print("  python scripts/architecture_review_config.py --save config.json      # 保存配置")


if __name__ == "__main__":
    main()
