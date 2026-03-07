"""
代码复杂度检查器

检查代码的圈复杂度、可维护性指数等复杂度指标。
"""

import ast
from typing import Dict, Any, Optional, Tuple

from ..core.base_checker import BaseChecker
from ..core.check_result import IssueSeverity


class ComplexityMetrics:
    """复杂度指标"""

    def __init__(self):
        self.cyclomatic_complexity = 0  # 圈复杂度
        self.lines_of_code = 0  # 代码行数
        self.comment_lines = 0  # 注释行数
        self.blank_lines = 0  # 空行数
        self.nesting_depth = 0  # 最大嵌套深度
        self.parameter_count = 0  # 参数数量
        self.variable_count = 0  # 变量数量
        self.function_count = 0  # 函数数量
        self.class_count = 0  # 类数量

    def calculate_maintainability_index(self) -> float:
        """
        计算可维护性指数

        Returns:
            float: 可维护性指数 (0-100)
        """
        # 简化的MI计算公式
        mi = max(0, (171 - 5.2 * self.cyclomatic_complexity
                     - 0.23 * self.lines_of_code
                     + 16.2 * (self.comment_lines / max(1, self.lines_of_code))) / 100)
        return min(100, mi * 100)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'cyclomatic_complexity': self.cyclomatic_complexity,
            'lines_of_code': self.lines_of_code,
            'comment_lines': self.comment_lines,
            'blank_lines': self.blank_lines,
            'nesting_depth': self.nesting_depth,
            'parameter_count': self.parameter_count,
            'variable_count': self.variable_count,
            'function_count': self.function_count,
            'class_count': self.class_count,
            'maintainability_index': self.calculate_maintainability_index()
        }


class ComplexityChecker(BaseChecker):
    """
    代码复杂度检查器

    分析代码的各种复杂度指标并提出改进建议。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def _setup_default_config(self) -> None:
        """设置默认配置"""
        defaults = {
            'max_cyclomatic_complexity': 10,  # 最大圈复杂度
            'max_lines_per_function': 50,  # 函数最大行数
            'max_nesting_depth': 4,  # 最大嵌套深度
            'max_parameters': 5,  # 最大参数数量
            'min_maintainability_index': 50,  # 最小可维护性指数
            'check_functions': True,  # 检查函数
            'check_classes': True,  # 检查类
            'check_modules': True,  # 检查模块
        }

        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

    @property
    def checker_name(self) -> str:
        return "complexity_checker"

    @property
    def checker_description(self) -> str:
        return "检查代码复杂度指标"

    def check(self, target_path: str) -> 'CheckResult':
        """
        执行复杂度检查

        Args:
            target_path: 检查目标路径

        Returns:
            CheckResult: 检查结果
        """
        result = self._create_result()

        try:
            # 收集Python文件
            python_files = self._collect_python_files(target_path)

            if not python_files:
                result.metadata['message'] = "未找到Python文件"
                result.set_end_time()
                return result

            total_functions = 0
            total_classes = 0

            # 检查每个文件
            for file_path in python_files:
                file_functions, file_classes = self._check_file_complexity(result, file_path)
                total_functions += file_functions
                total_classes += file_classes

            # 设置元数据
            result.metadata.update({
                'total_files': len(python_files),
                'total_functions': total_functions,
                'total_classes': total_classes
            })

        except Exception as e:
            self.logger.error(f"复杂度检查失败: {e}")
            result.add_issue(self._create_issue(
                file_path=target_path,
                message=f"复杂度检查失败: {e}",
                severity=IssueSeverity.ERROR,
                rule_id="COMPLEXITY_CHECK_FAILED"
            ))

        result.set_end_time()
        return result

    def _check_file_complexity(self, result: 'CheckResult', file_path: str) -> Tuple[int, int]:
        """
        检查单个文件的复杂度

        Args:
            result: 检查结果
            file_path: 文件路径

        Returns:
            Tuple[int, int]: (函数数量, 类数量)
        """
        try:
            content = self._read_file_content(file_path)
            if not content:
                return 0, 0

            # 解析AST
            tree = ast.parse(content, filename=file_path)

            # 计算文件级指标
            file_metrics = self._calculate_file_metrics(content)

            # 检查函数复杂度
            function_count = 0
            if self.config.get('check_functions', True):
                function_count = self._check_functions(result, tree, file_path)

            # 检查类复杂度
            class_count = 0
            if self.config.get('check_classes', True):
                class_count = self._check_classes(result, tree, file_path)

            # 检查模块级复杂度
            if self.config.get('check_modules', True):
                self._check_module_complexity(result, file_metrics, file_path)

            return function_count, class_count

        except Exception as e:
            self.logger.warning(f"检查文件复杂度失败 {file_path}: {e}")
            return 0, 0

    def _calculate_file_metrics(self, content: str) -> ComplexityMetrics:
        """
        计算文件级复杂度指标

        Args:
            content: 文件内容

        Returns:
            ComplexityMetrics: 复杂度指标
        """
        metrics = ComplexityMetrics()
        lines = content.split('\n')

        metrics.lines_of_code = len(lines)

        for line in lines:
            line = line.strip()
            if not line:
                metrics.blank_lines += 1
            elif line.startswith('#'):
                metrics.comment_lines += 1

        # 计算实际代码行数
        metrics.lines_of_code = metrics.lines_of_code - metrics.blank_lines - metrics.comment_lines

        return metrics

    def _check_functions(self, result: 'CheckResult', tree: ast.AST, file_path: str) -> int:
        """
        检查函数复杂度

        Args:
            result: 检查结果
            tree: AST树
            file_path: 文件路径

        Returns:
            int: 函数数量
        """
        function_count = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_count += 1
                self._check_single_function(result, node, file_path)

        return function_count

    def _check_single_function(self, result: 'CheckResult',
                               node: ast.FunctionDef, file_path: str) -> None:
        """
        检查单个函数的复杂度

        Args:
            result: 检查结果
            node: 函数节点
            file_path: 文件路径
        """
        function_name = node.name

        # 计算圈复杂度
        cyclomatic_complexity = self._calculate_cyclomatic_complexity(node)

        # 计算行数
        lines_count = self._calculate_function_lines(node)

        # 计算嵌套深度
        nesting_depth = self._calculate_nesting_depth(node)

        # 计算参数数量
        param_count = len(node.args.args)

        # 检查复杂度阈值
        if cyclomatic_complexity > self.config['max_cyclomatic_complexity']:
            result.add_issue(self._create_issue(
                file_path=file_path,
                message=f"函数 {function_name} 圈复杂度过高: {cyclomatic_complexity} (阈值: {self.config['max_cyclomatic_complexity']})",
                severity=IssueSeverity.WARNING,
                rule_id="HIGH_CYCLOMATIC_COMPLEXITY",
                line_number=node.lineno,
                details={
                    'function': function_name,
                    'complexity': cyclomatic_complexity,
                    'threshold': self.config['max_cyclomatic_complexity'],
                    'lines': lines_count,
                    'parameters': param_count
                }
            ))

        # 检查函数长度
        if lines_count > self.config['max_lines_per_function']:
            result.add_issue(self._create_issue(
                file_path=file_path,
                message=f"函数 {function_name} 过长: {lines_count}行 (阈值: {self.config['max_lines_per_function']}行)",
                severity=IssueSeverity.WARNING,
                rule_id="FUNCTION_TOO_LONG",
                line_number=node.lineno,
                details={
                    'function': function_name,
                    'lines': lines_count,
                    'threshold': self.config['max_lines_per_function']
                }
            ))

        # 检查嵌套深度
        if nesting_depth > self.config['max_nesting_depth']:
            result.add_issue(self._create_issue(
                file_path=file_path,
                message=f"函数 {function_name} 嵌套深度过深: {nesting_depth}层 (阈值: {self.config['max_nesting_depth']}层)",
                severity=IssueSeverity.WARNING,
                rule_id="DEEP_NESTING",
                line_number=node.lineno,
                details={
                    'function': function_name,
                    'depth': nesting_depth,
                    'threshold': self.config['max_nesting_depth']
                }
            ))

        # 检查参数数量
        if param_count > self.config['max_parameters']:
            result.add_issue(self._create_issue(
                file_path=file_path,
                message=f"函数 {function_name} 参数过多: {param_count}个 (阈值: {self.config['max_parameters']}个)",
                severity=IssueSeverity.INFO,
                rule_id="TOO_MANY_PARAMETERS",
                line_number=node.lineno,
                details={
                    'function': function_name,
                    'parameters': param_count,
                    'threshold': self.config['max_parameters']
                }
            ))

    def _check_classes(self, result: 'CheckResult', tree: ast.AST, file_path: str) -> int:
        """
        检查类复杂度

        Args:
            result: 检查结果
            tree: AST树
            file_path: 文件路径

        Returns:
            int: 类数量
        """
        class_count = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_count += 1
                self._check_single_class(result, node, file_path)

        return class_count

    def _check_single_class(self, result: 'CheckResult',
                            node: ast.ClassDef, file_path: str) -> None:
        """
        检查单个类的复杂度

        Args:
            result: 检查结果
            node: 类节点
            file_path: 文件路径
        """
        class_name = node.name

        # 计算类的方法数量
        method_count = sum(1 for item in node.body
                           if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)))

        # 计算类的行数
        lines_count = self._calculate_class_lines(node)

        # 检查类的大小
        if method_count > 20:  # 经验阈值
            result.add_issue(self._create_issue(
                file_path=file_path,
                message=f"类 {class_name} 方法过多: {method_count}个方法",
                severity=IssueSeverity.INFO,
                rule_id="LARGE_CLASS",
                line_number=node.lineno,
                details={
                    'class': class_name,
                    'method_count': method_count,
                    'lines': lines_count
                }
            ))

    def _check_module_complexity(self, result: 'CheckResult',
                                 metrics: ComplexityMetrics, file_path: str) -> None:
        """
        检查模块级复杂度

        Args:
            result: 检查结果
            metrics: 复杂度指标
            file_path: 文件路径
        """
        # 计算可维护性指数
        mi = metrics.calculate_maintainability_index()

        if mi < self.config['min_maintainability_index']:
            result.add_issue(self._create_issue(
                file_path=file_path,
                message=f"文件可维护性指数过低: {mi:.1f} (阈值: {self.config['min_maintainability_index']})",
                severity=IssueSeverity.WARNING,
                rule_id="LOW_MAINTAINABILITY_INDEX",
                details={
                    'maintainability_index': mi,
                    'threshold': self.config['min_maintainability_index'],
                    'lines_of_code': metrics.lines_of_code,
                    'cyclomatic_complexity': metrics.cyclomatic_complexity,
                    'comment_lines': metrics.comment_lines
                }
            ))

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """
        计算圈复杂度

        Args:
            node: 函数节点

        Returns:
            int: 圈复杂度
        """
        complexity = 1  # 基础复杂度

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With,
                                  ast.Try, ast.ExceptHandler, ast.Assert)):
                complexity += 1
            elif isinstance(child, ast.BoolOp) and isinstance(child.op, (ast.And, ast.Or)):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.Conditional) if hasattr(ast, 'Conditional') else False:
                complexity += 1

        return complexity

    def _calculate_function_lines(self, node: ast.FunctionDef) -> int:
        """
        计算函数的行数

        Args:
            node: 函数节点

        Returns:
            int: 行数
        """
        min_line = node.lineno
        max_line = node.lineno

        for child in ast.walk(node):
            if hasattr(child, 'lineno'):
                max_line = max(max_line, child.lineno)

        return max_line - min_line + 1

    def _calculate_class_lines(self, node: ast.ClassDef) -> int:
        """
        计算类的行数

        Args:
            node: 类节点

        Returns:
            int: 行数
        """
        min_line = node.lineno
        max_line = node.lineno

        for child in ast.walk(node):
            if hasattr(child, 'lineno'):
                max_line = max(max_line, child.lineno)

        return max_line - min_line + 1

    def _calculate_nesting_depth(self, node: ast.AST, depth: int = 0) -> int:
        """
        计算最大嵌套深度

        Args:
            node: AST节点
            depth: 当前深度

        Returns:
            int: 最大嵌套深度
        """
        max_depth = depth

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With,
                                  ast.Try, ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                child_depth = self._calculate_nesting_depth(child, depth + 1)
                max_depth = max(max_depth, child_depth)

        return max_depth
