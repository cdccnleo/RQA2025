"""
AST工具类

提供AST解析和分析的实用功能。
"""

import ast
from typing import Dict, List, Optional, Tuple


class ASTUtils:
    """
    AST工具类

    提供AST相关的实用功能。
    """

    @staticmethod
    def extract_functions(tree: ast.AST) -> List[ast.FunctionDef]:
        """
        提取所有函数定义

        Args:
            tree: AST树

        Returns:
            List[ast.FunctionDef]: 函数定义列表
        """
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node)
        return functions

    @staticmethod
    def extract_classes(tree: ast.AST) -> List[ast.ClassDef]:
        """
        提取所有类定义

        Args:
            tree: AST树

        Returns:
            List[ast.ClassDef]: 类定义列表
        """
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node)
        return classes

    @staticmethod
    def extract_method_calls(tree: ast.AST) -> List[Tuple[str, List[str]]]:
        """
        提取方法调用

        Args:
            tree: AST树

        Returns:
            List[Tuple[str, List[str]]]: (方法名, 参数列表)
        """
        calls = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                method_name = ASTUtils._get_call_name(node)
                args = ASTUtils._get_call_args(node)
                if method_name:
                    calls.append((method_name, args))

        return calls

    @staticmethod
    def extract_variables(tree: ast.AST) -> Dict[str, str]:
        """
        提取变量定义和使用

        Args:
            tree: AST树

        Returns:
            Dict[str, str]: 变量名到类型的映射
        """
        variables = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # 处理赋值语句
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_type = ASTUtils._infer_type(node.value)
                        variables[target.id] = var_type
            elif isinstance(node, ast.AnnAssign):
                # 处理类型注解赋值
                if isinstance(node.target, ast.Name):
                    var_type = ASTUtils._get_annotation_type(node.annotation)
                    variables[node.target.id] = var_type

        return variables

    @staticmethod
    def extract_control_flow(tree: ast.AST) -> List[str]:
        """
        提取控制流结构

        Args:
            tree: AST树

        Returns:
            List[str]: 控制流关键字列表
        """
        control_flow = []

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                control_flow.append('if')
            elif isinstance(node, ast.For):
                control_flow.append('for')
            elif isinstance(node, ast.While):
                control_flow.append('while')
            elif isinstance(node, ast.Try):
                control_flow.append('try')
            elif isinstance(node, ast.With):
                control_flow.append('with')

        return control_flow

    @staticmethod
    def calculate_complexity(tree: ast.AST) -> float:
        """
        计算代码复杂度

        Args:
            tree: AST树

        Returns:
            float: 复杂度分数
        """
        complexity = 0.0

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1.0
            elif isinstance(node, ast.FunctionDef):
                complexity += 0.5
            elif isinstance(node, ast.Call):
                complexity += 0.2
            elif isinstance(node, ast.BoolOp):
                complexity += 0.1

        return complexity

    @staticmethod
    def normalize_ast(tree: ast.AST) -> str:
        """
        将AST标准化为字符串表示

        Args:
            tree: AST树

        Returns:
            str: 标准化字符串
        """
        return ast.dump(tree, annotate_fields=False)

    @staticmethod
    def find_similar_subtrees(tree1: ast.AST, tree2: ast.AST) -> List[Tuple[ast.AST, ast.AST]]:
        """
        查找相似的子树

        Args:
            tree1: 第一个AST树
            tree2: 第二个AST树

        Returns:
            List[Tuple[ast.AST, ast.AST]]: 相似子树对
        """
        similar_pairs = []

        # 简单的相似性检查
        nodes1 = list(ast.walk(tree1))
        nodes2 = list(ast.walk(tree2))

        for node1 in nodes1:
            for node2 in nodes2:
                if type(node1) == type(node2):
                    # 类型相同的节点
                    dump1 = ast.dump(node1, annotate_fields=False)
                    dump2 = ast.dump(node2, annotate_fields=False)

                    if dump1 == dump2:
                        similar_pairs.append((node1, node2))

        return similar_pairs

    @staticmethod
    def _get_call_name(node: ast.Call) -> Optional[str]:
        """获取调用名称"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return f"{ASTUtils._get_attr_name(node.func)}"
        return None

    @staticmethod
    def _get_attr_name(node: ast.Attribute) -> str:
        """获取属性名称"""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{ASTUtils._get_attr_name(node.value)}.{node.attr}"
        return node.attr

    @staticmethod
    def _get_call_args(node: ast.Call) -> List[str]:
        """获取调用参数"""
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Name):
                args.append(arg.id)
            elif isinstance(arg, ast.Str):
                args.append(f'"{arg.s}"')
            elif isinstance(arg, ast.Num):
                args.append(str(arg.n))
            else:
                args.append('...')
        return args

    @staticmethod
    def _infer_type(node: ast.AST) -> str:
        """推断节点类型"""
        if isinstance(node, ast.Str):
            return 'str'
        elif isinstance(node, ast.Num):
            if isinstance(node.n, int):
                return 'int'
            elif isinstance(node.n, float):
                return 'float'
        elif isinstance(node, ast.List):
            return 'list'
        elif isinstance(node, ast.Dict):
            return 'dict'
        elif isinstance(node, ast.Name):
            return 'var'
        elif isinstance(node, ast.Call):
            return 'call'
        return 'unknown'

    @staticmethod
    def _get_annotation_type(annotation: ast.AST) -> str:
        """获取类型注解"""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Subscript):
            return 'generic'
        return 'unknown'
