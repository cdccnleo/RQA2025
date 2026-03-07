#!/usr/bin/env python3
"""
AI辅助开发工具

引入AI辅助进行架构设计和代码优化
"""

import os
import re
import json
import ast
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import openai


class AIAssistedDevelopment:
    """AI辅助开发工具"""

    def __init__(self, project_root: str, api_key: str = None):
        self.project_root = Path(project_root)
        self.infrastructure_dir = self.project_root / "src" / "infrastructure"

        # AI配置
        self.config = {
            "api_key": api_key or os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000,
            "enable_code_generation": True,
            "enable_architecture_review": True,
            "enable_optimization_suggestions": True
        }

        # AI分析历史
        self.analysis_history = []

    def analyze_architecture_with_ai(self, focus_areas: List[str] = None) -> Dict[str, Any]:
        """使用AI分析架构"""
        print("🤖 开始AI架构分析...")

        if not self.config["api_key"]:
            return {
                "success": False,
                "error": "未配置API密钥，请设置OPENAI_API_KEY环境变量"
            }

        # 收集架构信息
        architecture_info = self._collect_architecture_info()

        # 构建提示词
        prompt = self._build_architecture_analysis_prompt(architecture_info, focus_areas)

        try:
            # 调用AI API
            response = self._call_ai_api(prompt)

            # 解析AI响应
            analysis = self._parse_ai_response(response)

            # 记录到历史
            self.analysis_history.append({
                "timestamp": datetime.now(),
                "type": "architecture_analysis",
                "focus_areas": focus_areas,
                "analysis": analysis
            })

            print("✅ AI架构分析完成")
            return {
                "success": True,
                "analysis": analysis,
                "raw_response": response
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"AI分析失败: {e}"
            }

    def generate_code_with_ai(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """使用AI生成代码"""
        print("🤖 开始AI代码生成...")

        if not self.config["api_key"]:
            return {
                "success": False,
                "error": "未配置API密钥"
            }

        # 构建代码生成提示词
        prompt = self._build_code_generation_prompt(requirements)

        try:
            # 调用AI API
            response = self._call_ai_api(prompt)

            # 解析生成的代码
            generated_code = self._parse_generated_code(response)

            # 验证和优化代码
            validated_code = self._validate_generated_code(generated_code, requirements)

            print("✅ AI代码生成完成")
            return {
                "success": True,
                "code": validated_code,
                "raw_response": response
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"AI代码生成失败: {e}"
            }

    def optimize_code_with_ai(self, file_path: str, optimization_goals: List[str]) -> Dict[str, Any]:
        """使用AI优化代码"""
        print("🤖 开始AI代码优化...")

        if not self.config["api_key"]:
            return {
                "success": False,
                "error": "未配置API密钥"
            }

        try:
            # 读取原始代码
            with open(file_path, 'r', encoding='utf-8') as f:
                original_code = f.read()

            # 构建优化提示词
            prompt = self._build_code_optimization_prompt(original_code, optimization_goals)

            # 调用AI API
            response = self._call_ai_api(prompt)

            # 解析优化后的代码
            optimized_code = self._parse_optimized_code(response)

            # 应用优化
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(optimized_code)

            print("✅ AI代码优化完成")
            return {
                "success": True,
                "file": file_path,
                "original_length": len(original_code),
                "optimized_length": len(optimized_code),
                "improvements": optimization_goals
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"AI代码优化失败: {e}"
            }

    def review_architecture_with_ai(self, review_scope: str = "full") -> Dict[str, Any]:
        """使用AI审查架构"""
        print("🤖 开始AI架构审查...")

        if not self.config["api_key"]:
            return {
                "success": False,
                "error": "未配置API密钥"
            }

        # 收集架构信息
        architecture_info = self._collect_architecture_info()

        # 构建审查提示词
        prompt = self._build_architecture_review_prompt(architecture_info, review_scope)

        try:
            # 调用AI API
            response = self._call_ai_api(prompt)

            # 解析审查结果
            review = self._parse_review_response(response)

            print("✅ AI架构审查完成")
            return {
                "success": True,
                "review": review,
                "recommendations": review.get("recommendations", [])
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"AI架构审查失败: {e}"
            }

    def _collect_architecture_info(self) -> Dict[str, Any]:
        """收集架构信息"""
        info = {
            "directory_structure": {},
            "file_count": {},
            "interface_count": {},
            "responsibility_boundaries": {},
            "cross_layer_imports": [],
            "documentation_coverage": {}
        }

        # 分析目录结构
        for category_dir in self.infrastructure_dir.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith('_'):
                py_files = list(category_dir.glob("*.py"))
                info["directory_structure"][category_dir.name] = len(py_files)
                info["file_count"][category_dir.name] = len(py_files)

                # 统计接口
                interface_count = 0
                for py_file in py_files:
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        interface_count += len(re.findall(r'class I[A-Z]\w*Component', content))
                    except:
                        pass
                info["interface_count"][category_dir.name] = interface_count

        return info

    def _build_architecture_analysis_prompt(self, architecture_info: Dict[str, Any], focus_areas: List[str]) -> str:
        """构建架构分析提示词"""
        focus_text = ", ".join(focus_areas) if focus_areas else "整体架构"

        prompt = f"""
你是一个资深架构师，请分析以下基础设施层架构：

## 架构信息
- 目录结构: {json.dumps(architecture_info['directory_structure'], ensure_ascii=False)}
- 文件数量: {json.dumps(architecture_info['file_count'], ensure_ascii=False)}
- 接口数量: {json.dumps(architecture_info['interface_count'], ensure_ascii=False)}

## 分析重点
{focus_text}

## 分析要求
请从以下维度进行分析：

1. **架构合理性**: 目录结构是否合理，职责划分是否清晰
2. **扩展性**: 架构是否易于扩展和维护
3. **一致性**: 命名规范、代码风格是否一致
4. **性能**: 架构设计对性能的影响
5. **安全性**: 架构中的安全考虑
6. **可测试性**: 代码的可测试性设计

## 输出格式
请以JSON格式输出分析结果：
{{
    "overall_score": 85,
    "strengths": ["优势1", "优势2"],
    "weaknesses": ["问题1", "问题2"],
    "recommendations": [
        {{
            "priority": "high",
            "category": "架构优化",
            "description": "具体建议",
            "implementation": "实施方法"
        }}
    ],
    "architecture_patterns": ["模式1", "模式2"],
    "improvement_areas": ["改进方向1", "改进方向2"]
}}
"""

        return prompt

    def _build_code_generation_prompt(self, requirements: Dict[str, Any]) -> str:
        """构建代码生成提示词"""
        prompt = f"""
你是一个资深Python开发工程师，请根据以下需求生成高质量的代码：

## 需求说明
- 功能: {requirements.get('function', '未指定')}
- 类型: {requirements.get('type', '未指定')}
- 架构位置: {requirements.get('location', '基础设施层')}
- 依赖: {requirements.get('dependencies', '无')}

## 代码要求
1. 遵循现有的架构模式和命名规范
2. 使用I{Name}Component接口命名标准
3. 包含完整的文档字符串
4. 实现错误处理和日志记录
5. 考虑性能和资源使用
6. 提供使用示例

## 现有架构风格
- 接口命名: I{Name}Component
- 基类命名: Base{Name}Component
- 职责分离: 单一职责原则
- 依赖注入: 构造函数注入

请生成完整的Python代码：
"""

        return prompt

    def _build_code_optimization_prompt(self, original_code: str, optimization_goals: List[str]) -> str:
        """构建代码优化提示词"""
        prompt = f"""
你是一个资深Python开发工程师，请优化以下代码：

## 原始代码
```python
{original_code}
```

## 优化目标
{chr(10).join(f"- {goal}" for goal in optimization_goals)}

## 优化要求
1. 保持原有功能不变
2. 提高代码质量和可维护性
3. 优化性能和资源使用
4. 改进错误处理
5. 增强文档和注释
6. 统一代码风格

请提供优化后的完整代码：
"""

        return prompt

    def _build_architecture_review_prompt(self, architecture_info: Dict[str, Any], review_scope: str) -> str:
        """构建架构审查提示词"""
        prompt = f"""
你是一个资深架构师，请审查以下架构设计：

## 审查范围
{review_scope}

## 架构信息
{json.dumps(architecture_info, ensure_ascii=False, indent=2)}

## 审查要点
1. **架构一致性**: 是否遵循既定架构原则
2. **职责边界**: 各模块职责是否清晰明确
3. **依赖关系**: 模块间依赖是否合理
4. **扩展性**: 架构是否支持未来扩展
5. **可维护性**: 代码是否易于维护
6. **性能影响**: 架构设计对性能的影响

## 输出格式
请以JSON格式输出审查结果：
{{
    "review_score": 90,
    "issues": [
        {{
            "severity": "medium",
            "category": "架构设计",
            "description": "具体问题",
            "impact": "影响分析",
            "recommendation": "建议解决方案"
        }}
    ],
    "strengths": ["优势1", "优势2"],
    "recommendations": [
        {{
            "priority": "high",
            "area": "架构优化",
            "suggestion": "具体建议",
            "benefit": "预期收益"
        }}
    ]
}}
"""

        return prompt

    def _call_ai_api(self, prompt: str) -> str:
        """调用AI API"""
        try:
            client = openai.OpenAI(api_key=self.config["api_key"])

            response = client.chat.completions.create(
                model=self.config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"]
            )

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"AI API调用失败: {e}")

    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """解析AI响应"""
        try:
            # 尝试直接解析JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果不是JSON，尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # 如果无法解析，返回文本分析结果
                return {
                    "overall_score": 75,
                    "analysis": response,
                    "parse_error": "无法解析为JSON格式"
                }

    def _parse_generated_code(self, response: str) -> str:
        """解析生成的代码"""
        # 提取代码块
        code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0]

        # 如果没有代码块，提取所有Python代码
        python_code = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if python_code:
            return python_code[0]

        return response

    def _parse_optimized_code(self, response: str) -> str:
        """解析优化后的代码"""
        return self._parse_generated_code(response)

    def _parse_review_response(self, response: str) -> Dict[str, Any]:
        """解析审查响应"""
        return self._parse_ai_response(response)

    def _validate_generated_code(self, code: str, requirements: Dict[str, Any]) -> str:
        """验证生成的代码"""
        try:
            # 尝试解析AST以验证语法
            ast.parse(code)

            # 检查是否符合命名规范
            interface_count = len(re.findall(r'class I[A-Z]\w*Component', code))

            print(f"✅ 代码验证通过，包含 {interface_count} 个标准接口")
            return code

        except SyntaxError as e:
            print(f"❌ 代码语法错误: {e}")
            return f"# 代码生成失败：语法错误\n# {e}\n\n{code}"

    def generate_ai_report(self) -> Dict[str, Any]:
        """生成AI辅助开发报告"""
        report_data = {
            "timestamp": datetime.now(),
            "analysis_history": self.analysis_history[-10:],  # 最近10次分析
            "config": self.config.copy(),
            "summary": {
                "total_analyses": len(self.analysis_history),
                "success_rate": sum(1 for h in self.analysis_history if h.get("success", True)) / max(1, len(self.analysis_history)),
                "average_score": sum(h.get("analysis", {}).get("overall_score", 0) for h in self.analysis_history) / max(1, len(self.analysis_history))
            }
        }

        # 保存报告
        report_path = self.project_root / "reports" / \
            f"ai_assisted_development_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

        return {
            "success": True,
            "report_path": str(report_path),
            "data": report_data
        }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='AI辅助开发工具')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--analyze', help='AI架构分析，指定重点领域(逗号分隔)')
    parser.add_argument('--generate', help='AI代码生成，需求描述')
    parser.add_argument('--optimize', help='AI代码优化，文件路径')
    parser.add_argument('--review', help='AI架构审查，审查范围')
    parser.add_argument('--report', action='store_true', help='生成AI报告')
    parser.add_argument('--goals', help='优化目标(逗号分隔)')

    args = parser.parse_args()

    # 从环境变量获取API密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 请设置OPENAI_API_KEY环境变量")
        return

    ai_dev = AIAssistedDevelopment(args.project, api_key)

    if args.analyze:
        focus_areas = [area.strip() for area in args.analyze.split(',')]
        result = ai_dev.analyze_architecture_with_ai(focus_areas)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.generate:
        requirements = {"function": args.generate, "type": "interface"}
        result = ai_dev.generate_code_with_ai(requirements)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.optimize and args.goals:
        optimization_goals = [goal.strip() for goal in args.goals.split(',')]
        result = ai_dev.optimize_code_with_ai(args.optimize, optimization_goals)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.review:
        result = ai_dev.review_architecture_with_ai(args.review)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.report:
        result = ai_dev.generate_ai_report()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    else:
        print("🤖 AI辅助开发工具")
        print("使用 --help 查看可用命令")


if __name__ == "__main__":
    main()
