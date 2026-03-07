#!/usr/bin/env python3
"""
架构培训评估系统

评估培训效果，跟踪学习进度，生成培训报告
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


class TrainingAssessmentSystem:
    """培训评估系统"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.assessment_dir = self.project_root / "reports" / "training"
        self.assessment_dir.mkdir(parents=True, exist_ok=True)

        # 培训配置
        self.training_config = {
            "modules": [
                {
                    "id": "module_1",
                    "name": "架构设计理念",
                    "topics": ["业务流程驱动", "架构层次结构", "设计原则"],
                    "duration": 120,
                    "difficulty": "beginner"
                },
                {
                    "id": "module_2",
                    "name": "核心架构模式",
                    "topics": ["事件驱动架构", "依赖注入模式", "分层架构"],
                    "duration": 180,
                    "difficulty": "intermediate"
                },
                {
                    "id": "module_3",
                    "name": "架构工具使用",
                    "topics": ["一致性检查", "文档自动化", "质量仪表板"],
                    "duration": 120,
                    "difficulty": "intermediate"
                },
                {
                    "id": "module_4",
                    "name": "架构设计实践",
                    "topics": ["架构分析", "设计方法", "审查流程"],
                    "duration": 240,
                    "difficulty": "advanced"
                }
            ],
            "assessment_methods": [
                "theory_test",
                "practical_exercise",
                "peer_review",
                "project_work"
            ]
        }

    def generate_assessment_plan(self, trainee_info: Dict[str, Any]) -> Dict[str, Any]:
        """生成个性化评估计划"""

        plan = {
            "trainee": trainee_info,
            "assessment_plan": {
                "start_date": datetime.now().isoformat(),
                "modules": [],
                "total_duration": 0,
                "assessment_schedule": []
            },
            "evaluation_criteria": {
                "knowledge_mastery": 0.4,    # 理论知识掌握
                "practical_skills": 0.3,     # 实践技能
                "problem_solving": 0.2,      # 问题解决能力
                "communication": 0.1         # 沟通表达能力
            }
        }

        # 根据学员背景定制模块
        for module in self.training_config["modules"]:
            if self._should_include_module(module, trainee_info):
                plan["assessment_plan"]["modules"].append({
                    "module_id": module["id"],
                    "module_name": module["name"],
                    "assessment_method": self._select_assessment_method(module, trainee_info),
                    "weight": self._calculate_module_weight(module, trainee_info)
                })
                plan["assessment_plan"]["total_duration"] += module["duration"]

        return plan

    def conduct_assessment(self, assessment_plan: Dict[str, Any], responses: Dict[str, Any]) -> Dict[str, Any]:
        """执行评估"""

        results = {
            "assessment_id": f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "trainee_id": assessment_plan["trainee"]["id"],
            "completion_date": datetime.now().isoformat(),
            "module_results": [],
            "overall_score": 0.0,
            "grade": "",
            "feedback": "",
            "recommendations": []
        }

        total_score = 0.0
        total_weight = 0.0

        # 评估每个模块
        for module_plan in assessment_plan["assessment_plan"]["modules"]:
            module_result = self._assess_module(
                module_plan, responses.get(module_plan["module_id"], {}))
            results["module_results"].append(module_result)

            weighted_score = module_result["score"] * module_plan["weight"]
            total_score += weighted_score
            total_weight += module_plan["weight"]

        # 计算总分
        results["overall_score"] = total_score / total_weight if total_weight > 0 else 0
        results["grade"] = self._calculate_grade(results["overall_score"])
        results["feedback"] = self._generate_feedback(results)
        results["recommendations"] = self._generate_recommendations(results)

        return results

    def generate_training_report(self, assessment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成培训报告"""

        report = {
            "report_id": f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generation_date": datetime.now().isoformat(),
            "summary": {
                "total_trainees": len(assessment_results),
                "average_score": 0.0,
                "pass_rate": 0.0,
                "completion_rate": 0.0
            },
            "module_analysis": {},
            "trends": {},
            "recommendations": []
        }

        # 计算汇总数据
        total_score = 0
        pass_count = 0

        for result in assessment_results:
            total_score += result["overall_score"]
            if result["overall_score"] >= 60:
                pass_count += 1

        report["summary"]["average_score"] = total_score / len(assessment_results)
        report["summary"]["pass_rate"] = pass_count / len(assessment_results)
        report["summary"]["completion_rate"] = 1.0  # 假设都完成了

        # 模块分析
        report["module_analysis"] = self._analyze_modules(assessment_results)

        # 生成趋势分析
        report["trends"] = self._analyze_trends(assessment_results)

        # 生成建议
        report["recommendations"] = self._generate_training_recommendations(report)

        return report

    def _should_include_module(self, module: Dict[str, Any], trainee_info: Dict[str, Any]) -> bool:
        """判断是否应该包含某个模块"""
        experience_level = trainee_info.get("experience_level", "junior")

        # 根据经验水平过滤模块
        if experience_level == "junior" and module["difficulty"] == "advanced":
            return False
        elif experience_level == "senior" and module["difficulty"] == "beginner":
            return False

        return True

    def _select_assessment_method(self, module: Dict[str, Any], trainee_info: Dict[str, Any]) -> str:
        """选择评估方法"""
        methods = self.training_config["assessment_methods"]

        # 根据模块特点选择方法
        if module["difficulty"] == "advanced":
            return "project_work"
        elif "实践" in module["name"]:
            return "practical_exercise"
        else:
            return "theory_test"

    def _calculate_module_weight(self, module: Dict[str, Any], trainee_info: Dict[str, Any]) -> float:
        """计算模块权重"""
        base_weight = 1.0

        # 根据难度调整权重
        if module["difficulty"] == "advanced":
            base_weight *= 1.5
        elif module["difficulty"] == "beginner":
            base_weight *= 0.8

        # 根据学员背景调整
        experience_level = trainee_info.get("experience_level", "intermediate")
        if experience_level == "senior" and module["difficulty"] == "beginner":
            base_weight *= 0.7

        return base_weight

    def _assess_module(self, module_plan: Dict[str, Any], responses: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个模块"""
        # 简化的评估逻辑，实际应该更复杂
        score = 0.0

        # 理论测试评分
        theory_score = responses.get("theory_test", {}).get("score", 0)
        practical_score = responses.get("practical_exercise", {}).get("score", 0)

        if module_plan["assessment_method"] == "theory_test":
            score = theory_score
        elif module_plan["assessment_method"] == "practical_exercise":
            score = practical_score
        else:
            score = (theory_score + practical_score) / 2

        return {
            "module_id": module_plan["module_id"],
            "module_name": module_plan["module_name"],
            "assessment_method": module_plan["assessment_method"],
            "score": score,
            "max_score": 100,
            "percentage": score,
            "feedback": self._generate_module_feedback(module_plan, score)
        }

    def _calculate_grade(self, score: float) -> str:
        """计算等级"""
        if score >= 90:
            return "优秀"
        elif score >= 80:
            return "良好"
        elif score >= 70:
            return "中等"
        elif score >= 60:
            return "及格"
        else:
            return "不及格"

    def _generate_feedback(self, results: Dict[str, Any]) -> str:
        """生成反馈"""
        score = results["overall_score"]
        grade = results["grade"]

        feedback = f"总体表现{grade}，得分{score:.1f}分。"

        if score >= 90:
            feedback += "表现出色，对架构概念理解深刻，实践能力强。"
        elif score >= 80:
            feedback += "表现良好，基本掌握了核心概念和技能。"
        elif score >= 70:
            feedback += "表现一般，需要加强实践环节的练习。"
        elif score >= 60:
            feedback += "基本达到要求，但需要继续加强学习。"
        else:
            feedback += "需要大幅加强基础知识的学习和实践。"

        return feedback

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []

        for module_result in results["module_results"]:
            if module_result["score"] < 70:
                recommendations.append(f"建议加强{module_result['module_name']}的学习")
            if module_result["score"] < 60:
                recommendations.append(f"需要额外辅导{module_result['module_name']}相关内容")

        if results["overall_score"] < 70:
            recommendations.append("建议参加补习课程")
        if results["overall_score"] < 60:
            recommendations.append("需要重新参加培训")

        return recommendations

    def _analyze_modules(self, assessment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析模块表现"""
        module_scores = {}

        for result in assessment_results:
            for module_result in result["module_results"]:
                module_id = module_result["module_id"]
                if module_id not in module_scores:
                    module_scores[module_id] = []
                module_scores[module_id].append(module_result["score"])

        # 计算每个模块的平均分
        analysis = {}
        for module_id, scores in module_scores.items():
            analysis[module_id] = {
                "average_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "student_count": len(scores)
            }

        return analysis

    def _analyze_trends(self, assessment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析趋势"""
        # 简化的趋势分析
        return {
            "score_distribution": self._calculate_score_distribution(assessment_results),
            "improvement_trend": "需要更多数据来分析趋势",
            "common_weaknesses": self._identify_common_weaknesses(assessment_results)
        }

    def _calculate_score_distribution(self, assessment_results: List[Dict[str, Any]]) -> Dict[str, int]:
        """计算分数分布"""
        distribution = {"优秀": 0, "良好": 0, "中等": 0, "及格": 0, "不及格": 0}

        for result in assessment_results:
            grade = result["grade"]
            distribution[grade] += 1

        return distribution

    def _identify_common_weaknesses(self, assessment_results: List[Dict[str, Any]]) -> List[str]:
        """识别常见弱项"""
        weaknesses = []

        # 分析模块平均分
        module_analysis = self._analyze_modules(assessment_results)

        for module_id, stats in module_analysis.items():
            if stats["average_score"] < 70:
                weaknesses.append(f"模块 {module_id} 需要加强")

        return weaknesses

    def _generate_training_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """生成培训建议"""
        recommendations = []

        summary = report["summary"]

        if summary["average_score"] < 70:
            recommendations.append("整体培训效果需要改进")
        if summary["pass_rate"] < 0.8:
            recommendations.append("通过率偏低，需要优化教学方法")
        if summary["completion_rate"] < 0.9:
            recommendations.append("完成率有待提高")

        # 基于模块分析的建议
        module_analysis = report["module_analysis"]
        for module_id, stats in module_analysis.items():
            if stats["average_score"] < 70:
                recommendations.append(f"改进模块 {module_id} 的教学内容")

        return recommendations

    def generate_assessment_template(self) -> Dict[str, Any]:
        """生成评估模板"""
        template = {
            "assessment_template": {
                "version": "1.0",
                "modules": []
            }
        }

        for module in self.training_config["modules"]:
            module_template = {
                "module_id": module["id"],
                "module_name": module["name"],
                "assessment_criteria": {
                    "theory_test": {
                        "questions": [],
                        "weight": 0.4
                    },
                    "practical_exercise": {
                        "tasks": [],
                        "weight": 0.6
                    }
                }
            }

            # 为每个主题生成问题
            for topic in module["topics"]:
                if "theory_test" in module_template["assessment_criteria"]:
                    module_template["assessment_criteria"]["theory_test"]["questions"].extend(
                        self._generate_theory_questions(topic)
                    )

                if "practical_exercise" in module_template["assessment_criteria"]:
                    module_template["assessment_criteria"]["practical_exercise"]["tasks"].extend(
                        self._generate_practical_tasks(topic)
                    )

            template["assessment_template"]["modules"].append(module_template)

        return template

    def _generate_theory_questions(self, topic: str) -> List[Dict[str, Any]]:
        """生成理论问题"""
        return [
            {
                "question": f"请解释{topic}的核心概念",
                "type": "explanation",
                "difficulty": "medium"
            },
            {
                "question": f"{topic}的优势和局限性是什么？",
                "type": "analysis",
                "difficulty": "high"
            }
        ]

    def _generate_practical_tasks(self, topic: str) -> List[Dict[str, Any]]:
        """生成实践任务"""
        return [
            {
                "task": f"设计一个基于{topic}的架构组件",
                "deliverables": ["设计文档", "代码实现", "测试用例"],
                "time_estimate": "2小时"
            }
        ]

    def save_assessment_plan(self, plan: Dict[str, Any], filename: str):
        """保存评估计划"""
        filepath = self.assessment_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)

    def save_assessment_results(self, results: Dict[str, Any], filename: str):
        """保存评估结果"""
        filepath = self.assessment_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def generate_visualization(self, assessment_results: List[Dict[str, Any]], output_file: str):
        """生成可视化图表"""
        # 这里可以添加matplotlib/seaborn可视化代码


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    system = TrainingAssessmentSystem(project_root)

    # 示例：生成评估模板
    template = system.generate_assessment_template()

    # 保存模板
    system.save_assessment_plan(template, "assessment_template.json")

    print("✅ 培训评估系统已初始化")
    print("📋 评估模板已生成")


if __name__ == "__main__":
    main()
