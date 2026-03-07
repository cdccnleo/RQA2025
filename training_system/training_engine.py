#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2026 用户培训和文档系统
提供全面的培训课程、交互式学习和专业文档管理

培训特性:
1. 交互式培训模块 - 分层学习路径和进度跟踪
2. 多媒体教学内容 - 视频、演示、互动练习
3. 个性化学习推荐 - 基于用户水平和兴趣的定制学习
4. 评估和认证系统 - 知识考核和技能认证
5. 文档管理系统 - 智能搜索和版本控制
6. 帮助和支持中心 - 实时帮助和社区支持
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys
import random
import hashlib
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

class TrainingEngine:
    """培训引擎"""

    def __init__(self):
        self.courses = {}
        self.users = {}
        self.progress = {}
        self.assessments = {}
        self.certificates = {}

        # 培训内容库
        self.content_library = {
            'beginner': [],
            'intermediate': [],
            'advanced': [],
            'expert': []
        }

        # 学习路径
        self.learning_paths = {}

        self.load_training_content()

    def load_training_content(self):
        """加载培训内容"""
        # 基础课程
        self.courses['quantum_basics'] = {
            'id': 'quantum_basics',
            'title': '量子计算基础',
            'level': 'beginner',
            'duration_hours': 4,
            'modules': [
                {'title': '量子力学基础概念', 'type': 'video', 'duration': 45},
                {'title': '量子比特和量子门', 'type': 'interactive', 'duration': 60},
                {'title': '量子电路设计', 'type': 'demo', 'duration': 90},
                {'title': 'QAOA算法介绍', 'type': 'presentation', 'duration': 30}
            ],
            'prerequisites': [],
            'objectives': [
                '理解量子计算基本原理',
                '掌握量子比特操作',
                '学会设计简单量子电路',
                '了解QAOA算法应用'
            ],
            'assessment': {
                'type': 'quiz',
                'questions': 20,
                'passing_score': 80
            }
        }

        self.courses['ai_fundamentals'] = {
            'id': 'ai_fundamentals',
            'title': 'AI深度学习基础',
            'level': 'beginner',
            'duration_hours': 6,
            'modules': [
                {'title': '神经网络基础', 'type': 'video', 'duration': 60},
                {'title': '深度学习架构', 'type': 'interactive', 'duration': 90},
                {'title': '多模态学习', 'type': 'demo', 'duration': 120},
                {'title': '注意力机制', 'type': 'presentation', 'duration': 45}
            ],
            'prerequisites': [],
            'objectives': [
                '理解神经网络工作原理',
                '掌握深度学习架构',
                '学习多模态数据处理',
                '了解注意力机制应用'
            ],
            'assessment': {
                'type': 'project',
                'description': '构建简单多模态分类器',
                'evaluation_criteria': ['accuracy', 'efficiency', 'code_quality']
            }
        }

        self.courses['bci_introduction'] = {
            'id': 'bci_introduction',
            'title': '脑机接口入门',
            'level': 'intermediate',
            'duration_hours': 5,
            'modules': [
                {'title': '神经信号基础', 'type': 'video', 'duration': 50},
                {'title': '信号处理技术', 'type': 'interactive', 'duration': 75},
                {'title': '意识状态计算', 'type': 'demo', 'duration': 100},
                {'title': 'BCI应用案例', 'type': 'case_study', 'duration': 40}
            ],
            'prerequisites': ['quantum_basics'],
            'objectives': [
                '理解神经信号特征',
                '掌握信号处理算法',
                '学习意识状态计算',
                '了解BCI实际应用'
            ],
            'assessment': {
                'type': 'lab',
                'description': '模拟EEG信号处理实验',
                'passing_score': 85
            }
        }

        self.courses['fusion_engineering'] = {
            'id': 'fusion_engineering',
            'title': '创新引擎融合工程',
            'level': 'advanced',
            'duration_hours': 8,
            'modules': [
                {'title': '多引擎协同架构', 'type': 'presentation', 'duration': 60},
                {'title': '资源调度优化', 'type': 'interactive', 'duration': 120},
                {'title': '性能监控与调优', 'type': 'demo', 'duration': 150},
                {'title': '故障恢复机制', 'type': 'workshop', 'duration': 90}
            ],
            'prerequisites': ['ai_fundamentals', 'bci_introduction'],
            'objectives': [
                '掌握多引擎协同原理',
                '学习资源调度策略',
                '掌握性能优化技术',
                '理解故障恢复机制'
            ],
            'assessment': {
                'type': 'comprehensive',
                'components': ['theory_exam', 'practical_project', 'presentation'],
                'passing_score': 90
            }
        }

        # 学习路径
        self.learning_paths['quantum_specialist'] = {
            'name': '量子计算专家',
            'description': '成为量子计算和算法专家',
            'courses': ['quantum_basics', 'quantum_advanced', 'quantum_applications'],
            'duration_months': 6,
            'certification': 'Quantum Computing Specialist'
        }

        self.learning_paths['ai_engineer'] = {
            'name': 'AI系统工程师',
            'description': '掌握AI深度集成系统开发',
            'courses': ['ai_fundamentals', 'ai_advanced', 'multimodal_systems'],
            'duration_months': 8,
            'certification': 'AI Systems Engineer'
        }

        self.learning_paths['bci_researcher'] = {
            'name': '脑机接口研究员',
            'description': '成为脑机接口技术专家',
            'courses': ['bci_introduction', 'neural_signal_processing', 'consciousness_computing'],
            'duration_months': 7,
            'certification': 'BCI Research Specialist'
        }

        self.learning_paths['fusion_architect'] = {
            'name': '融合架构师',
            'description': '掌握三大创新引擎融合架构',
            'courses': ['fusion_engineering', 'system_integration', 'enterprise_deployment'],
            'duration_months': 9,
            'certification': 'Fusion Architecture Master'
        }

    def enroll_user(self, user_id, user_profile):
        """用户注册"""
        self.users[user_id] = {
            'profile': user_profile,
            'enrolled_courses': [],
            'completed_courses': [],
            'current_learning_path': None,
            'progress': {},
            'achievements': [],
            'enrollment_date': datetime.now().isoformat()
        }

        # 推荐学习路径
        recommended_path = self.recommend_learning_path(user_profile)
        if recommended_path:
            self.assign_learning_path(user_id, recommended_path)

        return self.users[user_id]

    def recommend_learning_path(self, user_profile):
        """推荐学习路径"""
        role = user_profile.get('role', 'general')
        experience = user_profile.get('experience_years', 0)
        interests = user_profile.get('interests', [])

        # 基于角色和经验推荐
        if role == 'quantum_physicist' or 'quantum' in interests:
            return 'quantum_specialist'
        elif role == 'ai_engineer' or 'ai' in interests:
            return 'ai_engineer'
        elif role == 'neuroscientist' or 'bci' in interests:
            return 'bci_researcher'
        elif experience > 5 and role in ['architect', 'lead_engineer']:
            return 'fusion_architect'
        else:
            # 默认推荐最受欢迎的路径
            return 'ai_engineer'

    def assign_learning_path(self, user_id, path_id):
        """分配学习路径"""
        if user_id not in self.users:
            return False

        if path_id not in self.learning_paths:
            return False

        self.users[user_id]['current_learning_path'] = path_id

        # 自动注册路径中的课程
        path = self.learning_paths[path_id]
        for course_id in path['courses']:
            if course_id in self.courses:
                self.enroll_course(user_id, course_id)

        return True

    def enroll_course(self, user_id, course_id):
        """注册课程"""
        if user_id not in self.users or course_id not in self.courses:
            return False

        user = self.users[user_id]
        course = self.courses[course_id]

        # 检查先修课程
        for prereq in course['prerequisites']:
            if prereq not in user['completed_courses']:
                return False  # 未满足先修条件

        if course_id not in user['enrolled_courses']:
            user['enrolled_courses'].append(course_id)

            # 初始化进度
            self.progress[f"{user_id}_{course_id}"] = {
                'completed_modules': [],
                'current_module': 0,
                'time_spent': 0,
                'score': 0,
                'status': 'enrolled',
                'start_date': datetime.now().isoformat()
            }

        return True

    def update_progress(self, user_id, course_id, module_index, time_spent, completed=False):
        """更新学习进度"""
        progress_key = f"{user_id}_{course_id}"

        if progress_key not in self.progress:
            return False

        progress = self.progress[progress_key]
        progress['time_spent'] += time_spent

        if completed and module_index not in progress['completed_modules']:
            progress['completed_modules'].append(module_index)
            progress['current_module'] = module_index + 1

        # 检查课程完成
        course = self.courses[course_id]
        if len(progress['completed_modules']) == len(course['modules']):
            progress['status'] = 'completed'
            progress['completion_date'] = datetime.now().isoformat()

            # 添加到用户完成课程
            self.users[user_id]['completed_courses'].append(course_id)

            # 检查是否完成学习路径
            self.check_learning_path_completion(user_id)

        return True

    def check_learning_path_completion(self, user_id):
        """检查学习路径完成"""
        user = self.users[user_id]
        current_path = user['current_learning_path']

        if not current_path:
            return

        path = self.learning_paths[current_path]
        required_courses = set(path['courses'])
        completed_courses = set(user['completed_courses'])

        if required_courses.issubset(completed_courses):
            # 学习路径完成
            user['achievements'].append({
                'type': 'learning_path_completed',
                'path_id': current_path,
                'path_name': path['name'],
                'certification': path['certification'],
                'completion_date': datetime.now().isoformat()
            })

            # 生成证书
            certificate_id = self.generate_certificate(user_id, current_path)
            user['certificates'] = user.get('certificates', [])
            user['certificates'].append(certificate_id)

    def generate_certificate(self, user_id, path_id):
        """生成证书"""
        certificate_id = f"cert_{user_id}_{path_id}_{int(datetime.now().timestamp())}"

        path = self.learning_paths[path_id]
        user = self.users[user_id]

        self.certificates[certificate_id] = {
            'certificate_id': certificate_id,
            'user_id': user_id,
            'user_name': user['profile'].get('name', 'Unknown'),
            'certification': path['certification'],
            'path_name': path['name'],
            'issue_date': datetime.now().isoformat(),
            'expiry_date': (datetime.now() + timedelta(days=365*2)).isoformat(),  # 2年有效期
            'verification_code': hashlib.sha256(f"{certificate_id}{user_id}".encode()).hexdigest()[:16]
        }

        return certificate_id

    def assess_user(self, user_id, course_id, assessment_data):
        """评估用户"""
        if course_id not in self.courses:
            return False

        course = self.courses[course_id]
        assessment_config = course['assessment']

        assessment_result = {
            'user_id': user_id,
            'course_id': course_id,
            'assessment_type': assessment_config['type'],
            'submitted_at': datetime.now().isoformat(),
            'data': assessment_data
        }

        # 计算分数
        if assessment_config['type'] == 'quiz':
            correct_answers = assessment_data.get('correct_answers', 0)
            total_questions = assessment_data.get('total_questions', 1)
            score = (correct_answers / total_questions) * 100
            assessment_result['score'] = score
            assessment_result['passed'] = score >= assessment_config['passing_score']

        elif assessment_config['type'] == 'project':
            # 项目评估
            evaluation = assessment_data.get('evaluation', {})
            avg_score = sum(evaluation.values()) / len(evaluation) if evaluation else 0
            assessment_result['score'] = avg_score
            assessment_result['passed'] = avg_score >= 80  # 默认80分通过

        elif assessment_config['type'] == 'lab':
            assessment_result['score'] = assessment_data.get('score', 0)
            assessment_result['passed'] = assessment_result['score'] >= assessment_config['passing_score']

        elif assessment_config['type'] == 'comprehensive':
            components = assessment_data.get('components', {})
            total_score = 0
            for component in components.values():
                if isinstance(component, dict):
                    total_score += component.get('score', 0)
                else:
                    total_score += component

            avg_score = total_score / len(components) if components else 0
            assessment_result['score'] = avg_score
            assessment_result['passed'] = avg_score >= assessment_config['passing_score']

        # 存储评估结果
        assessment_key = f"{user_id}_{course_id}"
        self.assessments[assessment_key] = assessment_result

        # 更新用户进度
        if assessment_result.get('passed', False):
            progress_key = f"{user_id}_{course_id}"
            if progress_key in self.progress:
                self.progress[progress_key]['assessment_score'] = assessment_result['score']
                self.progress[progress_key]['assessment_passed'] = True

        return assessment_result

    def get_user_dashboard(self, user_id):
        """获取用户学习仪表板"""
        if user_id not in self.users:
            return None

        user = self.users[user_id]
        dashboard = {
            'user_profile': user['profile'],
            'current_learning_path': user.get('current_learning_path'),
            'enrolled_courses': [],
            'completed_courses': [],
            'achievements': user.get('achievements', []),
            'certificates': user.get('certificates', []),
            'learning_stats': self.calculate_learning_stats(user_id)
        }

        # 课程详情
        for course_id in user['enrolled_courses']:
            if course_id in self.courses:
                course_info = self.courses[course_id].copy()
                progress_key = f"{user_id}_{course_id}"
                if progress_key in self.progress:
                    course_info['progress'] = self.progress[progress_key]
                dashboard['enrolled_courses'].append(course_info)

        for course_id in user['completed_courses']:
            if course_id in self.courses:
                course_info = self.courses[course_id].copy()
                progress_key = f"{user_id}_{course_id}"
                if progress_key in self.progress:
                    course_info['progress'] = self.progress[progress_key]
                dashboard['completed_courses'].append(course_info)

        return dashboard

    def calculate_learning_stats(self, user_id):
        """计算学习统计"""
        user = self.users[user_id]

        total_courses = len(user['enrolled_courses'])
        completed_courses = len(user['completed_courses'])
        completion_rate = (completed_courses / total_courses * 100) if total_courses > 0 else 0

        total_time = 0
        for course_id in user['enrolled_courses']:
            progress_key = f"{user_id}_{course_id}"
            if progress_key in self.progress:
                total_time += self.progress[progress_key].get('time_spent', 0)

        return {
            'total_courses_enrolled': total_courses,
            'courses_completed': completed_courses,
            'completion_rate': round(completion_rate, 1),
            'total_learning_time_hours': round(total_time / 3600, 1),
            'certifications_earned': len(user.get('certificates', [])),
            'achievements_unlocked': len(user.get('achievements', []))
        }

    def get_course_content(self, course_id, module_index=None):
        """获取课程内容"""
        if course_id not in self.courses:
            return None

        course = self.courses[course_id]

        if module_index is not None:
            if 0 <= module_index < len(course['modules']):
                return course['modules'][module_index]
            else:
                return None

        return course

    def search_content(self, query, filters=None):
        """搜索培训内容"""
        results = []

        # 搜索课程
        for course_id, course in self.courses.items():
            if query.lower() in course['title'].lower() or query.lower() in course.get('description', '').lower():
                if self._matches_filters(course, filters):
                    results.append({
                        'type': 'course',
                        'id': course_id,
                        'title': course['title'],
                        'level': course['level'],
                        'duration': course['duration_hours']
                    })

        # 搜索学习路径
        for path_id, path in self.learning_paths.items():
            if query.lower() in path['name'].lower() or query.lower() in path['description'].lower():
                results.append({
                    'type': 'learning_path',
                    'id': path_id,
                    'title': path['name'],
                    'description': path['description'],
                    'duration_months': path['duration_months']
                })

        return results

    def _matches_filters(self, course, filters):
        """检查是否匹配过滤条件"""
        if not filters:
            return True

        if 'level' in filters and course.get('level') != filters['level']:
            return False

        if 'duration_max' in filters and course.get('duration_hours', 0) > filters['duration_max']:
            return False

        if 'prerequisites' in filters and not set(course.get('prerequisites', [])).issubset(set(filters['prerequisites'])):
            return False

        return True


class DocumentationEngine:
    """文档引擎"""

    def __init__(self):
        self.documents = {}
        self.search_index = {}
        self.versions = defaultdict(list)
        self.categories = {
            'api_reference': 'API参考文档',
            'user_guide': '用户指南',
            'developer_guide': '开发者指南',
            'deployment_guide': '部署指南',
            'troubleshooting': '故障排除',
            'best_practices': '最佳实践',
            'release_notes': '发布说明'
        }

    def add_document(self, doc_id, document):
        """添加文档"""
        document['id'] = doc_id
        document['created_at'] = datetime.now().isoformat()
        document['version'] = document.get('version', '1.0')
        document['last_updated'] = document['created_at']

        self.documents[doc_id] = document

        # 添加到版本历史
        self.versions[doc_id].append({
            'version': document['version'],
            'updated_at': document['created_at'],
            'changes': document.get('changes', '初始版本')
        })

        # 更新搜索索引
        self._update_search_index(doc_id, document)

        return doc_id

    def update_document(self, doc_id, updates):
        """更新文档"""
        if doc_id not in self.documents:
            return False

        document = self.documents[doc_id]
        old_version = document['version']

        # 计算新版本号
        version_parts = old_version.split('.')
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        new_version = '.'.join(version_parts)

        # 应用更新
        document.update(updates)
        document['version'] = new_version
        document['last_updated'] = datetime.now().isoformat()

        # 添加版本历史
        self.versions[doc_id].append({
            'version': new_version,
            'updated_at': document['last_updated'],
            'changes': updates.get('changes', '内容更新')
        })

        # 更新搜索索引
        self._update_search_index(doc_id, document)

        return True

    def search_documents(self, query, category=None, tags=None):
        """搜索文档"""
        results = []

        for doc_id, document in self.documents.items():
            if category and document.get('category') != category:
                continue

            if tags:
                doc_tags = set(document.get('tags', []))
                if not set(tags).issubset(doc_tags):
                    continue

            # 检查标题和内容
            searchable_text = f"{document.get('title', '')} {document.get('content', '')} {document.get('summary', '')}"
            if query.lower() in searchable_text.lower():
                results.append({
                    'id': doc_id,
                    'title': document.get('title'),
                    'category': document.get('category'),
                    'summary': document.get('summary'),
                    'tags': document.get('tags', []),
                    'version': document.get('version'),
                    'last_updated': document.get('last_updated')
                })

        # 按相关性排序 (简单实现)
        results.sort(key=lambda x: len(query) / len(x['title']) if query in x['title'] else 0, reverse=True)

        return results

    def get_document(self, doc_id, version=None):
        """获取文档"""
        if doc_id not in self.documents:
            return None

        document = self.documents[doc_id]

        if version:
            # 获取指定版本
            for doc_version in self.versions[doc_id]:
                if doc_version['version'] == version:
                    # 这里需要实现版本控制逻辑
                    return document  # 简化实现

        return document

    def get_document_versions(self, doc_id):
        """获取文档版本历史"""
        return self.versions.get(doc_id, [])

    def _update_search_index(self, doc_id, document):
        """更新搜索索引"""
        # 简化实现 - 实际应该使用更复杂的索引
        title = document.get('title', '').lower()
        content = document.get('content', '').lower()

        self.search_index[doc_id] = {
            'title': title,
            'content': content,
            'category': document.get('category'),
            'tags': document.get('tags', [])
        }

    def get_related_documents(self, doc_id, limit=5):
        """获取相关文档"""
        if doc_id not in self.documents:
            return []

        document = self.documents[doc_id]
        doc_tags = set(document.get('tags', []))
        doc_category = document.get('category')

        related = []

        for other_id, other_doc in self.documents.items():
            if other_id == doc_id:
                continue

            # 计算相关性分数
            score = 0

            # 相同类别
            if other_doc.get('category') == doc_category:
                score += 2

            # 共同标签
            other_tags = set(other_doc.get('tags', []))
            common_tags = len(doc_tags.intersection(other_tags))
            score += common_tags

            if score > 0:
                related.append({
                    'id': other_id,
                    'title': other_doc.get('title'),
                    'score': score,
                    'category': other_doc.get('category')
                })

        # 按分数排序
        related.sort(key=lambda x: x['score'], reverse=True)

        return related[:limit]


def create_training_system():
    """创建培训系统"""
    training_engine = TrainingEngine()
    doc_engine = DocumentationEngine()

    # 添加示例用户
    training_engine.enroll_user('user_001', {
        'name': '张三',
        'role': 'ai_engineer',
        'experience_years': 3,
        'interests': ['ai', 'machine_learning'],
        'department': 'AI研发部'
    })

    training_engine.enroll_user('user_002', {
        'name': '李四',
        'role': 'quantum_physicist',
        'experience_years': 5,
        'interests': ['quantum', 'physics'],
        'department': '量子计算实验室'
    })

    # 添加示例文档
    doc_engine.add_document('api_overview', {
        'title': 'RQA2026 API 概览',
        'category': 'api_reference',
        'summary': 'RQA2026系统API的完整参考指南',
        'content': '详细介绍所有API端点、参数和使用示例...',
        'tags': ['api', 'reference', 'guide']
    })

    doc_engine.add_document('deployment_guide', {
        'title': '生产环境部署指南',
        'category': 'deployment_guide',
        'summary': '详细的生产环境部署和配置指南',
        'content': '包含Docker、Kubernetes、监控配置等...',
        'tags': ['deployment', 'production', 'kubernetes']
    })

    return {
        'training_engine': training_engine,
        'documentation_engine': doc_engine
    }


def main():
    """主函数"""
    print("🎓 启动 RQA2026 用户培训和文档系统")
    print("=" * 80)

    # 创建培训系统
    system = create_training_system()
    training_engine = system['training_engine']
    doc_engine = system['documentation_engine']

    print("✅ 培训系统初始化完成")
    print(f"   📚 课程数量: {len(training_engine.courses)}")
    print(f"   🛤️  学习路径: {len(training_engine.learning_paths)}")
    print(f"   👥 注册用户: {len(training_engine.users)}")

    print("\\n✅ 文档系统初始化完成")
    print(f"   📄 文档数量: {len(doc_engine.documents)}")
    print(f"   🏷️  分类数量: {len(doc_engine.categories)}")

    # 演示培训功能
    print("\\n🎯 培训系统演示:")

    # 用户学习路径推荐
    user_id = 'user_001'
    dashboard = training_engine.get_user_dashboard(user_id)
    if dashboard:
        print(f"👤 用户 {dashboard['user_profile']['name']} 学习仪表板:")
        print(f"   📊 注册课程: {len(dashboard['enrolled_courses'])}")
        print(f"   ✅ 完成课程: {len(dashboard['completed_courses'])}")
        print(f"   🎯 当前学习路径: {dashboard['current_learning_path']}")
        print(f"   📈 完成率: {dashboard['learning_stats']['completion_rate']}%")
        print(f"   🏆 获得认证: {dashboard['learning_stats']['certifications_earned']}")

    # 课程进度更新演示
    print("\\n📚 课程学习演示:")
    course_id = 'ai_fundamentals'
    course = training_engine.get_course_content(course_id)
    if course:
        print(f"🎓 课程: {course['title']}")
        print(f"   📖 模块数量: {len(course['modules'])}")
        print(f"   🎯 学习目标: {len(course['objectives'])}")

        # 模拟学习进度
        for i in range(len(course['modules'])):
            training_engine.update_progress(user_id, course_id, i, 1800, completed=True)  # 30分钟
            print(f"   ✅ 完成模块 {i+1}: {course['modules'][i]['title']}")

    # 重新获取仪表板
    dashboard = training_engine.get_user_dashboard(user_id)
    if dashboard:
        print(f"\\n📊 更新后统计:")
        print(f"   ✅ 完成课程: {len(dashboard['completed_courses'])}")
        print(f"   🏆 成就解锁: {len(dashboard['achievements'])}")

    # 文档搜索演示
    print("\\n📄 文档系统演示:")
    search_results = doc_engine.search_documents('API')
    print(f"🔍 搜索 'API' 找到 {len(search_results)} 个文档:")
    for result in search_results[:3]:
        print(f"   📄 {result['title']} ({result['category']})")

    # 相关文档推荐
    if search_results:
        related = doc_engine.get_related_documents(search_results[0]['id'])
        if related:
            print(f"\\n💡 相关文档推荐 ({len(related)} 个):")
            for doc in related:
                print(f"   📋 {doc['title']} (相关度: {doc['score']})")

    print("\\n✅ 用户培训和文档系统运行完成！")
    print("🎓 系统现已就绪，可提供全面的培训和文档服务")


if __name__ == "__main__":
    main()
