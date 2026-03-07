"""
经验推广与知识分享系统
提供项目经验总结、最佳实践推广、培训组织和知识传承机制
"""

import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperienceLesson:
    """经验教训"""
    lesson_id: str
    category: str  # 'success', 'challenge', 'failure', 'innovation'
    title: str
    description: str
    context: str
    impact: str
    root_cause: Optional[str] = None
    solution: Optional[str] = None
    prevention_measures: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    validated_by: Optional[str] = None


@dataclass
class BestPracticeShowcase:
    """最佳实践展示"""
    showcase_id: str
    practice_title: str
    practice_summary: str
    implementation_details: str
    benefits_achieved: List[str]
    success_metrics: List[str]
    replicability_score: int  # 1-10, 可复制性评分
    presenter: str
    presentation_date: datetime
    audience_size: int
    feedback_score: Optional[float] = None
    materials_url: Optional[str] = None


@dataclass
class KnowledgeSharingSession:
    """知识分享会议"""
    session_id: str
    title: str
    description: str
    presenter: str
    session_type: str  # 'workshop', 'webinar', 'presentation', 'panel'
    target_audience: List[str]
    scheduled_date: datetime
    duration_hours: float
    max_participants: int
    registered_participants: List[str] = field(default_factory=list)
    status: str = 'scheduled'  # 'scheduled', 'in_progress', 'completed', 'cancelled'
    materials: List[str] = field(default_factory=list)
    feedback: Dict[str, Any] = field(default_factory=dict)
    outcomes: List[str] = field(default_factory=list)


@dataclass
class CommunityEngagement:
    """社区参与"""
    engagement_id: str
    platform: str  # 'internal_portal', 'external_blog', 'conference', 'social_media'
    title: str
    content_summary: str
    target_audience: str
    engagement_metrics: Dict[str, int] = field(default_factory=dict)  # views, likes, shares, comments
    impact_assessment: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


class ExperienceSharingSystem:
    """经验分享系统"""

    def __init__(self):
        self.lessons = {}
        self.showcases = {}
        self.sharing_sessions = {}
        self.community_engagements = {}

    def document_experience_lesson(self, category: str, title: str, description: str,
                                 context: str, impact: str, author: str,
                                 root_cause: str = None, solution: str = None,
                                 prevention_measures: List[str] = None,
                                 tags: List[str] = None) -> str:
        """记录经验教训"""
        lesson_id = f"lesson_{int(time.time())}"

        lesson = ExperienceLesson(
            lesson_id=lesson_id,
            category=category,
            title=title,
            description=description,
            context=context,
            impact=impact,
            root_cause=root_cause,
            solution=solution,
            prevention_measures=prevention_measures or [],
            tags=tags or [],
            author=author
        )

        self.lessons[lesson_id] = lesson
        print(f"📝 记录经验教训: {title} ({lesson_id})")
        return lesson_id

    def create_best_practice_showcase(self, practice_title: str, practice_summary: str,
                                    implementation_details: str, benefits_achieved: List[str],
                                    success_metrics: List[str], replicability_score: int,
                                    presenter: str, presentation_date: datetime,
                                    audience_size: int) -> str:
        """创建最佳实践展示"""
        showcase_id = f"showcase_{int(time.time())}"

        showcase = BestPracticeShowcase(
            showcase_id=showcase_id,
            practice_title=practice_title,
            practice_summary=practice_summary,
            implementation_details=implementation_details,
            benefits_achieved=benefits_achieved,
            success_metrics=success_metrics,
            replicability_score=replicability_score,
            presenter=presenter,
            presentation_date=presentation_date,
            audience_size=audience_size
        )

        self.showcases[showcase_id] = showcase
        print(f"⭐ 创建最佳实践展示: {practice_title} ({showcase_id})")
        return showcase_id

    def schedule_knowledge_sharing_session(self, title: str, description: str,
                                         presenter: str, session_type: str,
                                         target_audience: List[str],
                                         scheduled_date: datetime,
                                         duration_hours: float,
                                         max_participants: int) -> str:
        """安排知识分享会议"""
        session_id = f"session_{int(time.time())}"

        session = KnowledgeSharingSession(
            session_id=session_id,
            title=title,
            description=description,
            presenter=presenter,
            session_type=session_type,
            target_audience=target_audience,
            scheduled_date=scheduled_date,
            duration_hours=duration_hours,
            max_participants=max_participants
        )

        self.sharing_sessions[session_id] = session
        print(f"📅 安排知识分享会议: {title} - {scheduled_date.strftime('%Y-%m-%d %H:%M')}")
        return session_id

    def register_for_session(self, session_id: str, participant: str) -> bool:
        """注册参加会议"""
        if session_id not in self.sharing_sessions:
            return False

        session = self.sharing_sessions[session_id]

        if len(session.registered_participants) >= session.max_participants:
            return False

        if participant not in session.registered_participants:
            session.registered_participants.append(participant)

        print(f"✅ 会议注册成功: {participant} -> {session.title}")
        return True

    def complete_sharing_session(self, session_id: str, materials: List[str] = None,
                               feedback: Dict[str, Any] = None,
                               outcomes: List[str] = None):
        """完成分享会议"""
        if session_id not in self.sharing_sessions:
            raise ValueError(f"会议不存在: {session_id}")

        session = self.sharing_sessions[session_id]
        session.status = 'completed'

        if materials:
            session.materials.extend(materials)

        if feedback:
            session.feedback = feedback

        if outcomes:
            session.outcomes.extend(outcomes)

        print(f"✅ 分享会议完成: {session.title} - 参与者: {len(session.registered_participants)}")

    def create_community_engagement(self, platform: str, title: str,
                                  content_summary: str, target_audience: str) -> str:
        """创建社区参与活动"""
        engagement_id = f"engagement_{int(time.time())}"

        engagement = CommunityEngagement(
            engagement_id=engagement_id,
            platform=platform,
            title=title,
            content_summary=content_summary,
            target_audience=target_audience
        )

        self.community_engagements[engagement_id] = engagement
        print(f"🌐 创建社区参与: {title} ({platform})")
        return engagement_id

    def update_engagement_metrics(self, engagement_id: str, metrics: Dict[str, int]):
        """更新参与度量"""
        if engagement_id not in self.community_engagements:
            return False

        engagement = self.community_engagements[engagement_id]
        engagement.engagement_metrics.update(metrics)

        print(f"📊 更新参与度量: {engagement.title} - {metrics}")
        return True

    def generate_experience_report(self) -> Dict[str, Any]:
        """生成经验报告"""
        report = {
            'summary': {
                'total_lessons': len(self.lessons),
                'total_showcases': len(self.showcases),
                'total_sessions': len(self.sharing_sessions),
                'total_engagements': len(self.community_engagements),
                'completed_sessions': len([s for s in self.sharing_sessions.values() if s.status == 'completed']),
                'total_participants': sum(len(s.registered_participants) for s in self.sharing_sessions.values())
            },
            'lessons_by_category': self._categorize_lessons(),
            'showcase_highlights': self._get_showcase_highlights(),
            'session_analytics': self._analyze_sessions(),
            'community_impact': self._assess_community_impact(),
            'recommendations': self._generate_sharing_recommendations()
        }

        return report

    def _categorize_lessons(self) -> Dict[str, List[Dict[str, Any]]]:
        """按类别整理经验教训"""
        categories = {}

        for lesson in self.lessons.values():
            if lesson.category not in categories:
                categories[lesson.category] = []

            categories[lesson.category].append({
                'lesson_id': lesson.lesson_id,
                'title': lesson.title,
                'impact': lesson.impact,
                'tags': lesson.tags,
                'author': lesson.author
            })

        return categories

    def _get_showcase_highlights(self) -> List[Dict[str, Any]]:
        """获取展示亮点"""
        highlights = []

        for showcase in self.showcases.values():
            highlights.append({
                'showcase_id': showcase.showcase_id,
                'title': showcase.practice_title,
                'replicability_score': showcase.replicability_score,
                'audience_size': showcase.audience_size,
                'feedback_score': showcase.feedback_score,
                'benefits': showcase.benefits_achieved
            })

        # 按可复制性评分排序
        highlights.sort(key=lambda x: x['replicability_score'], reverse=True)
        return highlights[:10]  # 返回前10个

    def _analyze_sessions(self) -> Dict[str, Any]:
        """分析会议数据"""
        completed_sessions = [s for s in self.sharing_sessions.values() if s.status == 'completed']

        if not completed_sessions:
            return {'total_completed': 0}

        total_participants = sum(len(s.registered_participants) for s in completed_sessions)
        avg_participants = total_participants / len(completed_sessions)

        session_types = {}
        for session in completed_sessions:
            session_type = session.session_type
            if session_type not in session_types:
                session_types[session_type] = 0
            session_types[session_type] += 1

        return {
            'total_completed': len(completed_sessions),
            'total_participants': total_participants,
            'average_participants': avg_participants,
            'session_types': session_types,
            'most_popular_type': max(session_types.items(), key=lambda x: x[1])[0] if session_types else None
        }

    def _assess_community_impact(self) -> Dict[str, Any]:
        """评估社区影响"""
        total_views = sum(e.engagement_metrics.get('views', 0) for e in self.community_engagements.values())
        total_likes = sum(e.engagement_metrics.get('likes', 0) for e in self.community_engagements.values())
        total_shares = sum(e.engagement_metrics.get('shares', 0) for e in self.community_engagements.values())
        total_comments = sum(e.engagement_metrics.get('comments', 0) for e in self.community_engagements.values())

        platforms = {}
        for engagement in self.community_engagements.values():
            platform = engagement.platform
            if platform not in platforms:
                platforms[platform] = 0
            platforms[platform] += 1

        return {
            'total_engagements': len(self.community_engagements),
            'total_views': total_views,
            'total_likes': total_likes,
            'total_shares': total_shares,
            'total_comments': total_comments,
            'engagement_rate': (total_likes + total_shares + total_comments) / total_views if total_views > 0 else 0,
            'platforms': platforms
        }

    def _generate_sharing_recommendations(self) -> List[str]:
        """生成分享建议"""
        recommendations = []

        # 基于经验教训分析
        success_count = len([l for l in self.lessons.values() if l.category == 'success'])
        challenge_count = len([l for l in self.lessons.values() if l.category == 'challenge'])

        if success_count > challenge_count:
            recommendations.append("🎉 项目成功经验丰富，建议重点推广成功实践")
        else:
            recommendations.append("⚠️ 项目挑战较多，建议深入分析问题根因并制定改进措施")

        # 基于会议分析
        session_analysis = self._analyze_sessions()
        if session_analysis.get('total_completed', 0) > 0:
            avg_participants = session_analysis.get('average_participants', 0)
            if avg_participants > 20:
                recommendations.append("👥 会议参与度高，建议扩大会议规模和频率")
            elif avg_participants < 10:
                recommendations.append("📈 会议参与度有待提升，建议改进会议主题和推广方式")

        # 基于社区影响
        community_impact = self._assess_community_impact()
        engagement_rate = community_impact.get('engagement_rate', 0)
        if engagement_rate > 0.1:
            recommendations.append("🌟 社区参与度良好，建议加强内容创作和发布频率")
        else:
            recommendations.append("📢 社区影响力有限，建议改进内容质量和传播策略")

        # 通用建议
        recommendations.extend([
            "📚 建立经验教训数据库，便于后续项目参考",
            "🎓 开发培训课程体系，系统化知识传承",
            "🤝 加强跨团队协作，促进经验共享",
            "🔄 建立持续改进机制，定期回顾和优化"
        ])

        return recommendations

    def search_experiences(self, query: str = "", category: str = None,
                          tags: List[str] = None) -> List[Dict[str, Any]]:
        """搜索经验"""
        results = []

        # 搜索经验教训
        for lesson in self.lessons.values():
            if self._matches_lesson_criteria(lesson, query, category, tags):
                results.append({
                    'type': 'lesson',
                    'id': lesson.lesson_id,
                    'title': lesson.title,
                    'category': lesson.category,
                    'impact': lesson.impact,
                    'tags': lesson.tags,
                    'author': lesson.author
                })

        # 搜索最佳实践展示
        for showcase in self.showcases.values():
            if query.lower() in showcase.practice_title.lower() or query.lower() in showcase.practice_summary.lower():
                results.append({
                    'type': 'showcase',
                    'id': showcase.showcase_id,
                    'title': showcase.practice_title,
                    'replicability_score': showcase.replicability_score,
                    'presenter': showcase.presenter
                })

        return results

    def _matches_lesson_criteria(self, lesson: ExperienceLesson, query: str,
                               category: str = None, tags: List[str] = None) -> bool:
        """检查经验教训是否匹配搜索条件"""
        query_match = not query or (query.lower() in lesson.title.lower() or
                                   query.lower() in lesson.description.lower())

        category_match = category is None or lesson.category == category

        tags_match = tags is None or any(tag in lesson.tags for tag in tags)

        return query_match and category_match and tags_match

    def export_sharing_materials(self, output_path: str = "./sharing_materials"):
        """导出分享材料"""
        import os
        os.makedirs(output_path, exist_ok=True)

        # 导出经验教训
        lessons_data = []
        for lesson in self.lessons.values():
            lessons_data.append({
                'lesson_id': lesson.lesson_id,
                'category': lesson.category,
                'title': lesson.title,
                'description': lesson.description,
                'context': lesson.context,
                'impact': lesson.impact,
                'solution': lesson.solution,
                'prevention_measures': lesson.prevention_measures,
                'tags': lesson.tags,
                'author': lesson.author
            })

        with open(f"{output_path}/lessons.json", 'w', encoding='utf-8') as f:
            json.dump(lessons_data, f, indent=2, ensure_ascii=False)

        # 导出最佳实践
        showcases_data = []
        for showcase in self.showcases.values():
            showcases_data.append({
                'showcase_id': showcase.showcase_id,
                'practice_title': showcase.practice_title,
                'practice_summary': showcase.practice_summary,
                'benefits_achieved': showcase.benefits_achieved,
                'success_metrics': showcase.success_metrics,
                'replicability_score': showcase.replicability_score,
                'presenter': showcase.presenter,
                'audience_size': showcase.audience_size
            })

        with open(f"{output_path}/showcases.json", 'w', encoding='utf-8') as f:
            json.dump(showcases_data, f, indent=2, ensure_ascii=False)

        # 导出会议记录
        sessions_data = []
        for session in self.sharing_sessions.values():
            sessions_data.append({
                'session_id': session.session_id,
                'title': session.title,
                'presenter': session.presenter,
                'session_type': session.session_type,
                'registered_participants': session.registered_participants,
                'status': session.status,
                'feedback': session.feedback,
                'outcomes': session.outcomes
            })

        with open(f"{output_path}/sessions.json", 'w', encoding='utf-8') as f:
            json.dump(sessions_data, f, indent=2, ensure_ascii=False)

        print(f"📤 分享材料已导出到: {output_path}")
        return output_path
