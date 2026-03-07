from typing import Dict, List, Any
from optimizations.short_term_optimizations import *


class UserFeedbackCollector(BaseComponent):
    """用户反馈收集器"""

    def __init__(self, feedback_dir: str = "data/feedback"):

        super().__init__("UserFeedbackCollector")
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "feedback.json"
        self.feedback: List[FeedbackItem] = []
        self._load_feedback()

        logger.info("用户反馈收集器初始化完成")

    def _load_feedback(self):
        """加载已有反馈"""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, "r", encoding="utf - 8") as f:
                    data = json.load(f)
                    self.feedback = [FeedbackItem(**item) for item in data]
                logger.info(f"加载了 {len(self.feedback)} 条反馈")
            except Exception as e:
                logger.error(f"加载反馈失败: {e}")

    def _save_feedback(self):
        """保存反馈"""
        try:
            data = [asdict(item) for item in self.feedback]
            with open(self.feedback_file, "w", encoding="utf - 8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存反馈失败: {e}")

    def collect_feedback(self) -> List[Dict[str, Any]]:
        """收集用户反馈"""
        logger.info("开始收集用户反馈")

        # 模拟收集反馈
        sample_feedback = [
            FeedbackItem(
                id=generate_id(),
                user="developer_001",
                category="performance",
                content="事件总线性能很好，但内存使用可以进一步优化",
                rating=4,
                timestamp=time.time(),
            ),
            FeedbackItem(
                id=generate_id(),
                user="developer_002",
                category="usability",
                content="依赖注入容器的API设计很清晰，使用起来很方便",
                rating=5,
                timestamp=time.time(),
            ),
            FeedbackItem(
                id=generate_id(),
                user="developer_003",
                category="documentation",
                content="文档很详细，但缺少一些实际使用示例",
                rating=3,
                timestamp=time.time(),
            ),
        ]

        self.feedback.extend(sample_feedback)
        self._save_feedback()

        logger.info(f"收集了 {len(sample_feedback)} 条新反馈")

        return [asdict(item) for item in sample_feedback]

    def get_feedback_summary(self) -> Dict[str, Any]:
        """获取反馈摘要"""
        if not self.feedback:
            return {"total": 0, "categories": {}, "average_rating": 0}

        categories = {}
        total_rating = 0

        for item in self.feedback:
            if item.category not in categories:
                categories[item.category] = {"count": 0, "rating": 0}
            categories[item.category]["count"] += 1
            categories[item.category]["rating"] += item.rating
            total_rating += item.rating

        average_rating = total_rating / len(self.feedback)

        return {
            "total": len(self.feedback),
            "categories": categories,
            "average_rating": round(average_rating, 2),
        }

    def shutdown(self) -> bool:
        """关闭用户反馈收集器"""
        try:
            logger.info("开始关闭用户反馈收集器")
            self._save_feedback()
            logger.info("用户反馈收集器关闭完成")
            return True
        except Exception as e:
            logger.error(f"关闭用户反馈收集器失败: {e}")
            return False


class UserFeedbackCollector(BaseComponent):
    """用户反馈收集器"""

    def __init__(self, feedback_dir: str = "data/feedback"):

        super().__init__("UserFeedbackCollector")
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "feedback.json"
        self.feedback: List[FeedbackItem] = []
        self._load_feedback()

        logger.info("用户反馈收集器初始化完成")

    def _load_feedback(self):
        """加载已有反馈"""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, "r", encoding="utf - 8") as f:
                    data = json.load(f)
                    self.feedback = [FeedbackItem(**item) for item in data]
                logger.info(f"加载了 {len(self.feedback)} 条反馈")
            except Exception as e:
                logger.error(f"加载反馈失败: {e}")

    def _save_feedback(self):
        """保存反馈"""
        try:
            data = [asdict(item) for item in self.feedback]
            with open(self.feedback_file, "w", encoding="utf - 8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存反馈失败: {e}")

    def collect_feedback(self) -> List[Dict[str, Any]]:
        """收集用户反馈"""
        logger.info("开始收集用户反馈")

        # 模拟收集反馈
        sample_feedback = [
            FeedbackItem(
                id=generate_id(),
                user="developer_001",
                category="performance",
                content="事件总线性能很好，但内存使用可以进一步优化",
                rating=4,
                timestamp=time.time(),
            ),
            FeedbackItem(
                id=generate_id(),
                user="developer_002",
                category="usability",
                content="依赖注入容器的API设计很清晰，使用起来很方便",
                rating=5,
                timestamp=time.time(),
            ),
            FeedbackItem(
                id=generate_id(),
                user="developer_003",
                category="documentation",
                content="文档很详细，但缺少一些实际使用示例",
                rating=3,
                timestamp=time.time(),
            ),
        ]

        self.feedback.extend(sample_feedback)
        self._save_feedback()

        logger.info(f"收集了 {len(sample_feedback)} 条新反馈")

        return [asdict(item) for item in sample_feedback]

    def get_feedback_summary(self) -> Dict[str, Any]:
        """获取反馈摘要"""
        if not self.feedback:
            return {"total": 0, "categories": {}, "average_rating": 0}

        categories = {}
        total_rating = 0

        for item in self.feedback:
            if item.category not in categories:
                categories[item.category] = {"count": 0, "rating": 0}
            categories[item.category]["count"] += 1
            categories[item.category]["rating"] += item.rating
            total_rating += item.rating

        average_rating = total_rating / len(self.feedback)

        return {
            "total": len(self.feedback),
            "categories": categories,
            "average_rating": round(average_rating, 2),
        }

    def shutdown(self) -> bool:
        """关闭用户反馈收集器"""
        try:
            logger.info("开始关闭用户反馈收集器")
            self._save_feedback()
            logger.info("用户反馈收集器关闭完成")
            return True
        except Exception as e:
            logger.error(f"关闭用户反馈收集器失败: {e}")
            return False


class UserFeedbackCollector(BaseComponent):
    """用户反馈收集器"""

    def __init__(self, feedback_dir: str = "data/feedback"):

        super().__init__("UserFeedbackCollector")
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "feedback.json"
        self.feedback: List[FeedbackItem] = []
        self._load_feedback()

        logger.info("用户反馈收集器初始化完成")

    def _load_feedback(self):
        """加载已有反馈"""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, "r", encoding="utf - 8") as f:
                    data = json.load(f)
                    self.feedback = [FeedbackItem(**item) for item in data]
                logger.info(f"加载了 {len(self.feedback)} 条反馈")
            except Exception as e:
                logger.error(f"加载反馈失败: {e}")

    def _save_feedback(self):
        """保存反馈"""
        try:
            data = [asdict(item) for item in self.feedback]
            with open(self.feedback_file, "w", encoding="utf - 8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存反馈失败: {e}")

    def collect_feedback(self) -> List[Dict[str, Any]]:
        """收集用户反馈"""
        logger.info("开始收集用户反馈")

        # 模拟收集反馈
        sample_feedback = [
            FeedbackItem(
                id=generate_id(),
                user="developer_001",
                category="performance",
                content="事件总线性能很好，但内存使用可以进一步优化",
                rating=4,
                timestamp=time.time(),
            ),
            FeedbackItem(
                id=generate_id(),
                user="developer_002",
                category="usability",
                content="依赖注入容器的API设计很清晰，使用起来很方便",
                rating=5,
                timestamp=time.time(),
            ),
            FeedbackItem(
                id=generate_id(),
                user="developer_003",
                category="documentation",
                content="文档很详细，但缺少一些实际使用示例",
                rating=3,
                timestamp=time.time(),
            ),
        ]

        self.feedback.extend(sample_feedback)
        self._save_feedback()

        logger.info(f"收集了 {len(sample_feedback)} 条新反馈")

        return [asdict(item) for item in sample_feedback]

    def get_feedback_summary(self) -> Dict[str, Any]:
        """获取反馈摘要"""
        if not self.feedback:
            return {"total": 0, "categories": {}, "average_rating": 0}

        categories = {}
        total_rating = 0

        for item in self.feedback:
            if item.category not in categories:
                categories[item.category] = {"count": 0, "rating": 0}
            categories[item.category]["count"] += 1
            categories[item.category]["rating"] += item.rating
            total_rating += item.rating

        average_rating = total_rating / len(self.feedback)

        return {
            "total": len(self.feedback),
            "categories": categories,
            "average_rating": round(average_rating, 2),
        }

    def shutdown(self) -> bool:
        """关闭用户反馈收集器"""
        try:
            logger.info("开始关闭用户反馈收集器")
            self._save_feedback()
            logger.info("用户反馈收集器关闭完成")
            return True
        except Exception as e:
            logger.error(f"关闭用户反馈收集器失败: {e}")
            return False


class UserFeedbackCollector(BaseComponent):
    """用户反馈收集器"""

    def __init__(self, feedback_dir: str = "data/feedback"):

        super().__init__("UserFeedbackCollector")
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "feedback.json"
        self.feedback: List[FeedbackItem] = []
        self._load_feedback()

        logger.info("用户反馈收集器初始化完成")

    def _load_feedback(self):
        """加载已有反馈"""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, "r", encoding="utf - 8") as f:
                    data = json.load(f)
                    self.feedback = [FeedbackItem(**item) for item in data]
                logger.info(f"加载了 {len(self.feedback)} 条反馈")
            except Exception as e:
                logger.error(f"加载反馈失败: {e}")

    def _save_feedback(self):
        """保存反馈"""
        try:
            data = [asdict(item) for item in self.feedback]
            with open(self.feedback_file, "w", encoding="utf - 8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存反馈失败: {e}")

    def collect_feedback(self) -> List[Dict[str, Any]]:
        """收集用户反馈"""
        logger.info("开始收集用户反馈")

        # 模拟收集反馈
        sample_feedback = [
            FeedbackItem(
                id=generate_id(),
                user="developer_001",
                category="performance",
                content="事件总线性能很好，但内存使用可以进一步优化",
                rating=4,
                timestamp=time.time(),
            ),
            FeedbackItem(
                id=generate_id(),
                user="developer_002",
                category="usability",
                content="依赖注入容器的API设计很清晰，使用起来很方便",
                rating=5,
                timestamp=time.time(),
            ),
            FeedbackItem(
                id=generate_id(),
                user="developer_003",
                category="documentation",
                content="文档很详细，但缺少一些实际使用示例",
                rating=3,
                timestamp=time.time(),
            ),
        ]

        self.feedback.extend(sample_feedback)
        self._save_feedback()

        logger.info(f"收集了 {len(sample_feedback)} 条新反馈")

        return [asdict(item) for item in sample_feedback]

    def get_feedback_summary(self) -> Dict[str, Any]:
        """获取反馈摘要"""
        if not self.feedback:
            return {"total": 0, "categories": {}, "average_rating": 0}

        categories = {}
        total_rating = 0

        for item in self.feedback:
            if item.category not in categories:
                categories[item.category] = {"count": 0, "rating": 0}
            categories[item.category]["count"] += 1
            categories[item.category]["rating"] += item.rating
            total_rating += item.rating

        average_rating = total_rating / len(self.feedback)

        return {
            "total": len(self.feedback),
            "categories": categories,
            "average_rating": round(average_rating, 2),
        }

    def shutdown(self) -> bool:
        """关闭用户反馈收集器"""
        try:
            logger.info("开始关闭用户反馈收集器")
            self._save_feedback()
            logger.info("用户反馈收集器关闭完成")
            return True
        except Exception as e:
            logger.error(f"关闭用户反馈收集器失败: {e}")
            return False


class UserFeedbackCollector(BaseComponent):
    """用户反馈收集器"""

    def __init__(self, feedback_dir: str = "data/feedback"):

        super().__init__("UserFeedbackCollector")
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "feedback.json"
        self.feedback: List[FeedbackItem] = []
        self._load_feedback()

        logger.info("用户反馈收集器初始化完成")

    def _load_feedback(self):
        """加载已有反馈"""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, "r", encoding="utf - 8") as f:
                    data = json.load(f)
                    self.feedback = [FeedbackItem(**item) for item in data]
                logger.info(f"加载了 {len(self.feedback)} 条反馈")
            except Exception as e:
                logger.error(f"加载反馈失败: {e}")

    def _save_feedback(self):
        """保存反馈"""
        try:
            data = [asdict(item) for item in self.feedback]
            with open(self.feedback_file, "w", encoding="utf - 8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存反馈失败: {e}")

    def collect_feedback(self) -> List[Dict[str, Any]]:
        """收集用户反馈"""
        logger.info("开始收集用户反馈")

        # 模拟收集反馈
        sample_feedback = [
            FeedbackItem(
                id=generate_id(),
                user="developer_001",
                category="performance",
                content="事件总线性能很好，但内存使用可以进一步优化",
                rating=4,
                timestamp=time.time(),
            ),
            FeedbackItem(
                id=generate_id(),
                user="developer_002",
                category="usability",
                content="依赖注入容器的API设计很清晰，使用起来很方便",
                rating=5,
                timestamp=time.time(),
            ),
            FeedbackItem(
                id=generate_id(),
                user="developer_003",
                category="documentation",
                content="文档很详细，但缺少一些实际使用示例",
                rating=3,
                timestamp=time.time(),
            ),
        ]

        self.feedback.extend(sample_feedback)
        self._save_feedback()

        logger.info(f"收集了 {len(sample_feedback)} 条新反馈")

        return [asdict(item) for item in sample_feedback]

    def get_feedback_summary(self) -> Dict[str, Any]:
        """获取反馈摘要"""
        if not self.feedback:
            return {"total": 0, "categories": {}, "average_rating": 0}

        categories = {}
        total_rating = 0

        for item in self.feedback:
            if item.category not in categories:
                categories[item.category] = {"count": 0, "rating": 0}
            categories[item.category]["count"] += 1
            categories[item.category]["rating"] += item.rating
            total_rating += item.rating

        average_rating = total_rating / len(self.feedback)

        return {
            "total": len(self.feedback),
            "categories": categories,
            "average_rating": round(average_rating, 2),
        }

    def shutdown(self) -> bool:
        """关闭用户反馈收集器"""
        try:
            logger.info("开始关闭用户反馈收集器")
            self._save_feedback()
            logger.info("用户反馈收集器关闭完成")
            return True
        except Exception as e:
            logger.error(f"关闭用户反馈收集器失败: {e}")
            return False


class UserFeedbackCollector(BaseComponent):
    """用户反馈收集器"""

    def __init__(self, feedback_dir: str = "data/feedback"):

        super().__init__("UserFeedbackCollector")
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "feedback.json"
        self.feedback: List[FeedbackItem] = []
        self._load_feedback()

        logger.info("用户反馈收集器初始化完成")

    def _load_feedback(self):
        """加载已有反馈"""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, "r", encoding="utf - 8") as f:
                    data = json.load(f)
                    self.feedback = [FeedbackItem(**item) for item in data]
                logger.info(f"加载了 {len(self.feedback)} 条反馈")
            except Exception as e:
                logger.error(f"加载反馈失败: {e}")

    def _save_feedback(self):
        """保存反馈"""
        try:
            data = [asdict(item) for item in self.feedback]
            with open(self.feedback_file, "w", encoding="utf - 8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存反馈失败: {e}")

    def collect_feedback(self) -> List[Dict[str, Any]]:
        """收集用户反馈"""
        logger.info("开始收集用户反馈")

        # 模拟收集反馈
        sample_feedback = [
            FeedbackItem(
                id=generate_id(),
                user="developer_001",
                category="performance",
                content="事件总线性能很好，但内存使用可以进一步优化",
                rating=4,
                timestamp=time.time(),
            ),
            FeedbackItem(
                id=generate_id(),
                user="developer_002",
                category="usability",
                content="依赖注入容器的API设计很清晰，使用起来很方便",
                rating=5,
                timestamp=time.time(),
            ),
            FeedbackItem(
                id=generate_id(),
                user="developer_003",
                category="documentation",
                content="文档很详细，但缺少一些实际使用示例",
                rating=3,
                timestamp=time.time(),
            ),
        ]

        self.feedback.extend(sample_feedback)
        self._save_feedback()

        logger.info(f"收集了 {len(sample_feedback)} 条新反馈")

        return [asdict(item) for item in sample_feedback]

    def get_feedback_summary(self) -> Dict[str, Any]:
        """获取反馈摘要"""
        if not self.feedback:
            return {"total": 0, "categories": {}, "average_rating": 0}

        categories = {}
        total_rating = 0

        for item in self.feedback:
            if item.category not in categories:
                categories[item.category] = {"count": 0, "rating": 0}
            categories[item.category]["count"] += 1
            categories[item.category]["rating"] += item.rating
            total_rating += item.rating

        average_rating = total_rating / len(self.feedback)

        return {
            "total": len(self.feedback),
            "categories": categories,
            "average_rating": round(average_rating, 2),
        }

    def shutdown(self) -> bool:
        """关闭用户反馈收集器"""
        try:
            logger.info("开始关闭用户反馈收集器")
            self._save_feedback()
            logger.info("用户反馈收集器关闭完成")
            return True
        except Exception as e:
            logger.error(f"关闭用户反馈收集器失败: {e}")
            return False
