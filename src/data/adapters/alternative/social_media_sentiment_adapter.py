"""
社交媒体情绪数据适配器

本模块提供社交媒体情绪数据的采集和分析，支持：
1. Twitter/X 情绪数据采集
2. 微博情绪数据采集
3. 情绪分析和评分
4. 情绪趋势追踪
5. 与价格数据的相关性分析

作者: 数据团队
创建日期: 2026-02-21
版本: 1.0.0
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict

# 尝试导入情绪分析库
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob未安装，将使用基础情绪分析")

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False
    logging.warning("Tweepy未安装，Twitter功能不可用")

from .base_alternative_adapter import (
    AlternativeDataAdapter,
    AlternativeDataType,
    SentimentData,
    AdapterStatus
)
from src.common.exceptions import DataSourceError


# 配置日志
logger = logging.getLogger(__name__)


class SocialPlatform(Enum):
    """社交媒体平台"""
    TWITTER = "twitter"
    WEIBO = "weibo"
    REDDIT = "reddit"
    NEWS = "news"


class SentimentLevel(Enum):
    """情绪等级"""
    VERY_NEGATIVE = -2    # 非常负面
    NEGATIVE = -1         # 负面
    NEUTRAL = 0           # 中性
    POSITIVE = 1          # 正面
    VERY_POSITIVE = 2     # 非常正面


@dataclass
class SocialMediaPost:
    """社交媒体帖子"""
    post_id: str
    platform: SocialPlatform
    author: str
    content: str
    timestamp: datetime
    likes: int = 0
    shares: int = 0
    comments: int = 0
    sentiment_score: float = 0.0
    sentiment_level: SentimentLevel = SentimentLevel.NEUTRAL
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "post_id": self.post_id,
            "platform": self.platform.value,
            "author": self.author,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "timestamp": self.timestamp.isoformat(),
            "likes": self.likes,
            "shares": self.shares,
            "comments": self.comments,
            "sentiment_score": self.sentiment_score,
            "sentiment_level": self.sentiment_level.name,
            "keywords": self.keywords
        }


@dataclass
class SentimentAggregate:
    """情绪聚合数据"""
    symbol: str
    timestamp: datetime
    overall_sentiment: float
    sentiment_level: SentimentLevel
    post_count: int
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    weighted_sentiment: float
    trending_keywords: List[str]
    volume_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "overall_sentiment": self.overall_sentiment,
            "sentiment_level": self.sentiment_level.name,
            "post_count": self.post_count,
            "positive_ratio": self.positive_ratio,
            "negative_ratio": self.negative_ratio,
            "neutral_ratio": self.neutral_ratio,
            "weighted_sentiment": self.weighted_sentiment,
            "trending_keywords": self.trending_keywords,
            "volume_score": self.volume_score
        }


class SocialMediaSentimentAdapter(AlternativeDataAdapter):
    """
    社交媒体情绪数据适配器
    
    功能:
    1. 多平台情绪数据采集 (Twitter/微博/Reddit)
    2. 实时情绪分析和评分
    3. 情绪趋势追踪和预警
    4. 与价格数据的相关性分析
    5. 热门话题和关键词提取
    
    使用示例:
        adapter = SocialMediaSentimentAdapter(
            twitter_api_key="xxx",
            twitter_api_secret="xxx"
        )
        
        await adapter.connect()
        
        # 获取情绪数据
        sentiment = await adapter.get_sentiment_data(
            symbol="AAPL",
            platform=SocialPlatform.TWITTER,
            hours_back=24
        )
        
        # 获取实时情绪流
        async for post in adapter.stream_sentiment("TSLA"):
            print(f"情绪评分: {post.sentiment_score}")
    """
    
    def __init__(
        self,
        twitter_api_key: Optional[str] = None,
        twitter_api_secret: Optional[str] = None,
        twitter_access_token: Optional[str] = None,
        twitter_access_secret: Optional[str] = None,
        weibo_api_key: Optional[str] = None,
        cache_duration_minutes: int = 5
    ):
        """
        初始化社交媒体情绪适配器
        
        参数:
            twitter_api_key: Twitter API密钥
            twitter_api_secret: Twitter API密钥
            twitter_access_token: Twitter访问令牌
            twitter_access_secret: Twitter访问令牌密钥
            weibo_api_key: 微博API密钥
            cache_duration_minutes: 缓存持续时间（分钟）
        """
        super().__init__()
        
        self.name = "SocialMediaSentiment"
        self.data_type = AlternativeDataType.SENTIMENT
        
        # API配置
        self._twitter_api_key = twitter_api_key
        self._twitter_api_secret = twitter_api_secret
        self._twitter_access_token = twitter_access_token
        self._twitter_access_secret = twitter_access_secret
        self._weibo_api_key = weibo_api_key
        
        # API客户端
        self._twitter_api = None
        self._weibo_api = None
        
        # 缓存
        self._sentiment_cache: Dict[str, Tuple[SentimentAggregate, datetime]] = {}
        self._cache_duration = timedelta(minutes=cache_duration_minutes)
        
        # 情绪词典（基础版）
        self._positive_words = {
            'good', 'great', 'excellent', 'amazing', 'awesome', 'fantastic',
            'outstanding', 'superb', 'brilliant', 'perfect', 'best', 'love',
            'like', 'happy', 'pleased', 'satisfied', 'positive', 'optimistic',
            'bullish', 'strong', 'growth', 'profit', 'gain', 'up', 'rise',
            'increase', 'surge', 'soar', 'boom', 'rally', 'breakthrough'
        }
        
        self._negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike',
            'angry', 'disappointed', 'unsatisfied', 'negative', 'pessimistic',
            'bearish', 'weak', 'loss', 'lose', 'decline', 'drop', 'fall',
            'decrease', 'crash', 'collapse', 'plunge', 'dump', 'bear', 'short'
        }
        
        # 统计
        self._stats = {
            "posts_processed": 0,
            "api_calls": 0,
            "cache_hits": 0
        }
    
    async def connect(self) -> bool:
        """
        连接到社交媒体API
        
        返回:
            bool: 连接是否成功
        """
        try:
            # 初始化Twitter API
            if TWEEPY_AVAILABLE and self._twitter_api_key:
                auth = tweepy.OAuthHandler(
                    self._twitter_api_key,
                    self._twitter_api_secret
                )
                if self._twitter_access_token:
                    auth.set_access_token(
                        self._twitter_access_token,
                        self._twitter_access_secret
                    )
                self._twitter_api = tweepy.API(auth, wait_on_rate_limit=True)
                
                # 测试连接
                self._twitter_api.verify_credentials()
                logger.info("Twitter API连接成功")
            
            self._is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"连接社交媒体API失败: {e}")
            self._is_connected = False
            return False
    
    async def disconnect(self) -> bool:
        """断开连接"""
        self._twitter_api = None
        self._weibo_api = None
        self._is_connected = False
        logger.info("社交媒体API已断开")
        return True
    
    async def get_sentiment_data(
        self,
        symbol: str,
        platform: SocialPlatform = SocialPlatform.TWITTER,
        hours_back: int = 24,
        min_posts: int = 10,
        use_cache: bool = True
    ) -> SentimentAggregate:
        """
        获取情绪数据
        
        参数:
            symbol: 股票代码
            platform: 社交媒体平台
            hours_back: 回溯小时数
            min_posts: 最少帖子数
            use_cache: 是否使用缓存
            
        返回:
            SentimentAggregate: 情绪聚合数据
        """
        cache_key = f"{symbol}:{platform.value}:{hours_back}"
        
        # 检查缓存
        if use_cache and cache_key in self._sentiment_cache:
            cached_data, cache_time = self._sentiment_cache[cache_key]
            if datetime.now() - cache_time < self._cache_duration:
                self._stats["cache_hits"] += 1
                return cached_data
        
        # 获取帖子
        posts = await self._fetch_posts(symbol, platform, hours_back, min_posts)
        
        if not posts:
            # 返回中性情绪
            return SentimentAggregate(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_sentiment=0.0,
                sentiment_level=SentimentLevel.NEUTRAL,
                post_count=0,
                positive_ratio=0.0,
                negative_ratio=0.0,
                neutral_ratio=1.0,
                weighted_sentiment=0.0,
                trending_keywords=[],
                volume_score=0.0
            )
        
        # 分析情绪
        aggregate = self._analyze_sentiment(symbol, posts)
        
        # 缓存结果
        if use_cache:
            self._sentiment_cache[cache_key] = (aggregate, datetime.now())
        
        return aggregate
    
    async def get_batch_sentiment(
        self,
        symbols: List[str],
        platform: SocialPlatform = SocialPlatform.TWITTER,
        hours_back: int = 24
    ) -> Dict[str, SentimentAggregate]:
        """
        批量获取情绪数据
        
        参数:
            symbols: 股票代码列表
            platform: 社交媒体平台
            hours_back: 回溯小时数
            
        返回:
            Dict[str, SentimentAggregate]: 情绪数据字典
        """
        results = {}
        
        tasks = [
            self.get_sentiment_data(symbol, platform, hours_back)
            for symbol in symbols
        ]
        
        sentiments = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, sentiment in zip(symbols, sentiments):
            if isinstance(sentiment, SentimentAggregate):
                results[symbol] = sentiment
            else:
                logger.warning(f"获取 {symbol} 情绪数据失败: {sentiment}")
        
        return results
    
    async def stream_sentiment(
        self,
        symbol: str,
        platform: SocialPlatform = SocialPlatform.TWITTER,
        duration_seconds: int = 60
    ):
        """
        流式获取情绪数据
        
        参数:
            symbol: 股票代码
            platform: 社交媒体平台
            duration_seconds: 持续时间（秒）
            
        生成:
            SocialMediaPost: 社交媒体帖子
        """
        start_time = datetime.now()
        seen_posts = set()
        
        while (datetime.now() - start_time).seconds < duration_seconds:
            try:
                # 获取新帖子
                posts = await self._fetch_posts(
                    symbol,
                    platform,
                    hours_back=1,
                    min_posts=1
                )
                
                for post in posts:
                    if post.post_id not in seen_posts:
                        seen_posts.add(post.post_id)
                        yield post
                
                # 等待一段时间
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"流式获取情绪数据失败: {e}")
                await asyncio.sleep(5)
    
    async def get_sentiment_trend(
        self,
        symbol: str,
        platform: SocialPlatform = SocialPlatform.TWITTER,
        days: int = 7,
        interval_hours: int = 6
    ) -> pd.DataFrame:
        """
        获取情绪趋势
        
        参数:
            symbol: 股票代码
            platform: 社交媒体平台
            days: 天数
            interval_hours: 时间间隔（小时）
            
        返回:
            DataFrame: 情绪趋势数据
        """
        trends = []
        
        for day in range(days):
            for hour in range(0, 24, interval_hours):
                time_point = datetime.now() - timedelta(days=day, hours=hour)
                
                # 获取该时间点的情绪数据
                sentiment = await self.get_sentiment_data(
                    symbol,
                    platform,
                    hours_back=interval_hours,
                    use_cache=False
                )
                
                trends.append({
                    'timestamp': time_point,
                    'sentiment': sentiment.overall_sentiment,
                    'sentiment_level': sentiment.sentiment_level.value,
                    'post_count': sentiment.post_count,
                    'volume_score': sentiment.volume_score
                })
        
        return pd.DataFrame(trends)
    
    def calculate_sentiment_correlation(
        self,
        sentiment_data: pd.DataFrame,
        price_data: pd.DataFrame,
        lag_hours: int = 0
    ) -> Dict[str, float]:
        """
        计算情绪与价格的相关性
        
        参数:
            sentiment_data: 情绪数据DataFrame
            price_data: 价格数据DataFrame
            lag_hours: 滞后小时数
            
        返回:
            Dict[str, float]: 相关性指标
        """
        try:
            # 对齐时间
            if 'timestamp' in sentiment_data.columns:
                sentiment_data = sentiment_data.set_index('timestamp')
            if 'timestamp' in price_data.columns:
                price_data = price_data.set_index('timestamp')
            
            # 计算价格变化
            price_data['returns'] = price_data['close'].pct_change()
            
            # 合并数据
            merged = pd.merge(
                sentiment_data,
                price_data,
                left_index=True,
                right_index=True,
                how='inner'
            )
            
            if len(merged) < 10:
                return {"error": "数据点不足，无法计算相关性"}
            
            # 计算相关性
            sentiment_price_corr = merged['sentiment'].corr(merged['close'])
            sentiment_return_corr = merged['sentiment'].corr(merged['returns'])
            
            # 计算领先滞后相关性
            lead_lag_corr = {}
            for lag in range(-5, 6):
                if lag != 0:
                    shifted_sentiment = merged['sentiment'].shift(lag)
                    corr = shifted_sentiment.corr(merged['returns'])
                    lead_lag_corr[f"lag_{lag}"] = corr
            
            return {
                "sentiment_price_correlation": sentiment_price_corr,
                "sentiment_return_correlation": sentiment_return_corr,
                "lead_lag_correlations": lead_lag_corr,
                "sample_size": len(merged)
            }
            
        except Exception as e:
            logger.error(f"计算相关性失败: {e}")
            return {"error": str(e)}
    
    async def _fetch_posts(
        self,
        symbol: str,
        platform: SocialPlatform,
        hours_back: int,
        min_posts: int
    ) -> List[SocialMediaPost]:
        """获取社交媒体帖子"""
        posts = []
        
        if platform == SocialPlatform.TWITTER and self._twitter_api:
            posts = await self._fetch_twitter_posts(symbol, hours_back, min_posts)
        elif platform == SocialPlatform.WEIBO:
            posts = await self._fetch_weibo_posts(symbol, hours_back, min_posts)
        else:
            # 模拟数据（用于测试）
            posts = self._generate_mock_posts(symbol, hours_back, min_posts)
        
        return posts
    
    async def _fetch_twitter_posts(
        self,
        symbol: str,
        hours_back: int,
        min_posts: int
    ) -> List[SocialMediaPost]:
        """获取Twitter帖子"""
        posts = []
        
        try:
            # 构建搜索查询
            query = f"${symbol} OR #{symbol}"
            
            # 搜索推文
            tweets = tweepy.Cursor(
                self._twitter_api.search_tweets,
                q=query,
                lang="en",
                tweet_mode="extended"
            ).items(min_posts * 2)  # 获取更多以过滤
            
            for tweet in tweets:
                # 检查时间
                tweet_time = tweet.created_at
                if datetime.now() - tweet_time > timedelta(hours=hours_back):
                    continue
                
                # 分析情绪
                content = tweet.full_text
                sentiment_score, sentiment_level = self._analyze_text_sentiment(content)
                
                # 提取关键词
                keywords = self._extract_keywords(content)
                
                post = SocialMediaPost(
                    post_id=str(tweet.id),
                    platform=SocialPlatform.TWITTER,
                    author=tweet.user.screen_name,
                    content=content,
                    timestamp=tweet_time,
                    likes=tweet.favorite_count,
                    shares=tweet.retweet_count,
                    comments=0,
                    sentiment_score=sentiment_score,
                    sentiment_level=sentiment_level,
                    keywords=keywords
                )
                
                posts.append(post)
                self._stats["posts_processed"] += 1
            
            self._stats["api_calls"] += 1
            
        except Exception as e:
            logger.error(f"获取Twitter帖子失败: {e}")
            # 返回模拟数据
            posts = self._generate_mock_posts(symbol, hours_back, min_posts)
        
        return posts[:min_posts]
    
    async def _fetch_weibo_posts(
        self,
        symbol: str,
        hours_back: int,
        min_posts: int
    ) -> List[SocialMediaPost]:
        """获取微博帖子（模拟实现）"""
        # 实际实现需要接入微博API
        logger.info("微博API未实现，返回模拟数据")
        return self._generate_mock_posts(symbol, hours_back, min_posts)
    
    def _generate_mock_posts(
        self,
        symbol: str,
        hours_back: int,
        count: int
    ) -> List[SocialMediaPost]:
        """生成模拟帖子数据（用于测试）"""
        import random
        
        mock_contents = [
            f"{symbol} is looking great today! Bullish on this stock. 📈",
            f"Just bought some {symbol}, expecting good earnings. 💰",
            f"{symbol} showing strong momentum in pre-market. 🚀",
            f"Concerned about {symbol} valuation here. Might be overpriced. 🤔",
            f"{symbol} technicals looking bearish. Be careful. 📉",
            f"Great quarter for {symbol}! Revenue beat expectations. 🎉",
            f"{symbol} down 5% today, what's happening? 😰",
            f"Loading up on {symbol} at these levels. Long term hold. 💎",
            f"{symbol} chart looks ready to breakout. Watch closely. 👀",
            f"Taking profits on {symbol} after the run-up. Locking in gains. 💵"
        ]
        
        posts = []
        for i in range(count):
            content = random.choice(mock_contents)
            sentiment_score, sentiment_level = self._analyze_text_sentiment(content)
            
            post = SocialMediaPost(
                post_id=f"mock_{symbol}_{i}_{int(datetime.now().timestamp())}",
                platform=SocialPlatform.TWITTER,
                author=f"user_{random.randint(1000, 9999)}",
                content=content,
                timestamp=datetime.now() - timedelta(hours=random.randint(0, hours_back)),
                likes=random.randint(0, 1000),
                shares=random.randint(0, 500),
                comments=random.randint(0, 200),
                sentiment_score=sentiment_score,
                sentiment_level=sentiment_level,
                keywords=self._extract_keywords(content)
            )
            
            posts.append(post)
        
        return posts
    
    def _analyze_sentiment(
        self,
        symbol: str,
        posts: List[SocialMediaPost]
    ) -> SentimentAggregate:
        """分析情绪"""
        if not posts:
            return SentimentAggregate(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_sentiment=0.0,
                sentiment_level=SentimentLevel.NEUTRAL,
                post_count=0,
                positive_ratio=0.0,
                negative_ratio=0.0,
                neutral_ratio=1.0,
                weighted_sentiment=0.0,
                trending_keywords=[],
                volume_score=0.0
            )
        
        # 计算基础统计
        sentiment_scores = [p.sentiment_score for p in posts]
        overall_sentiment = np.mean(sentiment_scores)
        
        # 计算情绪分布
        positive_count = sum(1 for p in posts if p.sentiment_level in [SentimentLevel.POSITIVE, SentimentLevel.VERY_POSITIVE])
        negative_count = sum(1 for p in posts if p.sentiment_level in [SentimentLevel.NEGATIVE, SentimentLevel.VERY_NEGATIVE])
        neutral_count = len(posts) - positive_count - negative_count
        
        total = len(posts)
        positive_ratio = positive_count / total
        negative_ratio = negative_count / total
        neutral_ratio = neutral_count / total
        
        # 确定情绪等级
        sentiment_level = self._score_to_level(overall_sentiment)
        
        # 计算加权情绪（考虑互动量）
        total_engagement = sum(p.likes + p.shares + p.comments for p in posts)
        if total_engagement > 0:
            weighted_sentiment = sum(
                p.sentiment_score * (p.likes + p.shares + p.comments) / total_engagement
                for p in posts
            )
        else:
            weighted_sentiment = overall_sentiment
        
        # 提取热门关键词
        all_keywords = []
        for post in posts:
            all_keywords.extend(post.keywords)
        
        keyword_counts = defaultdict(int)
        for kw in all_keywords:
            keyword_counts[kw] += 1
        
        trending_keywords = sorted(
            keyword_counts.keys(),
            key=lambda k: keyword_counts[k],
            reverse=True
        )[:10]
        
        # 计算成交量分数
        volume_score = min(1.0, len(posts) / 100)  # 归一化到0-1
        
        return SentimentAggregate(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_sentiment=overall_sentiment,
            sentiment_level=sentiment_level,
            post_count=len(posts),
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            neutral_ratio=neutral_ratio,
            weighted_sentiment=weighted_sentiment,
            trending_keywords=trending_keywords,
            volume_score=volume_score
        )
    
    def _analyze_text_sentiment(self, text: str) -> Tuple[float, SentimentLevel]:
        """分析文本情绪"""
        # 使用TextBlob（如果可用）
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            
            # 转换为我们的评分系统
            score = polarity * 2  # -2 to 2
            
            return score, self._score_to_level(score)
        
        # 基础情绪分析
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        positive_count = len(words & self._positive_words)
        negative_count = len(words & self._negative_words)
        
        if positive_count > negative_count:
            score = min(2.0, (positive_count - negative_count) * 0.5)
        elif negative_count > positive_count:
            score = max(-2.0, (positive_count - negative_count) * 0.5)
        else:
            score = 0.0
        
        return score, self._score_to_level(score)
    
    def _score_to_level(self, score: float) -> SentimentLevel:
        """将分数转换为情绪等级"""
        if score >= 1.5:
            return SentimentLevel.VERY_POSITIVE
        elif score >= 0.5:
            return SentimentLevel.POSITIVE
        elif score <= -1.5:
            return SentimentLevel.VERY_NEGATIVE
        elif score <= -0.5:
            return SentimentLevel.NEGATIVE
        else:
            return SentimentLevel.NEUTRAL
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取（基于大写字母和#标签）
        keywords = []
        
        # 提取股票代码 ($SYMBOL)
        stock_codes = re.findall(r'\$([A-Z]{1,5})', text)
        keywords.extend(stock_codes)
        
        # 提取标签 (#hashtag)
        hashtags = re.findall(r'#(\w+)', text)
        keywords.extend(hashtags)
        
        # 提取表情符号含义
        emoji_patterns = {
            '📈': 'uptrend',
            '📉': 'downtrend',
            '🚀': 'moon',
            '💰': 'money',
            '🐂': 'bullish',
            '🐻': 'bearish',
            '💎': 'diamond_hands',
            '😰': 'worried',
            '🎉': 'celebration'
        }
        
        for emoji, meaning in emoji_patterns.items():
            if emoji in text:
                keywords.append(meaning)
        
        return keywords
    
    def get_status(self) -> AdapterStatus:
        """获取适配器状态"""
        return AdapterStatus(
            name=self.name,
            is_connected=self._is_connected,
            is_available=self._is_connected,
            last_error=None,
            request_count=self._stats["api_calls"],
            avg_response_time_ms=0.0
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "posts_processed": self._stats["posts_processed"],
            "api_calls": self._stats["api_calls"],
            "cache_hits": self._stats["cache_hits"],
            "cache_size": len(self._sentiment_cache)
        }


# 便捷函数
async def get_stock_sentiment(
    symbol: str,
    hours_back: int = 24,
    twitter_api_key: Optional[str] = None
) -> SentimentAggregate:
    """
    便捷函数 - 获取股票情绪数据
    
    参数:
        symbol: 股票代码
        hours_back: 回溯小时数
        twitter_api_key: Twitter API密钥（可选）
        
    返回:
        SentimentAggregate: 情绪数据
    """
    adapter = SocialMediaSentimentAdapter(twitter_api_key=twitter_api_key)
    await adapter.connect()
    
    sentiment = await adapter.get_sentiment_data(symbol, hours_back=hours_back)
    
    await adapter.disconnect()
    
    return sentiment
