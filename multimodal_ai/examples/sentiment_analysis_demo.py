#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2026多模态AI情感分析演示

使用多模态AI技术分析市场情绪和新闻情感。

功能：
- 文本情感分析
- 新闻情绪量化
- 市场情绪指标计算
- 多模态数据融合

作者: RQA2026多模态AI引擎团队
时间: 2025年12月3日
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import re
from collections import defaultdict
import matplotlib.pyplot as plt


class MultimodalSentimentAnalyzer:
    """多模态情感分析器"""

    def __init__(self):
        # 情感词典
        self.positive_words = {
            'bullish', 'bull', 'up', 'rise', 'gain', 'profit', 'growth', 'strong',
            'surge', 'soar', 'rally', 'boost', 'climb', 'jump', 'increase', 'positive',
            'optimistic', 'confident', 'bullish', 'buy', 'long', 'momentum', 'breakout'
        }

        self.negative_words = {
            'bearish', 'bear', 'down', 'fall', 'loss', 'decline', 'weak', 'drop',
            'crash', 'plunge', 'slump', 'tumble', 'decrease', 'negative', 'pessimistic',
            'worried', 'bearish', 'sell', 'short', 'correction', 'recession', 'crash'
        }

        # 情感强度权重
        self.intensity_multipliers = {
            'very': 1.5, 'extremely': 2.0, 'highly': 1.3, 'slightly': 0.7,
            'moderately': 1.0, 'strongly': 1.4, 'weakly': 0.6
        }

        # 市场情绪历史
        self.sentiment_history = []

    def analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """
        分析文本情感

        Args:
            text: 输入文本

        Returns:
            Dict: 情感分析结果
        """
        # 预处理文本
        text = self._preprocess_text(text)

        # 分词
        words = text.lower().split()

        # 计算情感得分
        positive_score = 0
        negative_score = 0
        intensity_modifier = 1.0

        for i, word in enumerate(words):
            # 检查强度修饰词
            if word in self.intensity_multipliers:
                intensity_modifier = self.intensity_multipliers[word]
                continue

            # 检查情感词
            if word in self.positive_words:
                positive_score += 1 * intensity_modifier
                intensity_modifier = 1.0  # 重置修饰符
            elif word in self.negative_words:
                negative_score += 1 * intensity_modifier
                intensity_modifier = 1.0  # 重置修饰符

        # 计算综合情感得分
        total_score = positive_score - negative_score
        total_words = len(words)

        if total_words == 0:
            sentiment_score = 0
        else:
            sentiment_score = total_score / total_words

        # 确定情感标签
        if sentiment_score > 0.1:
            sentiment_label = 'positive'
        elif sentiment_score < -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'

        # 计算置信度
        confidence = min(abs(sentiment_score) * 2, 1.0)

        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'confidence': confidence,
            'positive_words': positive_score,
            'negative_words': negative_score,
            'total_words': total_words
        }

    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 移除特殊字符和数字
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        # 移除多余空格
        text = ' '.join(text.split())

        return text

    def analyze_news_batch(self, news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量分析新闻情感

        Args:
            news_list: 新闻列表 [{'title': str, 'content': str, 'timestamp': datetime}]

        Returns:
            List: 情感分析结果
        """
        results = []

        for news in news_list:
            # 合并标题和内容进行分析
            full_text = f"{news.get('title', '')} {news.get('content', '')}"

            # 分析情感
            sentiment_result = self.analyze_text_sentiment(full_text)

            # 添加时间戳和元数据
            result = {
                **sentiment_result,
                'timestamp': news.get('timestamp'),
                'title': news.get('title', ''),
                'source': news.get('source', 'unknown')
            }

            results.append(result)

        return results

    def calculate_market_sentiment_index(self,
                                       sentiment_results: List[Dict[str, Any]],
                                       time_window_hours: int = 24) -> Dict[str, Any]:
        """
        计算市场情绪指数

        Args:
            sentiment_results: 情感分析结果列表
            time_window_hours: 时间窗口(小时)

        Returns:
            Dict: 市场情绪指数
        """
        if not sentiment_results:
            return {
                'market_sentiment_index': 0,
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                'average_confidence': 0,
                'volatility': 0
            }

        # 计算时间窗口内的情感
        current_time = datetime.now()
        window_start = current_time - timedelta(hours=time_window_hours)

        window_results = [
            r for r in sentiment_results
            if r.get('timestamp') and r['timestamp'] >= window_start
        ]

        if not window_results:
            window_results = sentiment_results[-100:]  # 使用最近100条新闻

        # 计算加权情绪指数
        total_weighted_score = 0
        total_weight = 0
        sentiment_counts = defaultdict(int)

        for result in window_results:
            score = result['sentiment_score']
            confidence = result['confidence']

            # 基于置信度和时间衰减的权重
            time_weight = 1.0  # 可以根据时间距离计算衰减权重

            weight = confidence * time_weight
            total_weighted_score += score * weight
            total_weight += weight

            sentiment_counts[result['sentiment_label']] += 1

        # 计算市场情绪指数 (-1 到 1)
        market_sentiment_index = total_weighted_score / total_weight if total_weight > 0 else 0

        # 计算平均置信度
        avg_confidence = np.mean([r['confidence'] for r in window_results])

        # 计算情绪波动率
        scores = [r['sentiment_score'] for r in window_results]
        volatility = np.std(scores) if scores else 0

        return {
            'market_sentiment_index': market_sentiment_index,
            'sentiment_distribution': dict(sentiment_counts),
            'average_confidence': avg_confidence,
            'volatility': volatility,
            'sample_size': len(window_results),
            'time_window_hours': time_window_hours
        }

    def generate_trading_signals(self, market_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于市场情绪生成交易信号

        Args:
            market_sentiment: 市场情绪指数

        Returns:
            Dict: 交易信号
        """
        sentiment_index = market_sentiment['market_sentiment_index']
        volatility = market_sentiment['volatility']

        # 情绪阈值
        bullish_threshold = 0.3
        bearish_threshold = -0.3
        high_volatility_threshold = 0.5

        signals = []

        # 基于情绪指数的信号
        if sentiment_index > bullish_threshold:
            signals.append({
                'type': 'BULLISH_SENTIMENT',
                'strength': min(sentiment_index * 2, 1.0),
                'description': f'市场情绪乐观 (指数: {sentiment_index:.2f})'
            })
        elif sentiment_index < bearish_threshold:
            signals.append({
                'type': 'BEARISH_SENTIMENT',
                'strength': min(-sentiment_index * 2, 1.0),
                'description': f'市场情绪悲观 (指数: {sentiment_index:.2f})'
            })

        # 基于波动率的信号
        if volatility > high_volatility_threshold:
            signals.append({
                'type': 'HIGH_SENTIMENT_VOLATILITY',
                'strength': min(volatility, 1.0),
                'description': f'情绪波动较大 (波动率: {volatility:.2f})'
            })

        # 基于分布的信号
        distribution = market_sentiment['sentiment_distribution']
        total = sum(distribution.values())

        if total > 0:
            positive_ratio = distribution.get('positive', 0) / total
            negative_ratio = distribution.get('negative', 0) / total

            if positive_ratio > 0.6:
                signals.append({
                    'type': 'OVERWHELMING_POSITIVE',
                    'strength': positive_ratio,
                    'description': f'正面情绪占主导 ({positive_ratio:.1%})'
                })
            elif negative_ratio > 0.6:
                signals.append({
                    'type': 'OVERWHELMING_NEGATIVE',
                    'strength': negative_ratio,
                    'description': f'负面情绪占主导 ({negative_ratio:.1%})'
                })

        return {
            'signals': signals,
            'sentiment_index': sentiment_index,
            'volatility': volatility,
            'signal_count': len(signals)
        }

    def create_sample_news_data(self) -> List[Dict[str, Any]]:
        """创建示例新闻数据"""
        news_data = [
            {
                'title': 'Tech stocks surge as AI breakthrough announced',
                'content': 'Major technology companies reported strong earnings growth driven by artificial intelligence innovations. Investors are bullish on the sector.',
                'timestamp': datetime.now() - timedelta(hours=2),
                'source': 'Financial Times'
            },
            {
                'title': 'Market correction expected amid economic uncertainty',
                'content': 'Analysts warn of potential market downturn due to rising interest rates and geopolitical tensions. Bearish sentiment dominates trading floors.',
                'timestamp': datetime.now() - timedelta(hours=4),
                'source': 'Wall Street Journal'
            },
            {
                'title': 'Federal Reserve signals potential rate cuts',
                'content': 'Central bank officials indicated possible monetary policy easing in upcoming months. This positive development boosted investor confidence.',
                'timestamp': datetime.now() - timedelta(hours=6),
                'source': 'Bloomberg'
            },
            {
                'title': 'Oil prices stabilize after supply chain disruptions',
                'content': 'Crude oil markets showed resilience despite ongoing global supply challenges. Trading remained relatively stable with moderate volatility.',
                'timestamp': datetime.now() - timedelta(hours=8),
                'source': 'Reuters'
            },
            {
                'title': 'Cryptocurrency market experiences sharp decline',
                'content': 'Digital assets faced significant selling pressure as regulatory concerns intensified. The bearish trend continues to impact broader markets.',
                'timestamp': datetime.now() - timedelta(hours=12),
                'source': 'CoinDesk'
            }
        ]

        return news_data

    def visualize_sentiment_analysis(self, sentiment_results: List[Dict[str, Any]],
                                   market_sentiment: Dict[str, Any]):
        """可视化情感分析结果"""
        try:
            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # 1. 情感得分时间序列
            timestamps = [r['timestamp'] for r in sentiment_results if r.get('timestamp')]
            scores = [r['sentiment_score'] for r in sentiment_results if r.get('timestamp')]

            if timestamps and scores:
                axes[0, 0].plot(timestamps, scores, 'b-o', linewidth=2, markersize=4)
                axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
                axes[0, 0].set_title('Sentiment Score Over Time')
                axes[0, 0].set_ylabel('Sentiment Score')
                axes[0, 0].tick_params(axis='x', rotation=45)

            # 2. 情感分布饼图
            distribution = market_sentiment['sentiment_distribution']
            labels = distribution.keys()
            sizes = distribution.values()

            if sizes:
                axes[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                axes[0, 1].set_title('Sentiment Distribution')

            # 3. 置信度分布
            confidences = [r['confidence'] for r in sentiment_results]
            if confidences:
                axes[1, 0].hist(confidences, bins=10, alpha=0.7, color='green', edgecolor='black')
                axes[1, 0].set_title('Confidence Distribution')
                axes[1, 0].set_xlabel('Confidence')
                axes[1, 0].set_ylabel('Frequency')

            # 4. 市场情绪仪表盘
            sentiment_index = market_sentiment['market_sentiment_index']
            volatility = market_sentiment['volatility']

            # 创建情绪仪表盘
            theta = np.linspace(0, 2*np.pi, 100)
            r = 1
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            axes[1, 1].plot(x, y, 'k-', alpha=0.3)

            # 绘制情绪指针
            sentiment_angle = (sentiment_index + 1) * np.pi  # -1到1映射到0到2π
            pointer_x = 0.8 * np.cos(sentiment_angle)
            pointer_y = 0.8 * np.sin(sentiment_angle)
            axes[1, 1].arrow(0, 0, pointer_x, pointer_y, head_width=0.1, head_length=0.1, fc='red', ec='red')

            axes[1, 1].set_title(f'Market Sentiment Gauge\\nIndex: {sentiment_index:.3f}')
            axes[1, 1].set_xlim(-1.2, 1.2)
            axes[1, 1].set_ylim(-1.2, 1.2)
            axes[1, 1].set_aspect('equal')

            plt.tight_layout()
            plt.savefig('multimodal_sentiment_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

            print("📊 情感分析可视化图已保存为: multimodal_sentiment_analysis.png")

        except Exception as e:
            print(f"⚠️ 可视化失败: {e}")


def demonstrate_multimodal_sentiment_analysis():
    """演示多模态情感分析"""
    print("🎭 RQA2026多模态AI情感分析演示")
    print("=" * 60)

    # 创建情感分析器
    analyzer = MultimodalSentimentAnalyzer()

    # 创建示例新闻数据
    news_data = analyzer.create_sample_news_data()
    print(f"📄 加载了{len(news_data)}条示例新闻数据")

    # 批量分析新闻情感
    print("\n🔍 正在分析新闻情感...")
    sentiment_results = analyzer.analyze_news_batch(news_data)

    # 显示单条新闻分析结果
    print("\n📊 单条新闻情感分析结果:")
    for i, result in enumerate(sentiment_results[:3]):  # 只显示前3条
        print(f"\n新闻 {i+1}: {result['title'][:50]}...")
        print(f"  情感得分: {result['sentiment_score']:.3f}")
        print(f"  置信度: {result['confidence']:.3f}")
        print(f"  情感标签: {result['sentiment_label']}")

    # 计算市场情绪指数
    print("\n🌡️ 计算市场情绪指数...")
    market_sentiment = analyzer.calculate_market_sentiment_index(sentiment_results)
    print(f"  市场情绪指数: {market_sentiment['market_sentiment_index']:.3f}")
    print(f"  情绪波动率: {market_sentiment['volatility']:.3f}")
    print(f"  样本数量: {market_sentiment['sample_size']}")
    print(f"  情绪分布: {market_sentiment['sentiment_distribution']}")

    # 生成交易信号
    print("\n📈 生成交易信号...")
    trading_signals = analyzer.generate_trading_signals(market_sentiment)

    if trading_signals['signals']:
        print(f"🎯 检测到{len(trading_signals['signals'])}个交易信号:")
        for signal in trading_signals['signals']:
            print(f"  • {signal['type']}: {signal['description']} (强度: {signal['strength']:.2f})")
    else:
        print("📊 当前市场情绪相对中性，未检测到明确交易信号")

    # 可视化结果
    try:
        analyzer.visualize_sentiment_analysis(sentiment_results, market_sentiment)
    except Exception as e:
        print(f"⚠️ 可视化过程中出现错误: {e}")

    print("\n💡 多模态AI情感分析优势:")
    print("  • 实时市场情绪监测")
    print("  • 量化新闻影响评估")
    print("  • 多源数据融合分析")
    print("  • 交易信号自动生成")
    print("  • 风险情绪预警机制")

    print("\n🎊 RQA2026多模态AI情感分析演示完成!")
    print("为量化交易提供了强大的情绪分析能力！")

    return {
        'sentiment_results': sentiment_results,
        'market_sentiment': market_sentiment,
        'trading_signals': trading_signals
    }


if __name__ == "__main__":
    results = demonstrate_multimodal_sentiment_analysis()
