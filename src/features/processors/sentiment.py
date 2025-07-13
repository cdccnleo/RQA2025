"""
情感分析处理器模块
提供情感分析功能的统一接口
"""

from src.features.sentiment.analyzer import SentimentAnalyzer


class SentimentProcessor:
    """情感分析处理器"""
    
    def __init__(self, **kwargs):
        """初始化情感分析处理器"""
        self.analyzer = SentimentAnalyzer(**kwargs)
    
    def process(self, data, text_col="content", **kwargs):
        """处理情感分析数据"""
        return self.analyzer.generate_features(data, text_col=text_col, **kwargs)
    
    def analyze_sentiment(self, text, **kwargs):
        """分析单个文本的情感"""
        return self.analyzer.analyze(text, **kwargs)
    
    def batch_analyze(self, texts, **kwargs):
        """批量分析文本情感"""
        return self.analyzer.batch_analyze(texts, **kwargs)


# 为了向后兼容，导出SentimentAnalyzer
__all__ = ['SentimentProcessor', 'SentimentAnalyzer'] 