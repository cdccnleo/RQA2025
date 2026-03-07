"""
修复SentimentNewsLoader抽象类问题
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def fix_sentiment_news_loader():
    """修复SentimentNewsLoader的抽象方法实现"""

    news_loader_path = Path("src/data/loader/news_loader.py")

    if not news_loader_path.exists():
        print(f"❌ 文件不存在: {news_loader_path}")
        return False

    # 读取原文件
    with open(news_loader_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否已经修复
    if "_connect(self) -> bool:" in content and "_disconnect(self) -> bool:" in content:
        print("✅ SentimentNewsLoader已经修复")
        return True

    # 在SentimentNewsLoader类中添加抽象方法实现
    # 找到SentimentNewsLoader类的结束位置
    class_end_marker = "    def get_required_config_fields(self) -> list:"

    # 在get_required_config_fields方法之前插入抽象方法实现
    abstract_methods = '''
    def _connect(self) -> bool:
        """实现BaseDataAdapter的抽象方法 - 建立连接"""
        # 对于新闻加载器，连接总是成功的
        return True
    
    def _disconnect(self) -> bool:
        """实现BaseDataAdapter的抽象方法 - 断开连接"""
        # 对于新闻加载器，断开连接总是成功的
        return True
    
    def _validate_connection(self) -> bool:
        """实现BaseDataAdapter的抽象方法 - 验证连接状态"""
        # 对于新闻加载器，连接总是有效的
        return True
    
    def _load_data(self, request) -> pd.DataFrame:
        """实现BaseDataAdapter的抽象方法 - 实际加载数据"""
        # 调用父类的load_data方法
        if hasattr(request, 'start_date') and hasattr(request, 'end_date'):
            return self.load_data(request.start_date, request.end_date)
        else:
            # 如果没有日期参数，使用默认日期
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            return self.load_data(start_date, end_date)
    
    def _get_symbols(self) -> List[str]:
        """实现BaseDataAdapter的抽象方法 - 获取股票代码列表"""
        # 对于新闻加载器，返回空列表
        return []
    
    def _get_info(self, symbol: str) -> Dict[str, Any]:
        """实现BaseDataAdapter的抽象方法 - 获取数据信息"""
        # 对于新闻加载器，返回基本信息
        return {
            "symbol": symbol,
            "data_type": "news",
            "supports_sentiment": True
        }
    
    '''

    # 替换内容
    new_content = content.replace(
        class_end_marker,
        abstract_methods + "\n    " + class_end_marker
    )

    # 添加必要的导入
    if "from typing import List, Dict, Any" not in new_content:
        # 在文件开头的导入部分添加
        import_section = "from typing import Union, Dict, Any, Optional, List"
        new_content = new_content.replace(
            "from typing import Union, Dict, Any, Optional",
            import_section
        )

    # 写回文件
    with open(news_loader_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print("✅ 成功修复SentimentNewsLoader抽象方法")
    return True


def main():
    """主函数"""
    print("🔧 开始修复SentimentNewsLoader抽象类问题...")

    try:
        success = fix_sentiment_news_loader()
        if success:
            print("✅ 修复完成")
        else:
            print("❌ 修复失败")
            return 1
    except Exception as e:
        print(f"❌ 修复过程中出现错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
