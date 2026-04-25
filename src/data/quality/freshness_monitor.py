"""
RQA2025 数据新鲜度监控系统 (P2-2)
自动监控所有数据源的数据新鲜度，发现过期数据自动告警
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio

logger = logging.getLogger(__name__)

# 新鲜度阈值配置（小时）
FRESHNESS_THRESHOLDS = {
    "critical": 6,      # 6小时内未更新 → 严重告警
    "warning": 12,      # 12小时内未更新 → 警告
    "attention": 24,    # 24小时内未更新 → 关注
}

# 数据源类型优先级
SOURCE_PRIORITY = {
    "股票数据": 1,
    "指数数据": 2,
    "大宗商品": 3,
    "新闻数据": 4,
    "宏观经济": 5,
    "其他": 6,
}


class FreshnessMonitor:
    """数据新鲜度监控器"""
    
    def __init__(self, db_pool=None):
        self.db_pool = db_pool
        self._alert_history: Dict[str, datetime] = {}
        self._cooldown_hours = 4  # 告警冷却时间（小时）
    
    async def check_all_sources(self) -> Dict:
        """检查所有数据源的新鲜度"""
        results = {
            "checked_at": datetime.now().isoformat(),
            "total_sources": 0,
            "healthy": 0,
            "warning": 0,
            "critical": 0,
            "details": []
        }
        
        # 从数据库获取所有数据源的最后采集时间
        sources_data = await self._get_sources_last_collection()
        
        results["total_sources"] = len(sources_data)
        
        for source in sources_data:
            source_id = source["source_id"]
            last_collection = source.get("last_collection")
            data_type = source.get("data_type", "其他")
            record_count = source.get("record_count", 0)
            
            # 计算新鲜度状态
            freshness_status, hours_since = self._calculate_freshness(last_collection, data_type)
            
            detail = {
                "source_id": source_id,
                "data_type": data_type,
                "last_collection": last_collection,
                "hours_since": hours_since,
                "freshness_status": freshness_status,
                "record_count": record_count,
                "should_alert": False
            }
            
            # 判断是否需要告警（冷却期内不重复告警）
            if freshness_status in ["critical", "warning"] and record_count > 0:
                if self._should_send_alert(source_id):
                    detail["should_alert"] = True
                    self._alert_history[source_id] = datetime.now()
            
            results["details"].append(detail)
            
            # 统计
            if freshness_status == "critical":
                results["critical"] += 1
            elif freshness_status == "warning":
                results["warning"] += 1
            else:
                results["healthy"] += 1
        
        # 按严重程度排序
        results["details"].sort(key=lambda x: (
            SOURCE_PRIORITY.get(x["data_type"], 6),
            -x["hours_since"]
        ))
        
        return results
    
    async def _get_sources_last_collection(self) -> List[Dict]:
        """从数据库获取各数据源的最后采集时间"""
        if not self.db_pool:
            return []
        
        query = """
        SELECT 
            'akshare_hk_stock_data' as source_id,
            '股票数据' as data_type,
            MAX(date) as last_collection,
            COUNT(*) as record_count
        FROM akshare_hk_stock_data
        UNION ALL
        SELECT 
            'akshare_index_data' as source_id,
            '指数数据' as data_type,
            MAX(date) as last_collection,
            COUNT(*) as record_count
        FROM akshare_index_data
        UNION ALL
        SELECT 
            'akshare_commodity_gold' as source_id,
            '大宗商品' as data_type,
            MAX(date) as last_collection,
            COUNT(*) as record_count
        FROM akshare_commodity_gold
        UNION ALL
        SELECT 
            'akshare_commodity_energy' as source_id,
            '大宗商品' as data_type,
            MAX(date) as last_collection,
            COUNT(*) as record_count
        FROM akshare_commodity_energy
        UNION ALL
        SELECT 
            'akshare_commodity_crude' as source_id,
            '大宗商品' as data_type,
            MAX(date) as last_collection,
            COUNT(*) as record_count
        FROM akshare_commodity_crude
        UNION ALL
        SELECT 
            'akshare_news' as source_id,
            '新闻数据' as data_type,
            MAX(collected_at) as last_collection,
            COUNT(*) as record_count
        FROM akshare_news_data
        UNION ALL
        SELECT 
            'akshare_stock_a' as source_id,
            '股票数据' as data_type,
            MAX(date) as last_collection,
            COUNT(*) as record_count
        FROM akshare_stock_data;
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"获取数据源新鲜度失败: {e}")
            return []
    
    def _calculate_freshness(self, last_collection: Optional[datetime], data_type: str) -> Tuple[str, int]:
        """计算数据新鲜度状态"""
        if last_collection is None:
            return "critical", 999
        
        now = datetime.now()
        hours_since = int((now - last_collection).total_seconds() / 3600)
        
        # 根据数据类型确定阈值
        if data_type in ["股票数据", "指数数据"]:
            if hours_since <= FRESHNESS_THRESHOLDS["critical"]:
                return "healthy", hours_since
            elif hours_since <= FRESHNESS_THRESHOLDS["warning"]:
                return "warning", hours_since
            elif hours_since <= FRESHNESS_THRESHOLDS["attention"]:
                return "attention", hours_since
            else:
                return "critical", hours_since
        elif data_type == "新闻数据":
            # 新闻数据要求更高新鲜度
            if hours_since <= 1:
                return "healthy", hours_since
            elif hours_since <= 3:
                return "warning", hours_since
            elif hours_since <= 6:
                return "attention", hours_since
            else:
                return "critical", hours_since
        else:
            # 其他数据
            if hours_since <= FRESHNESS_THRESHOLDS["warning"]:
                return "healthy", hours_since
            elif hours_since <= FRESHNESS_THRESHOLDS["attention"]:
                return "warning", hours_since
            else:
                return "critical", hours_since
    
    def _should_send_alert(self, source_id: str) -> bool:
        """判断是否应该发送告警（防止重复告警）"""
        if source_id not in self._alert_history:
            return True
        
        last_alert = self._alert_history[source_id]
        hours_since_alert = (datetime.now() - last_alert).total_seconds() / 3600
        
        return hours_since_alert >= self._cooldown_hours
    
    def get_critical_issues(self, results: Dict) -> List[Dict]:
        """获取需要立即处理的关键问题"""
        return [
            d for d in results["details"]
            if d["freshness_status"] == "critical" 
            and d["record_count"] > 0
            and d["should_alert"]
        ]
    
    def generate_report(self, results: Dict) -> str:
        """生成新鲜度监控报告"""
        critical_issues = self.get_critical_issues(results)
        
        report = f"""
# 📊 RQA2025 数据新鲜度监控报告

**检查时间**: {results['checked_at']}
**数据源总数**: {results['total_sources']}

## 状态概览

| 状态 | 数量 | 说明 |
|------|------|------|
| ✅ 健康 | {results['healthy']} | 正常更新 |
| ⚠️ 警告 | {results['warning']} | 需要关注 |
| ❌ 严重 | {results['critical']} | 需要立即处理 |

## 需要关注的数据源

"""
        if critical_issues:
            report += "| 数据源 | 类型 | 最后更新 | 超时(小时) |\n"
            report += "|--------|------|----------|------------|\n"
            for issue in critical_issues[:10]:  # 最多显示10个
                report += f"| {issue['source_id']} | {issue['data_type']} | {issue['last_collection']} | {issue['hours_since']}h |\n"
        else:
            report += "✅ 所有数据源状态正常\n"
        
        return report


# 全局实例
_freshness_monitor: Optional[FreshnessMonitor] = None


def get_freshness_monitor(db_pool=None) -> FreshnessMonitor:
    """获取新鲜度监控器实例"""
    global _freshness_monitor
    if _freshness_monitor is None:
        _freshness_monitor = FreshnessMonitor(db_pool)
    return _freshness_monitor


async def run_freshness_check() -> Dict:
    """运行一次完整的新鲜度检查"""
    monitor = get_freshness_monitor()
    results = await monitor.check_all_sources()
    return results


if __name__ == "__main__":
    # 独立运行测试
    async def test():
        print("运行数据新鲜度检查...")
        results = await run_freshness_check()
        print(f"检查完成: {results['healthy']} 健康, {results['warning']} 警告, {results['critical']} 严重")
        
        # 生成报告
        monitor = get_freshness_monitor()
        report = monitor.generate_report(results)
        print(report)
    
    asyncio.run(test())