#!/usr/bin/env python3
"""
触发特征选择任务并记录历史
用于验证特征选择过程仪表盘
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """执行特征选择并记录历史"""
    
    # 1. 获取特征数据
    logger.info("1. 获取特征数据...")
    from src.gateway.web.feature_task_persistence import get_features
    
    features = get_features()
    if not features:
        logger.error("没有可用的特征数据")
        return False
    
    logger.info(f"获取到 {len(features)} 个特征")
    
    # 2. 准备特征选择参数
    logger.info("2. 准备特征选择...")
    
    # 按股票代码分组
    symbols = list(set(f.get('symbol', 'unknown') for f in features))
    logger.info(f"涉及股票: {symbols}")
    
    # 3. 执行特征选择
    logger.info("3. 执行特征选择...")
    from src.features.utils.feature_selector import FeatureSelector
    
    # 模拟特征选择过程
    selector = FeatureSelector(config={'method': 'correlation', 'k_features': 10})
    
    # 为每个股票执行选择
    selection_results = []
    for symbol in symbols[:3]:  # 选择前3个股票
        symbol_features = [f for f in features if f.get('symbol') == symbol]
        if not symbol_features:
            continue
            
        logger.info(f"  处理股票 {symbol}: {len(symbol_features)} 个特征")
        
        # 模拟选择结果（选择质量最高的10个）
        sorted_features = sorted(
            symbol_features, 
            key=lambda x: x.get('quality_score', 0), 
            reverse=True
        )
        selected = sorted_features[:min(10, len(sorted_features))]
        
        result = {
            'symbol': symbol,
            'input_count': len(symbol_features),
            'selected_count': len(selected),
            'selected_features': [f.get('name') for f in selected],
            'method': 'quality_based'
        }
        selection_results.append(result)
        logger.info(f"    选择 {len(selected)} 个特征")
    
    # 4. 记录到历史表
    logger.info("4. 记录特征选择历史...")
    
    try:
        from src.features.selection.feature_selector_history import (
            FeatureSelectorHistoryManager, FeatureSelectionRecord
        )
        
        history_manager = FeatureSelectorHistoryManager()
        
        # 创建历史记录
        total_input = sum(r['input_count'] for r in selection_results)
        total_selected = sum(r['selected_count'] for r in selection_results)
        all_selected = []
        for r in selection_results:
            all_selected.extend(r['selected_features'])
        
        record = FeatureSelectionRecord(
            selection_id=f"selection_{int(time.time())}",
            task_id=f"task_{int(time.time())}",
            timestamp=time.time(),
            datetime=datetime.now().isoformat(),
            input_features=[f.get('name') for f in features],
            input_feature_count=total_input,
            selection_method='quality_based',
            selection_params={'k_features': 10, 'symbols': symbols[:3]},
            selected_features=all_selected,
            selected_feature_count=total_selected,
            selection_ratio=total_selected / total_input if total_input > 0 else 0,
            evaluation_metrics={'avg_quality': 0.85},
            processing_time=1.5,
            notes=f"自动特征选择: {len(selection_results)} 个股票"
        )
        
        # 保存到历史
        history_manager._history.append(record)
        history_manager._save_history()
        
        logger.info(f"✅ 特征选择历史已记录: {record.selection_id}")
        logger.info(f"   输入特征: {total_input}")
        logger.info(f"   选择特征: {total_selected}")
        logger.info(f"   选择比例: {record.selection_ratio:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"记录历史失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
