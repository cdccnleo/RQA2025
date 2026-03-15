#!/usr/bin/env python3
"""同步特征选择历史记录到PostgreSQL"""

from src.features.selection.feature_selector_history import get_feature_selector_history_manager

manager = get_feature_selector_history_manager()
print(f'内存中的记录数: {len(manager._history)}')

# 同步到PostgreSQL
manager._sync_to_postgresql()
print('同步完成')
