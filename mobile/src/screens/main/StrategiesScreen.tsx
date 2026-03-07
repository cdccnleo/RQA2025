/**
 * RQA 2.0 策略选择屏幕
 *
 * 展示可用的量化策略列表，支持筛选、排序、搜索
 * 提供策略详情查看和性能对比功能
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import React, {useEffect, useState} from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  TextInput,
  ScrollView,
} from 'react-native';
import {useDispatch, useSelector} from 'react-redux';
import Icon from 'react-native-vector-icons/MaterialIcons';

// 组件
import StrategyCard from '../../components/strategies/StrategyCard';
import FilterModal from '../../components/strategies/FilterModal';
import LoadingSpinner from '../../components/common/LoadingSpinner';
import EmptyState from '../../components/common/EmptyState';

// Redux actions and selectors
import {
  fetchStrategies,
  selectStrategies,
  selectStrategiesLoading,
  selectStrategiesError,
  selectStrategiesFilters,
  selectStrategiesHasMore,
  updateFilters,
  clearFilters,
  resetStrategies,
} from '../../store/slices/strategiesSlice';

// 主题和工具
import {useTheme} from '../../theme/ThemeProvider';
import {colors, typography, spacing, borderRadius} from '../../theme/theme';

const StrategiesScreen: React.FC = () => {
  const dispatch = useDispatch();
  const {theme} = useTheme();

  // Redux state
  const strategies = useSelector(selectStrategies);
  const isLoading = useSelector(selectStrategiesLoading);
  const error = useSelector(selectStrategiesError);
  const filters = useSelector(selectStrategiesFilters);
  const hasMore = useSelector(selectStrategiesHasMore);

  // 本地状态
  const [searchQuery, setSearchQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [filteredStrategies, setFilteredStrategies] = useState(strategies);

  // 初始化数据
  useEffect(() => {
    dispatch(fetchStrategies({page: 1, limit: 20}) as any);
  }, [dispatch]);

  // 应用搜索筛选
  useEffect(() => {
    let filtered = strategies;

    if (searchQuery.trim()) {
      filtered = strategies.filter(strategy =>
        strategy.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        strategy.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        strategy.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
      );
    }

    setFilteredStrategies(filtered);
  }, [strategies, searchQuery]);

  // 加载更多数据
  const loadMoreStrategies = () => {
    if (hasMore && !isLoading) {
      const nextPage = Math.floor(strategies.length / 20) + 1;
      dispatch(fetchStrategies({page: nextPage, limit: 20}) as any);
    }
  };

  // 刷新数据
  const refreshStrategies = () => {
    dispatch(resetStrategies());
    dispatch(fetchStrategies({page: 1, limit: 20}) as any);
  };

  // 应用筛选
  const applyFilters = (newFilters: any) => {
    dispatch(updateFilters(newFilters) as any);
    setShowFilters(false);
  };

  // 清除筛选
  const clearAllFilters = () => {
    dispatch(clearFilters());
    setSearchQuery('');
  };

  // 获取活跃筛选数量
  const getActiveFiltersCount = () => {
    let count = 0;
    if (filters.category) count++;
    if (filters.riskLevel) count++;
    if (filters.timeHorizon) count++;
    if (filters.minReturn !== undefined) count++;
    if (filters.maxVolatility !== undefined) count++;
    if (searchQuery.trim()) count++;
    return count;
  };

  // 渲染策略项
  const renderStrategyItem = ({item}: {item: any}) => (
    <StrategyCard
      strategy={item}
      onPress={() => {
        // 导航到策略详情
        console.log('Navigate to strategy details:', item.id);
      }}
    />
  );

  // 渲染筛选标签
  const renderFilterTags = () => {
    const tags = [];

    if (filters.category) {
      tags.push(
        <TouchableOpacity
          key="category"
          style={[styles.filterTag, {backgroundColor: theme.primary}]}
          onPress={() => dispatch(updateFilters({category: undefined}) as any)}>
          <Text style={styles.filterTagText}>{getCategoryName(filters.category)}</Text>
          <Icon name="close" size={14} color={colors.text.primary} />
        </TouchableOpacity>
      );
    }

    if (filters.riskLevel) {
      tags.push(
        <TouchableOpacity
          key="risk"
          style={[styles.filterTag, {backgroundColor: theme.primary}]}
          onPress={() => dispatch(updateFilters({riskLevel: undefined}) as any)}>
          <Text style={styles.filterTagText}>{getRiskLevelName(filters.riskLevel)}</Text>
          <Icon name="close" size={14} color={colors.text.primary} />
        </TouchableOpacity>
      );
    }

    return tags;
  };

  // 辅助函数
  const getCategoryName = (category: string) => {
    const categories: {[key: string]: string} = {
      momentum: '动量策略',
      mean_reversion: '均值回归',
      arbitrage: '套利策略',
      ml_based: '机器学习',
      risk_parity: '风险平价',
    };
    return categories[category] || category;
  };

  const getRiskLevelName = (riskLevel: string) => {
    const levels: {[key: string]: string} = {
      low: '低风险',
      medium: '中风险',
      high: '高风险',
    };
    return levels[riskLevel] || riskLevel;
  };

  return (
    <View style={[styles.container, {backgroundColor: theme.colors.background.primary}]}>
      {/* 搜索栏 */}
      <View style={styles.searchContainer}>
        <View style={[styles.searchBar, {backgroundColor: theme.colors.background.secondary}]}>
          <Icon name="search" size={20} color={colors.text.secondary} />
          <TextInput
            style={[styles.searchInput, {color: theme.colors.text.primary}]}
            placeholder="搜索策略名称或标签"
            placeholderTextColor={colors.text.secondary}
            value={searchQuery}
            onChangeText={setSearchQuery}
          />
          {searchQuery ? (
            <TouchableOpacity onPress={() => setSearchQuery('')}>
              <Icon name="clear" size={20} color={colors.text.secondary} />
            </TouchableOpacity>
          ) : null}
        </View>

        <TouchableOpacity
          style={[styles.filterButton, getActiveFiltersCount() > 0 && styles.filterButtonActive]}
          onPress={() => setShowFilters(true)}>
          <Icon
            name="filter-list"
            size={20}
            color={getActiveFiltersCount() > 0 ? colors.text.primary : colors.text.secondary}
          />
          {getActiveFiltersCount() > 0 && (
            <View style={styles.filterBadge}>
              <Text style={styles.filterBadgeText}>{getActiveFiltersCount()}</Text>
            </View>
          )}
        </TouchableOpacity>
      </View>

      {/* 筛选标签 */}
      {renderFilterTags().length > 0 && (
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          style={styles.tagsContainer}
          contentContainerStyle={styles.tagsContent}>
          {renderFilterTags()}
          <TouchableOpacity
            style={[styles.clearFiltersButton]}
            onPress={clearAllFilters}>
            <Text style={[styles.clearFiltersText, {color: theme.primary}]}>清除全部</Text>
          </TouchableOpacity>
        </ScrollView>
      )}

      {/* 策略列表 */}
      {filteredStrategies.length > 0 ? (
        <FlatList
          data={filteredStrategies}
          renderItem={renderStrategyItem}
          keyExtractor={item => item.id}
          onEndReached={loadMoreStrategies}
          onEndReachedThreshold={0.1}
          refreshing={isLoading}
          onRefresh={refreshStrategies}
          contentContainerStyle={styles.strategiesList}
          ItemSeparatorComponent={() => <View style={styles.separator} />}
          ListFooterComponent={
            isLoading && strategies.length > 0 ? (
              <LoadingSpinner size="small" />
            ) : null
          }
        />
      ) : isLoading ? (
        <LoadingSpinner />
      ) : (
        <EmptyState
          icon="analytics"
          title="未找到策略"
          message="尝试调整筛选条件或搜索关键词"
          actionText="清除筛选"
          onAction={clearAllFilters}
        />
      )}

      {/* 筛选模态框 */}
      <FilterModal
        visible={showFilters}
        filters={filters}
        onApply={applyFilters}
        onClose={() => setShowFilters(false)}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  searchBar: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: borderRadius.lg,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    marginRight: spacing.md,
  },
  searchInput: {
    flex: 1,
    marginLeft: spacing.sm,
    ...typography.body,
  },
  filterButton: {
    width: 44,
    height: 44,
    borderRadius: borderRadius.lg,
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
  },
  filterButtonActive: {
    backgroundColor: 'rgba(0, 122, 255, 0.1)',
  },
  filterBadge: {
    position: 'absolute',
    top: 6,
    right: 6,
    backgroundColor: colors.error,
    borderRadius: 8,
    minWidth: 16,
    height: 16,
    justifyContent: 'center',
    alignItems: 'center',
  },
  filterBadgeText: {
    color: colors.text.primary,
    fontSize: 10,
    fontWeight: 'bold',
  },
  tagsContainer: {
    maxHeight: 50,
    marginBottom: spacing.sm,
  },
  tagsContent: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.xs,
  },
  filterTag: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: borderRadius.lg,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    marginRight: spacing.sm,
  },
  filterTagText: {
    ...typography.caption,
    color: colors.text.primary,
    marginRight: spacing.xs,
    fontWeight: '600',
  },
  clearFiltersButton: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  clearFiltersText: {
    ...typography.caption,
    fontWeight: '600',
  },
  strategiesList: {
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.xl,
  },
  separator: {
    height: spacing.md,
  },
});

export default StrategiesScreen;




