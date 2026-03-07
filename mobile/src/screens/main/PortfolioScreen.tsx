/**
 * RQA 2.0 投资组合屏幕
 *
 * 展示用户的投资组合概览、资产列表、收益情况
 * 支持资产详情查看和组合分析
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import React, {useEffect, useState} from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  FlatList,
  RefreshControl,
  Dimensions,
} from 'react-native';
import {useDispatch, useSelector} from 'react-redux';
import Icon from 'react-native-vector-icons/MaterialIcons';

// 组件
import AssetCard from '../../components/portfolio/AssetCard';
import PortfolioSummary from '../../components/portfolio/PortfolioSummary';
import LoadingSpinner from '../../components/common/LoadingSpinner';
import EmptyState from '../../components/common/EmptyState';

// Redux actions
import {
  fetchPortfolio,
  updateAssetPrices,
  selectPortfolioAssets,
  selectPortfolioSummary,
  selectPortfolioLoading,
  selectPortfolioError,
} from '../../store/slices/portfolioSlice';

// 主题和工具
import {useTheme} from '../../theme/ThemeProvider';
import {colors, typography, spacing, borderRadius, shadows} from '../../theme/theme';

const {width} = Dimensions.get('window');

const PortfolioScreen: React.FC = () => {
  const dispatch = useDispatch();
  const {theme} = useTheme();

  // Redux state
  const assets = useSelector(selectPortfolioAssets);
  const summary = useSelector(selectPortfolioSummary);
  const isLoading = useSelector(selectPortfolioLoading);
  const error = useSelector(selectPortfolioError);

  // 本地状态
  const [refreshing, setRefreshing] = useState(false);
  const [sortBy, setSortBy] = useState<'value' | 'gainLoss' | 'name'>('value');

  // 初始化数据
  useEffect(() => {
    dispatch(fetchPortfolio() as any);
  }, [dispatch]);

  // 刷新处理
  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await dispatch(updateAssetPrices() as any);
    } catch (error) {
      console.error('刷新价格失败:', error);
    } finally {
      setRefreshing(false);
    }
  };

  // 资产排序
  const sortedAssets = [...assets].sort((a, b) => {
    switch (sortBy) {
      case 'value':
        return b.marketValue - a.marketValue;
      case 'gainLoss':
        return b.gainLossPercent - a.gainLossPercent;
      case 'name':
        return a.name.localeCompare(b.name);
      default:
        return 0;
    }
  });

  // 资产类型图标
  const getAssetTypeIcon = (type: string) => {
    switch (type) {
      case 'stock':
        return 'trending-up';
      case 'bond':
        return 'account-balance';
      case 'crypto':
        return 'currency-bitcoin';
      case 'fund':
        return 'pie-chart';
      case 'option':
        return 'call';
      default:
        return 'help';
    }
  };

  // 渲染资产项
  const renderAssetItem = ({item}: {item: any}) => (
    <AssetCard
      asset={item}
      onPress={() => {
        // 导航到资产详情
        console.log('Navigate to asset details:', item.id);
      }}
    />
  );

  // 排序按钮
  const renderSortButtons = () => (
    <View style={styles.sortContainer}>
      <TouchableOpacity
        style={[
          styles.sortButton,
          sortBy === 'value' && {backgroundColor: theme.primary},
        ]}
        onPress={() => setSortBy('value')}>
        <Text
          style={[
            styles.sortButtonText,
            sortBy === 'value' && {color: colors.text.primary},
          ]}>
          市值
        </Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[
          styles.sortButton,
          sortBy === 'gainLoss' && {backgroundColor: theme.primary},
        ]}
        onPress={() => setSortBy('gainLoss')}>
        <Text
          style={[
            styles.sortButtonText,
            sortBy === 'gainLoss' && {color: colors.text.primary},
          ]}>
          收益
        </Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[
          styles.sortButton,
          sortBy === 'name' && {backgroundColor: theme.primary},
        ]}
        onPress={() => setSortBy('name')}>
        <Text
          style={[
            styles.sortButtonText,
            sortBy === 'name' && {color: colors.text.primary},
          ]}>
          名称
        </Text>
      </TouchableOpacity>
    </View>
  );

  // 加载状态
  if (isLoading && !summary) {
    return <LoadingSpinner />;
  }

  // 错误状态
  if (error && !summary) {
    return (
      <View style={styles.centerContainer}>
        <EmptyState
          icon="error"
          title="加载失败"
          message={error}
          actionText="重试"
          onAction={() => dispatch(fetchPortfolio() as any)}
        />
      </View>
    );
  }

  return (
    <View style={[styles.container, {backgroundColor: theme.colors.background.primary}]}>
      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={handleRefresh}
            colors={[theme.primary]}
            tintColor={theme.primary}
          />
        }>

        {/* 投资组合汇总 */}
        {summary && <PortfolioSummary summary={summary} />}

        {/* 资产列表标题 */}
        <View style={styles.sectionHeader}>
          <Text style={[styles.sectionTitle, {color: theme.colors.text.primary}]}>
            资产明细 ({assets.length})
          </Text>
          {renderSortButtons()}
        </View>

        {/* 资产列表 */}
        {sortedAssets.length > 0 ? (
          <FlatList
            data={sortedAssets}
            renderItem={renderAssetItem}
            keyExtractor={item => item.id}
            scrollEnabled={false}
            contentContainerStyle={styles.assetsList}
            ItemSeparatorComponent={() => <View style={styles.separator} />}
          />
        ) : (
          <EmptyState
            icon="account-balance-wallet"
            title="暂无资产"
            message="开始您的投资之旅"
            actionText="添加资产"
            onAction={() => {
              // 导航到添加资产页面
              console.log('Navigate to add asset');
            }}
          />
        )}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  sectionTitle: {
    ...typography.h2,
    fontWeight: 'bold',
  },
  sortContainer: {
    flexDirection: 'row',
    backgroundColor: colors.background.secondary,
    borderRadius: borderRadius.lg,
    padding: spacing.xs,
  },
  sortButton: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.md,
    marginHorizontal: spacing.xs,
  },
  sortButtonText: {
    ...typography.caption,
    fontWeight: '600',
    color: colors.text.secondary,
  },
  assetsList: {
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.xl,
  },
  separator: {
    height: spacing.sm,
  },
});

export default PortfolioScreen;




