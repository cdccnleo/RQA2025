/**
 * RQA 2.0 交易屏幕
 *
 * 提供完整的交易功能：市场数据展示、订单管理、快速交易
 * 支持实时报价、交易执行、订单跟踪
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
  Dimensions,
  TextInput,
} from 'react-native';
import {useDispatch, useSelector} from 'react-redux';
import Icon from 'react-native-vector-icons/MaterialIcons';

// Redux actions and selectors
import {
  fetchMarketData,
  selectMarketData,
  selectWatchlist,
  selectOrders,
  selectPositions,
  addToWatchlist,
  removeFromWatchlist,
  selectTradingLoading,
} from '../../store/slices/tradingSlice';

// 组件
import MarketDataCard from '../../components/trading/MarketDataCard';
import OrderCard from '../../components/trading/OrderCard';
import QuickTradeModal from '../../components/trading/QuickTradeModal';
import LoadingSpinner from '../../components/common/LoadingSpinner';

// 主题和工具
import {useTheme} from '../../theme/ThemeProvider';
import {colors, typography, spacing, borderRadius} from '../../theme/theme';

const {width} = Dimensions.get('window');

const TradingScreen: React.FC = () => {
  const dispatch = useDispatch();
  const {theme} = useTheme();

  // Redux state
  const marketData = useSelector(selectMarketData);
  const watchlist = useSelector(selectWatchlist);
  const orders = useSelector(selectOrders);
  const positions = useSelector(selectPositions);
  const isLoading = useSelector(selectTradingLoading);

  // 本地状态
  const [activeTab, setActiveTab] = useState<'market' | 'orders' | 'positions'>('market');
  const [searchSymbol, setSearchSymbol] = useState('');
  const [showQuickTrade, setShowQuickTrade] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);

  // 初始化数据
  useEffect(() => {
    if (watchlist.length > 0) {
      dispatch(fetchMarketData(watchlist) as any);
    }
  }, [dispatch, watchlist]);

  // 定期更新市场数据
  useEffect(() => {
    const interval = setInterval(() => {
      if (watchlist.length > 0) {
        dispatch(fetchMarketData(watchlist) as any);
      }
    }, 5000); // 每5秒更新一次

    return () => clearInterval(interval);
  }, [dispatch, watchlist]);

  // 过滤的自选股数据
  const watchlistData = watchlist
    .map(symbol => ({
      symbol,
      ...marketData[symbol],
    }))
    .filter(item => item.price !== undefined);

  // 搜索过滤
  const filteredWatchlist = watchlistData.filter(item =>
    item.symbol.toLowerCase().includes(searchSymbol.toLowerCase())
  );

  // 渲染市场数据项
  const renderMarketItem = ({item}: {item: any}) => (
    <MarketDataCard
      data={item}
      onPress={() => {
        setSelectedSymbol(item.symbol);
        setShowQuickTrade(true);
      }}
      onToggleWatchlist={(symbol) => {
        if (watchlist.includes(symbol)) {
          dispatch(removeFromWatchlist(symbol));
        } else {
          dispatch(addToWatchlist(symbol));
        }
      }}
      isInWatchlist={watchlist.includes(item.symbol)}
    />
  );

  // 渲染订单项
  const renderOrderItem = ({item}: {item: any}) => (
    <OrderCard order={item} />
  );

  // 渲染持仓项
  const renderPositionItem = ({item}: {item: any}) => (
    <View style={[styles.positionCard, {backgroundColor: theme.colors.background.secondary}]}>
      <View style={styles.positionHeader}>
        <Text style={[styles.positionSymbol, {color: theme.colors.text.primary}]}>
          {item.symbol}
        </Text>
        <View style={styles.positionValue}>
          <Text style={[styles.positionQuantity, {color: theme.colors.text.secondary}]}>
            {item.quantity} 股
          </Text>
          <Text style={[styles.positionPrice, {color: theme.colors.text.primary}]}>
            ¥{item.averagePrice.toFixed(2)}
          </Text>
        </View>
      </View>

      <View style={styles.positionMetrics}>
        <View style={styles.metric}>
          <Text style={[styles.metricLabel, {color: theme.colors.text.secondary}]}>
            市值
          </Text>
          <Text style={[styles.metricValue, {color: theme.colors.text.primary}]}>
            ¥{item.marketValue.toFixed(2)}
          </Text>
        </View>

        <View style={styles.metric}>
          <Text style={[styles.metricLabel, {color: theme.colors.text.secondary}]}>
            未实现盈亏
          </Text>
          <Text style={[
            styles.metricValue,
            {color: item.unrealizedPnL >= 0 ? colors.success : colors.error}
          ]}>
            {item.unrealizedPnL >= 0 ? '+' : ''}¥{item.unrealizedPnL.toFixed(2)}
          </Text>
        </View>
      </View>
    </View>
  );

  // 标签页配置
  const tabs = [
    {key: 'market', label: '市场', icon: 'trending-up'},
    {key: 'orders', label: '订单', icon: 'receipt'},
    {key: 'positions', label: '持仓', icon: 'account-balance'},
  ];

  return (
    <View style={[styles.container, {backgroundColor: theme.colors.background.primary}]}>
      {/* 标签页切换 */}
      <View style={styles.tabContainer}>
        {tabs.map(tab => (
          <TouchableOpacity
            key={tab.key}
            style={[
              styles.tab,
              activeTab === tab.key && {backgroundColor: theme.primary}
            ]}
            onPress={() => setActiveTab(tab.key as any)}>
            <Icon
              name={tab.icon}
              size={20}
              color={activeTab === tab.key ? colors.text.primary : colors.text.secondary}
            />
            <Text style={[
              styles.tabText,
              activeTab === tab.key && {color: colors.text.primary}
            ]}>
              {tab.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* 内容区域 */}
      {activeTab === 'market' && (
        <View style={styles.marketContent}>
          {/* 搜索栏 */}
          <View style={[styles.searchContainer, {backgroundColor: theme.colors.background.secondary}]}>
            <Icon name="search" size={20} color={colors.text.secondary} />
            <TextInput
              style={[styles.searchInput, {color: theme.colors.text.primary}]}
              placeholder="搜索股票代码"
              placeholderTextColor={colors.text.secondary}
              value={searchSymbol}
              onChangeText={setSearchSymbol}
            />
            {searchSymbol ? (
              <TouchableOpacity onPress={() => setSearchSymbol('')}>
                <Icon name="clear" size={20} color={colors.text.secondary} />
              </TouchableOpacity>
            ) : null}
          </View>

          {/* 市场数据列表 */}
          {filteredWatchlist.length > 0 ? (
            <FlatList
              data={filteredWatchlist}
              renderItem={renderMarketItem}
              keyExtractor={item => item.symbol}
              contentContainerStyle={styles.marketList}
              showsVerticalScrollIndicator={false}
            />
          ) : (
            <View style={styles.emptyContainer}>
              <Text style={[styles.emptyText, {color: theme.colors.text.secondary}]}>
                {searchSymbol ? '未找到相关股票' : '暂无市场数据'}
              </Text>
            </View>
          )}
        </View>
      )}

      {activeTab === 'orders' && (
        <View style={styles.ordersContent}>
          {orders.length > 0 ? (
            <FlatList
              data={orders}
              renderItem={renderOrderItem}
              keyExtractor={item => item.id}
              contentContainerStyle={styles.ordersList}
              showsVerticalScrollIndicator={false}
            />
          ) : (
            <View style={styles.emptyContainer}>
              <Text style={[styles.emptyText, {color: theme.colors.text.secondary}]}>
                暂无订单记录
              </Text>
            </View>
          )}
        </View>
      )}

      {activeTab === 'positions' && (
        <View style={styles.positionsContent}>
          {positions.length > 0 ? (
            <FlatList
              data={positions}
              renderItem={renderPositionItem}
              keyExtractor={item => item.symbol}
              contentContainerStyle={styles.positionsList}
              showsVerticalScrollIndicator={false}
            />
          ) : (
            <View style={styles.emptyContainer}>
              <Text style={[styles.emptyText, {color: theme.colors.text.secondary}]}>
                暂无持仓记录
              </Text>
            </View>
          )}
        </View>
      )}

      {/* 快速交易模态框 */}
      <QuickTradeModal
        visible={showQuickTrade}
        symbol={selectedSymbol}
        onClose={() => {
          setShowQuickTrade(false);
          setSelectedSymbol(null);
        }}
      />

      {/* 加载状态 */}
      {isLoading && (
        <View style={styles.loadingOverlay}>
          <LoadingSpinner />
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  tabContainer: {
    flexDirection: 'row',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.md,
    borderRadius: borderRadius.lg,
    marginHorizontal: spacing.xs,
  },
  tabText: {
    ...typography.caption,
    fontWeight: '600',
    marginLeft: spacing.xs,
    color: colors.text.secondary,
  },
  marketContent: {
    flex: 1,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: spacing.lg,
    marginBottom: spacing.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.lg,
  },
  searchInput: {
    flex: 1,
    marginLeft: spacing.sm,
    ...typography.body,
  },
  marketList: {
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.xl,
  },
  ordersContent: {
    flex: 1,
  },
  ordersList: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  positionsContent: {
    flex: 1,
  },
  positionsList: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  positionCard: {
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.md,
  },
  positionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  positionSymbol: {
    ...typography.h3,
    fontWeight: 'bold',
  },
  positionValue: {
    alignItems: 'flex-end',
  },
  positionQuantity: {
    ...typography.caption,
  },
  positionPrice: {
    ...typography.body,
    fontWeight: '600',
  },
  positionMetrics: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  metric: {
    alignItems: 'center',
  },
  metricLabel: {
    ...typography.caption,
    marginBottom: spacing.xs,
  },
  metricValue: {
    ...typography.body,
    fontWeight: '600',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyText: {
    ...typography.body,
    textAlign: 'center',
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default TradingScreen;




