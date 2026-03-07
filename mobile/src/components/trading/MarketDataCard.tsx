/**
 * RQA 2.0 市场数据卡片组件
 *
 * 展示股票实时报价、涨跌幅、成交量等市场信息
 * 支持点击交易和添加到自选股
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';

// 主题和工具
import {useTheme} from '../../theme/ThemeProvider';
import {colors, typography, spacing, borderRadius, shadows} from '../../theme/theme';

// 类型定义
interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  close: number;
  timestamp: string;
}

interface MarketDataCardProps {
  data: MarketData;
  onPress: () => void;
  onToggleWatchlist: (symbol: string) => void;
  isInWatchlist: boolean;
}

const MarketDataCard: React.FC<MarketDataCardProps> = ({
  data,
  onPress,
  onToggleWatchlist,
  isInWatchlist,
}) => {
  const {theme} = useTheme();

  // 格式化价格
  const formatPrice = (price: number) => {
    return price.toFixed(2);
  };

  // 格式化百分比
  const formatPercent = (percent: number) => {
    return `${percent >= 0 ? '+' : ''}${percent.toFixed(2)}%`;
  };

  // 格式化成交量
  const formatVolume = (volume: number) => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(1)}M`;
    } else if (volume >= 1000) {
      return `${(volume / 1000).toFixed(1)}K`;
    }
    return volume.toString();
  };

  // 获取涨跌颜色
  const getChangeColor = (change: number) => {
    return change >= 0 ? colors.success : colors.error;
  };

  // 获取涨跌图标
  const getChangeIcon = (change: number) => {
    return change >= 0 ? 'trending-up' : 'trending-down';
  };

  return (
    <TouchableOpacity
      style={[styles.container, {backgroundColor: theme.colors.background.secondary}]}
      onPress={onPress}
      activeOpacity={0.7}>

      {/* 股票基本信息 */}
      <View style={styles.stockInfo}>
        <View style={styles.symbolContainer}>
          <Text style={[styles.symbol, {color: theme.colors.text.primary}]}>
            {data.symbol}
          </Text>
          <TouchableOpacity
            style={styles.watchlistButton}
            onPress={() => onToggleWatchlist(data.symbol)}>
            <Icon
              name={isInWatchlist ? "star" : "star-border"}
              size={20}
              color={isInWatchlist ? colors.warning : colors.text.secondary}
            />
          </TouchableOpacity>
        </View>

        {/* 价格和涨跌 */}
        <View style={styles.priceContainer}>
          <Text style={[styles.price, {color: theme.colors.text.primary}]}>
            ¥{formatPrice(data.price)}
          </Text>

          <View style={styles.changeContainer}>
            <Icon
              name={getChangeIcon(data.change)}
              size={16}
              color={getChangeColor(data.change)}
            />
            <Text style={[
              styles.change,
              {color: getChangeColor(data.change)}
            ]}>
              {formatPrice(Math.abs(data.change))}
            </Text>
            <Text style={[
              styles.changePercent,
              {color: getChangeColor(data.change)}
            ]}>
              ({formatPercent(data.changePercent)})
            </Text>
          </View>
        </View>
      </View>

      {/* 市场统计 */}
      <View style={styles.marketStats}>
        <View style={styles.statItem}>
          <Text style={[styles.statLabel, {color: theme.colors.text.secondary}]}>
            最高
          </Text>
          <Text style={[styles.statValue, {color: theme.colors.text.primary}]}>
            ¥{formatPrice(data.high)}
          </Text>
        </View>

        <View style={styles.statItem}>
          <Text style={[styles.statLabel, {color: theme.colors.text.secondary}]}>
            最低
          </Text>
          <Text style={[styles.statValue, {color: theme.colors.text.primary}]}>
            ¥{formatPrice(data.low)}
          </Text>
        </View>

        <View style={styles.statItem}>
          <Text style={[styles.statLabel, {color: theme.colors.text.secondary}]}>
            成交量
          </Text>
          <Text style={[styles.statValue, {color: theme.colors.text.primary}]}>
            {formatVolume(data.volume)}
          </Text>
        </View>
      </View>

      {/* 操作按钮 */}
      <View style={styles.actions}>
        <TouchableOpacity
          style={[styles.actionButton, styles.buyButton]}
          onPress={() => {
            // 直接触发买入操作
            onPress();
          }}>
          <Text style={styles.buyButtonText}>买入</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.sellButton]}
          onPress={() => {
            // 直接触发卖出操作
            onPress();
          }}>
          <Text style={styles.sellButtonText}>卖出</Text>
        </TouchableOpacity>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginVertical: spacing.xs,
    ...shadows.sm,
  },
  stockInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: spacing.md,
  },
  symbolContainer: {
    flex: 1,
  },
  symbol: {
    ...typography.h2,
    fontWeight: 'bold',
  },
  watchlistButton: {
    marginTop: spacing.xs,
  },
  priceContainer: {
    alignItems: 'flex-end',
  },
  price: {
    ...typography.h2,
    fontWeight: 'bold',
  },
  changeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing.xs,
  },
  change: {
    ...typography.body,
    fontWeight: '600',
    marginLeft: spacing.xs,
  },
  changePercent: {
    ...typography.caption,
    marginLeft: spacing.xs,
  },
  marketStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: spacing.md,
  },
  statItem: {
    alignItems: 'center',
    flex: 1,
  },
  statLabel: {
    ...typography.caption,
    marginBottom: spacing.xs,
  },
  statValue: {
    ...typography.body,
    fontWeight: '600',
  },
  actions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  actionButton: {
    flex: 1,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
    alignItems: 'center',
    marginHorizontal: spacing.xs,
  },
  buyButton: {
    backgroundColor: colors.success,
  },
  buyButtonText: {
    ...typography.button,
    color: colors.text.primary,
    fontWeight: 'bold',
  },
  sellButton: {
    backgroundColor: colors.error,
  },
  sellButtonText: {
    ...typography.button,
    color: colors.text.primary,
    fontWeight: 'bold',
  },
});

export default MarketDataCard;




