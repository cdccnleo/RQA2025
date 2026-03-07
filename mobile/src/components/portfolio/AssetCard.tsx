/**
 * RQA 2.0 资产卡片组件
 *
 * 展示单个资产的详细信息，包括价格、收益、持有量等
 * 支持点击查看详情和快速操作
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
interface Asset {
  id: string;
  symbol: string;
  name: string;
  type: 'stock' | 'bond' | 'crypto' | 'fund' | 'option';
  quantity: number;
  averagePrice: number;
  currentPrice: number;
  marketValue: number;
  gainLoss: number;
  gainLossPercent: number;
  lastUpdated: string;
}

interface AssetCardProps {
  asset: Asset;
  onPress: () => void;
}

const AssetCard: React.FC<AssetCardProps> = ({asset, onPress}) => {
  const {theme} = useTheme();

  // 格式化金额
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('zh-CN', {
      style: 'currency',
      currency: 'CNY',
      minimumFractionDigits: 2,
    }).format(amount);
  };

  // 格式化百分比
  const formatPercent = (percent: number) => {
    return `${percent >= 0 ? '+' : ''}${percent.toFixed(2)}%`;
  };

  // 获取资产类型图标
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

  // 获取资产类型颜色
  const getAssetTypeColor = (type: string) => {
    switch (type) {
      case 'stock':
        return '#007AFF';
      case 'bond':
        return '#34C759';
      case 'crypto':
        return '#FF9500';
      case 'fund':
        return '#5856D6';
      case 'option':
        return '#FF3B30';
      default:
        return colors.text.secondary;
    }
  };

  // 获取收益颜色
  const getGainLossColor = (value: number) => {
    return value >= 0 ? colors.success : colors.error;
  };

  // 获取收益图标
  const getGainLossIcon = (value: number) => {
    return value >= 0 ? 'trending-up' : 'trending-down';
  };

  return (
    <TouchableOpacity
      style={[styles.container, {backgroundColor: theme.colors.background.secondary}]}
      onPress={onPress}
      activeOpacity={0.7}>

      {/* 资产基本信息 */}
      <View style={styles.assetInfo}>
        <View style={styles.assetHeader}>
          <View style={[styles.assetTypeIcon, {backgroundColor: getAssetTypeColor(asset.type)}]}>
            <Icon name={getAssetTypeIcon(asset.type)} size={16} color={colors.text.primary} />
          </View>
          <View style={styles.assetBasic}>
            <Text style={[styles.assetSymbol, {color: theme.colors.text.primary}]}>
              {asset.symbol}
            </Text>
            <Text style={[styles.assetName, {color: theme.colors.text.secondary}]}>
              {asset.name}
            </Text>
          </View>
          <TouchableOpacity style={styles.moreButton}>
            <Icon name="more-vert" size={20} color={colors.text.secondary} />
          </TouchableOpacity>
        </View>

        {/* 持有信息 */}
        <View style={styles.holdingInfo}>
          <Text style={[styles.holdingText, {color: theme.colors.text.secondary}]}>
            持有: {asset.quantity.toFixed(4)} 股
          </Text>
          <Text style={[styles.avgPriceText, {color: theme.colors.text.secondary}]}>
            均价: {formatCurrency(asset.averagePrice)}
          </Text>
        </View>
      </View>

      {/* 价格和收益信息 */}
      <View style={styles.priceInfo}>
        <View style={styles.currentPrice}>
          <Text style={[styles.priceText, {color: theme.colors.text.primary}]}>
            {formatCurrency(asset.currentPrice)}
          </Text>
          <View style={styles.gainLossContainer}>
            <Icon
              name={getGainLossIcon(asset.gainLoss)}
              size={14}
              color={getGainLossColor(asset.gainLoss)}
            />
            <Text style={[
              styles.gainLossPercent,
              {color: getGainLossColor(asset.gainLoss)}
            ]}>
              {formatPercent(asset.gainLossPercent)}
            </Text>
          </View>
        </View>

        <View style={styles.marketValue}>
          <Text style={[styles.marketValueLabel, {color: theme.colors.text.secondary}]}>
            市值
          </Text>
          <Text style={[styles.marketValueText, {color: theme.colors.text.primary}]}>
            {formatCurrency(asset.marketValue)}
          </Text>
        </View>
      </View>

      {/* 收益详情 */}
      <View style={styles.gainLossDetail}>
        <Text style={[
          styles.gainLossAmount,
          {color: getGainLossColor(asset.gainLoss)}
        ]}>
          {formatCurrency(asset.gainLoss)}
        </Text>
        <Text style={[styles.lastUpdated, {color: theme.colors.text.tertiary}]}>
          {new Date(asset.lastUpdated).toLocaleTimeString('zh-CN', {
            hour: '2-digit',
            minute: '2-digit'
          })}
        </Text>
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
  assetInfo: {
    marginBottom: spacing.md,
  },
  assetHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  assetTypeIcon: {
    width: 32,
    height: 32,
    borderRadius: borderRadius.md,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.md,
  },
  assetBasic: {
    flex: 1,
  },
  assetSymbol: {
    ...typography.h3,
    fontWeight: 'bold',
  },
  assetName: {
    ...typography.caption,
    marginTop: spacing.xs,
  },
  moreButton: {
    padding: spacing.xs,
  },
  holdingInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  holdingText: {
    ...typography.caption,
  },
  avgPriceText: {
    ...typography.caption,
  },
  priceInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  currentPrice: {
    alignItems: 'flex-start',
  },
  priceText: {
    ...typography.h2,
    fontWeight: 'bold',
  },
  gainLossContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing.xs,
  },
  gainLossPercent: {
    ...typography.body,
    fontWeight: '600',
    marginLeft: spacing.xs,
  },
  marketValue: {
    alignItems: 'flex-end',
  },
  marketValueLabel: {
    ...typography.caption,
    color: colors.text.secondary,
  },
  marketValueText: {
    ...typography.h3,
    fontWeight: 'bold',
  },
  gainLossDetail: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: spacing.sm,
    borderTopWidth: 1,
    borderTopColor: colors.border.light,
  },
  gainLossAmount: {
    ...typography.body,
    fontWeight: '600',
  },
  lastUpdated: {
    ...typography.caption,
  },
});

export default AssetCard;




