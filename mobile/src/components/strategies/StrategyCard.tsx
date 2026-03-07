/**
 * RQA 2.0 策略卡片组件
 *
 * 展示量化策略的基本信息、性能指标和风险等级
 * 支持快速预览和详情导航
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
interface Strategy {
  id: string;
  name: string;
  description: string;
  category: 'momentum' | 'mean_reversion' | 'arbitrage' | 'ml_based' | 'risk_parity';
  riskLevel: 'low' | 'medium' | 'high';
  timeHorizon: 'short' | 'medium' | 'long';
  expectedReturn: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  backtestPeriod: string;
  isActive: boolean;
  popularity: number;
  author: string;
  createdAt: string;
  lastUpdated: string;
  tags: string[];
}

interface StrategyCardProps {
  strategy: Strategy;
  onPress: () => void;
}

const StrategyCard: React.FC<StrategyCardProps> = ({strategy, onPress}) => {
  const {theme} = useTheme();

  // 格式化百分比
  const formatPercent = (percent: number) => {
    return `${percent >= 0 ? '+' : ''}${percent.toFixed(2)}%`;
  };

  // 获取策略类别信息
  const getCategoryInfo = (category: string) => {
    const categories = {
      momentum: {name: '动量策略', color: '#007AFF', icon: 'trending-up'},
      mean_reversion: {name: '均值回归', color: '#34C759', icon: 'undo'},
      arbitrage: {name: '套利策略', color: '#FF9500', icon: 'compare-arrows'},
      ml_based: {name: '机器学习', color: '#5856D6', icon: 'memory'},
      risk_parity: {name: '风险平价', color: '#FF3B30', icon: 'balance'},
    };
    return categories[category as keyof typeof categories] || categories.momentum;
  };

  // 获取风险等级信息
  const getRiskLevelInfo = (riskLevel: string) => {
    const levels = {
      low: {name: '低风险', color: colors.success, bgColor: 'rgba(52, 199, 89, 0.1)'},
      medium: {name: '中风险', color: '#FF9500', bgColor: 'rgba(255, 149, 0, 0.1)'},
      high: {name: '高风险', color: colors.error, bgColor: 'rgba(255, 59, 48, 0.1)'},
    };
    return levels[riskLevel as keyof typeof levels] || levels.medium;
  };

  // 获取时间周期信息
  const getTimeHorizonInfo = (timeHorizon: string) => {
    const horizons = {
      short: {name: '短期', icon: 'schedule'},
      medium: {name: '中期', icon: 'date-range'},
      long: {name: '长期', icon: 'timeline'},
    };
    return horizons[timeHorizon as keyof typeof horizons] || horizons.medium;
  };

  const categoryInfo = getCategoryInfo(strategy.category);
  const riskInfo = getRiskLevelInfo(strategy.riskLevel);
  const timeInfo = getTimeHorizonInfo(strategy.timeHorizon);

  return (
    <TouchableOpacity
      style={[styles.container, {backgroundColor: theme.colors.background.secondary}]}
      onPress={onPress}
      activeOpacity={0.7}>

      {/* 策略头部信息 */}
      <View style={styles.header}>
        <View style={styles.categoryContainer}>
          <View style={[styles.categoryIcon, {backgroundColor: categoryInfo.color}]}>
            <Icon name={categoryInfo.icon} size={16} color={colors.text.primary} />
          </View>
          <Text style={[styles.categoryName, {color: categoryInfo.color}]}>
            {categoryInfo.name}
          </Text>
        </View>

        <View style={styles.statusContainer}>
          {strategy.isActive && (
            <View style={[styles.statusBadge, {backgroundColor: colors.success}]}>
              <Text style={styles.statusText}>活跃</Text>
            </View>
          )}
          <TouchableOpacity style={styles.favoriteButton}>
            <Icon name="favorite-border" size={20} color={colors.text.secondary} />
          </TouchableOpacity>
        </View>
      </View>

      {/* 策略名称和描述 */}
      <View style={styles.content}>
        <Text style={[styles.strategyName, {color: theme.colors.text.primary}]}>
          {strategy.name}
        </Text>
        <Text style={[styles.strategyDescription, {color: theme.colors.text.secondary}]}>
          {strategy.description}
        </Text>
      </View>

      {/* 关键指标 */}
      <View style={styles.metrics}>
        <View style={styles.metricItem}>
          <Text style={[styles.metricLabel, {color: theme.colors.text.secondary}]}>
            预期收益
          </Text>
          <Text style={[styles.metricValue, {color: colors.success}]}>
            {formatPercent(strategy.expectedReturn)}
          </Text>
        </View>

        <View style={styles.metricItem}>
          <Text style={[styles.metricLabel, {color: theme.colors.text.secondary}]}>
            夏普比率
          </Text>
          <Text style={[styles.metricValue, {color: theme.colors.text.primary}]}>
            {strategy.sharpeRatio.toFixed(2)}
          </Text>
        </View>

        <View style={styles.metricItem}>
          <Text style={[styles.metricLabel, {color: theme.colors.text.secondary}]}>
            胜率
          </Text>
          <Text style={[styles.metricValue, {color: theme.colors.text.primary}]}>
            {formatPercent(strategy.winRate)}
          </Text>
        </View>
      </View>

      {/* 风险和时间信息 */}
      <View style={styles.footer}>
        <View style={[styles.riskBadge, {backgroundColor: riskInfo.bgColor}]}>
          <Text style={[styles.riskText, {color: riskInfo.color}]}>
            {riskInfo.name}
          </Text>
        </View>

        <View style={styles.timeInfo}>
          <Icon name={timeInfo.icon} size={14} color={colors.text.secondary} />
          <Text style={[styles.timeText, {color: theme.colors.text.secondary}]}>
            {timeInfo.name}
          </Text>
        </View>

        <View style={styles.popularity}>
          <Icon name="people" size={14} color={colors.text.secondary} />
          <Text style={[styles.popularityText, {color: theme.colors.text.secondary}]}>
            {strategy.popularity}
          </Text>
        </View>
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
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  categoryContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  categoryIcon: {
    width: 28,
    height: 28,
    borderRadius: borderRadius.md,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.sm,
  },
  categoryName: {
    ...typography.caption,
    fontWeight: '600',
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusBadge: {
    borderRadius: borderRadius.sm,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    marginRight: spacing.sm,
  },
  statusText: {
    ...typography.caption,
    color: colors.text.primary,
    fontWeight: '600',
  },
  favoriteButton: {
    padding: spacing.xs,
  },
  content: {
    marginBottom: spacing.md,
  },
  strategyName: {
    ...typography.h3,
    fontWeight: 'bold',
    marginBottom: spacing.xs,
  },
  strategyDescription: {
    ...typography.body,
    lineHeight: 20,
  },
  metrics: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: spacing.md,
  },
  metricItem: {
    alignItems: 'center',
    flex: 1,
  },
  metricLabel: {
    ...typography.caption,
    marginBottom: spacing.xs,
  },
  metricValue: {
    ...typography.body,
    fontWeight: 'bold',
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  riskBadge: {
    borderRadius: borderRadius.sm,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
  },
  riskText: {
    ...typography.caption,
    fontWeight: '600',
  },
  timeInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  timeText: {
    ...typography.caption,
    marginLeft: spacing.xs,
  },
  popularity: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  popularityText: {
    ...typography.caption,
    marginLeft: spacing.xs,
  },
});

export default StrategyCard;




