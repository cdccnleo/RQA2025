/**
 * RQA 2.0 投资组合汇总组件
 *
 * 展示投资组合的总价值、收益情况、日变化等关键指标
 * 支持实时数据更新和视觉化展示
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
import LinearGradient from 'react-native-linear-gradient';

// 主题和工具
import {useTheme} from '../../theme/ThemeProvider';
import {colors, typography, spacing, borderRadius, shadows} from '../../theme/theme';

// 类型定义
interface PortfolioSummaryData {
  totalValue: number;
  totalGainLoss: number;
  totalGainLossPercent: number;
  dayChange: number;
  dayChangePercent: number;
  cashBalance: number;
  buyingPower: number;
}

interface PortfolioSummaryProps {
  summary: PortfolioSummaryData;
}

const PortfolioSummary: React.FC<PortfolioSummaryProps> = ({summary}) => {
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

  // 获取收益颜色
  const getGainLossColor = (value: number) => {
    return value >= 0 ? colors.success : colors.error;
  };

  // 获取收益图标
  const getGainLossIcon = (value: number) => {
    return value >= 0 ? 'trending-up' : 'trending-down';
  };

  return (
    <LinearGradient
      colors={theme.gradient.background}
      style={styles.container}
      start={{x: 0, y: 0}}
      end={{x: 1, y: 1}}>

      {/* 总价值 */}
      <View style={styles.totalValueSection}>
        <Text style={styles.totalValueLabel}>总资产</Text>
        <Text style={styles.totalValueAmount}>
          {formatCurrency(summary.totalValue)}
        </Text>

        {/* 总收益 */}
        <View style={styles.gainLossContainer}>
          <Icon
            name={getGainLossIcon(summary.totalGainLoss)}
            size={16}
            color={getGainLossColor(summary.totalGainLoss)}
          />
          <Text style={[
            styles.gainLossAmount,
            {color: getGainLossColor(summary.totalGainLoss)}
          ]}>
            {formatCurrency(summary.totalGainLoss)}
          </Text>
          <Text style={[
            styles.gainLossPercent,
            {color: getGainLossColor(summary.totalGainLoss)}
          ]}>
            ({formatPercent(summary.totalGainLossPercent)})
          </Text>
        </View>
      </View>

      {/* 指标卡片 */}
      <View style={styles.metricsContainer}>
        {/* 日变化 */}
        <View style={[styles.metricCard, {backgroundColor: 'rgba(255, 255, 255, 0.1)'}]}>
          <View style={styles.metricHeader}>
            <Icon name="schedule" size={20} color={colors.text.primary} />
            <Text style={styles.metricLabel}>今日变化</Text>
          </View>
          <Text style={[
            styles.metricValue,
            {color: getGainLossColor(summary.dayChange)}
          ]}>
            {formatCurrency(summary.dayChange)}
          </Text>
          <Text style={[
            styles.metricSubtext,
            {color: getGainLossColor(summary.dayChange)}
          ]}>
            {formatPercent(summary.dayChangePercent)}
          </Text>
        </View>

        {/* 现金余额 */}
        <View style={[styles.metricCard, {backgroundColor: 'rgba(255, 255, 255, 0.1)'}]}>
          <View style={styles.metricHeader}>
            <Icon name="account-balance-wallet" size={20} color={colors.text.primary} />
            <Text style={styles.metricLabel}>现金余额</Text>
          </View>
          <Text style={styles.metricValue}>
            {formatCurrency(summary.cashBalance)}
          </Text>
          <TouchableOpacity style={styles.depositButton}>
            <Text style={styles.depositButtonText}>充值</Text>
          </TouchableOpacity>
        </View>

        {/* 购买力 */}
        <View style={[styles.metricCard, {backgroundColor: 'rgba(255, 255, 255, 0.1)'}]}>
          <View style={styles.metricHeader}>
            <Icon name="shopping-cart" size={20} color={colors.text.primary} />
            <Text style={styles.metricLabel}>购买力</Text>
          </View>
          <Text style={styles.metricValue}>
            {formatCurrency(summary.buyingPower)}
          </Text>
          <Text style={styles.metricSubtext}>
            可用于交易
          </Text>
        </View>
      </View>

      {/* 操作按钮 */}
      <View style={styles.actionsContainer}>
        <TouchableOpacity style={[styles.actionButton, {backgroundColor: theme.primary}]}>
          <Icon name="add" size={20} color={colors.text.primary} />
          <Text style={styles.actionButtonText}>买入</Text>
        </TouchableOpacity>

        <TouchableOpacity style={[styles.actionButton, styles.secondaryButton]}>
          <Icon name="remove" size={20} color={colors.primary} />
          <Text style={[styles.actionButtonText, styles.secondaryButtonText]}>
            卖出
          </Text>
        </TouchableOpacity>

        <TouchableOpacity style={[styles.actionButton, styles.secondaryButton]}>
          <Icon name="swap-horiz" size={20} color={colors.primary} />
          <Text style={[styles.actionButtonText, styles.secondaryButtonText]}>
            调仓
          </Text>
        </TouchableOpacity>
      </View>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    margin: spacing.lg,
    borderRadius: borderRadius.xl,
    padding: spacing.lg,
    ...shadows.lg,
  },
  totalValueSection: {
    alignItems: 'center',
    marginBottom: spacing.xl,
  },
  totalValueLabel: {
    ...typography.caption,
    color: colors.text.secondary,
    marginBottom: spacing.xs,
  },
  totalValueAmount: {
    ...typography.h1,
    color: colors.text.primary,
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: spacing.sm,
  },
  gainLossContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  gainLossAmount: {
    ...typography.body,
    fontWeight: '600',
    marginLeft: spacing.xs,
  },
  gainLossPercent: {
    ...typography.caption,
    marginLeft: spacing.xs,
  },
  metricsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: spacing.xl,
  },
  metricCard: {
    flex: 1,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginHorizontal: spacing.xs,
    alignItems: 'center',
  },
  metricHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.xs,
  },
  metricLabel: {
    ...typography.caption,
    color: colors.text.secondary,
    marginLeft: spacing.xs,
  },
  metricValue: {
    ...typography.h3,
    color: colors.text.primary,
    fontWeight: 'bold',
    marginBottom: spacing.xs,
  },
  metricSubtext: {
    ...typography.caption,
    color: colors.text.secondary,
  },
  depositButton: {
    backgroundColor: colors.success,
    borderRadius: borderRadius.sm,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    marginTop: spacing.xs,
  },
  depositButtonText: {
    ...typography.caption,
    color: colors.text.primary,
    fontWeight: '600',
  },
  actionsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.primary,
    borderRadius: borderRadius.lg,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    ...shadows.md,
  },
  actionButtonText: {
    ...typography.button,
    color: colors.text.primary,
    fontWeight: '600',
    marginLeft: spacing.xs,
  },
  secondaryButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderWidth: 1,
    borderColor: colors.primary,
  },
  secondaryButtonText: {
    color: colors.primary,
  },
});

export default PortfolioSummary;




