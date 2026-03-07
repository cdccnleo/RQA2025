/**
 * RQA 2.0 分析屏幕
 *
 * 提供投资分析和数据可视化功能：
 * - 投资组合表现分析
 * - 策略回测结果
 * - 市场数据图表
 * - 风险分析报告
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import React, {useState} from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';

// 组件
import PerformanceChart from '../../components/analytics/PerformanceChart';
import RiskMetrics from '../../components/analytics/RiskMetrics';
import StrategyComparison from '../../components/analytics/StrategyComparison';
import MarketOverview from '../../components/analytics/MarketOverview';

// 主题和工具
import {useTheme} from '../../theme/ThemeProvider';
import {colors, typography, spacing, borderRadius} from '../../theme/theme';

const {width} = Dimensions.get('window');

const AnalyticsScreen: React.FC = () => {
  const {theme} = useTheme();

  // 本地状态
  const [activeTab, setActiveTab] = useState<'performance' | 'risk' | 'strategies' | 'market'>('performance');

  // 标签页配置
  const tabs = [
    {key: 'performance', label: '表现', icon: 'show-chart'},
    {key: 'risk', label: '风险', icon: 'security'},
    {key: 'strategies', label: '策略', icon: 'analytics'},
    {key: 'market', label: '市场', icon: 'trending-up'},
  ];

  // 渲染标签页内容
  const renderTabContent = () => {
    switch (activeTab) {
      case 'performance':
        return <PerformanceChart />;
      case 'risk':
        return <RiskMetrics />;
      case 'strategies':
        return <StrategyComparison />;
      case 'market':
        return <MarketOverview />;
      default:
        return <PerformanceChart />;
    }
  };

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
      <ScrollView
        style={styles.content}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.contentContainer}>

        {/* 标题 */}
        <View style={styles.header}>
          <Text style={[styles.title, {color: theme.colors.text.primary}]}>
            {tabs.find(tab => tab.key === activeTab)?.label}分析
          </Text>
          <Text style={[styles.subtitle, {color: theme.colors.text.secondary}]}>
            {getTabDescription(activeTab)}
          </Text>
        </View>

        {/* 标签页内容 */}
        {renderTabContent()}

        {/* 时间范围选择器 */}
        <View style={styles.timeRangeContainer}>
          <Text style={[styles.timeRangeLabel, {color: theme.colors.text.primary}]}>
            时间范围:
          </Text>
          <View style={styles.timeRangeButtons}>
            {['1W', '1M', '3M', '6M', '1Y', 'ALL'].map(range => (
              <TouchableOpacity
                key={range}
                style={[styles.timeRangeButton, {backgroundColor: theme.colors.background.secondary}]}>
                <Text style={[styles.timeRangeText, {color: theme.colors.text.primary}]}>
                  {range}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      </ScrollView>
    </View>
  );
};

// 获取标签页描述
const getTabDescription = (tab: string) => {
  const descriptions = {
    performance: '投资组合历史表现和收益分析',
    risk: '风险指标评估和压力测试',
    strategies: '量化策略对比和回测分析',
    market: '市场概览和行业趋势分析',
  };
  return descriptions[tab as keyof typeof descriptions] || '';
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
  content: {
    flex: 1,
  },
  contentContainer: {
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.xl,
  },
  header: {
    marginBottom: spacing.lg,
  },
  title: {
    ...typography.h1,
    fontWeight: 'bold',
    marginBottom: spacing.xs,
  },
  subtitle: {
    ...typography.body,
    lineHeight: 20,
  },
  timeRangeContainer: {
    marginTop: spacing.lg,
  },
  timeRangeLabel: {
    ...typography.body,
    fontWeight: '600',
    marginBottom: spacing.md,
  },
  timeRangeButtons: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  timeRangeButton: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.lg,
    marginRight: spacing.sm,
    marginBottom: spacing.sm,
  },
  timeRangeText: {
    ...typography.caption,
    fontWeight: '600',
  },
});

export default AnalyticsScreen;




