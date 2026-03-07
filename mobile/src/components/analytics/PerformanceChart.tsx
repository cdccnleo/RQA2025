/**
 * RQA 2.0 性能图表组件
 *
 * 展示投资组合的历史表现和收益曲线
 * 支持多种图表类型和时间范围选择
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import React, {useState} from 'react';
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  TouchableOpacity,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';

// 这里使用简单的图表展示，实际项目中可以使用react-native-chart-kit或其他图表库
import SimpleChart from './SimpleChart';

// 主题和工具
import {useTheme} from '../../theme/ThemeProvider';
import {colors, typography, spacing, borderRadius} from '../../theme/theme';

const {width} = Dimensions.get('window');

const PerformanceChart: React.FC = () => {
  const {theme} = useTheme();

  // 本地状态
  const [chartType, setChartType] = useState<'line' | 'area' | 'bar'>('area');
  const [timeRange, setTimeRange] = useState('1M');

  // 模拟性能数据
  const performanceData = {
    labels: ['1月', '2月', '3月', '4月', '5月', '6月'],
    datasets: [
      {
        data: [100, 105, 98, 112, 108, 125],
        color: colors.success,
        strokeWidth: 2,
      },
      {
        data: [100, 102, 99, 101, 103, 105],
        color: colors.primary,
        strokeWidth: 1,
      }
    ]
  };

  // 关键指标
  const metrics = [
    {
      label: '总收益率',
      value: '+25.0%',
      change: '+2.1%',
      color: colors.success,
    },
    {
      label: '年化收益率',
      value: '18.5%',
      change: '+1.2%',
      color: colors.success,
    },
    {
      label: '最大回撤',
      value: '8.2%',
      change: '-1.5%',
      color: colors.success,
    },
    {
      label: '夏普比率',
      value: '1.85',
      change: '+0.12',
      color: colors.primary,
    },
  ];

  return (
    <View style={[styles.container, {backgroundColor: theme.colors.background.secondary}]}>
      {/* 图表头部 */}
      <View style={styles.header}>
        <Text style={[styles.title, {color: theme.colors.text.primary}]}>
          投资组合表现
        </Text>

        {/* 图表类型切换 */}
        <View style={styles.chartTypeContainer}>
          {[
            {type: 'area', icon: 'area-chart', label: '面积图'},
            {type: 'line', icon: 'show-chart', label: '线图'},
            {type: 'bar', icon: 'bar-chart', label: '柱状图'},
          ].map(item => (
            <TouchableOpacity
              key={item.type}
              style={[
                styles.chartTypeButton,
                chartType === item.type && {backgroundColor: theme.primary}
              ]}
              onPress={() => setChartType(item.type as any)}>
              <Icon
                name={item.icon as any}
                size={16}
                color={chartType === item.type ? colors.text.primary : colors.text.secondary}
              />
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* 图表区域 */}
      <View style={styles.chartContainer}>
        <SimpleChart
          data={performanceData}
          type={chartType}
          width={width - spacing.lg * 4}
          height={200}
        />

        {/* 图例 */}
        <View style={styles.legend}>
          <View style={styles.legendItem}>
            <View style={[styles.legendColor, {backgroundColor: colors.success}]} />
            <Text style={[styles.legendText, {color: theme.colors.text.primary}]}>
              投资组合
            </Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendColor, {backgroundColor: colors.primary}]} />
            <Text style={[styles.legendText, {color: theme.colors.text.primary}]}>
              基准指数
            </Text>
          </View>
        </View>
      </View>

      {/* 关键指标 */}
      <View style={styles.metricsContainer}>
        {metrics.map((metric, index) => (
          <View key={index} style={styles.metricCard}>
            <Text style={[styles.metricLabel, {color: theme.colors.text.secondary}]}>
              {metric.label}
            </Text>
            <Text style={[styles.metricValue, {color: metric.color}]}>
              {metric.value}
            </Text>
            <View style={styles.metricChange}>
              <Icon
                name={metric.change.startsWith('+') ? 'trending-up' : 'trending-down'}
                size={12}
                color={metric.change.startsWith('+') ? colors.success : colors.error}
              />
              <Text style={[
                styles.metricChangeText,
                {color: metric.change.startsWith('+') ? colors.success : colors.error}
              ]}>
                {metric.change}
              </Text>
            </View>
          </View>
        ))}
      </View>

      {/* 时间范围统计 */}
      <View style={styles.periodStats}>
        <View style={styles.periodStat}>
          <Text style={[styles.periodLabel, {color: theme.colors.text.secondary}]}>
            期间最高
          </Text>
          <Text style={[styles.periodValue, {color: colors.success}]}>
            +28.5%
          </Text>
        </View>

        <View style={styles.periodStat}>
          <Text style={[styles.periodLabel, {color: theme.colors.text.secondary}]}>
            期间最低
          </Text>
          <Text style={[styles.periodValue, {color: colors.error}]}>
            -8.2%
          </Text>
        </View>

        <View style={styles.periodStat}>
          <Text style={[styles.periodLabel, {color: theme.colors.text.secondary}]}>
            波动率
          </Text>
          <Text style={[styles.periodValue, {color: colors.warning}]}>
            12.3%
          </Text>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginVertical: spacing.md,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  title: {
    ...typography.h2,
    fontWeight: 'bold',
  },
  chartTypeContainer: {
    flexDirection: 'row',
  },
  chartTypeButton: {
    width: 36,
    height: 36,
    borderRadius: borderRadius.md,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: spacing.sm,
  },
  chartContainer: {
    marginBottom: spacing.lg,
  },
  legend: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: spacing.sm,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: spacing.md,
  },
  legendColor: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: spacing.sm,
  },
  legendText: {
    ...typography.caption,
    fontWeight: '600',
  },
  metricsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  metricCard: {
    width: (width - spacing.lg * 4 - spacing.md) / 2,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.md,
  },
  metricLabel: {
    ...typography.caption,
    marginBottom: spacing.xs,
  },
  metricValue: {
    ...typography.h3,
    fontWeight: 'bold',
    marginBottom: spacing.xs,
  },
  metricChange: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  metricChangeText: {
    ...typography.caption,
    fontWeight: '600',
    marginLeft: spacing.xs,
  },
  periodStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border.light,
  },
  periodStat: {
    alignItems: 'center',
    flex: 1,
  },
  periodLabel: {
    ...typography.caption,
    marginBottom: spacing.xs,
  },
  periodValue: {
    ...typography.body,
    fontWeight: 'bold',
  },
});

export default PerformanceChart;




