/**
 * RQA 2.0 简单图表组件
 *
 * 简化的图表实现，用于展示趋势数据
 * 实际项目中应使用专业的图表库如react-native-chart-kit
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
} from 'react-native';
import Svg, {Path, Line, Text as SvgText, Circle} from 'react-native-svg';

// 主题和工具
import {useTheme} from '../../theme/ThemeProvider';
import {colors, typography, spacing} from '../../theme/theme';

interface ChartData {
  labels: string[];
  datasets: Array<{
    data: number[];
    color: string;
    strokeWidth: number;
  }>;
}

interface SimpleChartProps {
  data: ChartData;
  type: 'line' | 'area' | 'bar';
  width: number;
  height: number;
}

const SimpleChart: React.FC<SimpleChartProps> = ({
  data,
  type,
  width,
  height,
}) => {
  const {theme} = useTheme();

  // 计算图表参数
  const padding = 40;
  const chartWidth = width - padding * 2;
  const chartHeight = height - padding * 2;

  // 获取数据范围
  const allData = data.datasets.flatMap(dataset => dataset.data);
  const minValue = Math.min(...allData);
  const maxValue = Math.max(...allData);
  const valueRange = maxValue - minValue || 1;

  // 转换为坐标
  const getPoint = (index: number, value: number) => {
    const x = padding + (index / (data.labels.length - 1)) * chartWidth;
    const y = padding + ((maxValue - value) / valueRange) * chartHeight;
    return {x, y};
  };

  // 生成路径
  const generatePath = (dataset: ChartData['datasets'][0]) => {
    if (dataset.data.length === 0) return '';

    const points = dataset.data.map((value, index) => getPoint(index, value));

    if (type === 'area') {
      const pathData = points.map((point, index) =>
        `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`
      ).join(' ');

      // 封闭面积图
      const bottomY = padding + chartHeight;
      return `${pathData} L ${points[points.length - 1].x} ${bottomY} L ${points[0].x} ${bottomY} Z`;
    } else {
      return points.map((point, index) =>
        `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`
      ).join(' ');
    }
  };

  // 生成柱状图
  const renderBars = () => {
    if (type !== 'bar') return null;

    const barWidth = chartWidth / data.labels.length * 0.8;
    const dataset = data.datasets[0];

    return dataset.data.map((value, index) => {
      const {x, y} = getPoint(index, value);
      const barHeight = ((value - minValue) / valueRange) * chartHeight;
      const barX = x - barWidth / 2;

      return (
        <React.Fragment key={index}>
          <SvgText
            x={x}
            y={padding + chartHeight + 20}
            fontSize="10"
            fill={theme.colors.text.secondary}
            textAnchor="middle">
            {data.labels[index]}
          </SvgText>
          <View
            style={{
              position: 'absolute',
              left: barX,
              top: y,
              width: barWidth,
              height: barHeight,
              backgroundColor: dataset.color,
              borderRadius: 2,
            }}
          />
        </React.Fragment>
      );
    });
  };

  return (
    <View style={styles.container}>
      <Svg width={width} height={height}>
        {/* 网格线 */}
        {[0, 0.25, 0.5, 0.75, 1].map(ratio => {
          const y = padding + ratio * chartHeight;
          return (
            <Line
              key={ratio}
              x1={padding}
              y1={y}
              x2={width - padding}
              y2={y}
              stroke={colors.border.light}
              strokeWidth="1"
              opacity="0.3"
            />
          );
        })}

        {/* 数据线/面积 */}
        {data.datasets.map((dataset, datasetIndex) => (
          <Path
            key={datasetIndex}
            d={generatePath(dataset)}
            fill={type === 'area' ? dataset.color : 'none'}
            fillOpacity={type === 'area' ? 0.3 : 0}
            stroke={dataset.color}
            strokeWidth={dataset.strokeWidth}
          />
        ))}

        {/* 数据点 */}
        {type !== 'bar' && data.datasets.map((dataset, datasetIndex) =>
          dataset.data.map((value, index) => {
            const {x, y} = getPoint(index, value);
            return (
              <Circle
                key={`${datasetIndex}-${index}`}
                cx={x}
                cy={y}
                r="3"
                fill={dataset.color}
                stroke={colors.text.primary}
                strokeWidth="1"
              />
            );
          })
        )}

        {/* X轴标签 */}
        {type !== 'bar' && data.labels.map((label, index) => {
          const {x} = getPoint(index, minValue);
          return (
            <SvgText
              key={index}
              x={x}
              y={padding + chartHeight + 20}
              fontSize="10"
              fill={theme.colors.text.secondary}
              textAnchor="middle">
              {label}
            </SvgText>
          );
        })}

        {/* Y轴标签 */}
        {[0, 0.5, 1].map(ratio => {
          const value = minValue + (maxValue - minValue) * ratio;
          const y = padding + (1 - ratio) * chartHeight;
          return (
            <SvgText
              key={ratio}
              x={padding - 10}
              y={y + 3}
              fontSize="10"
              fill={theme.colors.text.secondary}
              textAnchor="end">
              {value.toFixed(0)}
            </SvgText>
          );
        })}
      </Svg>

      {/* 柱状图 */}
      {renderBars()}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'relative',
  },
});

export default SimpleChart;




