/**
 * 模型指标卡片组件
 *
 * 在Dashboard展示模型性能指标概览
 */

import React from 'react';
import { TrendingUp, TrendingDown, Activity, AlertTriangle } from 'lucide-react';
import { useMonitoring } from '../../hooks/useMonitoring';
import Link from 'next/link';

interface MetricItemProps {
  label: string;
  value: number;
  unit?: string;
  status: 'normal' | 'warning' | 'critical';
  trend?: 'up' | 'down' | 'neutral';
}

const MetricItem: React.FC<MetricItemProps> = ({ label, value, unit, status, trend }) => {
  const statusColors = {
    normal: 'text-green-600 bg-green-50',
    warning: 'text-yellow-600 bg-yellow-50',
    critical: 'text-red-600 bg-red-50',
  };

  return (
    <div className={`p-3 rounded-lg ${statusColors[status]}`}>
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className="flex items-center justify-between">
        <span className="text-lg font-bold">
          {value.toFixed(2)}
          {unit && <span className="text-xs ml-1">{unit}</span>}
        </span>
        {trend && (
          <span>
            {trend === 'up' ? (
              <TrendingUp className="w-4 h-4" />
            ) : trend === 'down' ? (
              <TrendingDown className="w-4 h-4" />
            ) : null}
          </span>
        )}
      </div>
    </div>
  );
};

export const ModelMetricsCard: React.FC = () => {
  const { metrics, loading, error } = useMonitoring('default_model');

  // 模拟数据（实际应从metrics中提取）
  const mockMetrics = {
    accuracy: { value: 0.85, status: 'normal' as const },
    sharpeRatio: { value: 1.25, status: 'normal' as const },
    maxDrawdown: { value: 0.12, status: 'warning' as const },
    winRate: { value: 0.58, status: 'normal' as const },
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">模型性能指标</h3>
        <Link
          href="/monitoring"
          className="text-sm text-blue-600 hover:text-blue-800 transition-colors"
        >
          详细监控 →
        </Link>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-32">
          <Activity className="w-8 h-8 text-blue-500 animate-pulse" />
        </div>
      ) : error ? (
        <div className="flex items-center justify-center h-32 text-red-500">
          <AlertTriangle className="w-6 h-6 mr-2" />
          加载失败
        </div>
      ) : (
        <>
          {/* 关键指标 */}
          <div className="grid grid-cols-2 gap-3 mb-4">
            <MetricItem
              label="准确率"
              value={mockMetrics.accuracy.value * 100}
              unit="%"
              status={mockMetrics.accuracy.status}
              trend="up"
            />
            <MetricItem
              label="夏普比率"
              value={mockMetrics.sharpeRatio.value}
              status={mockMetrics.sharpeRatio.status}
              trend="up"
            />
            <MetricItem
              label="最大回撤"
              value={mockMetrics.maxDrawdown.value * 100}
              unit="%"
              status={mockMetrics.maxDrawdown.status}
              trend="down"
            />
            <MetricItem
              label="胜率"
              value={mockMetrics.winRate.value * 100}
              unit="%"
              status={mockMetrics.winRate.status}
              trend="neutral"
            />
          </div>

          {/* 状态指示器 */}
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
              <span className="text-sm text-gray-600">监控运行中</span>
            </div>
            <span className="text-xs text-gray-400">
              更新于 {new Date().toLocaleTimeString()}
            </span>
          </div>
        </>
      )}
    </div>
  );
};

export default ModelMetricsCard;
