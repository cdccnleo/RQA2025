/**
 * 监控总览页面
 */

import React, { useState } from 'react';
import { NextPage } from 'next';
import Head from 'next/head';
import { Activity, TrendingUp, TrendingDown, AlertTriangle, RefreshCw } from 'lucide-react';
import { useMonitoring, useDriftDetection } from '../../hooks/useMonitoring';

const MonitoringPage: NextPage = () => {
  const { metrics, loading, refresh } = useMonitoring('default_model');
  const { summary: driftSummary } = useDriftDetection();

  // 模拟指标数据
  const mockMetrics = {
    technical: {
      accuracy: { value: 0.85, threshold: 0.7, status: 'normal' as const },
      f1Score: { value: 0.82, threshold: 0.65, status: 'normal' as const },
      precision: { value: 0.84, threshold: 0.6, status: 'normal' as const },
      recall: { value: 0.80, threshold: 0.6, status: 'normal' as const },
    },
    business: {
      totalReturn: { value: 0.15, threshold: 0, status: 'normal' as const },
      annualizedReturn: { value: 0.18, threshold: 0.1, status: 'normal' as const },
      sharpeRatio: { value: 1.25, threshold: 1.0, status: 'normal' as const },
      maxDrawdown: { value: 0.12, threshold: 0.15, status: 'warning' as const },
      winRate: { value: 0.58, threshold: 0.5, status: 'normal' as const },
    },
    resource: {
      avgLatencyMs: { value: 85, threshold: 100, status: 'normal' as const },
      p95LatencyMs: { value: 150, threshold: 200, status: 'normal' as const },
      errorRate: { value: 0.02, threshold: 0.05, status: 'normal' as const },
    },
  };

  const MetricCard = ({ label, value, threshold, status, unit }: any) => {
    const statusColors = {
      normal: 'border-green-500 bg-green-50',
      warning: 'border-yellow-500 bg-yellow-50',
      critical: 'border-red-500 bg-red-50',
    };

    return (
      <div className={`p-4 rounded-lg border-l-4 ${statusColors[status]}`}>
        <div className="text-sm text-gray-500">{label}</div>
        <div className="text-2xl font-bold text-gray-900">
          {value.toFixed(2)}
          {unit && <span className="text-sm ml-1">{unit}</span>}
        </div>
        <div className="text-xs text-gray-400">阈值: {threshold}</div>
      </div>
    );
  };

  return (
    <>
      <Head>
        <title>模型性能监控 - 量化交易系统</title>
      </Head>

      <div className="min-h-screen bg-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* 页面标题 */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">模型性能监控</h1>
              <p className="mt-2 text-gray-600">实时监控模型性能和数据漂移</p>
            </div>
            <button
              onClick={() => refresh()}
              className="flex items-center px-4 py-2 bg-white text-gray-700 rounded-lg shadow hover:bg-gray-50 transition-colors"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              刷新
            </button>
          </div>

          {loading ? (
            <div className="flex items-center justify-center h-64">
              <Activity className="w-8 h-8 text-blue-500 animate-pulse" />
            </div>
          ) : (
            <>
              {/* 漂移检测警告 */}
              {driftSummary?.hasHighSeverity && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
                  <div className="flex items-center">
                    <AlertTriangle className="w-5 h-5 text-red-500 mr-2" />
                    <span className="text-red-800 font-medium">
                      检测到严重数据漂移，建议重新训练模型
                    </span>
                  </div>
                </div>
              )}

              {/* 技术指标 */}
              <div className="bg-white rounded-lg shadow-md p-6 mb-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">技术指标</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <MetricCard label="准确率" {...mockMetrics.technical.accuracy} unit="%" />
                  <MetricCard label="F1分数" {...mockMetrics.technical.f1Score} />
                  <MetricCard label="精确率" {...mockMetrics.technical.precision} />
                  <MetricCard label="召回率" {...mockMetrics.technical.recall} />
                </div>
              </div>

              {/* 业务指标 */}
              <div className="bg-white rounded-lg shadow-md p-6 mb-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">业务指标</h2>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                  <MetricCard label="总收益率" {...mockMetrics.business.totalReturn} unit="%" />
                  <MetricCard label="年化收益率" {...mockMetrics.business.annualizedReturn} unit="%" />
                  <MetricCard label="夏普比率" {...mockMetrics.business.sharpeRatio} />
                  <MetricCard label="最大回撤" {...mockMetrics.business.maxDrawdown} unit="%" />
                  <MetricCard label="胜率" {...mockMetrics.business.winRate} unit="%" />
                </div>
              </div>

              {/* 资源指标 */}
              <div className="bg-white rounded-lg shadow-md p-6 mb-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">资源指标</h2>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <MetricCard label="平均延迟" {...mockMetrics.resource.avgLatencyMs} unit="ms" />
                  <MetricCard label="P95延迟" {...mockMetrics.resource.p95LatencyMs} unit="ms" />
                  <MetricCard label="错误率" {...mockMetrics.resource.errorRate} unit="%" />
                </div>
              </div>

              {/* 漂移检测汇总 */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">漂移检测</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-4 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-gray-900">
                      {driftSummary?.totalDetections || 0}
                    </div>
                    <div className="text-sm text-gray-500">总检测次数</div>
                  </div>
                  <div className="text-center p-4 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-gray-900">
                      {(driftSummary?.latestDriftScore || 0).toFixed(2)}
                    </div>
                    <div className="text-sm text-gray-500">最新漂移分数</div>
                  </div>
                  <div className="text-center p-4 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-gray-900">
                      {driftSummary?.latestSeverity || 'none'}
                    </div>
                    <div className="text-sm text-gray-500">严重程度</div>
                  </div>
                  <div className="text-center p-4 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-gray-900">
                      {driftSummary?.shouldTriggerRetraining ? '是' : '否'}
                    </div>
                    <div className="text-sm text-gray-500">建议重新训练</div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
};

export default MonitoringPage;
