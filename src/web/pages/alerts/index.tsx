/**
 * 告警中心页面
 */

import React, { useState } from 'react';
import { NextPage } from 'next';
import Head from 'next/head';
import {
  Bell,
  AlertTriangle,
  AlertCircle,
  Info,
  CheckCircle,
  RefreshCw,
  Check,
  X,
  Clock,
} from 'lucide-react';
import { useAlerts, useAlertStatistics, useRollback } from '../../hooks/useAlerts';
import type { AlertSeverity, AlertStatus } from '../../types/alert';

const AlertsPage: NextPage = () => {
  const [filterSeverity, setFilterSeverity] = useState<AlertSeverity[]>([]);
  const [filterStatus, setFilterStatus] = useState<AlertStatus[]>(['active']);

  const { alerts, total, bySeverity, loading, refresh, acknowledge, resolve } = useAlerts({
    severity: filterSeverity,
    status: filterStatus,
  });

  const { statistics } = useAlertStatistics();
  const { decision, status: rollbackStatus, execute: executeRollback } = useRollback();

  const getSeverityIcon = (severity: AlertSeverity) => {
    switch (severity) {
      case 'critical':
        return <AlertTriangle className="w-5 h-5 text-red-500" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-orange-500" />;
      case 'warning':
        return <AlertCircle className="w-5 h-5 text-yellow-500" />;
      default:
        return <Info className="w-5 h-5 text-blue-500" />;
    }
  };

  const getSeverityColor = (severity: AlertSeverity) => {
    const colors: Record<AlertSeverity, string> = {
      critical: 'bg-red-100 text-red-800 border-red-200',
      error: 'bg-orange-100 text-orange-800 border-orange-200',
      warning: 'bg-yellow-100 text-yellow-800 border-yellow-200',
      info: 'bg-blue-100 text-blue-800 border-blue-200',
      debug: 'bg-gray-100 text-gray-800 border-gray-200',
    };
    return colors[severity];
  };

  const handleAcknowledge = async (alertId: string) => {
    try {
      await acknowledge(alertId, 'admin');
    } catch (err) {
      console.error('确认告警失败:', err);
    }
  };

  const handleResolve = async (alertId: string) => {
    try {
      await resolve(alertId);
    } catch (err) {
      console.error('解决告警失败:', err);
    }
  };

  return (
    <>
      <Head>
        <title>告警中心 - 量化交易系统</title>
      </Head>

      <div className="min-h-screen bg-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* 页面标题 */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">告警中心</h1>
              <p className="mt-2 text-gray-600">管理和处理系统告警</p>
            </div>
            <button
              onClick={() => refresh()}
              className="flex items-center px-4 py-2 bg-white text-gray-700 rounded-lg shadow hover:bg-gray-50 transition-colors"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              刷新
            </button>
          </div>

          {/* 统计卡片 */}
          <div className="grid grid-cols-5 gap-4 mb-6">
            <div className="bg-white rounded-lg shadow p-4">
              <div className="text-sm text-gray-500">活跃告警</div>
              <div className="text-2xl font-bold text-gray-900">
                {statistics?.activeAlerts || 0}
              </div>
            </div>
            <div className="bg-red-50 rounded-lg shadow p-4">
              <div className="text-sm text-red-500">严重</div>
              <div className="text-2xl font-bold text-red-600">{bySeverity.critical || 0}</div>
            </div>
            <div className="bg-orange-50 rounded-lg shadow p-4">
              <div className="text-sm text-orange-500">错误</div>
              <div className="text-2xl font-bold text-orange-600">{bySeverity.error || 0}</div>
            </div>
            <div className="bg-yellow-50 rounded-lg shadow p-4">
              <div className="text-sm text-yellow-500">警告</div>
              <div className="text-2xl font-bold text-yellow-600">{bySeverity.warning || 0}</div>
            </div>
            <div className="bg-blue-50 rounded-lg shadow p-4">
              <div className="text-sm text-blue-500">信息</div>
              <div className="text-2xl font-bold text-blue-600">{bySeverity.info || 0}</div>
            </div>
          </div>

          {/* 回滚建议 */}
          {decision?.shouldRollback && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <AlertTriangle className="w-5 h-5 text-red-500 mr-2" />
                  <div>
                    <span className="text-red-800 font-medium">建议执行回滚</span>
                    <p className="text-red-600 text-sm mt-1">
                      置信度: {(decision.confidence * 100).toFixed(0)}% | 原因: {decision.reasons.join(', ')}
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => executeRollback()}
                  disabled={rollbackStatus?.isRollbackInProgress}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors disabled:opacity-50"
                >
                  {rollbackStatus?.isRollbackInProgress ? '回滚中...' : '执行回滚'}
                </button>
              </div>
            </div>
          )}

          {/* 过滤器 */}
          <div className="bg-white rounded-lg shadow-md p-4 mb-6">
            <div className="flex items-center space-x-4">
              <span className="text-sm font-medium text-gray-700">状态过滤:</span>
              {(['active', 'acknowledged', 'resolved'] as AlertStatus[]).map((status) => (
                <button
                  key={status}
                  onClick={() => {
                    if (filterStatus.includes(status)) {
                      setFilterStatus(filterStatus.filter((s) => s !== status));
                    } else {
                      setFilterStatus([...filterStatus, status]);
                    }
                  }}
                  className={`px-3 py-1 rounded-full text-sm transition-colors ${
                    filterStatus.includes(status)
                      ? 'bg-blue-100 text-blue-800'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {status === 'active' && '活跃'}
                  {status === 'acknowledged' && '已确认'}
                  {status === 'resolved' && '已解决'}
                </button>
              ))}
            </div>
          </div>

          {/* 告警列表 */}
          <div className="bg-white rounded-lg shadow-md">
            <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900">告警列表</h2>
              <span className="text-sm text-gray-500">共 {total} 条告警</span>
            </div>

            {loading ? (
              <div className="flex items-center justify-center h-64">
                <Bell className="w-8 h-8 text-blue-500 animate-pulse" />
              </div>
            ) : alerts.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-64 text-gray-500">
                <CheckCircle className="w-12 h-12 mb-4 text-green-500" />
                <p>暂无告警</p>
              </div>
            ) : (
              <div className="divide-y divide-gray-200">
                {alerts.map((alert) => (
                  <div
                    key={alert.id}
                    className="px-6 py-4 hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-start space-x-4">
                      <div className="flex-shrink-0 mt-1">
                        {getSeverityIcon(alert.severity)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <h3 className="text-lg font-medium text-gray-900">{alert.title}</h3>
                          <span
                            className={`px-2 py-1 rounded-full text-xs font-medium border ${getSeverityColor(
                              alert.severity
                            )}`}
                          >
                            {alert.severity}
                          </span>
                        </div>
                        <p className="mt-1 text-gray-600">{alert.message}</p>
                        <div className="mt-2 flex items-center space-x-4 text-sm text-gray-500">
                          <span className="flex items-center">
                            <Clock className="w-4 h-4 mr-1" />
                            {new Date(alert.timestamp).toLocaleString()}
                          </span>
                          {alert.metricName && (
                            <span>
                              {alert.metricName}: {alert.metricValue?.toFixed(2)} (阈值:{' '}
                              {alert.threshold})
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        {alert.status === 'active' &&