/**
 * 告警摘要卡片组件
 *
 * 在Dashboard展示告警概览
 */

import React from 'react';
import { Bell, AlertTriangle, AlertCircle, Info, CheckCircle } from 'lucide-react';
import { useAlerts, useAlertStatistics } from '../../hooks/useAlerts';
import Link from 'next/link';

export const AlertSummaryCard: React.FC = () => {
  const { alerts, total, bySeverity, loading } = useAlerts({ status: ['active'] });
  const { statistics } = useAlertStatistics();

  const getSeverityIcon = (severity: string) => {
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

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-100 text-red-800';
      case 'error':
        return 'bg-orange-100 text-orange-800';
      case 'warning':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-blue-100 text-blue-800';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">告警中心</h3>
        <Link
          href="/alerts"
          className="text-sm text-blue-600 hover:text-blue-800 transition-colors"
        >
          查看全部 →
        </Link>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-32">
          <Bell className="w-8 h-8 text-blue-500 animate-pulse" />
        </div>
      ) : (
        <>
          {/* 告警统计 */}
          <div className="grid grid-cols-4 gap-2 mb-4">
            <div className="text-center p-2 bg-red-50 rounded-lg">
              <div className="text-xl font-bold text-red-600">
                {bySeverity.critical || 0}
              </div>
              <div className="text-xs text-red-500">严重</div>
            </div>
            <div className="text-center p-2 bg-orange-50 rounded-lg">
              <div className="text-xl font-bold text-orange-600">
                {bySeverity.error || 0}
              </div>
              <div className="text-xs text-orange-500">错误</div>
            </div>
            <div className="text-center p-2 bg-yellow-50 rounded-lg">
              <div className="text-xl font-bold text-yellow-600">
                {bySeverity.warning || 0}
              </div>
              <div className="text-xs text-yellow-500">警告</div>
            </div>
            <div className="text-center p-2 bg-blue-50 rounded-lg">
              <div className="text-xl font-bold text-blue-600">
                {bySeverity.info || 0}
              </div>
              <div className="text-xs text-blue-500">信息</div>
            </div>
          </div>

          {/* 最近告警 */}
          <div className="space-y-2">
            {alerts.slice(0, 3).map((alert) => (
              <div
                key={alert.id}
                className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                {getSeverityIcon(alert.severity)}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-gray-800 truncate">
                      {alert.title}
                    </span>
                    <span
                      className={`text-xs px-2 py-1 rounded-full ${getSeverityColor(
                        alert.severity
                      )}`}
                    >
                      {alert.severity}
                    </span>
                  </div>
                  <p className="text-sm text-gray-500 mt-1 truncate">
                    {alert.message}
                  </p>
                  <span className="text-xs text-gray-400 mt-1">
                    {new Date(alert.timestamp).toLocaleString()}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* 统计信息 */}
          {statistics && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <div className="flex items-center justify-between text-sm text-gray-600">
                <span>活跃告警: {statistics.activeAlerts}</span>
                <span>告警规则: {statistics.enabledRules}/{statistics.totalRules}</span>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default AlertSummaryCard;
