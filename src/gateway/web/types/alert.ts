/**
 * 告警类型定义
 * 
 * 告警管理和回滚的TypeScript类型定义
 */

/** 告警严重级别 */
export type AlertSeverity = 'debug' | 'info' | 'warning' | 'error' | 'critical';

/** 告警状态 */
export type AlertStatus = 'active' | 'acknowledged' | 'resolved' | 'suppressed';

/** 告警 */
export interface Alert {
  id: string;
  title: string;
  message: string;
  severity: AlertSeverity;
  status: AlertStatus;
  source: string;
  metricName?: string;
  metricValue?: number;
  threshold?: number;
  timestamp: string;
  acknowledgedBy?: string;
  acknowledgedAt?: string;
  resolvedAt?: string;
  durationSeconds?: number;
}

/** 告警列表响应 */
export interface AlertListResponse {
  alerts: Alert[];
  total: number;
  bySeverity: Record<string, number>;
  byStatus: Record<string, number>;
  page: number;
  pageSize: number;
}

/** 告警统计 */
export interface AlertStatistics {
  totalRules: number;
  enabledRules: number;
  activeAlerts: number;
  totalAlertsHistory: number;
  alertsBySeverity: Record<string, number>;
  suppressedMetrics: string[];
}

/** 确认告警请求 */
export interface AcknowledgeAlertRequest {
  acknowledgedBy: string;
}

/** 确认告警响应 */
export interface AcknowledgeAlertResponse {
  success: boolean;
  message: string;
  alertId: string;
}

/** 解决告警响应 */
export interface ResolveAlertResponse {
  success: boolean;
  message: string;
  alertId: string;
}

/** 抑制告警请求 */
export interface SuppressAlertRequest {
  durationMinutes: number;
}

/** 抑制告警响应 */
export interface SuppressAlertResponse {
  success: boolean;
  message: string;
  alertId: string;
  suppressedUntil: string;
}

/** 回滚决策 */
export interface RollbackDecision {
  shouldRollback: boolean;
  trigger?: string;
  confidence: number;
  reasons: string[];
  recommendedAction: string;
  timestamp: string;
}

/** 执行回滚请求 */
export interface ExecuteRollbackRequest {
  force?: boolean;
  targetVersion?: string;
}

/** 执行回滚响应 */
export interface ExecuteRollbackResponse {
  success: boolean;
  status: string;
  previousVersion?: string;
  currentVersion?: string;
  startTime: string;
  endTime?: string;
  message: string;
}

/** 回滚状态 */
export interface RollbackStatus {
  modelId: string;
  isRollbackInProgress: boolean;
  thresholds: Record<string, number>;
  baselineMetrics: Record<string, number>;
  rollbackCount: number;
  backupAvailable: boolean;
}

/** WebSocket告警消息 */
export interface AlertWebSocketMessage {
  type: 'alert' | 'alert_status_update';
  timestamp: string;
  data: {
    alertId: string;
    title?: string;
    message?: string;
    severity?: AlertSeverity;
    status?: AlertStatus;
    metricName?: string;
    metricValue?: number;
    threshold?: number;
    acknowledgedBy?: string;
  };
}

/** 告警过滤器 */
export interface AlertFilter {
  severity?: AlertSeverity[];
  status?: AlertStatus[];
  source?: string;
  startTime?: string;
  endTime?: string;
  page?: number;
  pageSize?: number;
}
