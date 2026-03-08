/**
 * 告警API服务
 * 
 * 提供告警管理和回滚的API调用方法
 */

import { get, post } from './api';
import type {
  Alert,
  AlertListResponse,
  AlertStatistics,
  AcknowledgeAlertRequest,
  AcknowledgeAlertResponse,
  ResolveAlertResponse,
  SuppressAlertRequest,
  SuppressAlertResponse,
  RollbackDecision,
  ExecuteRollbackRequest,
  ExecuteRollbackResponse,
  RollbackStatus,
  AlertFilter,
} from '../types/alert';

const BASE_URL = '/api/v1/alerts';

/**
 * 获取告警列表
 */
export const getAlerts = async (filter?: AlertFilter): Promise<AlertListResponse> => {
  const queryParams = new URLSearchParams();
  
  if (filter?.severity) queryParams.append('severity', filter.severity.join(','));
  if (filter?.status) queryParams.append('status', filter.status.join(','));
  if (filter?.source) queryParams.append('source', filter.source);
  if (filter?.startTime) queryParams.append('start_time', filter.startTime);
  if (filter?.endTime) queryParams.append('end_time', filter.endTime);
  if (filter?.page) queryParams.append('page', filter.page.toString());
  if (filter?.pageSize) queryParams.append('page_size', filter.pageSize.toString());
  
  const query = queryParams.toString();
  const url = `${BASE_URL}${query ? `?${query}` : ''}`;
  
  return get<AlertListResponse>(url);
};

/**
 * 获取告警统计
 */
export const getAlertStatistics = async (): Promise<AlertStatistics> => {
  return get<AlertStatistics>(`${BASE_URL}/statistics`);
};

/**
 * 获取告警详情
 */
export const getAlertDetail = async (alertId: string): Promise<Alert> => {
  return get<Alert>(`${BASE_URL}/${alertId}`);
};

/**
 * 确认告警
 */
export const acknowledgeAlert = async (alertId: string, request: AcknowledgeAlertRequest): Promise<AcknowledgeAlertResponse> => {
  return post<AcknowledgeAlertResponse>(`${BASE_URL}/${alertId}/acknowledge`, request);
};

/**
 * 解决告警
 */
export const resolveAlert = async (alertId: string): Promise<ResolveAlertResponse> => {
  return post<ResolveAlertResponse>(`${BASE_URL}/${alertId}/resolve`);
};

/**
 * 抑制告警
 */
export const suppressAlert = async (alertId: string, request: SuppressAlertRequest): Promise<SuppressAlertResponse> => {
  return post<SuppressAlertResponse>(`${BASE_URL}/${alertId}/suppress`, request);
};

/**
 * 获取回滚决策
 */
export const getRollbackDecision = async (): Promise<RollbackDecision> => {
  return get<RollbackDecision>(`${BASE_URL}/rollback/decision`);
};

/**
 * 执行回滚
 */
export const executeRollback = async (request: ExecuteRollbackRequest): Promise<ExecuteRollbackResponse> => {
  return post<ExecuteRollbackResponse>(`${BASE_URL}/rollback/execute`, request);
};

/**
 * 获取回滚状态
 */
export const getRollbackStatus = async (): Promise<RollbackStatus> => {
  return get<RollbackStatus>(`${BASE_URL}/rollback/status`);
};

/**
 * 获取回滚历史
 */
export const getRollbackHistory = async (): Promise<{ history: any[]; total: number }> => {
  return get<{ history: any[]; total: number }>(`${BASE_URL}/rollback/history`);
};

/**
 * 创建WebSocket连接
 */
export const createAlertsWebSocket = (): WebSocket => {
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
  return new WebSocket(`${wsUrl}${BASE_URL}/ws/alerts`);
};

export default {
  getAlerts,
  getAlertStatistics,
  getAlertDetail,
  acknowledgeAlert,
  resolveAlert,
  suppressAlert,
  getRollbackDecision,
  executeRollback,
  getRollbackStatus,
  getRollbackHistory,
  createAlertsWebSocket,
};
