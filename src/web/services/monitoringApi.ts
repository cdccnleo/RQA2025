/**
 * 监控API服务
 * 
 * 提供模型性能监控和漂移检测的API调用方法
 */

import { get, post } from './api';
import type {
  MetricsSnapshot,
  MetricsHistoryResponse,
  MetricsStatistics,
  MonitoringStatus,
  DriftReport,
  DriftSummary,
} from '../types/monitoring';

const BASE_URL = '/api/v1/monitoring';

/**
 * 获取监控状态
 */
export const getMonitoringStatus = async (): Promise<MonitoringStatus> => {
  return get<MonitoringStatus>(`${BASE_URL}/status`);
};

/**
 * 获取当前指标
 */
export const getCurrentMetrics = async (modelId: string): Promise<MetricsSnapshot> => {
  return get<MetricsSnapshot>(`${BASE_URL}/metrics/${modelId}`);
};

/**
 * 获取指标历史
 */
export const getMetricsHistory = async (
  modelId: string,
  params?: {
    startTime?: string;
    endTime?: string;
    metricNames?: string[];
    aggregation?: '1m' | '5m' | '1h' | '1d';
  }
): Promise<MetricsHistoryResponse> => {
  const queryParams = new URLSearchParams();
  
  if (params?.startTime) queryParams.append('start_time', params.startTime);
  if (params?.endTime) queryParams.append('end_time', params.endTime);
  if (params?.metricNames) queryParams.append('metric_names', params.metricNames.join(','));
  if (params?.aggregation) queryParams.append('aggregation', params.aggregation);
  
  const query = queryParams.toString();
  const url = `${BASE_URL}/metrics/${modelId}/history${query ? `?${query}` : ''}`;
  
  return get<MetricsHistoryResponse>(url);
};

/**
 * 获取指标统计
 */
export const getMetricsStatistics = async (modelId: string): Promise<MetricsStatistics> => {
  return get<MetricsStatistics>(`${BASE_URL}/metrics/${modelId}/statistics`);
};

/**
 * 手动收集指标
 */
export const collectMetrics = async (modelId: string): Promise<{ success: boolean; timestamp: string; metricsCount: number }> => {
  return post<{ success: boolean; timestamp: string; metricsCount: number }>(`${BASE_URL}/metrics/${modelId}/collect`);
};

/**
 * 启动监控
 */
export const startMonitoring = async (modelId: string): Promise<{ success: boolean; message: string; modelId: string; intervalSeconds: number }> => {
  return post<{ success: boolean; message: string; modelId: string; intervalSeconds: number }>(`${BASE_URL}/metrics/${modelId}/start`);
};

/**
 * 停止监控
 */
export const stopMonitoring = async (modelId: string): Promise<{ success: boolean; message: string; modelId: string }> => {
  return post<{ success: boolean; message: string; modelId: string }>(`${BASE_URL}/metrics/${modelId}/stop`);
};

/**
 * 获取漂移报告
 */
export const getDriftReports = async (params?: {
  startTime?: string;
  endTime?: string;
  severity?: string;
}): Promise<DriftReport[]> => {
  const queryParams = new URLSearchParams();
  
  if (params?.startTime) queryParams.append('start_time', params.startTime);
  if (params?.endTime) queryParams.append('end_time', params.endTime);
  if (params?.severity) queryParams.append('severity', params.severity);
  
  const query = queryParams.toString();
  const url = `${BASE_URL}/drift${query ? `?${query}` : ''}`;
  
  return get<DriftReport[]>(url);
};

/**
 * 获取漂移汇总
 */
export const getDriftSummary = async (): Promise<DriftSummary> => {
  return get<DriftSummary>(`${BASE_URL}/drift/summary`);
};

/**
 * 手动触发漂移检测
 */
export const detectDrift = async (): Promise<{ success: boolean; message: string; reportsCount: number }> => {
  return post<{ success: boolean; message: string; reportsCount: number }>(`${BASE_URL}/drift/detect`);
};

/**
 * 设置参考数据
 */
export const setReferenceData = async (): Promise<{ success: boolean; message: string }> => {
  return post<{ success: boolean; message: string }>(`${BASE_URL}/drift/reference`);
};

/**
 * 创建WebSocket连接
 */
export const createMetricsWebSocket = (modelId: string): WebSocket => {
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
  return new WebSocket(`${wsUrl}${BASE_URL}/ws/metrics/${modelId}`);
};

export default {
  getMonitoringStatus,
  getCurrentMetrics,
  getMetricsHistory,
  getMetricsStatistics,
  collectMetrics,
  startMonitoring,
  stopMonitoring,
  getDriftReports,
  getDriftSummary,
  detectDrift,
  setReferenceData,
  createMetricsWebSocket,
};
