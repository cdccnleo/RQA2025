/**
 * 监控类型定义
 * 
 * 模型性能监控和漂移检测的TypeScript类型定义
 */

/** 指标状态 */
export type MetricStatus = 'normal' | 'warning' | 'critical';

/** 指标值 */
export interface MetricValue {
  value: number;
  threshold?: number;
  status: MetricStatus;
  unit?: string;
}

/** 指标快照 */
export interface MetricsSnapshot {
  timestamp: string;
  modelId: string;
  metrics: Record<string, MetricValue>;
}

/** 指标历史数据点 */
export interface MetricsHistoryPoint {
  timestamp: string;
  values: Record<string, number>;
}

/** 指标历史响应 */
export interface MetricsHistoryResponse {
  modelId: string;
  data: MetricsHistoryPoint[];
  aggregation: string;
}

/** 指标统计 */
export interface MetricsStatistics {
  modelId: string;
  monitoringDurationSeconds: number;
  totalSnapshots: number;
  collectorsCount: number;
  isMonitoring: boolean;
}

/** 监控状态 */
export interface MonitoringStatus {
  performanceMonitoring: boolean;
  driftDetection: boolean;
  monitoringIntervalSeconds: number;
  modelsCount: number;
}

/** 漂移类型 */
export type DriftType = 'data_drift' | 'concept_drift' | 'feature_drift';

/** 漂移严重程度 */
export type DriftSeverity = 'none' | 'low' | 'medium' | 'high';

/** 漂移报告 */
export interface DriftReport {
  timestamp: string;
  driftType: DriftType;
  severity: DriftSeverity;
  driftScore: number;
  affectedFeatures: string[];
  statistics?: Record<string, any>;
  recommendations: string[];
}

/** 漂移汇总 */
export interface DriftSummary {
  status: string;
  totalDetections: number;
  recentSeverityDistribution: Record<string, number>;
  latestDriftScore: number;
  latestSeverity: string;
  hasHighSeverity: boolean;
  shouldTriggerRetraining: boolean;
}

/** WebSocket指标消息 */
export interface MetricsWebSocketMessage {
  type: 'metrics';
  timestamp: string;
  data: {
    modelId: string;
    metrics: Record<string, {
      value: number;
      threshold?: number;
      status: MetricStatus;
    }>;
  };
}

/** 技术指标 */
export interface TechnicalMetrics {
  accuracy: number;
  f1Score: number;
  rocAuc: number;
  precision: number;
  recall: number;
}

/** 业务指标 */
export interface BusinessMetrics {
  totalReturn: number;
  annualizedReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
}

/** 资源指标 */
export interface ResourceMetrics {
  avgLatencyMs: number;
  p95LatencyMs: number;
  p99LatencyMs: number;
  errorRate: number;
  throughputRps: number;
}
