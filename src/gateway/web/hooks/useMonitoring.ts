/**
 * 监控管理Hook
 * 
 * 提供模型性能监控和漂移检测功能
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import monitoringApi from '../services/monitoringApi';
import type {
  MetricsSnapshot,
  MetricsHistoryResponse,
  DriftSummary,
  MetricsWebSocketMessage,
} from '../types/monitoring';

interface UseMonitoringReturn {
  metrics: MetricsSnapshot | null;
  history: MetricsHistoryResponse | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

/**
 * 获取监控指标
 */
export const useMonitoring = (modelId: string): UseMonitoringReturn => {
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null);
  const [history, setHistory] = useState<MetricsHistoryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [metricsData, historyData] = await Promise.all([
        monitoringApi.getCurrentMetrics(modelId),
        monitoringApi.getMetricsHistory(modelId, { aggregation: '5m' }),
      ]);
      setMetrics(metricsData);
      setHistory(historyData);
    } catch (err) {
      setError(err instanceof Error ? err.message : '获取监控数据失败');
    } finally {
      setLoading(false);
    }
  }, [modelId]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  return { metrics, history, loading, error, refresh: fetchData };
};

interface UseMonitoringWebSocketReturn {
  metrics: MetricsSnapshot | null;
  connected: boolean;
}

/**
 * 监控WebSocket连接
 */
export const useMonitoringWebSocket = (modelId: string): UseMonitoringWebSocketReturn => {
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = monitoringApi.createMetricsWebSocket(modelId);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onmessage = (event) => {
      try {
        const message: MetricsWebSocketMessage = JSON.parse(event.data);
        if (message.type === 'metrics') {
          setMetrics({
            timestamp: message.timestamp,
            modelId: message.data.modelId,
            metrics: message.data.metrics,
          });
        }
      } catch (err) {
        console.error('WebSocket消息解析失败:', err);
      }
    };
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    return () => ws.close();
  }, [modelId]);

  return { metrics, connected };
};

interface UseDriftDetectionReturn {
  summary: DriftSummary | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

/**
 * 获取漂移检测
 */
export const useDriftDetection = (): UseDriftDetectionReturn => {
  const [summary, setSummary] = useState<DriftSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const data = await monitoringApi.getDriftSummary();
      setSummary(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : '获取漂移检测失败');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, [fetchData]);

  return { summary, loading, error, refresh: fetchData };
};

export default { useMonitoring, useMonitoringWebSocket, useDriftDetection };
