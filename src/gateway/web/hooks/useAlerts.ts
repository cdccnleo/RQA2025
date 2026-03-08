/**
 * 告警管理Hook
 *
 * 提供告警查询、确认、解决等功能
 */

import { useState, useEffect, useCallback } from 'react';
import alertApi from '../services/alertApi';
import type {
  Alert,
  AlertListResponse,
  AlertStatistics,
  AlertFilter,
  RollbackDecision,
  RollbackStatus,
} from '../types/alert';

interface UseAlertsReturn {
  alerts: Alert[];
  total: number;
  bySeverity: Record<string, number>;
  byStatus: Record<string, number>;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  acknowledge: (alertId: string, user: string) => Promise<void>;
  resolve: (alertId: string) => Promise<void>;
}

/**
 * 获取告警列表
 */
export const useAlerts = (filter?: AlertFilter): UseAlertsReturn => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [total, setTotal] = useState(0);
  const [bySeverity, setBySeverity] = useState<Record<string, number>>({});
  const [byStatus, setByStatus] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await alertApi.getAlerts(filter);
      setAlerts(response.alerts);
      setTotal(response.total);
      setBySeverity(response.bySeverity);
      setByStatus(response.byStatus);
    } catch (err) {
      setError(err instanceof Error ? err.message : '获取告警失败');
    } finally {
      setLoading(false);
    }
  }, [filter]);

  const acknowledge = useCallback(async (alertId: string, user: string) => {
    try {
      await alertApi.acknowledgeAlert(alertId, { acknowledgedBy: user });
      await fetchData();
    } catch (err) {
      throw err;
    }
  }, [fetchData]);

  const resolve = useCallback(async (alertId: string) => {
    try {
      await alertApi.resolveAlert(alertId);
      await fetchData();
    } catch (err) {
      throw err;
    }
  }, [fetchData]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [fetchData]);

  return {
    alerts,
    total,
    bySeverity,
    byStatus,
    loading,
    error,
    refresh: fetchData,
    acknowledge,
    resolve,
  };
};

interface UseAlertStatisticsReturn {
  statistics: AlertStatistics | null;
  loading: boolean;
  error: string | null;
}

/**
 * 获取告警统计
 */
export const useAlertStatistics = (): UseAlertStatisticsReturn => {
  const [statistics, setStatistics] = useState<AlertStatistics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const data = await alertApi.getAlertStatistics();
        setStatistics(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : '获取统计失败');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return { statistics, loading, error };
};

interface UseRollbackReturn {
  decision: RollbackDecision | null;
  status: RollbackStatus | null;
  loading: boolean;
  error: string | null;
  evaluate: () => Promise<void>;
  execute: (force?: boolean) => Promise<void>;
}

/**
 * 回滚管理
 */
export const useRollback = (): UseRollbackReturn => {
  const [decision, setDecision] = useState<RollbackDecision | null>(null);
  const [status, setStatus] = useState<RollbackStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const evaluate = useCallback(async () => {
    setLoading(true);
    try {
      const [decisionData, statusData] = await Promise.all([
        alertApi.getRollbackDecision(),
        alertApi.getRollbackStatus(),
      ]);
      setDecision(decisionData);
      setStatus(statusData);
    } catch (err) {
      setError(err instanceof Error ? err.message : '获取回滚信息失败');
    } finally {
      setLoading(false);
    }
  }, []);

  const execute = useCallback(async (force?: boolean) => {
    try {
      await alertApi.executeRollback({ force });
      await evaluate();
    } catch (err) {
      throw err;
    }
  }, [evaluate]);

  useEffect(() => {
    evaluate();
  }, [evaluate]);

  return { decision, status, loading, error, evaluate, execute };
};

export default { useAlerts, useAlertStatistics, useRollback };
