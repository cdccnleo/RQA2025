/**
 * 管道管理Hook
 * 
 * 提供管道状态管理和操作功能
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import pipelineApi from '../services/pipelineApi';
import type {
  Pipeline,
  PipelineListResponse,
  PipelineDetailsResponse,
  ExecutePipelineRequest,
  PipelineConfig,
  PipelineStatusWebSocketMessage,
} from '../types/pipeline';

interface UsePipelineReturn {
  pipelines: Pipeline[];
  total: number;
  running: number;
  completed: number;
  failed: number;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

/**
 * 获取管道列表
 */
export const usePipelineList = (): UsePipelineReturn => {
  const [pipelines, setPipelines] = useState<Pipeline[]>([]);
  const [total, setTotal] = useState(0);
  const [running, setRunning] = useState(0);
  const [completed, setCompleted] = useState(0);
  const [failed, setFailed] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchPipelines = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await pipelineApi.getPipelineStatus();
      setPipelines(response.pipelines);
      setTotal(response.total);
      setRunning(response.running);
      setCompleted(response.completed);
      setFailed(response.failed);
    } catch (err) {
      setError(err instanceof Error ? err.message : '获取管道列表失败');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPipelines();
    // 每5秒自动刷新
    const interval = setInterval(fetchPipelines, 5000);
    return () => clearInterval(interval);
  }, [fetchPipelines]);

  return {
    pipelines,
    total,
    running,
    completed,
    failed,
    loading,
    error,
    refresh: fetchPipelines,
  };
};

interface UsePipelineDetailsReturn {
  pipeline: Pipeline | null;
  stages: PipelineDetailsResponse['stages'];
  logs: PipelineDetailsResponse['logs'];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

/**
 * 获取管道详情
 */
export const usePipelineDetails = (pipelineId: string | null): UsePipelineDetailsReturn => {
  const [pipeline, setPipeline] = useState<Pipeline | null>(null);
  const [stages, setStages] = useState<PipelineDetailsResponse['stages']>([]);
  const [logs, setLogs] = useState<PipelineDetailsResponse['logs']>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDetails = useCallback(async () => {
    if (!pipelineId) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await pipelineApi.getPipelineDetails(pipelineId);
      setPipeline(response.pipeline);
      setStages(response.stages);
      setLogs(response.logs);
    } catch (err) {
      setError(err instanceof Error ? err.message : '获取管道详情失败');
    } finally {
      setLoading(false);
    }
  }, [pipelineId]);

  useEffect(() => {
    fetchDetails();
  }, [fetchDetails]);

  return {
    pipeline,
    stages,
    logs,
    loading,
    error,
    refresh: fetchDetails,
  };
};

interface UsePipelineWebSocketReturn {
  status: string | null;
  progress: number;
  currentStage: string | null;
  durationSeconds: number | null;
  connected: boolean;
}

/**
 * 管道WebSocket连接
 */
export const usePipelineWebSocket = (pipelineId: string | null): UsePipelineWebSocketReturn => {
  const [status, setStatus] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState<string | null>(null);
  const [durationSeconds, setDurationSeconds] = useState<number | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!pipelineId) return;

    const ws = pipelineApi.createPipelineWebSocket(pipelineId);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const message: PipelineStatusWebSocketMessage = JSON.parse(event.data);
        if (message.type === 'pipeline_status') {
          setStatus(message.data.status);
          setProgress(message.data.progress);
          setCurrentStage(message.data.currentStage || null);
          setDurationSeconds(message.data.durationSeconds || null);
        }
      } catch (err) {
        console.error('WebSocket消息解析失败:', err);
      }
    };

    ws.onclose = () => {
      setConnected(false);
    };

    ws.onerror = (error) => {
      console.error('WebSocket错误:', error);
      setConnected(false);
    };

    return () => {
      ws.close();
    };
  }, [pipelineId]);

  return {
    status,
    progress,
    currentStage,
    durationSeconds,
    connected,
  };
};

interface UseExecutePipelineReturn {
  execute: (request: ExecutePipelineRequest) => Promise<{ pipelineId: string }>;
  loading: boolean;
  error: string | null;
}

/**
 * 执行管道
 */
export const useExecutePipeline = (): UseExecutePipelineReturn => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const execute = useCallback(async (request: ExecutePipelineRequest) => {
    setLoading(true);
    setError(null);
    try {
      const response = await pipelineApi.executePipeline(request);
      return { pipelineId: response.pipelineId };
    } catch (err) {
      const message = err instanceof Error ? err.message : '执行管道失败';
      setError(message);
      throw new Error(message);
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    execute,
    loading,
    error,
  };
};

interface UsePipelineConfigsReturn {
  configs: PipelineConfig[];
  loading: boolean;
  error: string | null;
}

/**
 * 获取管道配置
 */
export const usePipelineConfigs = (): UsePipelineConfigsReturn => {
  const [configs, setConfigs] = useState<PipelineConfig[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchConfigs = async () => {
      setLoading(true);
      try {
        const response = await pipelineApi.getPipelineConfigs();
        setConfigs(response.configs);
      } catch (err) {
        setError(err instanceof Error ? err.message : '获取配置失败');
      } finally {
        setLoading(false);
      }
    };

    fetchConfigs();
  }, []);

  return {
    configs,
    loading,
    error,
  };
};

export default {
  usePipelineList,
  usePipelineDetails,
  usePipelineWebSocket,
  useExecutePipeline,
  usePipelineConfigs,
};
