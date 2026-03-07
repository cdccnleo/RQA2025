/**
 * 管道API服务
 * 
 * 提供管道相关的API调用方法
 */

import { get, post } from './api';
import type {
  Pipeline,
  PipelineListResponse,
  PipelineDetailsResponse,
  ExecutePipelineRequest,
  ExecutePipelineResponse,
  CancelPipelineResponse,
  PipelineConfig,
} from '../types/pipeline';

const BASE_URL = '/api/v1/pipeline';

/**
 * 获取管道状态列表
 */
export const getPipelineStatus = async (): Promise<PipelineListResponse> => {
  return get<PipelineListResponse>(`${BASE_URL}/status`);
};

/**
 * 获取管道详情
 */
export const getPipelineDetails = async (pipelineId: string): Promise<PipelineDetailsResponse> => {
  return get<PipelineDetailsResponse>(`${BASE_URL}/${pipelineId}/details`);
};

/**
 * 执行管道
 */
export const executePipeline = async (request: ExecutePipelineRequest): Promise<ExecutePipelineResponse> => {
  return post<ExecutePipelineResponse>(`${BASE_URL}/execute`, request);
};

/**
 * 取消管道执行
 */
export const cancelPipeline = async (pipelineId: string): Promise<CancelPipelineResponse> => {
  return post<CancelPipelineResponse>(`${BASE_URL}/${pipelineId}/cancel`);
};

/**
 * 获取管道配置列表
 */
export const getPipelineConfigs = async (): Promise<{ configs: PipelineConfig[] }> => {
  return get<{ configs: PipelineConfig[] }>(`${BASE_URL}/configs`);
};

/**
 * 创建WebSocket连接
 */
export const createPipelineWebSocket = (pipelineId: string): WebSocket => {
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
  return new WebSocket(`${wsUrl}${BASE_URL}/ws/${pipelineId}`);
};

export default {
  getPipelineStatus,
  getPipelineDetails,
  executePipeline,
  cancelPipeline,
  getPipelineConfigs,
  createPipelineWebSocket,
};
