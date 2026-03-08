/**
 * 管道类型定义
 * 
 * ML自动化训练管道的TypeScript类型定义
 */

/** 管道状态 */
export type PipelineStatus = 
  | 'pending' 
  | 'running' 
  | 'completed' 
  | 'failed' 
  | 'rolling_back' 
  | 'rolled_back' 
  | 'cancelled';

/** 阶段状态 */
export type StageStatus = 
  | 'pending' 
  | 'running' 
  | 'completed' 
  | 'failed' 
  | 'skipped';

/** 日志条目 */
export interface LogEntry {
  timestamp: string;
  level: 'debug' | 'info' | 'warning' | 'error';
  message: string;
}

/** 管道阶段 */
export interface PipelineStage {
  name: string;
  status: StageStatus;
  startTime?: string;
  endTime?: string;
  durationSeconds?: number;
  output?: Record<string, any>;
  error?: string;
  logs?: LogEntry[];
}

/** 管道 */
export interface Pipeline {
  id: string;
  name: string;
  version: string;
  status: PipelineStatus;
  currentStage?: string;
  progress: number;
  startTime?: string;
  endTime?: string;
  durationSeconds?: number;
  stages: PipelineStage[];
  context?: Record<string, any>;
  error?: string;
}

/** 管道列表响应 */
export interface PipelineListResponse {
  pipelines: Pipeline[];
  total: number;
  running: number;
  completed: number;
  failed: number;
}

/** 管道详情响应 */
export interface PipelineDetailsResponse {
  pipeline: Pipeline;
  stages: PipelineStage[];
  logs: LogEntry[];
  metrics: Record<string, any>;
}

/** 执行管道请求 */
export interface ExecutePipelineRequest {
  configId: string;
  context: {
    symbols: string[];
    startDate: string;
    endDate: string;
    [key: string]: any;
  };
}

/** 执行管道响应 */
export interface ExecutePipelineResponse {
  pipelineId: string;
  status: string;
  message: string;
}

/** 取消管道响应 */
export interface CancelPipelineResponse {
  success: boolean;
  message: string;
}

/** 管道配置 */
export interface PipelineConfig {
  id: string;
  name: string;
  description: string;
  stagesCount: number;
}

/** WebSocket管道状态消息 */
export interface PipelineStatusWebSocketMessage {
  type: 'pipeline_status';
  timestamp: string;
  data: {
    pipelineId: string;
    status: PipelineStatus;
    currentStage?: string;
    progress: number;
    durationSeconds?: number;
  };
}
