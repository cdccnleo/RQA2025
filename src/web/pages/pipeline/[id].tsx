/**
 * 管道详情页面
 *
 * 展示单个管道的详细执行信息和日志
 */

import React, { useState } from 'react';
import { NextPage } from 'next';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Link from 'next/link';
import {
  ArrowLeft,
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  Pause,
  Terminal,
  RefreshCw,
  Cancel,
} from 'lucide-react';
import { usePipelineDetails, usePipelineWebSocket } from '../../hooks/usePipeline';

const PipelineDetailPage: NextPage = () => {
  const router = useRouter();
  const { id } = router.query;
  const pipelineId = typeof id === 'string' ? id : null;

  const { pipeline, stages, logs, loading, error, refresh } = usePipelineDetails(pipelineId);
  const { status, progress, currentStage, connected } = usePipelineWebSocket(pipelineId);

  const [activeTab, setActiveTab] = useState<'overview' | 'stages' | 'logs'>('overview');

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <Loader2 className="w-6 h-6 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="w-6 h-6 text-green-500" />;
      case 'failed':
        return <XCircle className="w-6 h-6 text-red-500" />;
      default:
        return <Pause className="w-6 h-6 text-gray-400" />;
    }
  };

  const getStatusText = (status: string) => {
    const statusMap: Record<string, string> = {
      pending: '等待中',
      running: '运行中',
      completed: '已完成',
      failed: '失败',
      rolling_back: '回滚中',
      rolled_back: '已回滚',
      cancelled: '已取消',
    };
    return statusMap[status] || status;
  };

  const getStageStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'skipped':
        return <Pause className="w-5 h-5 text-gray-400" />;
      default:
        return <div className="w-5 h-5 rounded-full border-2 border-gray-300" />;
    }
  };

  if (!pipelineId) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-gray-500">加载中...</div>
      </div>
    );
  }

  return (
    <>
      <Head>
        <title>管道详情 - {pipeline?.name || '加载中'} - 量化交易系统</title>
      </Head>

      <div className="min-h-screen bg-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* 返回按钮 */}
          <Link
            href="/pipeline"
            className="inline-flex items-center text-gray-600 hover:text-gray-900 mb-6"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            返回管道列表
          </Link>

          {loading ? (
            <div className="flex items-center justify-center h-64">
              <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
            </div>
          ) : error ? (
            <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-red-700">
              {error}
            </div>
          ) : pipeline ? (
            <>
              {/* 管道头部信息 */}
              <div className="bg-white rounded-lg shadow-md p-6 mb-6">
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-4">
                    {getStatusIcon(status || pipeline.status)}
                    <div>
                      <h1 className="text-2xl font-bold text-gray-900">{pipeline.name}</h1>
                      <div className="mt-2 flex items-center space-x-4 text-sm text-gray-500">
                        <span>ID: {pipeline.id}</span>
                        <span>版本: {pipeline.version}</span>
                        <span
                          className={`px-2 py-1 rounded-full text-xs font-medium ${
                            (status || pipeline.status) === 'running'
                              ? 'bg-blue-100 text-blue-800'
                              : (status || pipeline.status) === 'completed'
                              ? 'bg-green-100 text-green-800'
                              : (status || pipeline.status) === 'failed'
                              ? 'bg-red-100 text-red-800'
                              : 'bg-gray-100 text-gray-800'
                          }`}
                        >
                          {getStatusText(status || pipeline.status)}
                        </span>
                        {connected && (
                          <span className="flex items-center text-green-600">
                            <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse" />
                            实时连接
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex space-x-3">
                    <button
                      onClick={() => refresh()}
                      className="flex items-center px-4 py-2 bg-white text-gray-700 rounded-lg shadow hover:bg-gray-50 transition-colors"
                    >
                      <RefreshCw className="w-4 h-4 mr-2" />
                      刷新
                    </button>
                    {(status || pipeline.status) === 'running' && (
                      <button className="flex items-center px-4 py-2 bg-red-600 text-white rounded-lg shadow hover:bg-red-700 transition-colors">
                        <Cancel className="w-4 h-4 mr-2" />
                        取消执行
                      </button>
                    )}
                  </div>
                </div>

                {/* 进度条 */}
                <div className="mt-6">
                  <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
                    <span>执行进度</span>
                    <span>{(progress || pipeline.progress).toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-3 bg-gray-200 rounded-full">
                    <div
                      className="h-full bg-blue-500 rounded-full transition-all duration-500"
                      style={{ width: `${progress || pipeline.progress}%` }}
                    />
                  </div>
                  <div className="mt-2 text-sm text-gray-500">
                    当前阶段: {currentStage || pipeline.currentStage || '等待开始'}
                  </div>
                </div>

                {/* 时间信息 */}
                <div className="mt-6 grid grid-cols-3 gap-4 pt-6 border-t border-gray-200">
                  <div>
                    <div className="text-sm text-gray-500">开始时间</div>
                    <div className="text-gray-900">
                      {pipeline.startTime
                        ? new Date(pipeline.startTime).toLocaleString()
                        : '-'}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-500">结束时间</div>
                    <div className="text-gray-900">
                      {pipeline.endTime
                        ? new Date(pipeline.endTime).toLocaleString()
                        : '-'}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-500">执行耗时</div>
                    <div className="text-gray-900">
                      {pipeline.durationSeconds
                        ? `${Math.round(pipeline.durationSeconds)} 秒`
                        : '-'}
                    </div>
                  </div>
                </div>
              </div>

              {/* 标签页切换 */}
              <div className="bg-white rounded-lg shadow-md">
                <div className="border-b border-gray-200">
                  <nav className="flex space-x-8 px-6">
                    {[
                      { id: 'overview', label: '概览' },
                      { id: 'stages', label: '执行阶段' },
                      { id: 'logs', label: '日志' },
                    ].map((tab) => (
                      <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id as any)}
                        className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                          activeTab === tab.id
                            ? 'border-blue-500 text-blue-600'
                            : 'border-transparent text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        {tab.label}
                      </button>
                    ))}
                  </nav>
                </div>

                <div className="p-6">
                  {/* 概览标签 */}
                  {activeTab === 'overview' && (
                    <div className="space-y-6">
                      <div>
                        <h3 className="text-lg font-medium text-gray-900 mb-4">执行上下文</h3>
                        <div className="bg-gray-50 rounded-lg p-4">
                          <pre className="text-sm text-gray-700 overflow-auto">
                            {JSON.stringify(pipeline.context, null, 2)}
                          </pre>
                        </div>
                      </div>

                      {pipeline.error && (
                        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                          <h3 className="text-lg font-medium text-red-900 mb-2">错误信息</h3>
                          <p className="text-red-700">{pipeline.error}</p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* 阶段标签 */}
                  {activeTab === 'stages' && (
                    <div className="space-y-4">
                      {stages.length === 0 ? (
                        <div className="text-center text-gray-500 py-8">
                          暂无阶段信息
                        </div>
                      ) : (
                        stages.map((stage, index) => (
                          <div
                            key={stage.name}
                            className="flex items-start space-x-4 p-4 bg-gray-50 rounded-lg"
                          >
                            <div className="flex-shrink-0 mt-1">
                              {getStageStatusIcon(stage.status)}
                            </div>
                            <div className="flex-1">
                              <div className="flex items-center justify-between">
                                <h4 className="font-medium text-gray-900">
                                  {index + 1}. {stage.name}
                                </h4>
                                <span
                                  className={`px-2 py-1 rounded-full text-xs font-medium ${
                                    stage.status === 'completed'
                                      ? 'bg-green-100 text-green-800'
                                      : stage.status === 'running'
                                      ? 'bg-blue-100 text-blue-800'
                                      : stage.status === 'failed'
                                      ? 'bg-red-100 text-red-800'
                                      : 'bg-gray-100 text-gray-800'
                                  }`}
                                >
                                  {stage.status}
                                </span>
                              </div>
                              {stage.durationSeconds && (
                                <div className="mt-1 text-sm text-gray-500">
                                  <Clock className="w-3 h-3 inline mr-1" />
                                  {Math.round(stage.durationSeconds)} 秒
                                </div>
                              )}
                              {stage.error && (
                                <div className="mt-2 text-sm text-red-600">
                                  错误: {stage.error}
                                </div>
                              )}
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  )}

                  {/* 日志标签 */}
                  {activeTab === 'logs' && (
                    <div className="bg-gray-900 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center text-gray-400">
                          <Terminal className="w-4 h-4 mr-2" />
                          <span className="text-sm">执行日志</span>
                        </div>
                        <span className="text-xs text-gray-500">
                          {logs.length} 条日志
                        </span>
                      </div>
                      <div className="space-y-2 max-h-96 overflow-auto font-mono text-sm">
                        {logs.length === 0 ? (
                          <div className="text-gray-500 text-center py-8">
                            暂无日志
                          </div>
                        ) : (
                          logs.map((log, index) => (
                            <div
                              key={index}
                              className={`flex space-x-3 ${
                                log.level === 'error'
                                  ? 'text-red-400'
                                  : log.level === 'warning'
                                  ? 'text-yellow-400'
                                  : 'text-gray-300'
                              }`}
                            >
                              <span className="text-gray-500 flex-shrink-0">
                                {new Date(log.timestamp).toLocaleTimeString()}
                              </span>
                              <span className="uppercase text-xs flex-shrink-0 w-12">
                                {log.level}
                              </span>
                              <span>{log.message}</span>
                            </div>
                          ))
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </>
          ) : null}
        </div>
      </div>
    </>
  );
};

export default PipelineDetailPage;
