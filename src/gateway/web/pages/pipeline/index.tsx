/**
 * 管道列表页面
 */

import React from 'react';
import { NextPage } from 'next';
import Head from 'next/head';
import Link from 'next/link';
import { Play, RefreshCw, Loader2, CheckCircle, XCircle, Pause } from 'lucide-react';
import { usePipelineList, useExecutePipeline } from '../../hooks/usePipeline';

const PipelinePage: NextPage = () => {
  const { pipelines, total, running, completed, failed, loading, refresh } = usePipelineList();
  const { execute, loading: executing } = useExecutePipeline();

  const handleExecute = async () => {
    try {
      await execute({
        configId: 'default',
        context: {
          symbols: ['AAPL', 'GOOGL', 'MSFT'],
          startDate: '2024-01-01',
          endDate: '2024-12-31',
        },
      });
      refresh();
    } catch (err) {
      console.error('执行失败:', err);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Pause className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusText = (status: string) => {
    const statusMap: Record<string, string> = {
      pending: '等待中',
      running: '运行中',
      completed: '已完成',
      failed: '失败',
      rolling_back: '回滚中',
      cancelled: '已取消',
    };
    return statusMap[status] || status;
  };

  return (
    <>
      <Head>
        <title>ML管道监控 - 量化交易系统</title>
      </Head>

      <div className="min-h-screen bg-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* 页面标题 */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">ML管道监控</h1>
              <p className="mt-2 text-gray-600">管理和监控自动化训练管道</p>
            </div>
            <div className="flex space-x-4">
              <button
                onClick={() => refresh()}
                className="flex items-center px-4 py-2 bg-white text-gray-700 rounded-lg shadow hover:bg-gray-50 transition-colors"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                刷新
              </button>
              <button
                onClick={handleExecute}
                disabled={executing}
                className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg shadow hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                {executing ? (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Play className="w-4 h-4 mr-2" />
                )}
                执行新管道
              </button>
            </div>
          </div>

          {/* 统计卡片 */}
          <div className="grid grid-cols-4 gap-6 mb-8">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="text-sm text-gray-500">总管道</div>
              <div className="text-3xl font-bold text-gray-900">{total}</div>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <div className="text-sm text-blue-500">运行中</div>
              <div className="text-3xl font-bold text-blue-600">{running}</div>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <div className="text-sm text-green-500">已完成</div>
              <div className="text-3xl font-bold text-green-600">{completed}</div>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <div className="text-sm text-red-500">失败</div>
              <div className="text-3xl font-bold text-red-600">{failed}</div>
            </div>
          </div>

          {/* 管道列表 */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">管道列表</h2>
            </div>

            {loading ? (
              <div className="flex items-center justify-center h-64">
                <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
              </div>
            ) : pipelines.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-64 text-gray-500">
                <Pause className="w-12 h-12 mb-4" />
                <p>暂无管道执行记录</p>
                <button
                  onClick={handleExecute}
                  className="mt-4 text-blue-600 hover:text-blue-800"
                >
                  执行第一个管道 →
                </button>
              </div>
            ) : (
              <div className="divide-y divide-gray-200">
                {pipelines.map((pipeline) => (
                  <div
                    key={pipeline.id}
                    className="px-6 py-4 hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        {getStatusIcon(pipeline.status)}
                        <div>
                          <Link
                            href={`/pipeline/${pipeline.id}`}
                            className="text-lg font-medium text-blue-600 hover:text-blue-800"
                          >
                            {pipeline.name}
                          </Link>
                          <div className="text-sm text-gray-500">
                            ID: {pipeline.id} | 版本: {pipeline.version}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium text-gray-900">
                          {getStatusText(pipeline.status)}
                        </div>
                        <div className="text-sm text-gray-500">
                          {pipeline.currentStage || '-'}
                        </div>
                      </div>
                    </div>

                    {/* 进度条 */}
                    <div className="mt-4">
                      <div className="flex items-center justify-between text-sm text-gray-600 mb-1">
                        <span>进度</span>
                        <span>{pipeline.progress.toFixed(1)}%</span>
                      </div>
                      <div className="w-full h-2 bg-gray-200 rounded-full">
                        <div
                          className="h-full bg-blue-500 rounded-full transition-all duration-300"
                          style={{ width: `${pipeline.progress}%` }}
                        />
                      </div>
                    </div>

                    {/* 时间信息 */}
                    <div className="mt-4 flex items-center space-x-6 text-sm text-gray-500">
                      {pipeline.startTime && (
                        <span>
                          开始: {new Date(pipeline.startTime).toLocaleString()}
                        </span>
                      )}
                      {pipeline.durationSeconds && (
                        <span>
                          耗时: {Math.round(pipeline.durationSeconds)}s
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
};

export default PipelinePage;
