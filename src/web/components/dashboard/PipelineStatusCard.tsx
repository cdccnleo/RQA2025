/**
 * 管道状态卡片组件
 *
 * 在Dashboard展示管道执行状态概览
 */

import React from 'react';
import { Play, Pause, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import { usePipelineList } from '../../hooks/usePipeline';
import Link from 'next/link';

export const PipelineStatusCard: React.FC = () => {
  const { pipelines, total, running, completed, failed, loading } = usePipelineList();

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

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">ML管道状态</h3>
        <Link
          href="/pipeline"
          className="text-sm text-blue-600 hover:text-blue-800 transition-colors"
        >
          查看全部 →
        </Link>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-32">
          <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
        </div>
      ) : (
        <>
          {/* 统计概览 */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-800">{total}</div>
              <div className="text-xs text-gray-500">总管道</div>
            </div>
            <div className="text-center p-3 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{running}</div>
              <div className="text-xs text-blue-500">运行中</div>
            </div>
            <div className="text-center p-3 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">{completed}</div>
              <div className="text-xs text-green-500">已完成</div>
            </div>
            <div className="text-center p-3 bg-red-50 rounded-lg">
              <div className="text-2xl font-bold text-red-600">{failed}</div>
              <div className="text-xs text-red-500">失败</div>
            </div>
          </div>

          {/* 最近管道列表 */}
          <div className="space-y-3">
            {pipelines.slice(0, 3).map((pipeline) => (
              <div
                key={pipeline.id}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center space-x-3">
                  {getStatusIcon(pipeline.status)}
                  <div>
                    <div className="font-medium text-gray-800">{pipeline.name}</div>
                    <div className="text-xs text-gray-500">
                      {pipeline.currentStage || '等待执行'}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium text-gray-700">
                    {pipeline.progress.toFixed(0)}%
                  </div>
                  <div className="w-20 h-2 bg-gray-200 rounded-full mt-1">
                    <div
                      className="h-full bg-blue-500 rounded-full transition-all duration-300"
                      style={{ width: `${pipeline.progress}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* 快速操作 */}
          <div className="mt-4 pt-4 border-t border-gray-200">
            <Link
              href="/pipeline"
              className="flex items-center justify-center w-full py-2 px-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Play className="w-4 h-4 mr-2" />
              执行新管道
            </Link>
          </div>
        </>
      )}
    </div>
  );
};

export default PipelineStatusCard;
