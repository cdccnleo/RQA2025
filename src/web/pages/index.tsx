import { useState, useEffect } from 'react';
import Head from 'next/head';

interface SystemStatus {
  message: string;
  status: string;
  environment: string;
  version: string;
  services: string[];
  timestamp: number;
}

interface HealthStatus {
  status: string;
  service: string;
  environment: string;
  container: boolean;
  timestamp: number;
}

export default function Home() {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const [systemRes, healthRes] = await Promise.all([
          fetch('http://localhost:8000/'),
          fetch('http://localhost:8000/health')
        ]);

        const systemData = await systemRes.json();
        const healthData = await healthRes.json();

        setSystemStatus(systemData);
        setHealthStatus(healthData);
      } catch (error) {
        console.error('Failed to fetch system status:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchStatus();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Head>
        <title>RQA2025 量化交易系统</title>
        <meta name="description" content="RQA2025 现代化量化交易AI自主系统" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            🚀 RQA2025 量化交易系统
          </h1>
          <p className="text-xl text-gray-600">
            现代化量化交易AI自主系统
          </p>
        </div>

        {/* Status Cards */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {/* System Status Card */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4 flex items-center">
              📊 系统状态
            </h2>
            {loading ? (
              <div className="animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-4 bg-gray-200 rounded w-1/2"></div>
              </div>
            ) : systemStatus ? (
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">状态:</span>
                  <span className={`font-semibold ${
                    systemStatus.status === 'running' ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {systemStatus.status === 'running' ? '🟢 运行中' : '🔴 停止'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">环境:</span>
                  <span className="font-semibold">{systemStatus.environment}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">版本:</span>
                  <span className="font-semibold">{systemStatus.version}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">服务数量:</span>
                  <span className="font-semibold">{systemStatus.services.length}</span>
                </div>
              </div>
            ) : (
              <p className="text-red-600">❌ 无法获取系统状态</p>
            )}
          </div>

          {/* Health Status Card */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4 flex items-center">
              ❤️ 健康检查
            </h2>
            {loading ? (
              <div className="animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-4 bg-gray-200 rounded w-1/2"></div>
              </div>
            ) : healthStatus ? (
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">健康状态:</span>
                  <span className={`font-semibold ${
                    healthStatus.status === 'healthy' ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {healthStatus.status === 'healthy' ? '🟢 健康' : '🔴 不健康'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">服务:</span>
                  <span className="font-semibold">{healthStatus.service}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">容器化:</span>
                  <span className="font-semibold">
                    {healthStatus.container ? '✅ 是' : '❌ 否'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">最后检查:</span>
                  <span className="font-semibold text-sm">
                    {new Date(healthStatus.timestamp * 1000).toLocaleString()}
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-red-600">❌ 无法获取健康状态</p>
            )}
          </div>
        </div>

        {/* Services Overview */}
        {systemStatus && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4 flex items-center">
              🔧 核心服务
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {systemStatus.services.map((service, index) => (
                <div key={index} className="bg-blue-50 rounded-lg p-4 text-center">
                  <div className="text-2xl mb-2">
                    {service === 'strategy' && '📈'}
                    {service === 'trading' && '💰'}
                    {service === 'risk' && '🛡️'}
                    {service === 'data' && '📊'}
                  </div>
                  <div className="font-semibold text-gray-800 capitalize">
                    {service.replace('_', ' ')}
                  </div>
                  <div className="text-sm text-green-600 mt-1">✅ 可用</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Quick Actions */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4 flex items-center">
            ⚡ 快速操作
          </h2>
          <div className="grid md:grid-cols-3 gap-4">
            <a
              href="http://localhost:9090"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-purple-600 hover:bg-purple-700 text-white font-semibold py-3 px-6 rounded-lg transition duration-300 flex items-center justify-center"
            >
              📊 Prometheus监控
            </a>
            <a
              href="http://localhost:3000"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition duration-300 flex items-center justify-center"
            >
              📈 Grafana仪表板
            </a>
            <button
              onClick={() => window.location.reload()}
              className="bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-6 rounded-lg transition duration-300 flex items-center justify-center"
            >
              🔄 刷新状态
            </button>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-12 text-gray-500">
          <p>© 2025 RQA2025 量化交易系统 | 现代化AI自主交易平台</p>
        </footer>
      </main>
    </div>
  );
}
