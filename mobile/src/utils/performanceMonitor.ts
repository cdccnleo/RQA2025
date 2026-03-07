/**
 * RQA 2.0 性能监控工具
 *
 * 监控应用性能指标，包括启动时间、渲染性能、内存使用等
 * 支持性能数据收集、分析和上报
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import {InteractionManager, PerformanceObserver, Platform} from 'react-native';

interface PerformanceMetric {
  name: string;
  value: number;
  timestamp: number;
  metadata?: Record<string, any>;
}

interface PerformanceReport {
  sessionId: string;
  startTime: number;
  endTime: number;
  metrics: PerformanceMetric[];
  deviceInfo: {
    platform: string;
    version: string;
    deviceId?: string;
  };
  appInfo: {
    version: string;
    buildNumber: string;
  };
}

class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private metrics: PerformanceMetric[] = [];
  private sessionId: string;
  private startTime: number;
  private observer: PerformanceObserver | null = null;

  private constructor() {
    this.sessionId = this.generateSessionId();
    this.startTime = Date.now();

    this.setupPerformanceObserver();
    this.setupInteractionManager();
  }

  public static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }

  /**
   * 生成会话ID
   */
  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * 设置性能观察器
   */
  private setupPerformanceObserver(): void {
    if (Platform.OS === 'web') {
      // Web平台使用Performance Observer API
      try {
        this.observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            this.recordMetric({
              name: entry.name,
              value: entry.duration,
              timestamp: Date.now(),
              metadata: {
                entryType: entry.entryType,
                startTime: entry.startTime,
              },
            });
          }
        });

        this.observer.observe({entryTypes: ['measure', 'navigation']});
      } catch (error) {
        console.warn('Performance Observer not supported:', error);
      }
    }
  }

  /**
   * 设置交互管理器监控
   */
  private setupInteractionManager(): void {
    InteractionManager.setDeadline(100); // 100ms deadline for interactions
  }

  /**
   * 记录性能指标
   */
  public recordMetric(metric: Omit<PerformanceMetric, 'timestamp'>): void {
    const fullMetric: PerformanceMetric = {
      ...metric,
      timestamp: Date.now(),
    };

    this.metrics.push(fullMetric);

    // 实时上报重要指标
    if (this.isCriticalMetric(metric.name)) {
      this.reportMetricImmediately(fullMetric);
    }

    // 本地存储（生产环境应该有大小限制）
    if (this.metrics.length > 1000) {
      this.metrics = this.metrics.slice(-500); // 保留最新的500条
    }
  }

  /**
   * 标记应用启动完成
   */
  public markAppLaunch(): void {
    const launchTime = Date.now() - this.startTime;
    this.recordMetric({
      name: 'app_launch',
      value: launchTime,
      metadata: {
        phase: 'complete',
      },
    });
  }

  /**
   * 标记页面加载完成
   */
  public markPageLoad(pageName: string, loadTime: number): void {
    this.recordMetric({
      name: 'page_load',
      value: loadTime,
      metadata: {
        page: pageName,
      },
    });
  }

  /**
   * 标记API请求性能
   */
  public markApiRequest(endpoint: string, duration: number, success: boolean): void {
    this.recordMetric({
      name: 'api_request',
      value: duration,
      metadata: {
        endpoint,
        success,
      },
    });
  }

  /**
   * 标记内存使用情况
   */
  public markMemoryUsage(): void {
    // React Native没有直接的内存API，这里记录一个标记
    this.recordMetric({
      name: 'memory_check',
      value: 0,
      metadata: {
        timestamp: Date.now(),
      },
    });
  }

  /**
   * 标记帧率下降
   */
  public markFrameDrop(frameRate: number): void {
    this.recordMetric({
      name: 'frame_drop',
      value: frameRate,
      metadata: {
        threshold: 60, // 预期60fps
      },
    });
  }

  /**
   * 获取性能指标摘要
   */
  public getMetricsSummary(): {
    totalMetrics: number;
    averageResponseTime: number;
    errorRate: number;
    slowRequests: number;
  } {
    const apiMetrics = this.metrics.filter(m => m.name === 'api_request');
    const slowRequests = apiMetrics.filter(m => m.value > 2000).length; // 超过2秒的请求
    const failedRequests = apiMetrics.filter(m => m.metadata?.success === false).length;

    return {
      totalMetrics: this.metrics.length,
      averageResponseTime: apiMetrics.length > 0
        ? apiMetrics.reduce((sum, m) => sum + m.value, 0) / apiMetrics.length
        : 0,
      errorRate: apiMetrics.length > 0 ? failedRequests / apiMetrics.length : 0,
      slowRequests,
    };
  }

  /**
   * 生成性能报告
   */
  public generateReport(): PerformanceReport {
    return {
      sessionId: this.sessionId,
      startTime: this.startTime,
      endTime: Date.now(),
      metrics: [...this.metrics],
      deviceInfo: {
        platform: Platform.OS,
        version: Platform.Version as string,
      },
      appInfo: {
        version: '2.0.0',
        buildNumber: '1',
      },
    };
  }

  /**
   * 导出性能数据（用于调试）
   */
  public exportMetrics(): string {
    return JSON.stringify(this.generateReport(), null, 2);
  }

  /**
   * 判断是否为关键指标
   */
  private isCriticalMetric(metricName: string): boolean {
    const criticalMetrics = [
      'app_launch',
      'page_load',
      'api_request',
      'frame_drop',
      'memory_check',
    ];
    return criticalMetrics.includes(metricName);
  }

  /**
   * 立即上报关键指标
   */
  private reportMetricImmediately(metric: PerformanceMetric): void {
    // 在生产环境中，这里应该发送到监控服务
    console.log('Critical metric:', metric);

    // 示例：发送到监控服务
    // monitoringService.reportMetric(metric);
  }

  /**
   * 清理资源
   */
  public cleanup(): void {
    if (this.observer) {
      this.observer.disconnect();
      this.observer = null;
    }
    this.metrics = [];
  }
}

// 导出单例实例
export const performanceMonitor = PerformanceMonitor.getInstance();

// 便捷的性能监控装饰器
export function withPerformanceMonitoring<T extends any[], R>(
  fn: (...args: T) => R,
  name: string
) {
  return (...args: T): R => {
    const startTime = Date.now();
    try {
      const result = fn(...args);
      const duration = Date.now() - startTime;
      performanceMonitor.recordMetric({
        name,
        value: duration,
        metadata: {success: true},
      });
      return result;
    } catch (error) {
      const duration = Date.now() - startTime;
      performanceMonitor.recordMetric({
        name,
        value: duration,
        metadata: {success: false, error: error.message},
      });
      throw error;
    }
  };
}

export default PerformanceMonitor;




