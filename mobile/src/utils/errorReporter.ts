/**
 * RQA 2.0 错误报告工具
 *
 * 捕获和报告应用错误、崩溃和异常
 * 支持错误分类、用户反馈和远程上报
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import React from 'react';
import {Alert, Platform, View, Text, TouchableOpacity} from 'react-native';

interface ErrorReport {
  id: string;
  timestamp: number;
  error: {
    name: string;
    message: string;
    stack?: string;
  };
  context: {
    userId?: string;
    screen?: string;
    action?: string;
    metadata?: Record<string, any>;
  };
  device: {
    platform: string;
    version: string;
    model?: string;
    appVersion: string;
  };
  severity: 'low' | 'medium' | 'high' | 'critical';
  userFeedback?: string;
}

class ErrorReporter {
  private static instance: ErrorReporter;
  private reports: ErrorReport[] = [];
  private isInitialized = false;

  private constructor() {}

  public static getInstance(): ErrorReporter {
    if (!ErrorReporter.instance) {
      ErrorReporter.instance = new ErrorReporter();
    }
    return ErrorReporter.instance;
  }

  /**
   * 初始化错误报告器
   */
  public initialize(): void {
    if (this.isInitialized) return;

    // 设置全局错误处理器
    this.setupGlobalErrorHandler();

    // 设置React错误边界处理器
    this.setupReactErrorHandler();

    // 设置Promise拒绝处理器
    this.setupPromiseRejectionHandler();

    this.isInitialized = true;
    console.log('ErrorReporter initialized');
  }

  /**
   * 设置全局错误处理器
   */
  private setupGlobalErrorHandler(): void {
    const originalHandler = ErrorUtils.getGlobalHandler();

    ErrorUtils.setGlobalHandler((error, isFatal) => {
      this.captureError(error, {
        severity: isFatal ? 'critical' : 'high',
        context: {
          action: 'global_error',
          metadata: {isFatal},
        },
      });

      // 调用原始处理器
      if (originalHandler) {
        originalHandler(error, isFatal);
      }

      // 对于致命错误，显示用户友好的提示
      if (isFatal) {
        Alert.alert(
          '应用出现问题',
          '应用遇到严重错误，即将重启。请联系客服获取帮助。',
          [{text: '确定'}],
          {cancelable: false}
        );
      }
    });
  }

  /**
   * 设置React错误边界处理器
   */
  private setupReactErrorHandler(): void {
    // 这个会在ErrorBoundary组件中使用
  }

  /**
   * 设置Promise拒绝处理器
   */
  private setupPromiseRejectionHandler(): void {
    const originalHandler = global.onunhandledrejection;

    global.onunhandledrejection = (event) => {
      this.captureError(new Error(event.reason), {
        severity: 'high',
        context: {
          action: 'unhandled_promise_rejection',
          metadata: {promise: event.promise},
        },
      });

      // 调用原始处理器
      if (originalHandler) {
        originalHandler(event);
      }
    };
  }

  /**
   * 捕获错误
   */
  public captureError(
    error: Error,
    options: {
      severity?: 'low' | 'medium' | 'high' | 'critical';
      context?: {
        screen?: string;
        action?: string;
        userId?: string;
        metadata?: Record<string, any>;
      };
      userFeedback?: string;
    } = {}
  ): string {
    const reportId = this.generateReportId();

    const report: ErrorReport = {
      id: reportId,
      timestamp: Date.now(),
      error: {
        name: error.name,
        message: error.message,
        stack: error.stack,
      },
      context: {
        ...options.context,
      },
      device: {
        platform: Platform.OS,
        version: Platform.Version as string,
        appVersion: '2.0.0',
      },
      severity: options.severity || 'medium',
      userFeedback: options.userFeedback,
    };

    this.reports.push(report);

    // 立即上报严重错误
    if (report.severity === 'high' || report.severity === 'critical') {
      this.reportErrorImmediately(report);
    }

    // 本地存储（生产环境应该有限制）
    if (this.reports.length > 100) {
      this.reports = this.reports.slice(-50); // 保留最新的50个报告
    }

    console.log('Error captured:', reportId, error.message);
    return reportId;
  }

  /**
   * 捕获消息（非错误）
   */
  public captureMessage(
    message: string,
    level: 'info' | 'warning' | 'error' = 'info',
    context?: Record<string, any>
  ): string {
    const severity = level === 'error' ? 'high' : level === 'warning' ? 'medium' : 'low';

    return this.captureError(new Error(message), {
      severity,
      context: {
        action: 'message',
        metadata: {level, ...context},
      },
    });
  }

  /**
   * 添加用户反馈
   */
  public addUserFeedback(reportId: string, feedback: string): void {
    const report = this.reports.find(r => r.id === reportId);
    if (report) {
      report.userFeedback = feedback;
      this.reportErrorImmediately(report);
    }
  }

  /**
   * 获取错误摘要
   */
  public getErrorSummary(): {
    totalErrors: number;
    criticalErrors: number;
    highErrors: number;
    recentErrors: ErrorReport[];
  } {
    const criticalErrors = this.reports.filter(r => r.severity === 'critical').length;
    const highErrors = this.reports.filter(r => r.severity === 'high').length;
    const recentErrors = this.reports
      .filter(r => Date.now() - r.timestamp < 24 * 60 * 60 * 1000) // 最近24小时
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, 10);

    return {
      totalErrors: this.reports.length,
      criticalErrors,
      highErrors,
      recentErrors,
    };
  }

  /**
   * 导出错误报告
   */
  public exportReports(): string {
    return JSON.stringify(this.reports, null, 2);
  }

  /**
   * 生成报告ID
   */
  private generateReportId(): string {
    return `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * 立即上报错误
   */
  private reportErrorImmediately(report: ErrorReport): void {
    // 在生产环境中，这里应该发送到错误监控服务
    console.log('Reporting error:', report.id, report.error.message);

    // 示例：发送到错误监控服务
    // errorMonitoringService.reportError(report);
  }

  /**
   * 清理资源
   */
  public cleanup(): void {
    this.reports = [];
  }
}

// 导出单例实例
export const errorReporter = ErrorReporter.getInstance();

// 高阶组件：错误边界
import React, {Component, ReactNode} from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: any) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  errorId?: string;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {hasError: false};
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    const errorId = errorReporter.captureError(error, {
      severity: 'high',
      context: {
        action: 'react_error_boundary',
      },
    });

    return {hasError: true, errorId};
  }

  componentDidCatch(error: Error, errorInfo: any) {
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <ErrorFallback
          errorId={this.state.errorId}
          onRetry={() => this.setState({hasError: false})}
        />
      );
    }

    return this.props.children;
  }
}

// 错误回退组件
interface ErrorFallbackProps {
  errorId?: string;
  onRetry: () => void;
}

const ErrorFallback: React.FC<ErrorFallbackProps> = ({errorId, onRetry}) => {
  return (
    <View style={{flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20}}>
      <Text style={{fontSize: 18, fontWeight: 'bold', marginBottom: 10}}>
        出现了一个错误
      </Text>
      <Text style={{textAlign: 'center', marginBottom: 20}}>
        应用遇到问题，请稍后重试。如果问题持续，请联系客服。
      </Text>
      {errorId && (
        <Text style={{fontSize: 12, color: '#666', marginBottom: 20}}>
          错误ID: {errorId}
        </Text>
      )}
      <TouchableOpacity
        onPress={onRetry}
        style={{
          backgroundColor: '#007AFF',
          paddingHorizontal: 20,
          paddingVertical: 10,
          borderRadius: 8,
        }}>
        <Text style={{color: 'white', fontWeight: 'bold'}}>重试</Text>
      </TouchableOpacity>
    </View>
  );
};

export default ErrorReporter;
