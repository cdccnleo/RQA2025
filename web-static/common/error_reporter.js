/**
 * 错误日志上报器
 * 实现前端错误集中收集和上报
 */

class ErrorReporter {
    /**
     * 构造函数
     * @param {Object} config - 配置对象
     * @param {string} config.apiEndpoint - 错误上报API端点
     * @param {number} config.maxQueueSize - 最大队列大小，默认50
     * @param {number} config.flushInterval - 批量上报间隔（毫秒），默认5000
     */
    constructor(config = {}) {
        this.apiEndpoint = config.apiEndpoint || '/api/v1/monitoring/errors';
        this.maxQueueSize = config.maxQueueSize || 50;
        this.flushInterval = config.flushInterval || 5000;
        this.errorQueue = [];
        this.reportedErrors = new Set(); // 用于去重
        this.flushTimer = null;
        
        // 监听全局错误
        this._setupErrorHandlers();
        
        // 启动定时刷新
        this.start();
    }

    /**
     * 设置错误处理器
     */
    _setupErrorHandlers() {
        // 监听JavaScript错误
        window.addEventListener('error', (event) => {
            this.reportError({
                type: 'javascript_error',
                message: event.message,
                filename: event.filename,
                lineno: event.lineno,
                colno: event.colno,
                stack: event.error?.stack,
                timestamp: Date.now()
            });
        });

        // 监听Promise rejection
        window.addEventListener('unhandledrejection', (event) => {
            this.reportError({
                type: 'promise_rejection',
                message: event.reason?.message || String(event.reason),
                stack: event.reason?.stack,
                timestamp: Date.now()
            });
        });
    }

    /**
     * 生成错误指纹（用于去重）
     * @param {Object} error - 错误对象
     * @returns {string} 错误指纹
     */
    _generateFingerprint(error) {
        const key = `${error.type}:${error.message}:${error.filename || ''}:${error.lineno || ''}`;
        return btoa(key).substring(0, 32);
    }

    /**
     * 报告错误
     * @param {Object} error - 错误对象
     */
    reportError(error) {
        // 添加环境信息
        const enrichedError = {
            ...error,
            userAgent: navigator.userAgent,
            url: window.location.href,
            referrer: document.referrer,
            viewport: {
                width: window.innerWidth,
                height: window.innerHeight
            },
            timestamp: error.timestamp || Date.now()
        };

        // 生成指纹并检查是否已上报
        const fingerprint = this._generateFingerprint(enrichedError);
        if (this.reportedErrors.has(fingerprint)) {
            return; // 已上报过，跳过
        }

        this.reportedErrors.add(fingerprint);
        this.errorQueue.push(enrichedError);

        // 如果队列已满，立即上报
        if (this.errorQueue.length >= this.maxQueueSize) {
            this.flush();
        }
    }

    /**
     * 批量上报错误
     */
    async flush() {
        if (this.errorQueue.length === 0) {
            return;
        }

        const errorsToReport = [...this.errorQueue];
        this.errorQueue = [];

        try {
            const response = await fetch(this.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    errors: errorsToReport,
                    timestamp: Date.now()
                })
            });

            if (response.ok) {
                console.log(`成功上报 ${errorsToReport.length} 个错误`);
            } else {
                // 上报失败，重新加入队列（限制重试次数）
                errorsToReport.forEach(error => {
                    if (!error.retryCount || error.retryCount < 3) {
                        error.retryCount = (error.retryCount || 0) + 1;
                        this.errorQueue.push(error);
                    }
                });
            }
        } catch (error) {
            console.error('错误上报失败:', error);
            // 上报失败，重新加入队列
            errorsToReport.forEach(error => {
                if (!error.retryCount || error.retryCount < 3) {
                    error.retryCount = (error.retryCount || 0) + 1;
                    this.errorQueue.push(error);
                }
            });
        }
    }

    /**
     * 开始自动上报
     */
    start() {
        if (this.flushTimer) {
            return; // 已经启动
        }

        this.flushTimer = setInterval(() => {
            this.flush();
        }, this.flushInterval);
    }

    /**
     * 停止自动上报
     */
    stop() {
        if (this.flushTimer) {
            clearInterval(this.flushTimer);
            this.flushTimer = null;
        }

        // 停止前上报剩余错误
        this.flush();
    }
}

// 创建全局实例
const errorReporter = new ErrorReporter();

// 页面卸载前上报剩余错误
window.addEventListener('beforeunload', () => {
    errorReporter.stop();
});

