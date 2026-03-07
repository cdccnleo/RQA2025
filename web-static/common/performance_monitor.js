/**
 * 前端性能监控
 * 实现前端性能指标采集和上报
 */

class PerformanceMonitor {
    /**
     * 构造函数
     * @param {Object} config - 配置对象
     * @param {string} config.apiEndpoint - 性能数据上报API端点
     * @param {number} config.reportInterval - 上报间隔（毫秒），默认60000（1分钟）
     */
    constructor(config = {}) {
        this.apiEndpoint = config.apiEndpoint || '/api/v1/monitoring/performance';
        this.reportInterval = config.reportInterval || 60000;
        this.metrics = {
            apiCalls: [],
            websocketLatency: [],
            pageLoadTime: null,
            renderTime: null
        };
        this.reportTimer = null;
        this.startTime = performance.now();
        
        // 监听页面加载性能
        this._measurePageLoad();
        
        // 监听API调用
        this._interceptFetch();
        
        // 监听WebSocket延迟
        this._monitorWebSocket();
    }

    /**
     * 测量页面加载性能
     */
    _measurePageLoad() {
        if (window.performance && window.performance.timing) {
            window.addEventListener('load', () => {
                const timing = window.performance.timing;
                this.metrics.pageLoadTime = timing.loadEventEnd - timing.navigationStart;
                this.metrics.renderTime = timing.domContentLoadedEventEnd - timing.navigationStart;
            });
        }
    }

    /**
     * 拦截fetch请求，统计API调用耗时
     */
    _interceptFetch() {
        const originalFetch = window.fetch;
        const self = this;
        
        window.fetch = async function(...args) {
            const startTime = performance.now();
            const url = args[0];
            
            try {
                const response = await originalFetch.apply(this, args);
                const endTime = performance.now();
                const duration = endTime - startTime;
                
                // 记录API调用
                self.metrics.apiCalls.push({
                    url: url,
                    method: args[1]?.method || 'GET',
                    duration: duration,
                    status: response.status,
                    timestamp: Date.now()
                });
                
                // 只保留最近100条记录
                if (self.metrics.apiCalls.length > 100) {
                    self.metrics.apiCalls.shift();
                }
                
                return response;
            } catch (error) {
                const endTime = performance.now();
                const duration = endTime - startTime;
                
                self.metrics.apiCalls.push({
                    url: url,
                    method: args[1]?.method || 'GET',
                    duration: duration,
                    status: 'error',
                    error: error.message,
                    timestamp: Date.now()
                });
                
                throw error;
            }
        };
    }

    /**
     * 监控WebSocket延迟
     */
    _monitorWebSocket() {
        const originalWebSocket = window.WebSocket;
        const self = this;
        
        window.WebSocket = function(...args) {
            const ws = new originalWebSocket(...args);
            
            // 监听消息，计算延迟
            const originalOnMessage = ws.onmessage;
            ws.onmessage = function(event) {
                if (event.data) {
                    try {
                        const message = JSON.parse(event.data);
                        if (message.timestamp) {
                            const latency = Date.now() - new Date(message.timestamp).getTime();
                            self.metrics.websocketLatency.push({
                                latency: latency,
                                timestamp: Date.now()
                            });
                            
                            // 只保留最近100条记录
                            if (self.metrics.websocketLatency.length > 100) {
                                self.metrics.websocketLatency.shift();
                            }
                        }
                    } catch (e) {
                        // 忽略解析错误
                    }
                }
                
                if (originalOnMessage) {
                    originalOnMessage.call(this, event);
                }
            };
            
            return ws;
        };
    }

    /**
     * 获取性能统计
     * @returns {Object} 性能统计对象
     */
    getStats() {
        const apiCalls = this.metrics.apiCalls;
        const websocketLatency = this.metrics.websocketLatency;
        
        const avgApiDuration = apiCalls.length > 0
            ? apiCalls.reduce((sum, call) => sum + call.duration, 0) / apiCalls.length
            : 0;
        
        const avgWebSocketLatency = websocketLatency.length > 0
            ? websocketLatency.reduce((sum, item) => sum + item.latency, 0) / websocketLatency.length
            : 0;
        
        return {
            pageLoadTime: this.metrics.pageLoadTime,
            renderTime: this.metrics.renderTime,
            apiCalls: {
                total: apiCalls.length,
                avgDuration: avgApiDuration,
                recent: apiCalls.slice(-10)
            },
            websocket: {
                avgLatency: avgWebSocketLatency,
                recent: websocketLatency.slice(-10)
            },
            uptime: performance.now() - this.startTime
        };
    }

    /**
     * 上报性能数据
     */
    async report() {
        const stats = this.getStats();
        
        try {
            const response = await fetch(this.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    ...stats,
                    userAgent: navigator.userAgent,
                    url: window.location.href,
                    timestamp: Date.now()
                })
            });
            
            if (response.ok) {
                console.log('性能数据上报成功');
            }
        } catch (error) {
            console.error('性能数据上报失败:', error);
        }
    }

    /**
     * 开始自动上报
     */
    start() {
        if (this.reportTimer) {
            return; // 已经启动
        }
        
        // 立即上报一次
        this.report();
        
        // 设置定时上报
        this.reportTimer = setInterval(() => {
            this.report();
        }, this.reportInterval);
    }

    /**
     * 停止自动上报
     */
    stop() {
        if (this.reportTimer) {
            clearInterval(this.reportTimer);
            this.reportTimer = null;
        }
    }
}

// 创建全局实例
const performanceMonitor = new PerformanceMonitor();

// 页面加载完成后自动开始监控
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        performanceMonitor.start();
    });
} else {
    performanceMonitor.start();
}

