/**
 * 统一的WebSocket管理器
 * 整合dashboard_websocket_helper.js和data_management_websocket_helper.js的功能
 * 提供统一的WebSocket连接管理、重连策略、错误处理和轮询回退机制
 */

class UnifiedWebSocketManager {
    /**
     * 构造函数
     * @param {Object} config - 配置对象
     * @param {number} config.pollingInterval - 默认轮询间隔（毫秒），默认15000
     * @param {number} config.maxReconnectAttempts - 最大重连次数，默认10
     * @param {number} config.baseReconnectDelay - 基础重连延迟（毫秒），默认5000
     */
    constructor(config = {}) {
        this.connections = new Map(); // 存储所有WebSocket连接
        this.defaultPollingInterval = config.pollingInterval || 15000; // 默认15秒轮询
        this.maxReconnectAttempts = config.maxReconnectAttempts || 10;
        this.baseReconnectDelay = config.baseReconnectDelay || 5000; // 5秒基础延迟
    }

    /**
     * 创建WebSocket连接
     * @param {string} endpoint - WebSocket端点路径（如 '/ws/dashboard-metrics'）
     * @param {string} channel - 频道名称（如 'dashboard_metrics'）
     * @param {Object} handlers - 处理器对象
     * @param {Function} handlers.onMessage - 消息处理函数
     * @param {Function} handlers.onError - 错误处理函数（可选）
     * @param {Function} handlers.onOpen - 连接打开处理函数（可选）
     * @param {Function} handlers.onClose - 连接关闭处理函数（可选）
     * @param {Function} handlers.fallbackLoad - 回退加载函数（轮询时调用）
     * @param {number} pollingInterval - 轮询间隔（毫秒），默认使用配置的默认值
     * @returns {Object} 连接管理器对象
     */
    connect(endpoint, channel, handlers, pollingInterval = null) {
        const pollInterval = pollingInterval || this.defaultPollingInterval;
        const maxReconnectAttempts = this.maxReconnectAttempts;
        const baseReconnectDelay = this.baseReconnectDelay;
        let websocket = null;
        let pollingTimer = null;
        let reconnectTimer = null;
        let reconnectAttempts = 0;

        const connectionManager = {
            channel,
            endpoint,
            isConnected: () => websocket && websocket.readyState === WebSocket.OPEN,
            disconnect: () => {
                disconnect();
            },
            reconnect: () => {
                reconnect();
            }
        };

        /**
         * 启动轮询模式
         */
        function startPolling() {
            if (pollingTimer || !handlers.fallbackLoad) {
                return; // 已经在轮询中或没有回退函数
            }

            console.log(`启动${channel}轮询模式，间隔${pollInterval}ms`);

            // 立即执行一次
            try {
                handlers.fallbackLoad();
            } catch (error) {
                console.error(`轮询加载失败 (${channel}):`, error);
            }

            // 设置定时轮询
            pollingTimer = setInterval(() => {
                if (handlers.fallbackLoad) {
                    try {
                        handlers.fallbackLoad();
                    } catch (error) {
                        console.error(`轮询加载失败 (${channel}):`, error);
                    }
                }
            }, pollInterval);
        }

        /**
         * 停止轮询
         */
        function stopPolling() {
            if (pollingTimer) {
                clearInterval(pollingTimer);
                pollingTimer = null;
            }
        }

        /**
         * 连接WebSocket
         */
        function connect() {
            // 如果已有连接，先关闭
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                return;
            }

            // 清除重连定时器
            if (reconnectTimer) {
                clearTimeout(reconnectTimer);
                reconnectTimer = null;
            }

            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsHost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
                ? 'localhost:8000'
                : window.location.host;
            const wsUrl = `${wsProtocol}//${wsHost}${endpoint}`;

            try {
                websocket = new WebSocket(wsUrl);

                websocket.onopen = function(event) {
                    console.log(`${channel} WebSocket连接已建立`);
                    reconnectAttempts = 0;

                    // 连接成功后，清除轮询定时器
                    stopPolling();

                    // 调用自定义onOpen处理器
                    if (handlers.onOpen) {
                        try {
                            handlers.onOpen(event);
                        } catch (error) {
                            console.error(`${channel} onOpen处理器错误:`, error);
                        }
                    }
                };

                websocket.onmessage = function(event) {
                    try {
                        const message = JSON.parse(event.data);
                        if (handlers.onMessage) {
                            handlers.onMessage(message, event);
                        }
                    } catch (error) {
                        console.error(`处理${channel} WebSocket消息失败:`, error);
                    }
                };

                websocket.onerror = function(error) {
                    console.error(`${channel} WebSocket错误:`, error);

                    // 调用自定义onError处理器
                    if (handlers.onError) {
                        try {
                            handlers.onError(error);
                        } catch (err) {
                            console.error(`${channel} onError处理器错误:`, err);
                        }
                    }

                    // WebSocket连接失败时，回退到轮询模式
                    if (!pollingTimer) {
                        console.warn(`WebSocket连接失败，使用轮询模式: ${channel}`);
                        startPolling();
                    }
                };

                websocket.onclose = function(event) {
                    console.log(`${channel} WebSocket连接已关闭 (code: ${event.code}, reason: ${event.reason || '未知'})`);
                    websocket = null;

                    // 调用自定义onClose处理器
                    if (handlers.onClose) {
                        try {
                            handlers.onClose(event);
                        } catch (error) {
                            console.error(`${channel} onClose处理器错误:`, error);
                        }
                    }

                    // 回退到轮询模式
                    if (!pollingTimer) {
                        startPolling();
                    }

                    // 尝试重连（指数退避）
                    if (reconnectAttempts < maxReconnectAttempts) {
                        const delay = baseReconnectDelay * Math.pow(2, Math.min(reconnectAttempts, 5)); // 最大延迟约160秒
                        console.log(`${channel} WebSocket将在${delay/1000}秒后尝试重连 (第${reconnectAttempts + 1}次)...`);
                        reconnectTimer = setTimeout(() => {
                            reconnectAttempts++;
                            connect();
                        }, delay);
                    } else {
                        console.warn(`${channel} WebSocket已达到最大重连次数，停止重连`);
                    }
                };
            } catch (error) {
                console.error(`WebSocket连接失败 (${channel}):`, error);
                // WebSocket连接失败时，回退到轮询模式
                if (!pollingTimer) {
                    startPolling();
                }
            }
        }

        /**
         * 断开连接
         */
        function disconnect() {
            // 清除重连定时器
            if (reconnectTimer) {
                clearTimeout(reconnectTimer);
                reconnectTimer = null;
            }

            // 停止轮询
            stopPolling();

            // 关闭WebSocket连接
            if (websocket) {
                if (websocket.readyState === WebSocket.OPEN || websocket.readyState === WebSocket.CONNECTING) {
                    websocket.close(1000, '主动断开连接');
                }
                websocket = null;
            }

            reconnectAttempts = 0;
            console.log(`${channel} WebSocket连接已断开`);
        }

        /**
         * 重新连接
         */
        function reconnect() {
            disconnect();
            reconnectAttempts = 0;
            connect();
        }

        // 存储连接管理器
        connectionManager._internal = {
            connect,
            disconnect,
            websocket,
            pollingTimer,
            reconnectTimer
        };

        this.connections.set(channel, connectionManager);

        // 立即连接
        connect();

        return connectionManager;
    }

    /**
     * 断开指定频道的连接
     * @param {string} channel - 频道名称
     */
    disconnect(channel) {
        const connection = this.connections.get(channel);
        if (connection && connection._internal) {
            connection._internal.disconnect();
            this.connections.delete(channel);
        }
    }

    /**
     * 断开所有连接
     */
    disconnectAll() {
        for (const [channel] of this.connections) {
            this.disconnect(channel);
        }
    }

    /**
     * 检查指定频道是否已连接
     * @param {string} channel - 频道名称
     * @returns {boolean} 是否已连接
     */
    isConnected(channel) {
        const connection = this.connections.get(channel);
        return connection ? connection.isConnected() : false;
    }
}

// 创建全局实例
const wsManager = new UnifiedWebSocketManager();

/**
 * 便捷函数：连接Dashboard指标WebSocket
 * @param {Function} onMessage - 消息处理函数
 * @param {Function} fallbackLoad - 回退加载函数
 * @returns {Object} 连接管理器对象
 */
function connectDashboardMetricsWebSocket(onMessage, fallbackLoad) {
    return wsManager.connect('/ws/dashboard-metrics', 'dashboard_metrics', {
        onMessage: (message) => {
            if (message.type === 'dashboard_metrics' && message.data) {
                onMessage(message.data);
            }
        },
        fallbackLoad: fallbackLoad
    }, 10000); // 10秒轮询间隔
}

/**
 * 便捷函数：连接Dashboard告警和事件WebSocket
 * @param {Function} onMessage - 消息处理函数
 * @param {Function} fallbackLoad - 回退加载函数
 * @returns {Object} 连接管理器对象
 */
function connectDashboardAlertsWebSocket(onMessage, fallbackLoad) {
    return wsManager.connect('/ws/dashboard-alerts', 'dashboard_alerts', {
        onMessage: (message) => {
            if (message.type === 'dashboard_alerts' && message.data) {
                onMessage(message.data);
            }
        },
        fallbackLoad: fallbackLoad
    }, 30000); // 30秒轮询间隔
}

/**
 * 便捷函数：连接架构状态WebSocket
 * @param {Function} onMessage - 消息处理函数
 * @param {Function} fallbackLoad - 回退加载函数
 * @returns {Object} 连接管理器对象
 */
function connectArchitectureStatusWebSocket(onMessage, fallbackLoad) {
    return wsManager.connect('/ws/architecture-status', 'architecture_status', {
        onMessage: (message) => {
            if (message.type === 'architecture_status' && message.data) {
                onMessage(message.data);
            }
        },
        fallbackLoad: fallbackLoad
    }, 15000); // 15秒轮询间隔
}

/**
 * 便捷函数：连接数据质量监控WebSocket
 * @param {Function} onMessage - 消息处理函数
 * @param {Function} fallbackLoad - 回退加载函数
 * @returns {Object} 连接管理器对象
 */
function connectDataQualityWebSocket(onMessage, fallbackLoad) {
    return wsManager.connect('/ws/data-quality', 'data_quality', {
        onMessage: (message) => {
            if (message.type === 'data_quality' && message.data) {
                onMessage(message.data);
            }
        },
        fallbackLoad: fallbackLoad
    }, 15000); // 15秒轮询间隔
}

/**
 * 便捷函数：连接缓存系统监控WebSocket
 * @param {Function} onMessage - 消息处理函数
 * @param {Function} fallbackLoad - 回退加载函数
 * @returns {Object} 连接管理器对象
 */
function connectDataCacheWebSocket(onMessage, fallbackLoad) {
    return wsManager.connect('/ws/data-cache', 'data_cache', {
        onMessage: (message) => {
            if (message.type === 'data_cache' && message.data) {
                onMessage(message.data);
            }
        },
        fallbackLoad: fallbackLoad
    }, 15000); // 15秒轮询间隔
}

/**
 * 便捷函数：连接数据湖管理WebSocket
 * @param {Function} onMessage - 消息处理函数
 * @param {Function} fallbackLoad - 回退加载函数
 * @returns {Object} 连接管理器对象
 */
function connectDataLakeWebSocket(onMessage, fallbackLoad) {
    return wsManager.connect('/ws/data-lake', 'data_lake', {
        onMessage: (message) => {
            if (message.type === 'data_lake' && message.data) {
                onMessage(message.data);
            }
        },
        fallbackLoad: fallbackLoad
    }, 15000); // 15秒轮询间隔
}

/**
 * 便捷函数：连接数据性能监控WebSocket
 * @param {Function} onMessage - 消息处理函数
 * @param {Function} fallbackLoad - 回退加载函数
 * @returns {Object} 连接管理器对象
 */
function connectDataPerformanceWebSocket(onMessage, fallbackLoad) {
    return wsManager.connect('/ws/data-performance', 'data_performance', {
        onMessage: (message) => {
            if (message.type === 'data_performance' && message.data) {
                onMessage(message.data);
            }
        },
        fallbackLoad: fallbackLoad
    }, 10000); // 10秒轮询间隔
}


