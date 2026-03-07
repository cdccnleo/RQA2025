/**
 * 数据管理层WebSocket连接辅助函数
 * 统一管理数据质量、缓存系统、数据湖管理的WebSocket连接
 */

/**
 * 创建统一的WebSocket连接函数
 * @param {string} endpoint - WebSocket端点路径（如 '/ws/data-quality'）
 * @param {string} channel - 频道名称（如 'data_quality'）
 * @param {Function} onMessage - 消息处理函数
 * @param {Function} fallbackLoad - 回退加载函数（轮询时调用）
 * @param {number} pollingInterval - 轮询间隔（毫秒），默认15000（15秒）
 * @returns {Object} WebSocket连接管理器对象
 */
function createDataManagementWebSocket(endpoint, channel, onMessage, fallbackLoad, pollingInterval = 15000) {
    let websocket = null;
    let pollingTimer = null;
    let reconnectTimer = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 10;
    const baseReconnectDelay = 5000; // 5秒基础重连延迟

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

            websocket.onopen = function() {
                console.log(`${channel} WebSocket连接已建立`);
                reconnectAttempts = 0;
                
                // 连接成功后，清除轮询定时器
                if (pollingTimer) {
                    clearInterval(pollingTimer);
                    pollingTimer = null;
                }
            };

            websocket.onmessage = function(event) {
                try {
                    const message = JSON.parse(event.data);
                    if (message.type === channel && message.data) {
                        onMessage(message.data);
                    }
                } catch (error) {
                    console.error(`处理${channel} WebSocket消息失败:`, error);
                }
            };

            websocket.onerror = function(error) {
                console.error(`${channel} WebSocket错误:`, error);
                // WebSocket连接失败时，回退到轮询模式
                if (!pollingTimer) {
                    console.warn(`WebSocket连接失败，使用轮询模式: ${channel}`);
                    startPolling();
                }
            };

            websocket.onclose = function(event) {
                console.log(`${channel} WebSocket连接已关闭 (code: ${event.code}, reason: ${event.reason})`);
                websocket = null;
                
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
     * 启动轮询模式
     */
    function startPolling() {
        if (pollingTimer) {
            return; // 已经在轮询中
        }
        
        console.log(`启动${channel}轮询模式，间隔${pollingInterval}ms`);
        
        // 立即执行一次
        if (fallbackLoad) {
            try {
                fallbackLoad();
            } catch (error) {
                console.error(`轮询加载失败 (${channel}):`, error);
            }
        }
        
        // 设置定时轮询
        pollingTimer = setInterval(() => {
            if (fallbackLoad) {
                try {
                    fallbackLoad();
                } catch (error) {
                    console.error(`轮询加载失败 (${channel}):`, error);
                }
            }
        }, pollingInterval);
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
     * 检查连接状态
     */
    function isConnected() {
        return websocket && websocket.readyState === WebSocket.OPEN;
    }

    /**
     * 手动重连
     */
    function reconnect() {
        disconnect();
        reconnectAttempts = 0;
        connect();
    }

    return {
        connect,
        disconnect,
        reconnect,
        isConnected,
        startPolling,
        stopPolling
    };
}

/**
 * 连接数据质量监控WebSocket
 * @param {Function} onMessage - 消息处理函数
 * @param {Function} fallbackLoad - 回退加载函数
 * @returns {Object} WebSocket连接管理器对象
 */
function connectDataQualityWebSocket(onMessage, fallbackLoad) {
    return createDataManagementWebSocket(
        '/ws/data-quality',
        'data_quality',
        onMessage,
        fallbackLoad,
        15000 // 15秒轮询间隔
    );
}

/**
 * 连接缓存系统监控WebSocket
 * @param {Function} onMessage - 消息处理函数
 * @param {Function} fallbackLoad - 回退加载函数
 * @returns {Object} WebSocket连接管理器对象
 */
function connectDataCacheWebSocket(onMessage, fallbackLoad) {
    return createDataManagementWebSocket(
        '/ws/data-cache',
        'data_cache',
        onMessage,
        fallbackLoad,
        15000 // 15秒轮询间隔
    );
}

/**
 * 连接数据湖管理WebSocket
 * @param {Function} onMessage - 消息处理函数
 * @param {Function} fallbackLoad - 回退加载函数
 * @returns {Object} WebSocket连接管理器对象
 */
function connectDataLakeWebSocket(onMessage, fallbackLoad) {
    return createDataManagementWebSocket(
        '/ws/data-lake',
        'data_lake',
        onMessage,
        fallbackLoad,
        15000 // 15秒轮询间隔
    );
}

