// Dashboard WebSocket连接管理辅助函数
// 用于管理Dashboard的实时更新WebSocket连接

// Dashboard指标WebSocket连接管理
let dashboardMetricsWebSocket = null;
let dashboardMetricsPollingInterval = null;

// Dashboard告警和事件WebSocket连接管理
let dashboardAlertsWebSocket = null;
let dashboardAlertsPollingInterval = null;

/**
 * 连接Dashboard指标WebSocket（系统性能+数据流）
 */
function connectDashboardMetricsWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'localhost:8000'
        : window.location.host;
    const wsUrl = `${wsProtocol}//${wsHost}/ws/dashboard-metrics`;

    try {
        dashboardMetricsWebSocket = new WebSocket(wsUrl);

        dashboardMetricsWebSocket.onopen = function() {
            console.log('Dashboard指标WebSocket连接已建立');
            if (dashboardMetricsPollingInterval) {
                clearInterval(dashboardMetricsPollingInterval);
                dashboardMetricsPollingInterval = null;
            }
        };

        dashboardMetricsWebSocket.onmessage = function(event) {
            try {
                const message = JSON.parse(event.data);
                if (message.type === 'dashboard_metrics' && message.data) {
                    // 更新图表数据
                    if (typeof updateCharts === 'function') {
                        updateChartsFromMetrics(message.data);
                    }
                }
            } catch (error) {
                console.error('处理Dashboard指标WebSocket消息失败:', error);
            }
        };

        dashboardMetricsWebSocket.onerror = function(error) {
            console.error('Dashboard指标WebSocket错误:', error);
            // WebSocket连接失败时，回退到轮询模式
            if (!dashboardMetricsPollingInterval) {
                dashboardMetricsPollingInterval = setInterval(function() {
                    if (typeof updateCharts === 'function') {
                        updateCharts();
                    }
                }, 10000); // 每10秒轮询一次
            }
            // 5秒后尝试重连
            setTimeout(connectDashboardMetricsWebSocket, 5000);
        };

        dashboardMetricsWebSocket.onclose = function() {
            console.log('Dashboard指标WebSocket连接已关闭，5秒后尝试重连...');
            // 尝试重连
            if (!dashboardMetricsPollingInterval) {
                dashboardMetricsPollingInterval = setInterval(function() {
                    if (typeof updateCharts === 'function') {
                        updateCharts();
                    }
                }, 10000); // 每10秒轮询一次
            }
            setTimeout(connectDashboardMetricsWebSocket, 5000);
        };
    } catch (error) {
        console.error('Dashboard指标WebSocket连接失败:', error);
        // WebSocket连接失败时，回退到轮询模式
        if (!dashboardMetricsPollingInterval) {
            dashboardMetricsPollingInterval = setInterval(function() {
                if (typeof updateCharts === 'function') {
                    updateCharts();
                }
            }, 10000); // 每10秒轮询一次
        }
    }
}

/**
 * 连接Dashboard告警和事件WebSocket
 */
function connectDashboardAlertsWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'localhost:8000'
        : window.location.host;
    const wsUrl = `${wsProtocol}//${wsHost}/ws/dashboard-alerts`;

    try {
        dashboardAlertsWebSocket = new WebSocket(wsUrl);

        dashboardAlertsWebSocket.onopen = function() {
            console.log('Dashboard告警和事件WebSocket连接已建立');
            if (dashboardAlertsPollingInterval) {
                clearInterval(dashboardAlertsPollingInterval);
                dashboardAlertsPollingInterval = null;
            }
        };

        dashboardAlertsWebSocket.onmessage = function(event) {
            try {
                const message = JSON.parse(event.data);
                if (message.type === 'dashboard_alerts' && message.data) {
                    // 更新告警和事件
                    if (typeof loadAlerts === 'function' && message.data.alerts) {
                        updateAlertsFromData(message.data.alerts);
                    }
                    if (typeof loadRecentEvents === 'function' && message.data.events) {
                        updateEventsFromData(message.data.events);
                    }
                }
            } catch (error) {
                console.error('处理Dashboard告警和事件WebSocket消息失败:', error);
            }
        };

        dashboardAlertsWebSocket.onerror = function(error) {
            console.error('Dashboard告警和事件WebSocket错误:', error);
            // WebSocket连接失败时，回退到轮询模式
            if (!dashboardAlertsPollingInterval) {
                dashboardAlertsPollingInterval = setInterval(function() {
                    if (typeof loadAlerts === 'function') {
                        loadAlerts();
                    }
                    if (typeof loadRecentEvents === 'function') {
                        loadRecentEvents();
                    }
                }, 30000); // 每30秒轮询一次
            }
            // 5秒后尝试重连
            setTimeout(connectDashboardAlertsWebSocket, 5000);
        };

        dashboardAlertsWebSocket.onclose = function() {
            console.log('Dashboard告警和事件WebSocket连接已关闭，5秒后尝试重连...');
            // 尝试重连
            if (!dashboardAlertsPollingInterval) {
                dashboardAlertsPollingInterval = setInterval(function() {
                    if (typeof loadAlerts === 'function') {
                        loadAlerts();
                    }
                    if (typeof loadRecentEvents === 'function') {
                        loadRecentEvents();
                    }
                }, 30000); // 每30秒轮询一次
            }
            setTimeout(connectDashboardAlertsWebSocket, 5000);
        };
    } catch (error) {
        console.error('Dashboard告警和事件WebSocket连接失败:', error);
        // WebSocket连接失败时，回退到轮询模式
        if (!dashboardAlertsPollingInterval) {
            dashboardAlertsPollingInterval = setInterval(function() {
                if (typeof loadAlerts === 'function') {
                    loadAlerts();
                }
                if (typeof loadRecentEvents === 'function') {
                    loadRecentEvents();
                }
            }, 30000); // 每30秒轮询一次
        }
    }
}

/**
 * 从指标数据更新图表（辅助函数）
 */
function updateChartsFromMetrics(metricsData) {
    if (!metricsData) return;
    
    // 更新性能图表
    if (window.performanceChart && metricsData.system_metrics) {
        const systemMetrics = metricsData.system_metrics;
        const hours = Array.from({length: 6}, (_, i) => {
            const date = new Date();
            date.setHours(date.getHours() - (5 - i));
            return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
        });
        
        if (performanceChart.data.labels.length === 0) {
            performanceChart.data.labels = hours;
        }
        
        // 添加新数据点
        const systemLoad = systemMetrics.avg_response_time || 0;
        const memoryUsage = systemMetrics.avg_throughput || 0;
        
        performanceChart.data.datasets[0].data.push(systemLoad);
        performanceChart.data.datasets[1].data.push(memoryUsage);
        
        // 保持最多6个数据点
        if (performanceChart.data.datasets[0].data.length > 6) {
            performanceChart.data.datasets[0].data.shift();
            performanceChart.data.datasets[1].data.shift();
        }
        
        performanceChart.update();
    }
    
    // 更新数据流图表
    if (window.dataFlowChart && metricsData.throughput_data) {
        const throughputData = metricsData.throughput_data;
        const stages = ['数据采集', '特征工程', '模型推理', '交易执行', '风险评估'];
        const stageKeys = ['data_collection', 'feature_engineering', 'model_inference', 'trading_execution', 'risk_assessment'];
        
        const data = stageKeys.map(key => {
            const stageData = throughputData[key] || {};
            return stageData.throughput || 0;
        });
        
        dataFlowChart.data.datasets[0].data = data;
        dataFlowChart.update();
    }
}

/**
 * 从告警数据更新告警显示（辅助函数）
 */
function updateAlertsFromData(alertsData) {
    if (!alertsData || !alertsData.risk_alerts) return;
    
    const container = document.getElementById('active-alerts');
    if (!container) return;
    
    const riskAlerts = alertsData.risk_alerts || 0;
    
    if (riskAlerts === 0) {
        container.innerHTML = `
            <div class="text-center text-green-500 py-4">
                <i class="fas fa-check-circle text-2xl mb-2"></i>
                <div class="text-sm">无告警</div>
                <div class="text-xs text-gray-400 mt-1">系统运行正常</div>
            </div>
        `;
    } else {
        container.innerHTML = `
            <div class="text-center text-red-500 py-4">
                <i class="fas fa-exclamation-triangle text-2xl mb-2"></i>
                <div class="text-sm">有 ${riskAlerts} 个告警</div>
                <div class="text-xs text-gray-400 mt-1">请及时处理</div>
            </div>
        `;
    }
}

/**
 * 从事件数据更新事件显示（辅助函数）
 */
function updateEventsFromData(eventsData) {
    if (!eventsData || !Array.isArray(eventsData)) return;
    
    const container = document.getElementById('recent-events');
    if (!container) return;
    
    if (eventsData.length === 0) {
        container.innerHTML = `
            <div class="text-center text-gray-500 py-4">
                <i class="fas fa-info-circle text-2xl mb-2"></i>
                <div class="text-sm">暂无事件</div>
                <div class="text-xs text-gray-400 mt-1">系统运行正常</div>
            </div>
        `;
    } else {
        container.innerHTML = eventsData.map(event => {
            let iconColor = 'blue';
            if (event.level === 'warning') {
                iconColor = 'yellow';
            } else if (event.level === 'error') {
                iconColor = 'red';
            }
            
            const eventTime = new Date(event.timestamp);
            const timeStr = eventTime.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
            
            return `
                <div class="flex items-center justify-between p-2 border-b border-gray-100">
                    <div class="flex items-center flex-1">
                        <i class="fas fa-circle text-${iconColor}-500 text-xs mr-2"></i>
                        <div class="flex-1">
                            <span class="text-sm">${event.message || event.type}</span>
                            ${event.source ? `<span class="text-xs text-gray-400 ml-2">(${event.source})</span>` : ''}
                        </div>
                    </div>
                    <span class="text-xs text-gray-500 ml-2">${timeStr}</span>
                </div>
            `;
        }).join('');
    }
}

