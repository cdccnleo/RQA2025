// RQA2026 Web界面主要JavaScript功能

// 全局变量
let refreshInterval;
let charts = {};

// DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// 应用初始化
function initializeApp() {
    // 设置全局错误处理
    window.addEventListener('error', handleGlobalError);
    window.addEventListener('unhandledrejection', handleUnhandledRejection);

    // 初始化工具提示
    initializeTooltips();

    // 初始化模态框
    initializeModals();

    // 设置自动刷新 (如果在仪表板页面)
    if (document.getElementById('engine-status-container')) {
        startAutoRefresh();
    }

    // 页面可见性API - 当页面重新获得焦点时刷新数据
    document.addEventListener('visibilitychange', function() {
        if (!document.hidden) {
            refreshAllData();
        }
    });

    console.log('RQA2026 Web界面初始化完成');
}

// 启动自动刷新
function startAutoRefresh() {
    refreshInterval = setInterval(refreshAllData, 30000); // 每30秒刷新一次
}

// 停止自动刷新
function stopAutoRefresh() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
        refreshInterval = null;
    }
}

// 刷新所有数据
async function refreshAllData() {
    try {
        // 更新引擎状态
        await updateEngineStatus();

        // 更新性能指标
        await updatePerformanceMetrics();

        // 更新系统健康状态
        await updateSystemHealth();

        // 显示最后更新时间
        updateLastRefreshTime();

    } catch (error) {
        console.error('数据刷新失败:', error);
        showNotification('数据刷新失败，请检查网络连接', 'warning');
    }
}

// 更新引擎状态
async function updateEngineStatus() {
    try {
        const response = await fetch('/api/engine-status');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();

        // 更新状态显示
        updateEngineStatusDisplay(data);

        // 更新概览统计
        updateOverviewStats(data);

    } catch (error) {
        console.error('引擎状态更新失败:', error);
        showEngineStatusError();
    }
}

// 更新引擎状态显示
function updateEngineStatusDisplay(data) {
    const container = document.getElementById('engine-status-container');
    if (!container) return;

    const lastUpdate = document.getElementById('last-update');
    if (lastUpdate && data.last_update) {
        lastUpdate.textContent = `最后更新: ${new Date(data.last_update).toLocaleTimeString()}`;
    }

    let html = '';
    let healthyCount = 0;

    for (const [engineName, status] of Object.entries(data.engines)) {
        const isHealthy = status.status === 'healthy';
        if (isHealthy) healthyCount++;

        const statusClass = getStatusClass(status.status);
        const statusIcon = getStatusIcon(status.status);
        const statusText = getStatusText(status.status);
        const displayName = getEngineDisplayName(engineName);

        html += `
            <div class="engine-item d-flex justify-content-between align-items-center mb-3 p-3 border rounded">
                <div class="d-flex align-items-center">
                    <span class="engine-status ${statusClass} me-2"></span>
                    <div>
                        <strong>${displayName}</strong>
                        <br>
                        <small class="text-muted">${statusText}</small>
                    </div>
                </div>
                <div class="text-end">
                    <div class="fw-bold">${statusIcon}</div>
                    ${status.response_time ? `<small class="text-muted">${status.response_time}ms</small>` : ''}
                    ${status.error ? `<br><small class="text-danger">${status.error}</small>` : ''}
                </div>
            </div>
        `;
    }

    container.innerHTML = html;

    // 更新健康引擎计数
    const healthyCounter = document.getElementById('healthy-engines');
    if (healthyCounter) {
        healthyCounter.textContent = `${healthyCount}/4`;
        healthyCounter.className = healthyCount === 4 ? 'text-success' : 'text-warning';
    }
}

// 更新概览统计
function updateOverviewStats(data) {
    let totalResponseTime = 0;
    let engineCount = 0;

    for (const status of Object.values(data.engines)) {
        if (status.response_time) {
            totalResponseTime += status.response_time;
            engineCount++;
        }
    }

    const avgResponseTime = engineCount > 0 ? totalResponseTime / engineCount : 0;
    const avgElement = document.getElementById('avg-response-time');
    if (avgElement) {
        avgElement.textContent = `${avgResponseTime.toFixed(1)}ms`;
    }
}

// 更新性能指标
async function updatePerformanceMetrics() {
    try {
        const response = await fetch('/api/performance-metrics');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();

        // 更新性能图表
        updatePerformanceCharts(data);

        // 更新指标显示
        updateMetricsDisplay(data);

    } catch (error) {
        console.error('性能指标更新失败:', error);
    }
}

// 更新性能图表
function updatePerformanceCharts(data) {
    // 这里可以实现更复杂的图表更新逻辑
    // 目前简化处理，由各页面具体实现
}

// 更新指标显示
function updateMetricsDisplay(data) {
    // 更新请求总数
    const totalRequests = Object.values(data.engines)
        .reduce((sum, engine) => sum + engine.requests_per_second, 0);

    const requestsElement = document.getElementById('total-requests');
    if (requestsElement) {
        requestsElement.textContent = totalRequests.toFixed(1);
    }

    // 更新系统指标
    updateSystemMetrics(data.system);
}

// 更新系统指标
function updateSystemMetrics(systemData) {
    // CPU使用率
    const cpuElement = document.getElementById('cpu-usage');
    if (cpuElement) {
        cpuElement.textContent = `${systemData.cpu_usage.toFixed(1)}%`;
        cpuElement.className = systemData.cpu_usage > 80 ? 'text-danger' : 'text-success';
    }

    // 内存使用率
    const memoryElement = document.getElementById('memory-usage');
    if (memoryElement) {
        memoryElement.textContent = `${systemData.memory_usage.toFixed(1)}%`;
        memoryElement.className = systemData.memory_usage > 90 ? 'text-danger' : 'text-success';
    }
}

// 更新系统健康状态
async function updateSystemHealth() {
    try {
        const response = await fetch('/api/system-health');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();

        // 更新健康状态指示器
        updateHealthIndicators(data);

    } catch (error) {
        console.error('系统健康状态更新失败:', error);
    }
}

// 更新健康状态指示器
function updateHealthIndicators(data) {
    const healthIndicator = document.getElementById('system-health-indicator');
    if (healthIndicator) {
        const isHealthy = data.overall_status === 'healthy';
        healthIndicator.className = `badge ${isHealthy ? 'bg-success' : 'bg-danger'}`;
        healthIndicator.textContent = isHealthy ? '系统正常' : '系统异常';
    }
}

// 显示引擎状态错误
function showEngineStatusError() {
    const container = document.getElementById('engine-status-container');
    if (container) {
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle"></i>
                无法获取引擎状态，请检查服务连接
            </div>
        `;
    }
}

// 更新最后刷新时间
function updateLastRefreshTime() {
    const lastRefreshElement = document.getElementById('last-refresh-time');
    if (lastRefreshElement) {
        lastRefreshElement.textContent = new Date().toLocaleTimeString();
    }
}

// 获取状态样式类
function getStatusClass(status) {
    const classes = {
        'healthy': 'healthy',
        'warning': 'warning',
        'error': 'error',
        'unreachable': 'error'
    };
    return classes[status] || 'warning';
}

// 获取状态图标
function getStatusIcon(status) {
    const icons = {
        'healthy': '<i class="fas fa-check-circle text-success"></i>',
        'warning': '<i class="fas fa-exclamation-triangle text-warning"></i>',
        'error': '<i class="fas fa-times-circle text-danger"></i>',
        'unreachable': '<i class="fas fa-question-circle text-secondary"></i>'
    };
    return icons[status] || '<i class="fas fa-question-circle text-secondary"></i>';
}

// 获取状态文本
function getStatusText(status) {
    const texts = {
        'healthy': '运行正常',
        'warning': '存在警告',
        'error': '运行异常',
        'unreachable': '无法连接'
    };
    return texts[status] || '未知状态';
}

// 获取引擎显示名称
function getEngineDisplayName(engineName) {
    const names = {
        'quantum': '量子计算引擎',
        'ai': 'AI深度集成引擎',
        'bci': '脑机接口引擎',
        'fusion': '融合引擎'
    };
    return names[engineName] || engineName;
}

// 显示通知
function showNotification(message, type = 'info', duration = 5000) {
    // 创建通知元素
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    // 添加到页面
    document.body.appendChild(notification);

    // 自动移除
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, duration);
}

// 运行引擎演示
async function runEngineDemo(engineName) {
    try {
        // 显示加载状态
        showNotification(`正在运行 ${getEngineDisplayName(engineName)} 演示...`, 'info');

        const response = await fetch(`/api/engine-demo/${engineName}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();

        // 显示结果
        showDemoResult(engineName, result);

    } catch (error) {
        console.error('演示运行失败:', error);
        showNotification(`演示运行失败: ${error.message}`, 'danger');
    }
}

// 显示演示结果
function showDemoResult(engineName, result) {
    const modal = new bootstrap.Modal(document.getElementById('demoModal'));
    const title = document.getElementById('demoModalTitle');
    const content = document.getElementById('demoResult');

    title.textContent = `${getEngineDisplayName(engineName)} 演示结果`;
    content.textContent = JSON.stringify(result, null, 2);

    modal.show();
}

// 初始化工具提示
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// 初始化模态框
function initializeModals() {
    // 为所有模态框添加事件监听
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        modal.addEventListener('shown.bs.modal', function () {
            // 模态框显示时的处理
        });
        modal.addEventListener('hidden.bs.modal', function () {
            // 模态框隐藏时的处理
        });
    });
}

// 全局错误处理
function handleGlobalError(event) {
    console.error('全局错误:', event.error);
    showNotification('应用程序发生错误，请刷新页面重试', 'danger');
}

// 未处理的Promise拒绝处理
function handleUnhandledRejection(event) {
    console.error('未处理的Promise拒绝:', event.reason);
    showNotification('网络请求失败，请检查连接', 'warning');
}

// 导出主要函数供页面使用
window.RQA2026 = {
    runEngineDemo: runEngineDemo,
    refreshAllData: refreshAllData,
    showNotification: showNotification
};
