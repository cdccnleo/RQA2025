/**
 * RQA2025 统一Web管理界面 JavaScript
 * 提供实时数据更新、交互功能和用户体验优化
 */

class DashboardManager {
    constructor() {
        this.ws = null;
        this.charts = {};
        this.modules = [];
        this.alerts = [];
        this.updateInterval = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        
        this.init();
    }

    /**
     * 初始化仪表板
     */
    init() {
        this.initializeCharts();
        this.updateCurrentTime();
        this.connectWebSocket();
        this.loadModules();
        this.setupEventListeners();
        
        // 定时更新
        setInterval(() => this.updateCurrentTime(), 1000);
        setInterval(() => this.updateSystemMetrics(), 5000);
        
        // 模拟实时数据更新
        this.startMockDataUpdates();
    }

    /**
     * 设置事件监听器
     */
    setupEventListeners() {
        // 模块卡片点击事件
        document.addEventListener('click', (e) => {
            if (e.target.closest('.module-card')) {
                const moduleName = e.target.closest('.module-card').dataset.module;
                this.openModule(moduleName);
            }
        });

        // 告警关闭事件
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('alert-close')) {
                e.target.closest('.alert').remove();
            }
        });

        // 搜索功能
        const searchInput = document.getElementById('search-modules');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.filterModules(e.target.value);
            });
        }

        // 主题切换
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                this.toggleTheme();
            });
        }
    }

    /**
     * 初始化图表
     */
    initializeCharts() {
        // CPU使用率图表
        const cpuCtx = document.getElementById('cpu-chart');
        if (cpuCtx) {
            this.charts.cpu = new Chart(cpuCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU使用率',
                        data: [],
                        borderColor: 'rgb(59, 130, 246)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.1)'
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            }
                        }
                    },
                    elements: {
                        point: {
                            radius: 0
                        }
                    }
                }
            });
        }

        // 内存使用率图表
        const memoryCtx = document.getElementById('memory-chart');
        if (memoryCtx) {
            this.charts.memory = new Chart(memoryCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: '内存使用率',
                        data: [],
                        borderColor: 'rgb(245, 158, 11)',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.1)'
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            }
                        }
                    },
                    elements: {
                        point: {
                            radius: 0
                        }
                    }
                }
            });
        }

        // 系统资源饼图
        const resourceCtx = document.getElementById('resource-chart');
        if (resourceCtx) {
            this.charts.resource = new Chart(resourceCtx.getContext('2d'), {
                type: 'doughnut',
                data: {
                    labels: ['CPU', '内存', '磁盘', 'GPU'],
                    datasets: [{
                        data: [45, 62, 38, 15],
                        backgroundColor: [
                            'rgb(59, 130, 246)',
                            'rgb(245, 158, 11)',
                            'rgb(139, 92, 246)',
                            'rgb(16, 185, 129)'
                        ],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        }
    }

    /**
     * 连接WebSocket
     */
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket连接已建立');
            this.updateConnectionStatus('已连接', 'success');
            this.reconnectAttempts = 0;
            
            // 订阅实时数据
            this.ws.send(JSON.stringify({
                type: 'subscribe',
                topics: ['system_metrics', 'module_updates', 'alerts']
            }));
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            } catch (error) {
                console.error('解析WebSocket消息失败:', error);
            }
        };

        this.ws.onclose = () => {
            console.log('WebSocket连接已关闭');
            this.updateConnectionStatus('连接断开', 'error');
            
            // 自动重连
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                setTimeout(() => this.connectWebSocket(), 5000);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket错误:', error);
            this.updateConnectionStatus('连接错误', 'error');
        };
    }

    /**
     * 处理WebSocket消息
     */
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'system_metrics':
                this.updateSystemMetrics(data.data);
                break;
            case 'module_update':
                this.updateModuleStatus(data.data);
                break;
            case 'alert':
                this.addAlert(data.data);
                break;
            case 'heartbeat':
                this.updateConnectionStatus('已连接', 'success');
                break;
            default:
                console.log('未知消息类型:', data.type);
        }
    }

    /**
     * 更新连接状态
     */
    updateConnectionStatus(status, type) {
        const indicator = document.getElementById('ws-indicator');
        const statusElement = document.getElementById('ws-status');
        
        if (indicator && statusElement) {
            indicator.textContent = status;
            
            statusElement.className = 'fixed bottom-4 right-4 px-4 py-2 rounded-lg shadow-lg text-white';
            statusElement.classList.add(type === 'success' ? 'bg-green-600' : 'bg-red-600');
        }
    }

    /**
     * 更新系统指标
     */
    updateSystemMetrics(metrics) {
        // 更新显示值
        if (metrics.cpu_usage !== undefined) {
            this.updateMetricDisplay('cpu-usage', `${metrics.cpu_usage}%`);
            this.updateChart(this.charts.cpu, metrics.cpu_usage);
        }
        
        if (metrics.memory_usage !== undefined) {
            this.updateMetricDisplay('memory-usage', `${metrics.memory_usage}%`);
            this.updateChart(this.charts.memory, metrics.memory_usage);
        }
        
        if (metrics.disk_usage !== undefined) {
            this.updateMetricDisplay('disk-usage', `${metrics.disk_usage}%`);
        }

        if (metrics.gpu_usage !== undefined) {
            this.updateMetricDisplay('gpu-usage', `${metrics.gpu_usage}%`);
        }

        // 更新资源饼图
        if (this.charts.resource && metrics.cpu_usage && metrics.memory_usage && metrics.disk_usage) {
            this.charts.resource.data.datasets[0].data = [
                metrics.cpu_usage,
                metrics.memory_usage,
                metrics.disk_usage,
                metrics.gpu_usage || 0
            ];
            this.charts.resource.update();
        }
    }

    /**
     * 更新指标显示
     */
    updateMetricDisplay(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            // 添加动画效果
            element.style.transform = 'scale(1.1)';
            element.style.transition = 'transform 0.2s ease';
            
            setTimeout(() => {
                element.textContent = value;
                element.style.transform = 'scale(1)';
            }, 100);
        }
    }

    /**
     * 更新图表
     */
    updateChart(chart, value) {
        if (!chart) return;

        const now = new Date().toLocaleTimeString('zh-CN');
        
        chart.data.labels.push(now);
        chart.data.datasets[0].data.push(value);
        
        // 保持最近20个数据点
        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        
        chart.update('none'); // 使用 'none' 模式提高性能
    }

    /**
     * 加载模块
     */
    async loadModules() {
        try {
            const response = await fetch('/api/modules');
            const data = await response.json();
            this.modules = data.modules || [];
            this.renderModules();
        } catch (error) {
            console.error('加载模块失败:', error);
            this.showNotification('加载模块失败', 'error');
        }
    }

    /**
     * 渲染模块
     */
    renderModules() {
        const container = document.getElementById('modules-container');
        if (!container) return;

        container.innerHTML = '';

        this.modules.forEach(module => {
            const moduleCard = this.createModuleCard(module);
            container.appendChild(moduleCard);
        });
    }

    /**
     * 创建模块卡片
     */
    createModuleCard(module) {
        const card = document.createElement('div');
        card.className = 'module-card rounded-lg p-6 card-hover cursor-pointer';
        card.dataset.module = module.name;

        const statusClass = this.getStatusClass(module.status);
        const iconClass = this.getModuleIcon(module.name);

        card.innerHTML = `
            <div class="flex items-center justify-between mb-4">
                <div class="flex items-center space-x-3">
                    <i class="${iconClass} text-2xl text-blue-500"></i>
                    <div>
                        <h3 class="text-lg font-semibold">${module.display_name}</h3>
                        <p class="text-sm text-gray-600">${module.description}</p>
                    </div>
                </div>
                <div class="w-3 h-3 rounded-full ${statusClass}"></div>
            </div>
            <div class="text-sm text-gray-500">
                <p>路由: ${module.route}</p>
                <p>权限: ${module.permissions.join(', ')}</p>
            </div>
        `;

        return card;
    }

    /**
     * 获取状态样式类
     */
    getStatusClass(status) {
        switch (status) {
            case 'online':
                return 'status-online';
            case 'warning':
                return 'status-warning';
            case 'error':
                return 'status-error';
            default:
                return 'status-warning';
        }
    }

    /**
     * 获取模块图标
     */
    getModuleIcon(moduleName) {
        const iconMap = {
            'config': 'fas fa-cog',
            'fpga_monitoring': 'fas fa-microchip',
            'resource_monitoring': 'fas fa-server',
            'features_monitoring': 'fas fa-chart-line',
            'strategy_management': 'fas fa-chess',
            'data_management': 'fas fa-database',
            'backtest_management': 'fas fa-chart-bar',
            'alert_management': 'fas fa-bell'
        };
        return iconMap[moduleName] || 'fas fa-cube';
    }

    /**
     * 打开模块
     */
    openModule(moduleName) {
        const module = this.modules.find(m => m.name === moduleName);
        if (module) {
            // 在新窗口打开模块
            window.open(`/api/modules/${moduleName}`, '_blank');
            
            // 记录访问日志
            this.logModuleAccess(moduleName);
        }
    }

    /**
     * 更新模块状态
     */
    updateModuleStatus(moduleData) {
        const module = this.modules.find(m => m.name === moduleData.name);
        if (module) {
            module.status = moduleData.status;
            this.renderModules();
            
            // 显示状态变化通知
            this.showNotification(`${module.display_name} 状态已更新`, 'info');
        }
    }

    /**
     * 添加告警
     */
    addAlert(alert) {
        this.alerts.unshift(alert);
        
        // 保持最多10个告警
        if (this.alerts.length > 10) {
            this.alerts.pop();
        }

        this.renderAlerts();
        this.showNotification(alert.title, alert.level);
    }

    /**
     * 渲染告警
     */
    renderAlerts() {
        const container = document.getElementById('alerts-container');
        if (!container) return;

        container.innerHTML = '';

        this.alerts.forEach(alert => {
            const alertElement = document.createElement('div');
            const alertClass = this.getAlertClass(alert.level);

            alertElement.className = `alert ${alertClass} border-l-4 p-4 mb-4`;
            alertElement.innerHTML = `
                <div class="flex items-center justify-between">
                    <div>
                        <p class="font-semibold">${alert.title}</p>
                        <p class="text-sm">${alert.message}</p>
                    </div>
                    <div class="flex items-center space-x-2">
                        <span class="text-xs">${new Date(alert.timestamp).toLocaleString('zh-CN')}</span>
                        <button class="alert-close text-gray-400 hover:text-gray-600">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                </div>
            `;

            container.appendChild(alertElement);
        });
    }

    /**
     * 获取告警样式类
     */
    getAlertClass(level) {
        switch (level) {
            case 'error':
                return 'alert-error';
            case 'warning':
                return 'alert-warning';
            case 'success':
                return 'alert-success';
            default:
                return 'alert-info';
        }
    }

    /**
     * 显示通知
     */
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 px-4 py-2 rounded-lg shadow-lg text-white z-50 transform transition-all duration-300 translate-x-full`;
        
        const bgClass = type === 'error' ? 'bg-red-600' : 
                       type === 'warning' ? 'bg-yellow-600' : 
                       type === 'success' ? 'bg-green-600' : 'bg-blue-600';
        
        notification.classList.add(bgClass);
        notification.textContent = message;

        document.body.appendChild(notification);

        // 显示动画
        setTimeout(() => {
            notification.classList.remove('translate-x-full');
        }, 100);

        // 自动隐藏
        setTimeout(() => {
            notification.classList.add('translate-x-full');
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    /**
     * 更新当前时间
     */
    updateCurrentTime() {
        const now = new Date();
        const timeElement = document.getElementById('current-time');
        if (timeElement) {
            timeElement.textContent = now.toLocaleString('zh-CN');
        }
    }

    /**
     * 过滤模块
     */
    filterModules(searchTerm) {
        const filteredModules = this.modules.filter(module => 
            module.display_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            module.description.toLowerCase().includes(searchTerm.toLowerCase())
        );
        
        this.renderFilteredModules(filteredModules);
    }

    /**
     * 渲染过滤后的模块
     */
    renderFilteredModules(modules) {
        const container = document.getElementById('modules-container');
        if (!container) return;

        container.innerHTML = '';

        if (modules.length === 0) {
            container.innerHTML = `
                <div class="col-span-full text-center py-8 text-gray-500">
                    <i class="fas fa-search text-4xl mb-4"></i>
                    <p>未找到匹配的模块</p>
                </div>
            `;
            return;
        }

        modules.forEach(module => {
            const moduleCard = this.createModuleCard(module);
            container.appendChild(moduleCard);
        });
    }

    /**
     * 切换主题
     */
    toggleTheme() {
        const body = document.body;
        const isDark = body.classList.contains('dark');
        
        if (isDark) {
            body.classList.remove('dark');
            localStorage.setItem('theme', 'light');
        } else {
            body.classList.add('dark');
            localStorage.setItem('theme', 'dark');
        }
    }

    /**
     * 记录模块访问
     */
    logModuleAccess(moduleName) {
        // 这里可以发送访问日志到服务器
        console.log(`用户访问模块: ${moduleName}`);
    }

    /**
     * 开始模拟数据更新
     */
    startMockDataUpdates() {
        // 模拟实时数据更新
        setInterval(() => {
            const mockData = {
                cpu_usage: Math.floor(Math.random() * 30) + 40,
                memory_usage: Math.floor(Math.random() * 20) + 50,
                disk_usage: Math.floor(Math.random() * 15) + 30,
                gpu_usage: Math.floor(Math.random() * 10) + 5
            };
            
            this.updateSystemMetrics(mockData);
        }, 5000);
    }

    /**
     * 更新系统指标（模拟）
     */
    updateSystemMetrics() {
        // 这个方法会被定时器调用，用于模拟数据更新
        // 实际环境中应该通过WebSocket获取真实数据
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    window.dashboardManager = new DashboardManager();
});

// 导出类供其他模块使用
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DashboardManager;
} 