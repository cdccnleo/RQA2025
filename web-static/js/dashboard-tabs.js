/**
 * 仪表板标签页管理器
 * 
 * 提供标签页切换、内容缓存、按需加载功能
 * 
 * 作者: Claude
 * 创建日期: 2026-02-21
 */

class DashboardTabs {
    constructor() {
        // 标签页配置
        this.tabs = [
            { id: 'overview', name: '系统总览', icon: 'fa-tachometer-alt' },
            { id: 'business', name: '业务驱动', icon: 'fa-project-diagram' },
            { id: 'strategy', name: '策略监控', icon: 'fa-chess' },
            { id: 'trading', name: '交易监控', icon: 'fa-exchange-alt' },
            { id: 'data', name: '数据监控', icon: 'fa-database' },
            { id: 'alert', name: '告警中心', icon: 'fa-bell' },
            { id: 'architecture', name: '架构监控', icon: 'fa-sitemap' }
        ];
        
        this.activeTab = 'overview';
        this.loadedTabs = new Set(['overview']); // 默认加载总览
        this.cache = new Map();
        this.tabContents = new Map();
        
        // 从本地存储恢复上次活动的标签页
        this.restoreLastTab();
        
        // 初始化
        this.init();
    }
    
    /**
     * 初始化标签页
     */
    init() {
        this.renderTabs();
        this.renderTabContents();
        this.bindEvents();
        this.switchTab(this.activeTab, false);
    }
    
    /**
     * 渲染标签页导航
     */
    renderTabs() {
        const container = document.getElementById('dashboard-tabs');
        if (!container) return;
        
        const tabsHtml = this.tabs.map(tab => `
            <button 
                class="dashboard-tab ${tab.id === this.activeTab ? 'active' : ''}" 
                data-tab="${tab.id}"
                title="${tab.name}"
            >
                <i class="fas ${tab.icon}"></i>
                <span class="tab-name">${tab.name}</span>
            </button>
        `).join('');
        
        container.innerHTML = tabsHtml;
    }
    
    /**
     * 渲染标签页内容容器
     */
    renderTabContents() {
        const container = document.getElementById('dashboard-tab-contents');
        if (!container) return;
        
        const contentsHtml = this.tabs.map(tab => `
            <div 
                id="tab-content-${tab.id}" 
                class="tab-content ${tab.id === this.activeTab ? 'active' : 'hidden'}"
                data-tab="${tab.id}"
            >
                ${tab.id === 'overview' ? this.getOverviewContent() : 
                  `<div class="tab-loading">
                    <i class="fas fa-spinner fa-spin"></i>
                    <span>正在加载${tab.name}...</span>
                  </div>`}
            </div>
        `).join('');
        
        container.innerHTML = contentsHtml;
    }
    
    /**
     * 获取系统总览内容（精简版 - 只显示关键指标和概要）
     */
    getOverviewContent() {
        return `
            <div class="mb-8">
                <h1 class="text-3xl font-bold text-gray-900 mb-2">系统总览</h1>
                <p class="text-gray-600">基于21层级架构的量化交易系统实时监控</p>
            </div>

            <!-- 关键指标卡片 -->
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 mb-8">
                <div class="bg-white rounded-lg shadow p-6 card-hover">
                    <div class="flex items-center">
                        <div class="flex-shrink-0">
                            <div class="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                                <i class="fas fa-chart-line text-blue-600"></i>
                            </div>
                        </div>
                        <div class="ml-4">
                            <dt class="text-sm font-medium text-gray-500 truncate">活跃策略</dt>
                            <dd class="text-2xl font-semibold text-gray-900" id="active-strategies">加载中...</dd>
                        </div>
                    </div>
                </div>

                <div class="bg-white rounded-lg shadow p-6 card-hover">
                    <div class="flex items-center">
                        <div class="flex-shrink-0">
                            <div class="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                                <i class="fas fa-hand-holding-usd text-green-600"></i>
                            </div>
                        </div>
                        <div class="ml-4">
                            <dt class="text-sm font-medium text-gray-500 truncate">今日收益</dt>
                            <dd class="text-2xl font-semibold text-gray-900" id="daily-pnl">加载中...</dd>
                        </div>
                    </div>
                </div>

                <div class="bg-white rounded-lg shadow p-6 card-hover">
                    <div class="flex items-center">
                        <div class="flex-shrink-0">
                            <div class="w-10 h-10 bg-yellow-100 rounded-lg flex items-center justify-center">
                                <i class="fas fa-clock text-yellow-600"></i>
                            </div>
                        </div>
                        <div class="ml-4">
                            <dt class="text-sm font-medium text-gray-500 truncate">数据延迟</dt>
                            <dd class="text-2xl font-semibold text-gray-900" id="data-latency">加载中...</dd>
                        </div>
                    </div>
                </div>

                <div class="bg-white rounded-lg shadow p-6 card-hover">
                    <div class="flex items-center">
                        <div class="flex-shrink-0">
                            <div class="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center">
                                <i class="fas fa-shield-alt text-red-600"></i>
                            </div>
                        </div>
                        <div class="ml-4">
                            <dt class="text-sm font-medium text-gray-500 truncate">风险等级</dt>
                            <dd class="text-2xl font-semibold text-gray-900" id="risk-level">加载中...</dd>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 系统健康摘要 -->
            <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
                <h3 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                    <i class="fas fa-heartbeat text-green-600 mr-2"></i>
                    系统健康状态
                </h3>
                <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div class="text-center p-4 bg-green-50 rounded-lg">
                        <div class="text-3xl font-bold text-green-600" id="system-health-score">98%</div>
                        <div class="text-sm text-gray-600 mt-1">系统健康度</div>
                    </div>
                    <div class="text-center p-4 bg-blue-50 rounded-lg">
                        <div class="text-3xl font-bold text-blue-600" id="active-layers">21/21</div>
                        <div class="text-sm text-gray-600 mt-1">正常层级</div>
                    </div>
                    <div class="text-center p-4 bg-purple-50 rounded-lg">
                        <div class="text-3xl font-bold text-purple-600" id="uptime">99.9%</div>
                        <div class="text-sm text-gray-600 mt-1">可用性</div>
                    </div>
                </div>
            </div>

            <!-- 最近告警 -->
            <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
                <h3 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                    <i class="fas fa-bell text-yellow-600 mr-2"></i>
                    最近告警
                </h3>
                <div id="recent-alerts-list" class="space-y-2">
                    <div class="text-gray-500 text-center py-4">加载中...</div>
                </div>
            </div>

            <!-- 快速导航 -->
            <div class="bg-white rounded-lg shadow-lg p-6">
                <h3 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                    <i class="fas fa-compass text-indigo-600 mr-2"></i>
                    快速导航
                </h3>
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <a href="/strategy-management" target="_blank" rel="noopener noreferrer" 
                       class="flex items-center p-4 bg-purple-50 rounded-lg hover:bg-purple-100 transition-colors">
                        <i class="fas fa-chess text-purple-600 mr-3"></i>
                        <span class="font-medium">策略管理</span>
                    </a>
                    <a href="/trading-execution" target="_blank" rel="noopener noreferrer"
                       class="flex items-center p-4 bg-green-50 rounded-lg hover:bg-green-100 transition-colors">
                        <i class="fas fa-exchange-alt text-green-600 mr-3"></i>
                        <span class="font-medium">交易执行</span>
                    </a>
                    <a href="/data-sources-config" target="_blank" rel="noopener noreferrer"
                       class="flex items-center p-4 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors">
                        <i class="fas fa-database text-blue-600 mr-3"></i>
                        <span class="font-medium">数据源配置</span>
                    </a>
                    <a href="/risk-reporting" target="_blank" rel="noopener noreferrer"
                       class="flex items-center p-4 bg-red-50 rounded-lg hover:bg-red-100 transition-colors">
                        <i class="fas fa-shield-alt text-red-600 mr-3"></i>
                        <span class="font-medium">风险监控</span>
                    </a>
                </div>
            </div>
        `;
    }
    
    /**
     * 绑定事件
     */
    bindEvents() {
        const tabsContainer = document.getElementById('dashboard-tabs');
        if (!tabsContainer) return;
        
        tabsContainer.addEventListener('click', (e) => {
            const tabBtn = e.target.closest('.dashboard-tab');
            if (tabBtn) {
                const tabId = tabBtn.dataset.tab;
                this.switchTab(tabId);
            }
        });
        
        // 键盘导航支持
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                const currentIndex = this.tabs.findIndex(t => t.id === this.activeTab);
                let newIndex = currentIndex;
                
                if (e.key === 'ArrowLeft') {
                    newIndex = currentIndex > 0 ? currentIndex - 1 : this.tabs.length - 1;
                } else if (e.key === 'ArrowRight') {
                    newIndex = currentIndex < this.tabs.length - 1 ? currentIndex + 1 : 0;
                }
                
                if (newIndex !== currentIndex) {
                    e.preventDefault();
                    this.switchTab(this.tabs[newIndex].id);
                }
            }
        });
    }
    
    /**
     * 切换标签页
     * @param {string} tabId - 标签页ID
     * @param {boolean} saveState - 是否保存状态
     */
    async switchTab(tabId, saveState = true) {
        if (tabId === this.activeTab) return;
        
        // 更新UI
        this.updateTabUI(tabId);
        
        // 隐藏当前内容
        const currentContent = document.getElementById(`tab-content-${this.activeTab}`);
        if (currentContent) {
            currentContent.classList.add('hidden');
            currentContent.classList.remove('active');
        }
        
        // 显示新内容
        const newContent = document.getElementById(`tab-content-${tabId}`);
        if (newContent) {
            newContent.classList.remove('hidden');
            newContent.classList.add('active');
        }
        
        // 检查是否需要加载内容
        if (!this.loadedTabs.has(tabId)) {
            await this.loadTabContent(tabId);
            this.loadedTabs.add(tabId);
        }
        
        this.activeTab = tabId;
        
        // 保存状态
        if (saveState) {
            this.saveTabState();
        }
        
        // 触发自定义事件
        window.dispatchEvent(new CustomEvent('tabChanged', { 
            detail: { tabId, previousTab: this.activeTab } 
        }));
    }
    
    /**
     * 更新标签页UI
     */
    updateTabUI(activeTabId) {
        const tabs = document.querySelectorAll('.dashboard-tab');
        tabs.forEach(tab => {
            if (tab.dataset.tab === activeTabId) {
                tab.classList.add('active');
            } else {
                tab.classList.remove('active');
            }
        });
    }
    
    /**
     * 加载标签页内容（简化版 - 同步加载）
     * @param {string} tabId - 标签页ID
     */
    loadTabContent(tabId) {
        console.log(`[DashboardTabs] 开始加载标签页: ${tabId}`);
        
        const contentContainer = document.getElementById(`tab-content-${tabId}`);
        if (!contentContainer) {
            console.error(`[DashboardTabs] 错误: 容器未找到 tab-content-${tabId}`);
            return;
        }
        
        // 显示加载状态
        const tab = this.tabs.find(t => t.id === tabId);
        const tabName = tab ? tab.name : tabId;
        contentContainer.innerHTML = `
            <div class="tab-loading">
                <i class="fas fa-spinner fa-spin"></i>
                <span>正在加载${tabName}...</span>
            </div>
        `;
        console.log(`[DashboardTabs] 加载状态已显示`);
        
        try {
            // 检查缓存
            if (this.cache.has(tabId)) {
                console.log(`[DashboardTabs] 从缓存加载内容: ${tabId}`);
                const cachedContent = this.cache.get(tabId);
                contentContainer.innerHTML = cachedContent;
                this.initTabComponents(tabId);
                console.log(`[DashboardTabs] 缓存内容加载完成: ${tabId}`);
                return;
            }
            
            // 直接生成内容（同步，不使用异步模拟）
            console.log(`[DashboardTabs] 生成内容: ${tabId}`);
            const content = this.generateTabContent(tabId);
            console.log(`[DashboardTabs] 内容生成完成，长度: ${content.length}`);
            
            // 缓存内容
            this.cache.set(tabId, content);
            
            // 替换内容
            contentContainer.innerHTML = content;
            console.log(`[DashboardTabs] DOM更新完成`);
            
            // 确保内容可见
            contentContainer.classList.remove('hidden');
            contentContainer.classList.add('active');
            console.log(`[DashboardTabs] 样式类更新完成: hidden removed, active added`);
            
            // 初始化该标签页的组件
            this.initTabComponents(tabId);
            
            console.log(`[DashboardTabs] 标签页 ${tabId} 内容加载完成`);
            
        } catch (error) {
            console.error(`[DashboardTabs] 加载标签页 ${tabId} 失败:`, error);
            contentContainer.innerHTML = `
                <div class="tab-error">
                    <i class="fas fa-exclamation-triangle"></i>
                    <span>加载失败: ${error.message}</span>
                    <button onclick="dashboardTabs.loadTabContent('${tabId}')" style="margin-top: 10px; padding: 8px 16px; background: #4f46e5; color: white; border: none; border-radius: 4px; cursor: pointer;">重试</button>
                </div>
            `;
        }
    }
    
    /**
     * 模拟加载延迟
     */
    simulateLoad(tabId) {
        return new Promise(resolve => {
            // 模拟网络延迟
            setTimeout(resolve, 300 + Math.random() * 200);
        });
    }
    
    /**
     * 生成标签页内容
     * @param {string} tabId - 标签页ID
     */
    generateTabContent(tabId) {
        const contentGenerators = {
            'business': () => this.generateBusinessContent(),
            'strategy': () => this.generateStrategyContent(),
            'trading': () => this.generateTradingContent(),
            'data': () => this.generateDataContent(),
            'alert': () => this.generateAlertContent(),
            'architecture': () => this.generateArchitectureContent()
        };
        
        const generator = contentGenerators[tabId];
        return generator ? generator() : '<div class="tab-empty">暂无内容</div>';
    }
    
    /**
     * 生成业务驱动内容（完整业务流程监控）
     */
    generateBusinessContent() {
        return `
            <div class="mb-8">
                <h1 class="text-3xl font-bold text-gray-900 mb-2">业务驱动监控</h1>
                <p class="text-gray-600">量化策略完整业务流程实时监控</p>
            </div>

            <!-- 业务流程总览 -->
            <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
                <h3 class="text-xl font-semibold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-project-diagram text-indigo-600 mr-2"></i>
                    量化策略完整业务流程
                </h3>
                
                <!-- 流程步骤可视化 -->
                <div class="business-flow">
                    <!-- 第一阶段：策略研发 -->
                    <div class="flow-phase mb-6">
                        <h4 class="text-lg font-medium text-gray-700 mb-3 flex items-center">
                            <span class="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center mr-2 text-sm">1</span>
                            策略研发
                        </h4>
                        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
                            ${this.generateDashboardCard('/strategy-conception', '策略构思', 'fa-lightbulb', 'purple')}
                            ${this.generateDashboardCard('/strategy-management', '策略管理', 'fa-cogs', 'indigo')}
                            ${this.generateDashboardCard('/feature-engineering-monitor', '特征工程', 'fa-cogs', 'green')}
                            ${this.generateDashboardCard('/model-training-monitor', '模型训练', 'fa-robot', 'orange')}
                        </div>
                    </div>
                    
                    <!-- 第二阶段：策略验证 -->
                    <div class="flow-phase mb-6">
                        <h4 class="text-lg font-medium text-gray-700 mb-3 flex items-center">
                            <span class="w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center mr-2 text-sm">2</span>
                            策略验证
                        </h4>
                        <div class="grid grid-cols-2 sm:grid-cols-3 gap-3">
                            ${this.generateDashboardCard('/strategy-backtest', '策略回测', 'fa-chart-area', 'red')}
                            ${this.generateDashboardCard('/strategy-optimizer', '策略优化', 'fa-sliders-h', 'purple')}
                            ${this.generateDashboardCard('/strategy-performance-evaluation', '性能评估', 'fa-chart-bar', 'indigo')}
                        </div>
                    </div>
                    
                    <!-- 第三阶段：策略部署 -->
                    <div class="flow-phase mb-6">
                        <h4 class="text-lg font-medium text-gray-700 mb-3 flex items-center">
                            <span class="w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center mr-2 text-sm">3</span>
                            策略部署
                        </h4>
                        <div class="grid grid-cols-2 gap-3">
                            ${this.generateDashboardCard('/strategy-lifecycle', '策略部署', 'fa-rocket', 'teal')}
                            ${this.generateDashboardCard('/strategy-execution-monitor', '执行监控', 'fa-tachometer-alt', 'pink')}
                        </div>
                    </div>
                    
                    <!-- 第四阶段：交易执行 -->
                    <div class="flow-phase mb-6">
                        <h4 class="text-lg font-medium text-gray-700 mb-3 flex items-center">
                            <span class="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center mr-2 text-sm">4</span>
                            交易执行
                        </h4>
                        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
                            ${this.generateDashboardCard('/trading-execution', '市场监控', 'fa-eye', 'green')}
                            ${this.generateDashboardCard('/trading-execution', '信号生成', 'fa-signal', 'blue')}
                            ${this.generateDashboardCard('/order-routing-monitor', '智能路由', 'fa-route', 'purple')}
                            ${this.generateDashboardCard('/trading-execution', '成交执行', 'fa-check-circle', 'indigo')}
                        </div>
                    </div>
                    
                    <!-- 第五阶段：风险控制 -->
                    <div class="flow-phase mb-6">
                        <h4 class="text-lg font-medium text-gray-700 mb-3 flex items-center">
                            <span class="w-8 h-8 bg-red-500 text-white rounded-full flex items-center justify-center mr-2 text-sm">5</span>
                            风险控制
                        </h4>
                        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
                            ${this.generateDashboardCard('/risk-reporting', '实时监测', 'fa-search', 'red')}
                            ${this.generateDashboardCard('/risk-reporting', '风险评估', 'fa-calculator', 'orange')}
                            ${this.generateDashboardCard('/risk-reporting', '风险拦截', 'fa-hand-paper', 'yellow')}
                            ${this.generateDashboardCard('/risk-reporting', '预警通知', 'fa-bell', 'purple')}
                        </div>
                    </div>
                    
                    <!-- 第六阶段：持仓管理 -->
                    <div class="flow-phase">
                        <h4 class="text-lg font-medium text-gray-700 mb-3 flex items-center">
                            <span class="w-8 h-8 bg-teal-500 text-white rounded-full flex items-center justify-center mr-2 text-sm">6</span>
                            持仓管理
                        </h4>
                        <div class="grid grid-cols-2 sm:grid-cols-3 gap-3">
                            ${this.generateDashboardCard('/trading-execution', '持仓监控', 'fa-balance-scale', 'teal')}
                            ${this.generateDashboardCard('/trading-execution', '结果反馈', 'fa-bell', 'blue')}
                            ${this.generateDashboardCard('/risk-reporting', '报告生成', 'fa-file-alt', 'indigo')}
                        </div>
                    </div>
                </div>
            </div>

            <!-- 业务流程状态总览 -->
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div class="bg-blue-50 rounded-lg p-4 text-center">
                    <div class="text-2xl font-bold text-blue-600" id="business-strategy-count">0</div>
                    <div class="text-sm text-gray-600">策略研发中</div>
                </div>
                <div class="bg-green-50 rounded-lg p-4 text-center">
                    <div class="text-2xl font-bold text-green-600" id="business-validation-count">0</div>
                    <div class="text-sm text-gray-600">策略验证中</div>
                </div>
                <div class="bg-purple-50 rounded-lg p-4 text-center">
                    <div class="text-2xl font-bold text-purple-600" id="business-trading-count">0</div>
                    <div class="text-sm text-gray-600">交易执行中</div>
                </div>
                <div class="bg-red-50 rounded-lg p-4 text-center">
                    <div class="text-2xl font-bold text-red-600" id="business-risk-count">0</div>
                    <div class="text-sm text-gray-600">风险预警</div>
                </div>
            </div>
        `;
    }
    
    /**
     * 生成策略监控内容（精简版 - 移除业务流程相关内容）
     */
    generateStrategyContent() {
        return `
            <div class="mb-8">
                <h1 class="text-3xl font-bold text-gray-900 mb-2">策略监控</h1>
                <p class="text-gray-600">策略性能指标和实时状态监控</p>
            </div>

            <!-- 策略概览 -->
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
                <div class="bg-purple-50 rounded-lg p-4 text-center">
                    <div class="text-2xl font-bold text-purple-600" id="strategy-active-count">0</div>
                    <div class="text-sm text-gray-600">活跃策略</div>
                </div>
                <div class="bg-blue-50 rounded-lg p-4 text-center">
                    <div class="text-2xl font-bold text-blue-600" id="strategy-total-pnl">0%</div>
                    <div class="text-sm text-gray-600">总收益率</div>
                </div>
                <div class="bg-green-50 rounded-lg p-4 text-center">
                    <div class="text-2xl font-bold text-green-600" id="strategy-sharpe">0</div>
                    <div class="text-sm text-gray-600">夏普比率</div>
                </div>
                <div class="bg-orange-50 rounded-lg p-4 text-center">
                    <div class="text-2xl font-bold text-orange-600" id="strategy-max-dd">0%</div>
                    <div class="text-sm text-gray-600">最大回撤</div>
                </div>
            </div>

            <!-- 策略工具 -->
            <div class="tab-section">
                <h3><i class="fas fa-tools text-purple-600 mr-2"></i>策略工具</h3>
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    ${this.generateDashboardCard('/strategy-execution-monitor', '执行监控', 'fa-tachometer-alt', 'green')}
                    ${this.generateDashboardCard('/strategy-realtime-monitor', '实时监控', 'fa-stream', 'blue')}
                    ${this.generateDashboardCard('/strategy-optimizer', '参数优化', 'fa-sliders-h', 'purple')}
                    ${this.generateDashboardCard('/strategy-ai-optimizer', 'AI优化', 'fa-brain', 'indigo')}
                </div>
            </div>

            <!-- 策略列表 -->
            <div class="tab-section mt-6">
                <h3><i class="fas fa-list text-blue-600 mr-2"></i>策略列表</h3>
                <div id="strategy-list" class="space-y-2">
                    <div class="text-gray-500 text-center py-4">加载中...</div>
                </div>
            </div>
        `;
    }
    
    /**
     * 生成交易监控内容（精简版 - 移除业务流程相关内容）
     */
    generateTradingContent() {
        return `
            <div class="mb-8">
                <h1 class="text-3xl font-bold text-gray-900 mb-2">交易监控</h1>
                <p class="text-gray-600">实时交易数据和订单状态监控</p>
            </div>

            <!-- 交易概览 -->
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
                <div class="bg-green-50 rounded-lg p-4 text-center">
                    <div class="text-2xl font-bold text-green-600" id="trading-today-volume">0</div>
                    <div class="text-sm text-gray-600">今日成交量</div>
                </div>
                <div class="bg-blue-50 rounded-lg p-4 text-center">
                    <div class="text-2xl font-bold text-blue-600" id="trading-today-amount">0</div>
                    <div class="text-sm text-gray-600">今日成交额</div>
                </div>
                <div class="bg-purple-50 rounded-lg p-4 text-center">
                    <div class="text-2xl font-bold text-purple-600" id="trading-active-orders">0</div>
                    <div class="text-sm text-gray-600">活跃订单</div>
                </div>
                <div class="bg-orange-50 rounded-lg p-4 text-center">
                    <div class="text-2xl font-bold text-orange-600" id="trading-fill-rate">0%</div>
                    <div class="text-sm text-gray-600">成交率</div>
                </div>
            </div>

            <!-- 交易工具 -->
            <div class="tab-section">
                <h3><i class="fas fa-tools text-green-600 mr-2"></i>交易工具</h3>
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    ${this.generateDashboardCard('/trading-execution', '交易执行', 'fa-exchange-alt', 'green')}
                    ${this.generateDashboardCard('/order-routing-monitor', '订单路由', 'fa-route', 'purple')}
                    ${this.generateDashboardCard('/risk-reporting', '风险报告', 'fa-shield-alt', 'red')}
                </div>
            </div>

            <!-- 实时交易数据 -->
            <div class="tab-section mt-6">
                <h3><i class="fas fa-chart-line text-blue-600 mr-2"></i>实时交易数据</h3>
                <div id="trading-realtime-data" class="space-y-2">
                    <div class="text-gray-500 text-center py-4">加载中...</div>
                </div>
            </div>
        `;
    }
    
    /**
     * 生成数据监控内容
     */
    generateDataContent() {
        return `
            <div class="tab-section">
                <h3><i class="fas fa-database text-blue-600 mr-2"></i>数据管理层监控</h3>
                <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
                    ${this.generateDashboardCard('/data-quality-monitor', '数据质量', 'fa-check-circle', 'green')}
                    ${this.generateDashboardCard('/cache-monitor', '缓存监控', 'fa-memory', 'blue')}
                    ${this.generateDashboardCard('/data-lake-manager', '数据湖', 'fa-water', 'purple')}
                    ${this.generateDashboardCard('/data-performance-monitor', '性能监控', 'fa-tachometer-alt', 'yellow')}
                    ${this.generateDashboardCard('/data-sources-config', '数据源配置', 'fa-plug', 'indigo')}
                    ${this.generateDashboardCard('/data-collection-monitor.html', '数据采集监控', 'fa-download', 'teal')}
                </div>
            </div>
            <div class="tab-section mt-6">
                <h3><i class="fas fa-stream text-indigo-600 mr-2"></i>数据流监控</h3>
                <div class="data-flow-visualization" id="data-flow-viz">
                    <!-- 数据流可视化 -->
                </div>
            </div>
        `;
    }
    
    /**
     * 生成告警中心内容
     */
    generateAlertContent() {
        return `
            <div class="tab-section">
                <h3><i class="fas fa-bell text-yellow-600 mr-2"></i>告警和事件监控</h3>
                <div class="alert-dashboard" id="alert-dashboard">
                    <div class="alert-summary">
                        <div class="alert-stat critical">
                            <span class="count">0</span>
                            <span class="label">严重告警</span>
                        </div>
                        <div class="alert-stat warning">
                            <span class="count">0</span>
                            <span class="label">警告</span>
                        </div>
                        <div class="alert-stat info">
                            <span class="count">0</span>
                            <span class="label">信息</span>
                        </div>
                    </div>
                    <div class="alert-list" id="alert-list">
                        <!-- 告警列表 -->
                    </div>
                </div>
            </div>
        `;
    }
    
    /**
     * 生成架构监控内容（精简版 - 移除业务流程相关内容）
     */
    generateArchitectureContent() {
        return `
            <div class="mb-8">
                <h1 class="text-3xl font-bold text-gray-900 mb-2">架构监控</h1>
                <p class="text-gray-600">21层级架构状态和系统组件健康度</p>
            </div>

            <!-- 架构概览 -->
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
                <div class="bg-indigo-50 rounded-lg p-4 text-center">
                    <div class="text-2xl font-bold text-indigo-600" id="arch-total-layers">21</div>
                    <div class="text-sm text-gray-600">总层级数</div>
                </div>
                <div class="bg-green-50 rounded-lg p-4 text-center">
                    <div class="text-2xl font-bold text-green-600" id="arch-healthy-layers">21</div>
                    <div class="text-sm text-gray-600">健康层级</div>
                </div>
                <div class="bg-yellow-50 rounded-lg p-4 text-center">
                    <div class="text-2xl font-bold text-yellow-600" id="arch-warning-layers">0</div>
                    <div class="text-sm text-gray-600">警告层级</div>
                </div>
                <div class="bg-red-50 rounded-lg p-4 text-center">
                    <div class="text-2xl font-bold text-red-600" id="arch-error-layers">0</div>
                    <div class="text-sm text-gray-600">异常层级</div>
                </div>
            </div>

            <!-- 21层级架构状态 -->
            <div class="tab-section">
                <h3><i class="fas fa-layer-group text-indigo-600 mr-2"></i>21层级架构状态</h3>
                <div class="architecture-layers" id="architecture-layers">
                    <div class="grid grid-cols-3 sm:grid-cols-5 lg:grid-cols-7 gap-2" id="layer-status-grid">
                        <!-- 21层级状态由JavaScript动态生成 -->
                    </div>
                </div>
            </div>

            <!-- 系统组件健康度 -->
            <div class="tab-section mt-6">
                <h3><i class="fas fa-heartbeat text-green-600 mr-2"></i>系统组件健康度</h3>
                <div id="component-health" class="space-y-2">
                    <div class="text-gray-500 text-center py-4">加载中...</div>
                </div>
            </div>
        `;
    }
    
    /**
     * 生成仪表盘卡片HTML
     */
    generateDashboardCard(href, title, icon, color) {
        const colorClasses = {
            'green': 'bg-green-100 text-green-600',
            'blue': 'bg-blue-100 text-blue-600',
            'purple': 'bg-purple-100 text-purple-600',
            'indigo': 'bg-indigo-100 text-indigo-600',
            'teal': 'bg-teal-100 text-teal-600',
            'orange': 'bg-orange-100 text-orange-600',
            'red': 'bg-red-100 text-red-600',
            'yellow': 'bg-yellow-100 text-yellow-600',
            'pink': 'bg-pink-100 text-pink-600'
        };
        
        const colorClass = colorClasses[color] || colorClasses['blue'];
        
        return `
            <a href="${href}" 
               target="_blank" 
               rel="noopener noreferrer"
               class="dashboard-card ${colorClass} rounded-lg p-4 card-hover text-center block"
               title="${title} - 在新窗口打开">
                <div class="w-12 h-12 ${colorClass.split(' ')[0].replace('100', '500')} rounded-full flex items-center justify-center mx-auto mb-2">
                    <i class="fas ${icon} text-white"></i>
                </div>
                <h4 class="font-semibold text-sm">${title}</h4>
                <i class="fas fa-external-link-alt external-icon text-xs mt-2 opacity-50"></i>
            </a>
        `;
    }
    
    /**
     * 初始化标签页组件
     * @param {string} tabId - 标签页ID
     */
    initTabComponents(tabId) {
        // 根据标签页ID初始化对应的图表和组件
        switch(tabId) {
            case 'business':
                this.initBusinessDashboard();
                break;
            case 'strategy':
                this.initStrategyCharts();
                break;
            case 'trading':
                this.initTradingCharts();
                break;
            case 'data':
                this.initDataCharts();
                break;
            case 'alert':
                this.initAlertDashboard();
                break;
            case 'architecture':
                this.initArchitectureView();
                break;
        }
    }
    
    /**
     * 初始化业务驱动仪表板
     */
    initBusinessDashboard() {
        // 业务驱动仪表板初始化
        console.log('初始化业务驱动仪表板');
        // 这里可以添加业务数据的加载和更新逻辑
    }
    
    /**
     * 初始化策略图表
     */
    initStrategyCharts() {
        // 策略监控图表初始化
        console.log('初始化策略监控图表');
    }
    
    /**
     * 初始化交易图表
     */
    initTradingCharts() {
        // 交易监控图表初始化
        console.log('初始化交易监控图表');
    }
    
    /**
     * 初始化数据图表
     */
    initDataCharts() {
        // 数据监控图表初始化
        console.log('初始化数据监控图表');
    }
    
    /**
     * 初始化告警仪表板
     */
    initAlertDashboard() {
        // 告警仪表板初始化
        console.log('初始化告警仪表板');
    }
    
    /**
     * 初始化架构视图
     */
    initArchitectureView() {
        // 架构视图初始化
        console.log('初始化架构视图');
    }
    
    /**
     * 保存标签页状态到本地存储
     */
    saveTabState() {
        try {
            localStorage.setItem('dashboard_active_tab', this.activeTab);
            localStorage.setItem('dashboard_loaded_tabs', JSON.stringify([...this.loadedTabs]));
        } catch (e) {
            console.warn('保存标签页状态失败:', e);
        }
    }
    
    /**
     * 从本地存储恢复标签页状态
     */
    restoreLastTab() {
        try {
            const savedTab = localStorage.getItem('dashboard_active_tab');
            if (savedTab && this.tabs.find(t => t.id === savedTab)) {
                this.activeTab = savedTab;
            }
            
            // 页面刷新后，只保留 overview 作为已加载（因为它是同步渲染的）
            // 其他标签页需要重新加载内容
            this.loadedTabs = new Set(['overview']);
            
            // 清除本地存储的 loadedTabs，确保下次刷新重新加载
            localStorage.removeItem('dashboard_loaded_tabs');
        } catch (e) {
            console.warn('恢复标签页状态失败:', e);
        }
    }
    
    /**
     * 刷新当前标签页
     */
    refreshCurrentTab() {
        // 清除当前标签页的缓存
        this.cache.delete(this.activeTab);
        
        // 重新加载
        this.loadTabContent(this.activeTab);
    }
    
    /**
     * 刷新所有标签页
     */
    refreshAllTabs() {
        // 清除所有缓存
        this.cache.clear();
        this.loadedTabs.clear();
        this.loadedTabs.add('overview');
        
        // 重新加载当前标签页
        this.loadTabContent(this.activeTab);
    }
}

// 全局实例
let dashboardTabs;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    console.log('[DashboardTabs] DOMContentLoaded 事件触发，开始初始化...');
    try {
        dashboardTabs = new DashboardTabs();
        console.log('[DashboardTabs] 初始化完成');
    } catch (error) {
        console.error('[DashboardTabs] 初始化失败:', error);
    }
});

// 导出供其他模块使用
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DashboardTabs;
}
