/**
 * RQA2025 用户体验优化工具
 * 提供页面加载性能优化、数据刷新策略和自定义监控面板功能
 */

// 防止重复声明
if (typeof window.UXOptimization === 'undefined') {
    // 用户体验优化工具
    window.UXOptimization = {
        // 页面加载性能优化
        optimizePageLoad: () => {
            // 延迟加载非关键资源
            UXOptimization.lazyLoadNonCritical();

            // 启用服务端缓存策略
            UXOptimization.enableServiceWorkerCache();
        },

        // 预加载关键资源（暂时禁用，等待API实现）
        preloadCriticalResources: () => {
            // TODO: 等监控概览API实现后启用
            // const criticalAPIs = [
            //     '/api/v1/monitoring/overview',
            //     '/api/v1/monitoring/alerts/overview'
            // ];
            // criticalAPIs.forEach(api => {
            //     const link = document.createElement('link');
            //     link.rel = 'preload';
            //     link.as = 'fetch';
            //     link.href = api;
            //     link.crossOrigin = 'anonymous';
            //     document.head.appendChild(link);
            // });
        },

        // 延迟加载非关键资源
        lazyLoadNonCritical: () => {
            // 延迟加载复杂图表库
            const lazyLoadChart = () => {
                if ('IntersectionObserver' in window) {
                    const chartObserver = new IntersectionObserver((entries) => {
                        entries.forEach(entry => {
                            if (entry.isIntersecting) {
                                // 动态加载Chart.js扩展
                                UXOptimization.loadChartExtensions();
                                chartObserver.disconnect();
                            }
                        });
                    });

                    // 观察图表容器
                    document.querySelectorAll('.chart-container').forEach(container => {
                        chartObserver.observe(container);
                    });
                }
            };

            // 页面加载完成后延迟执行
            setTimeout(lazyLoadChart, 1000);
        },

        // 优化资源加载顺序（已禁用，避免脚本重复加载）
        optimizeResourceLoading: () => {
            // 此功能已禁用，因为会导致脚本重复加载和执行
            // 现代浏览器已经优化了脚本加载顺序
            console.log('optimizeResourceLoading: 已禁用，避免脚本重复加载');
        },

        // 启用Service Worker缓存
        enableServiceWorkerCache: () => {
            if ('serviceWorker' in navigator) {
                navigator.serviceWorker.register('/sw.js')
                    .then(registration => {
                        console.log('Service Worker registered:', registration);
                    })
                    .catch(error => {
                        console.log('Service Worker registration failed:', error);
                    });
            }
        },

        // 数据刷新策略优化
        optimizeDataRefresh: () => {
            // 实现增量数据更新
            UXOptimization.enableIncrementalUpdates();

            // 自适应刷新频率
            UXOptimization.adaptiveRefreshRate();

            // 后台数据同步
            UXOptimization.backgroundDataSync();
        },

        // 增量数据更新
        enableIncrementalUpdates: () => {
            // 存储上次更新时间戳
            const lastUpdateKey = 'rqa_last_update';
            const lastUpdates = JSON.parse(localStorage.getItem(lastUpdateKey) || '{}');

            // 为API请求添加增量参数
            const originalFetch = window.fetch;
            window.fetch = function(...args) {
                const [url] = args;
                if (url.includes('/api/v1/')) {
                    const endpoint = url.replace('/api/v1/', '').replace(/\?.*/, '');
                    const lastUpdate = lastUpdates[endpoint];

                    if (lastUpdate) {
                        const separator = url.includes('?') ? '&' : '?';
                        args[0] = url + separator + 'since=' + lastUpdate;
                    }
                }

                return originalFetch.apply(this, args).then(response => {
                    // 更新时间戳
                    if (response.ok && url.includes('/api/v1/')) {
                        const endpoint = url.replace('/api/v1/', '').replace(/\?.*/, '');
                        lastUpdates[endpoint] = Date.now();
                        localStorage.setItem(lastUpdateKey, JSON.stringify(lastUpdates));
                    }
                    return response;
                });
            };
        },

        // 自适应刷新频率
        adaptiveRefreshRate: () => {
            let currentInterval = 30000; // 默认30秒

            const adjustInterval = (networkStatus, userActivity) => {
                let newInterval = 30000;

                // 基于网络状态调整
                if (networkStatus) {
                    switch (networkStatus.effectiveType) {
                        case 'slow-2g':
                        case '2g':
                            newInterval = 120000; // 2分钟
                            break;
                        case '3g':
                            newInterval = 60000; // 1分钟
                            break;
                        case '4g':
                        default:
                            newInterval = 30000; // 30秒
                            break;
                    }
                }

                // 基于用户活动调整
                if (!userActivity) {
                    newInterval *= 2; // 用户不活跃时减半频率
                }

                // 平滑调整间隔
                if (Math.abs(newInterval - currentInterval) > 5000) {
                    currentInterval = newInterval;
                    UXOptimization.updateRefreshInterval(currentInterval);
                }
            };

            // 监听网络状态变化
            if ('connection' in navigator) {
                navigator.connection.addEventListener('change', () => {
                    adjustInterval(navigator.connection, UXOptimization.isUserActive());
                });
            }

            // 监听用户活动
            let lastActivity = Date.now();
            ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'].forEach(event => {
                document.addEventListener(event, () => {
                    lastActivity = Date.now();
                }, { passive: true });
            });

            // 定期检查用户活动
            setInterval(() => {
                adjustInterval(navigator.connection, UXOptimization.isUserActive());
            }, 60000); // 每分钟检查一次
        },

        // 检查用户是否活跃
        isUserActive: () => {
            const now = Date.now();
            const lastActivity = parseInt(localStorage.getItem('rqa_last_activity') || '0');
            return (now - lastActivity) < 300000; // 5分钟内有活动
        },

        // 更新刷新间隔
        updateRefreshInterval: (interval) => {
            // 发送自定义事件，通知其他组件更新间隔
            const event = new CustomEvent('refreshIntervalChanged', {
                detail: { interval }
            });
            document.dispatchEvent(event);
        },

        // 后台数据同步
        backgroundDataSync: () => {
            if ('BackgroundSyncManager' in window) {
                navigator.serviceWorker.ready.then(registration => {
                    // 注册后台同步
                    registration.sync.register('data-sync')
                        .then(() => console.log('Background sync registered'))
                        .catch(err => console.log('Background sync failed:', err));
                });
            }
        },

        // 自定义监控面板功能
        enableCustomDashboard: () => {
            // 只在dashboard页面启用自定义面板功能
            if (!window.location.pathname.includes('dashboard')) {
                return;
            }
            UXOptimization.createDashboardCustomizer();
            UXOptimization.loadUserPreferences();
            UXOptimization.enableDragDrop();
        },

        // 创建仪表板自定义器
        createDashboardCustomizer: () => {
            const customizer = document.createElement('div');
            customizer.id = 'dashboard-customizer';
            customizer.className = 'fixed top-4 right-4 z-50 hidden';

            customizer.innerHTML = `
                <div class="bg-white rounded-lg shadow-lg p-4 min-w-64">
                    <div class="flex items-center justify-between mb-4">
                        <h3 class="text-lg font-semibold text-gray-900">自定义面板</h3>
                        <button onclick="UXOptimization.toggleCustomizer()" class="text-gray-400 hover:text-gray-600">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    <div class="space-y-3">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">显示指标</label>
                            <div class="space-y-2">
                                <label class="flex items-center">
                                    <input type="checkbox" class="rounded border-gray-300" checked>
                                    <span class="ml-2 text-sm">系统健康度</span>
                                </label>
                                <label class="flex items-center">
                                    <input type="checkbox" class="rounded border-gray-300" checked>
                                    <span class="ml-2 text-sm">活跃策略</span>
                                </label>
                                <label class="flex items-center">
                                    <input type="checkbox" class="rounded border-gray-300" checked>
                                    <span class="ml-2 text-sm">告警统计</span>
                                </label>
                            </div>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">图表类型</label>
                            <select class="w-full border border-gray-300 rounded-md px-3 py-2">
                                <option value="line">线形图</option>
                                <option value="bar">柱状图</option>
                                <option value="area">面积图</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">刷新频率</label>
                            <select class="w-full border border-gray-300 rounded-md px-3 py-2">
                                <option value="15000">15秒</option>
                                <option value="30000">30秒</option>
                                <option value="60000">1分钟</option>
                                <option value="300000">5分钟</option>
                            </select>
                        </div>
                        <div class="flex space-x-2">
                            <button onclick="UXOptimization.savePreferences()" class="flex-1 bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700">
                                保存
                            </button>
                            <button onclick="UXOptimization.resetPreferences()" class="flex-1 bg-gray-600 text-white px-4 py-2 rounded-md hover:bg-gray-700">
                                重置
                            </button>
                        </div>
                    </div>
                </div>
            `;

            document.body.appendChild(customizer);

            // 添加自定义按钮到导航栏
            const nav = document.querySelector('nav');
            if (nav) {
                const navContainer = nav.querySelector('.flex.items-center.space-x-2') || 
                                     nav.querySelector('.flex.items-center.space-x-4');
                if (navContainer) {
                    const customizeBtn = document.createElement('button');
                    customizeBtn.onclick = () => UXOptimization.toggleCustomizer();
                    customizeBtn.className = 'bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg transition duration-300 hidden sm:inline';
                    customizeBtn.innerHTML = '<i class="fas fa-cog mr-2"></i>自定义';
                    navContainer.appendChild(customizeBtn);
                }
            }
        },

        // 切换自定义器显示
        toggleCustomizer: () => {
            const customizer = document.getElementById('dashboard-customizer');
            customizer.classList.toggle('hidden');
        },

        // 加载用户偏好设置
        loadUserPreferences: () => {
            const preferences = JSON.parse(localStorage.getItem('rqa_user_preferences') || '{}');

            // 应用保存的偏好设置
            if (preferences.theme) {
                document.documentElement.setAttribute('data-theme', preferences.theme);
            }

            if (preferences.refreshRate) {
                UXOptimization.updateRefreshInterval(preferences.refreshRate);
            }

            if (preferences.visibleMetrics) {
                Object.entries(preferences.visibleMetrics).forEach(([metric, visible]) => {
                    const element = document.querySelector(`[data-metric="${metric}"]`);
                    if (element) {
                        element.style.display = visible ? 'block' : 'none';
                    }
                });
            }
        },

        // 保存用户偏好设置
        savePreferences: () => {
            const customizer = document.getElementById('dashboard-customizer');
            const preferences = {
                theme: document.documentElement.getAttribute('data-theme') || 'light',
                refreshRate: parseInt(customizer.querySelector('select:last-of-type').value),
                visibleMetrics: {}
            };

            // 保存可见指标设置
            customizer.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
                const metric = checkbox.nextElementSibling.textContent.trim();
                preferences.visibleMetrics[metric] = checkbox.checked;
            });

            localStorage.setItem('rqa_user_preferences', JSON.stringify(preferences));
            UXOptimization.loadUserPreferences();

            // 显示保存成功提示
            UXOptimization.showNotification('偏好设置已保存', 'success');
        },

        // 重置用户偏好设置
        resetPreferences: () => {
            localStorage.removeItem('rqa_user_preferences');
            UXOptimization.loadUserPreferences();

            // 重置自定义器中的复选框
            const customizer = document.getElementById('dashboard-customizer');
            customizer.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
                checkbox.checked = true;
            });

            UXOptimization.showNotification('偏好设置已重置', 'info');
        },

        // 启用拖拽功能
        enableDragDrop: () => {
            let draggedElement = null;

            document.addEventListener('dragstart', (e) => {
                if (e.target.classList.contains('draggable')) {
                    draggedElement = e.target;
                    e.dataTransfer.effectAllowed = 'move';
                }
            });

            document.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'move';
            });

            document.addEventListener('drop', (e) => {
                e.preventDefault();
                if (draggedElement && e.target.classList.contains('drop-zone')) {
                    e.target.appendChild(draggedElement);
                    UXOptimization.saveLayout();
                }
            });
        },

        // 保存布局
        saveLayout: () => {
            const layout = {};
            document.querySelectorAll('.drop-zone').forEach((zone, index) => {
                layout[`zone-${index}`] = Array.from(zone.children).map(child => child.id);
            });

            localStorage.setItem('rqa_dashboard_layout', JSON.stringify(layout));
        },

        // 显示通知
        showNotification: (message, type = 'info') => {
            const notification = document.createElement('div');
            notification.className = `fixed top-4 right-4 z-50 p-4 rounded-md shadow-lg ${
                type === 'success' ? 'bg-green-500 text-white' :
                type === 'error' ? 'bg-red-500 text-white' :
                type === 'warning' ? 'bg-yellow-500 text-black' :
                'bg-blue-500 text-white'
            }`;

            notification.innerHTML = `
                <div class="flex items-center">
                    <i class="fas ${
                        type === 'success' ? 'fa-check-circle' :
                        type === 'error' ? 'fa-exclamation-circle' :
                        type === 'warning' ? 'fa-exclamation-triangle' :
                        'fa-info-circle'
                    } mr-2"></i>
                    <span>${message}</span>
                    <button onclick="this.parentElement.parentElement.remove()" class="ml-4 text-current opacity-75 hover:opacity-100">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;

            document.body.appendChild(notification);

            // 3秒后自动移除
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, 3000);
        },

        // 性能监控
        enablePerformanceMonitoring: () => {
            // 监控页面加载性能
            window.addEventListener('load', () => {
                const perfData = performance.getEntriesByType('navigation')[0];

                console.log('页面加载性能:', {
                    domContentLoaded: perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart,
                    loadComplete: perfData.loadEventEnd - perfData.loadEventStart,
                    totalTime: perfData.loadEventEnd - perfData.fetchStart
                });
            });

            // 监控运行时性能
            let frameCount = 0;
            let lastTime = performance.now();

            const measureFPS = () => {
                frameCount++;
                const currentTime = performance.now();

                if (currentTime - lastTime >= 1000) {
                    const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
                    console.log('FPS:', fps);

                    // 如果FPS过低，启用性能优化
                    if (fps < 30) {
                        UXOptimization.enablePerformanceOptimizations();
                    }

                    frameCount = 0;
                    lastTime = currentTime;
                }

                requestAnimationFrame(measureFPS);
            };

            requestAnimationFrame(measureFPS);
        },

        // 启用性能优化
        enablePerformanceOptimizations: () => {
            // 减少动画
            document.documentElement.style.setProperty('--animation-duration', '0.01ms');

            // 简化图表
            document.querySelectorAll('.chart-container').forEach(container => {
                container.classList.add('simplified');
            });

            console.log('性能优化已启用');
        },

        // 初始化所有优化功能
        init: () => {
            console.log('初始化用户体验优化...');

            // 页面加载性能优化
            UXOptimization.optimizePageLoad();

            // 数据刷新策略优化
            UXOptimization.optimizeDataRefresh();

            // 自定义监控面板
            UXOptimization.enableCustomDashboard();

            // 性能监控
            UXOptimization.enablePerformanceMonitoring();

            // 监听刷新间隔变化
            document.addEventListener('refreshIntervalChanged', (e) => {
                console.log('刷新间隔已更新为:', e.detail.interval, 'ms');
            });

            console.log('用户体验优化初始化完成');
        }
    };

    // 挂载到 window 对象
    window.UXOptimization = UXOptimization;
}
