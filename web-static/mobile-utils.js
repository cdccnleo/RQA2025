/**
 * RQA2025 移动端工具函数
 * 提供移动端优化和响应式功能
 */

// 移动端检测和设备信息
const MobileUtils = {
    // 检测设备类型
    isMobile: () => window.innerWidth < 768,
    isTablet: () => window.innerWidth >= 768 && window.innerWidth < 1024,
    isDesktop: () => window.innerWidth >= 1024,

    // 检测触摸设备
    isTouchDevice: () => 'ontouchstart' in window || navigator.maxTouchPoints > 0,

    // 检测网络状态
    getNetworkStatus: () => {
        if ('connection' in navigator) {
            const connection = navigator.connection;
            return {
                effectiveType: connection.effectiveType,
                downlink: connection.downlink,
                rtt: connection.rtt,
                saveData: connection.saveData
            };
        }
        return null;
    },

    // 优化数据请求频率
    throttleDataRequests: (fn, delay = 30000) => {
        let lastCall = 0;
        return function(...args) {
            const now = Date.now();
            if (now - lastCall >= delay) {
                lastCall = now;
                return fn.apply(this, args);
            }
        };
    },

    // 减少移动端数据量
    optimizeDataForMobile: (data) => {
        if (!MobileUtils.isMobile()) return data;

        // 减少时间序列数据点
        if (data.labels && data.labels.length > 10) {
            const step = Math.floor(data.labels.length / 10);
            data.labels = data.labels.filter((_, i) => i % step === 0);
            if (data.datasets) {
                data.datasets.forEach(dataset => {
                    if (dataset.data) {
                        dataset.data = dataset.data.filter((_, i) => i % step === 0);
                    }
                });
            }
        }

        // 简化图例
        if (data.datasets && data.datasets.length > 3) {
            data.datasets = data.datasets.slice(0, 3);
        }

        return data;
    },

    // 优化图表配置
    getResponsiveChartOptions: (baseOptions = {}) => {
        const isMobile = MobileUtils.isMobile();
        const isTablet = MobileUtils.isTablet();

        return {
            ...baseOptions,
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                ...baseOptions.plugins,
                legend: {
                    ...baseOptions.plugins?.legend,
                    position: isMobile ? 'bottom' : (baseOptions.plugins?.legend?.position || 'top'),
                    labels: {
                        ...baseOptions.plugins?.legend?.labels,
                        font: {
                            size: isMobile ? 10 : (isTablet ? 11 : 12)
                        }
                    }
                }
            },
            scales: {
                ...baseOptions.scales,
                x: {
                    ...baseOptions.scales?.x,
                    ticks: {
                        ...baseOptions.scales?.x?.ticks,
                        font: {
                            size: isMobile ? 8 : (isTablet ? 9 : 12)
                        }
                    }
                },
                y: {
                    ...baseOptions.scales?.y,
                    ticks: {
                        ...baseOptions.scales?.y?.ticks,
                        font: {
                            size: isMobile ? 8 : (isTablet ? 9 : 12)
                        }
                    }
                }
            }
        };
    },

    // 懒加载非关键内容
    lazyLoadContent: (selector, callback) => {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    callback(entry.target);
                    observer.unobserve(entry.target);
                }
            });
        });

        document.querySelectorAll(selector).forEach(el => {
            observer.observe(el);
        });
    },

    // 防抖函数用于输入优化
    debounce: (fn, delay = 300) => {
        let timeoutId;
        return function(...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => fn.apply(this, args), delay);
        };
    },

    // 触摸事件优化
    enhanceTouchTargets: () => {
        // 为小按钮添加触摸目标
        document.querySelectorAll('button, a, .clickable').forEach(el => {
            if (!el.classList.contains('touch-target')) {
                el.classList.add('touch-target');
            }
        });

        // 优化触摸反馈
        document.addEventListener('touchstart', (e) => {
            const target = e.target.closest('.touch-target');
            if (target) {
                target.style.transform = 'scale(0.98)';
            }
        });

        document.addEventListener('touchend', (e) => {
            const target = e.target.closest('.touch-target');
            if (target) {
                setTimeout(() => {
                    target.style.transform = '';
                }, 150);
            }
        });
    },

    // 内存优化
    optimizeMemory: () => {
        // 定期清理不用的图表实例
        const cleanupCharts = () => {
            // 页面不可见时清理图表
            if (document.hidden) {
                // 可以在这里添加图表清理逻辑
                console.log('页面不可见，优化内存');
            }
        };

        document.addEventListener('visibilitychange', cleanupCharts);
    },

    // 网络状态监听
    monitorNetworkStatus: (callback) => {
        if ('connection' in navigator) {
            const connection = navigator.connection;
            const updateNetworkStatus = () => {
                const status = MobileUtils.getNetworkStatus();
                callback(status);
            };

            connection.addEventListener('change', updateNetworkStatus);
            updateNetworkStatus(); // 初始状态
        }
    },

    // 自适应刷新频率
    getAdaptiveRefreshInterval: () => {
        const networkStatus = MobileUtils.getNetworkStatus();

        if (MobileUtils.isMobile()) {
            if (networkStatus?.effectiveType === 'slow-2g' || networkStatus?.effectiveType === '2g') {
                return 120000; // 2分钟
            } else if (networkStatus?.effectiveType === '3g') {
                return 60000; // 1分钟
            } else {
                return 30000; // 30秒
            }
        } else {
            return 15000; // 桌面端15秒
        }
    },

    // 简化移动端视图
    simplifyMobileView: () => {
        if (!MobileUtils.isMobile()) return;

        // 隐藏复杂图表详情
        document.querySelectorAll('.complex-chart').forEach(chart => {
            chart.style.display = 'none';
        });

        // 简化表格显示
        document.querySelectorAll('table').forEach(table => {
            table.classList.add('mobile-table');
        });

        // 启用移动端专用视图
        document.querySelectorAll('.mobile-only').forEach(el => {
            el.style.display = 'block';
        });
    },

    // 初始化移动端优化
    init: () => {
        console.log('初始化移动端优化...');

        // 检测设备类型
        const deviceType = MobileUtils.isMobile() ? 'mobile' : (MobileUtils.isTablet() ? 'tablet' : 'desktop');
        document.documentElement.setAttribute('data-device', deviceType);

        // 应用移动端优化
        MobileUtils.enhanceTouchTargets();
        MobileUtils.optimizeMemory();
        MobileUtils.simplifyMobileView();

        // 监听窗口大小变化
        window.addEventListener('resize', MobileUtils.debounce(() => {
            const newDeviceType = MobileUtils.isMobile() ? 'mobile' : (MobileUtils.isTablet() ? 'tablet' : 'desktop');
            document.documentElement.setAttribute('data-device', newDeviceType);
        }, 250));

        // 网络状态监听
        MobileUtils.monitorNetworkStatus((status) => {
            if (status) {
                document.documentElement.setAttribute('data-network', status.effectiveType);
                console.log('网络状态变化:', status);
            }
        });

        console.log('移动端优化初始化完成');
    }
};

// 导出工具函数
window.MobileUtils = MobileUtils;
