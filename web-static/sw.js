/**
 * RQA2025 Service Worker - 缓存策略和后台同步
 */

const CACHE_NAME = 'rqa2025-v1.0.2';
const STATIC_CACHE = 'rqa2025-static-v1.0.2';
const API_CACHE = 'rqa2025-api-v1.0.2';

// 需要缓存的本地静态资源
const LOCAL_STATIC_ASSETS = [
    '/',
    '/dashboard',
    '/mobile-optimization.css',
    '/mobile-utils.js',
    '/ux-optimization.js'
];

// CDN资源（使用no-cors模式缓存）
const CDN_ASSETS = [
    'https://cdn.tailwindcss.com',
    'https://cdn.jsdelivr.net/npm/chart.js',
    'https://cdn.jsdelivr.net/npm/luxon@3.4.4/build/global/luxon.min.js',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css'
];

// API端点缓存配置
const API_ENDPOINTS = [
    '/api/v1/monitoring/overview',
    '/api/v1/monitoring/alerts/overview',
    '/api/v1/strategy/development/overview',
    '/api/v1/trading/execution/flow',
    '/api/v1/risk/control/flow'
];

// 安装Service Worker
self.addEventListener('install', event => {
    console.log('Service Worker installing...');
    event.waitUntil(
        Promise.all([
            // 缓存本地资源
            caches.open(STATIC_CACHE)
                .then(cache => {
                    console.log('Caching local static assets...');
                    return cache.addAll(LOCAL_STATIC_ASSETS);
                }),
            // 缓存CDN资源（使用no-cors模式）
            caches.open(STATIC_CACHE)
                .then(cache => {
                    console.log('Caching CDN assets...');
                    const cdnPromises = CDN_ASSETS.map(url => 
                        fetch(url, { mode: 'no-cors' })
                            .then(response => cache.put(url, response))
                            .catch(err => console.log('Failed to cache CDN asset:', url, err))
                    );
                    return Promise.allSettled(cdnPromises);
                })
        ]).then(() => self.skipWaiting())
    );
});

// 激活Service Worker
self.addEventListener('activate', event => {
    console.log('Service Worker activating...');
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    if (cacheName !== STATIC_CACHE && cacheName !== API_CACHE) {
                        console.log('Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        }).then(() => self.clients.claim())
    );
});

// 处理获取请求
self.addEventListener('fetch', event => {
    const { request } = event;
    const url = new URL(request.url);

    // 处理API请求
    if (url.pathname.startsWith('/api/v1/')) {
        event.respondWith(handleApiRequest(request));
        return;
    }

    // 处理CDN资源请求
    if (CDN_ASSETS.some(asset => url.href.includes(asset))) {
        event.respondWith(handleCdnRequest(request));
        return;
    }

    // 处理本地静态资源请求
    if (LOCAL_STATIC_ASSETS.some(asset => url.pathname === asset || url.pathname.startsWith(asset))) {
        event.respondWith(handleStaticRequest(request));
        return;
    }

    // 默认网络优先策略
    event.respondWith(
        fetch(request)
            .catch(() => caches.match(request))
    );
});

// 不缓存的API路径列表
const NO_CACHE_API_PATHS = [
    '/api/v1/strategy/conceptions',
    '/api/v1/strategy/lifecycle'
];

// 处理API请求 - 网络优先 + 缓存回退（排除特定API和POST请求）
async function handleApiRequest(request) {
    const url = new URL(request.url);

    // 检查是否是POST/PUT/DELETE等非GET请求（这些请求不能被缓存）
    if (request.method !== 'GET') {
        try {
            return await fetch(request);
        } catch (error) {
            return new Response(
                JSON.stringify({
                    error: '网络错误',
                    message: `${request.method} 请求失败，请检查网络连接`,
                    offline: true
                }),
                {
                    status: 503,
                    headers: { 'Content-Type': 'application/json' }
                }
            );
        }
    }

    // 检查是否是不缓存的API
    const shouldNotCache = NO_CACHE_API_PATHS.some(path => url.pathname.startsWith(path));

    try {
        // 尝试从网络获取
        const networkResponse = await fetch(request);

        // 如果成功且不是排除的API，缓存响应（只缓存GET请求）
        if (networkResponse.ok && !shouldNotCache && request.method === 'GET') {
            const cache = await caches.open(API_CACHE);
            cache.put(request, networkResponse.clone());
        }

        return networkResponse;
    } catch (error) {
        // 网络失败，从缓存获取（排除的API也不使用缓存）
        if (!shouldNotCache) {
            const cachedResponse = await caches.match(request);
            if (cachedResponse) {
                return cachedResponse;
            }
        }

        // 返回离线响应
        return new Response(
            JSON.stringify({
                error: '离线模式',
                message: '无法连接到服务器，请检查网络连接',
                offline: true
            }),
            {
                status: 503,
                headers: { 'Content-Type': 'application/json' }
            }
        );
    }
}

// 处理CDN资源请求 - 缓存优先，失败时网络获取（no-cors）
async function handleCdnRequest(request) {
    // 首先尝试从缓存获取
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
        return cachedResponse;
    }

    try {
        // 使用no-cors模式获取CDN资源
        const networkResponse = await fetch(request, { mode: 'no-cors' });
        
        // 缓存响应（即使是opaque响应）
        if (networkResponse) {
            const cache = await caches.open(STATIC_CACHE);
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
    } catch (error) {
        console.error('Failed to fetch CDN asset:', request.url, error);
        // 返回一个简单的错误响应
        return new Response('CDN resource not available', { status: 404 });
    }
}

// 处理本地静态资源请求 - 缓存优先
async function handleStaticRequest(request) {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
        return cachedResponse;
    }

    try {
        const networkResponse = await fetch(request);
        if (networkResponse.ok) {
            const cache = await caches.open(STATIC_CACHE);
            cache.put(request, networkResponse.clone());
        }
        return networkResponse;
    } catch (error) {
        console.error('Failed to fetch static asset:', error);
        return new Response('Resource not available offline', { status: 404 });
    }
}

// 处理后台同步
self.addEventListener('sync', event => {
    console.log('Background sync triggered:', event.tag);

    if (event.tag === 'data-sync') {
        event.waitUntil(syncData());
    }
});

// 数据同步函数
async function syncData() {
    console.log('Performing background data sync...');

    try {
        // 同步未发送的数据
        const unsentData = await getUnsentData();

        for (const data of unsentData) {
            await sendDataToServer(data);
        }

        // 预加载关键数据
        await preloadCriticalData();

        console.log('Background sync completed');
    } catch (error) {
        console.error('Background sync failed:', error);
    }
}

// 获取未发送的数据
async function getUnsentData() {
    // 从IndexedDB或其他存储中获取未同步的数据
    return [];
}

// 发送数据到服务器
async function sendDataToServer(data) {
    // 实现数据发送逻辑
    return fetch('/api/v1/sync', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });
}

// 预加载关键数据
async function preloadCriticalData() {
    const criticalAPIs = [
        '/api/v1/monitoring/overview',
        '/api/v1/monitoring/alerts/overview'
    ];

    for (const api of criticalAPIs) {
        try {
            const response = await fetch(api);
            if (response.ok) {
                const cache = await caches.open(API_CACHE);
                cache.put(api, response);
            }
        } catch (error) {
            console.log('Failed to preload:', api, error);
        }
    }
}

// 处理推送消息
self.addEventListener('push', event => {
    console.log('Push message received:', event);

    if (event.data) {
        const data = event.data.json();

        const options = {
            body: data.message,
            icon: '/icon-192x192.png',
            badge: '/badge-72x72.png',
            data: data.url,
            requireInteraction: data.urgent || false,
            silent: false,
            actions: [
                {
                    action: 'view',
                    title: '查看详情'
                },
                {
                    action: 'dismiss',
                    title: '忽略'
                }
            ]
        };

        event.waitUntil(
            self.registration.showNotification(data.title, options)
        );
    }
});

// 处理通知点击
self.addEventListener('notificationclick', event => {
    console.log('Notification click:', event);

    event.notification.close();

    if (event.action === 'view') {
        event.waitUntil(
            clients.openWindow(event.notification.data || '/dashboard')
        );
    }
});

// 定期清理过期缓存
self.addEventListener('message', event => {
    if (event.data && event.data.type === 'CLEAN_CACHE') {
        cleanExpiredCache();
    }
});

// 清理过期缓存
async function cleanExpiredCache() {
    const cache = await caches.open(API_CACHE);
    const keys = await cache.keys();

    // 删除超过1小时的缓存
    const oneHourAgo = Date.now() - (60 * 60 * 1000);

    for (const request of keys) {
        const response = await cache.match(request);
        if (response) {
            const date = response.headers.get('date');
            if (date && new Date(date).getTime() < oneHourAgo) {
                await cache.delete(request);
            }
        }
    }

    console.log('Cache cleanup completed');
}
