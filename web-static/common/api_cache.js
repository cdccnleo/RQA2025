/**
 * API响应缓存管理器
 * 实现前端API响应缓存，减少重复请求
 */

class APICache {
    /**
     * 构造函数
     * @param {Object} config - 配置对象
     * @param {number} config.defaultTTL - 默认TTL（毫秒），默认60000（1分钟）
     * @param {number} config.maxSize - 最大缓存条目数，默认100
     */
    constructor(config = {}) {
        this.cache = new Map(); // 使用Map存储缓存
        this.defaultTTL = config.defaultTTL || 60000; // 默认1分钟
        this.maxSize = config.maxSize || 100;
        this.timers = new Map(); // 存储定时器，用于自动清理
    }

    /**
     * 生成缓存键
     * @param {string} url - API URL
     * @param {Object} options - 请求选项（可选）
     * @returns {string} 缓存键
     */
    _generateKey(url, options = {}) {
        const method = (options.method || 'GET').toUpperCase();
        const body = options.body ? JSON.stringify(options.body) : '';
        return `${method}:${url}:${body}`;
    }

    /**
     * 获取缓存
     * @param {string} url - API URL
     * @param {Object} options - 请求选项（可选）
     * @returns {Object|null} 缓存的数据，如果不存在或已过期则返回null
     */
    get(url, options = {}) {
        const key = this._generateKey(url, options);
        const cached = this.cache.get(key);

        if (!cached) {
            return null;
        }

        // 检查是否过期
        if (Date.now() > cached.expiresAt) {
            this.delete(url, options);
            return null;
        }

        return cached.data;
    }

    /**
     * 设置缓存
     * @param {string} url - API URL
     * @param {Object} options - 请求选项（可选）
     * @param {*} data - 要缓存的数据
     * @param {number} ttl - TTL（毫秒），默认使用配置的默认值
     */
    set(url, options = {}, data, ttl = null) {
        const key = this._generateKey(url, options);
        const cacheTTL = ttl || this.defaultTTL;

        // 如果缓存已满，删除最旧的条目
        if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
            const firstKey = this.cache.keys().next().value;
            this.deleteByKey(firstKey);
        }

        // 清除旧的定时器（如果存在）
        if (this.timers.has(key)) {
            clearTimeout(this.timers.get(key));
        }

        // 设置缓存
        const expiresAt = Date.now() + cacheTTL;
        this.cache.set(key, {
            data: data,
            expiresAt: expiresAt,
            cachedAt: Date.now()
        });

        // 设置自动清理定时器
        const timer = setTimeout(() => {
            this.deleteByKey(key);
        }, cacheTTL);
        this.timers.set(key, timer);
    }

    /**
     * 删除缓存
     * @param {string} url - API URL
     * @param {Object} options - 请求选项（可选）
     */
    delete(url, options = {}) {
        const key = this._generateKey(url, options);
        this.deleteByKey(key);
    }

    /**
     * 根据键删除缓存
     * @param {string} key - 缓存键
     */
    deleteByKey(key) {
        if (this.timers.has(key)) {
            clearTimeout(this.timers.get(key));
            this.timers.delete(key);
        }
        this.cache.delete(key);
    }

    /**
     * 清空所有缓存
     */
    clear() {
        // 清除所有定时器
        for (const timer of this.timers.values()) {
            clearTimeout(timer);
        }
        this.timers.clear();
        this.cache.clear();
    }

    /**
     * 清理过期缓存
     */
    cleanup() {
        const now = Date.now();
        const keysToDelete = [];

        for (const [key, cached] of this.cache.entries()) {
            if (now > cached.expiresAt) {
                keysToDelete.push(key);
            }
        }

        for (const key of keysToDelete) {
            this.deleteByKey(key);
        }
    }

    /**
     * 获取缓存统计信息
     * @returns {Object} 统计信息
     */
    getStats() {
        return {
            size: this.cache.size,
            maxSize: this.maxSize,
            entries: Array.from(this.cache.entries()).map(([key, value]) => ({
                key: key,
                cachedAt: value.cachedAt,
                expiresAt: value.expiresAt,
                ttl: value.expiresAt - Date.now()
            }))
        };
    }
}

// 创建全局缓存实例
const apiCache = new APICache({
    defaultTTL: 10000, // 默认10秒
    maxSize: 100
});

// 定义不同API的TTL配置
const API_TTL_CONFIG = {
    // 架构状态：10秒
    '/architecture/status': 10000,
    '/architecture': 10000,
    
    // 数据质量：5秒
    '/data/quality': 5000,
    '/data/quality/metrics': 5000,
    '/data/quality/issues': 5000,
    
    // 性能指标：3秒
    '/data-sources/metrics': 3000,
    '/data/performance': 3000,
    
    // 告警事件：2秒
    '/risk/status': 2000,
    '/system/events': 2000,
    
    // 其他：使用默认值
    default: 10000
};

/**
 * 根据URL获取TTL配置
 * @param {string} url - API URL
 * @returns {number} TTL（毫秒）
 */
function getTTLForURL(url) {
    for (const [pattern, ttl] of Object.entries(API_TTL_CONFIG)) {
        if (pattern !== 'default' && url.includes(pattern)) {
            return ttl;
        }
    }
    return API_TTL_CONFIG.default;
}

