/**
 * API请求队列管理器
 * 实现前端API请求限流和去重
 */

class RequestQueue {
    /**
     * 构造函数
     * @param {Object} config - 配置对象
     * @param {number} config.maxConcurrent - 最大并发请求数，默认5
     * @param {number} config.dedupeWindow - 去重时间窗口（毫秒），默认1000
     * @param {number} config.rateLimit - 速率限制（请求/秒），默认10
     */
    constructor(config = {}) {
        this.maxConcurrent = config.maxConcurrent || 5;
        this.dedupeWindow = config.dedupeWindow || 1000;
        this.rateLimit = config.rateLimit || 10;
        
        this.queue = [];
        this.running = 0;
        this.pendingRequests = new Map(); // 用于去重
        this.tokenBucket = {
            tokens: this.rateLimit,
            lastRefill: Date.now()
        };
    }

    /**
     * 生成请求键（用于去重）
     * @param {string} url - 请求URL
     * @param {Object} options - 请求选项
     * @returns {string} 请求键
     */
    _generateRequestKey(url, options = {}) {
        const method = (options.method || 'GET').toUpperCase();
        const body = options.body ? JSON.stringify(options.body) : '';
        return `${method}:${url}:${body}`;
    }

    /**
     * 检查是否应该去重
     * @param {string} key - 请求键
     * @returns {boolean} 是否应该去重
     */
    _shouldDedupe(key) {
        const pending = this.pendingRequests.get(key);
        if (!pending) {
            return false;
        }

        const now = Date.now();
        if (now - pending.timestamp < this.dedupeWindow) {
            return true; // 在时间窗口内，应该去重
        }

        // 超过时间窗口，移除
        this.pendingRequests.delete(key);
        return false;
    }

    /**
     * 检查令牌桶是否有足够的令牌
     * @returns {boolean} 是否有足够的令牌
     */
    _checkTokenBucket() {
        const now = Date.now();
        const elapsed = (now - this.tokenBucket.lastRefill) / 1000; // 秒
        const tokensToAdd = elapsed * this.rateLimit;
        
        this.tokenBucket.tokens = Math.min(
            this.rateLimit,
            this.tokenBucket.tokens + tokensToAdd
        );
        this.tokenBucket.lastRefill = now;

        if (this.tokenBucket.tokens >= 1) {
            this.tokenBucket.tokens -= 1;
            return true;
        }

        return false;
    }

    /**
     * 执行请求
     * @param {Function} requestFn - 请求函数
     * @returns {Promise} Promise对象
     */
    async _executeRequest(requestFn) {
        this.running++;
        
        try {
            return await requestFn();
        } finally {
            this.running--;
            this._processQueue();
        }
    }

    /**
     * 处理队列
     */
    _processQueue() {
        // 检查并发限制和速率限制
        while (
            this.queue.length > 0 &&
            this.running < this.maxConcurrent &&
            this._checkTokenBucket()
        ) {
            const item = this.queue.shift();
            const key = item.key;
            
            // 再次检查去重（可能在队列中等待时已过期）
            if (this._shouldDedupe(key)) {
                // 合并到已存在的请求
                const pending = this.pendingRequests.get(key);
                pending.promise.then(
                    result => item.resolve(result),
                    error => item.reject(error)
                );
                continue;
            }

            // 创建新的请求
            const promise = this._executeRequest(item.requestFn);
            this.pendingRequests.set(key, {
                promise: promise,
                timestamp: Date.now()
            });

            promise
                .then(result => {
                    this.pendingRequests.delete(key);
                    item.resolve(result);
                })
                .catch(error => {
                    this.pendingRequests.delete(key);
                    item.reject(error);
                });
        }
    }

    /**
     * 添加请求到队列
     * @param {string} url - 请求URL
     * @param {Object} options - 请求选项
     * @param {Function} requestFn - 请求函数
     * @returns {Promise} Promise对象
     */
    async add(url, options = {}, requestFn) {
        const key = this._generateRequestKey(url, options);

        // 检查去重
        if (this._shouldDedupe(key)) {
            const pending = this.pendingRequests.get(key);
            return pending.promise;
        }

        // 创建新的请求
        return new Promise((resolve, reject) => {
            this.queue.push({
                key: key,
                url: url,
                options: options,
                requestFn: requestFn,
                resolve: resolve,
                reject: reject
            });

            this._processQueue();
        });
    }

    /**
     * 清空队列
     */
    clear() {
        this.queue = [];
        this.pendingRequests.clear();
    }

    /**
     * 获取队列状态
     * @returns {Object} 队列状态
     */
    getStatus() {
        return {
            queueLength: this.queue.length,
            running: this.running,
            maxConcurrent: this.maxConcurrent,
            pendingRequests: this.pendingRequests.size,
            tokens: this.tokenBucket.tokens,
            rateLimit: this.rateLimit
        };
    }
}

// 创建全局实例
const requestQueue = new RequestQueue();

/**
 * 包装fetch函数，使用请求队列
 * @param {string} url - 请求URL
 * @param {Object} options - 请求选项
 * @returns {Promise} Promise对象
 */
async function queuedFetch(url, options = {}) {
    return requestQueue.add(url, options, () => {
        return fetch(url, options);
    });
}

