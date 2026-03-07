/**
 * 统一的API客户端
 * 集成缓存功能，提供统一的API调用接口
 */

// 确保api_cache.js已加载（通过script标签加载）
// api_cache.js会在api_client.js之前加载

/**
 * 统一的API客户端类
 */
class APIClient {
    /**
     * 构造函数
     * @param {Object} config - 配置对象
     * @param {string} config.baseURL - 基础URL
     * @param {boolean} config.enableCache - 是否启用缓存，默认true
     * @param {Object} config.defaultHeaders - 默认请求头
     */
    constructor(config = {}) {
        this.baseURL = config.baseURL || '';
        this.enableCache = config.enableCache !== false;
        this.defaultHeaders = config.defaultHeaders || {
            'Content-Type': 'application/json'
        };
    }

    /**
     * 获取API基础URL（环境感知）
     * @param {string} endpoint - API端点
     * @returns {string} 完整的API URL
     */
    getApiBaseUrl(endpoint = '') {
        if (this.baseURL) {
            return this.baseURL + endpoint;
        }
        
        // 环境感知的URL生成
        const baseUrl = window.location.protocol === 'file:' || window.location.hostname === 'localhost'
            ? 'http://localhost:8000/api/v1'
            : '/api/v1';
        return baseUrl + endpoint;
    }

    /**
     * 执行API请求
     * @param {string} url - API URL（相对于baseURL或完整URL）
     * @param {Object} options - 请求选项
     * @param {boolean} options.useCache - 是否使用缓存，默认true（仅对GET请求有效）
     * @param {number} options.cacheTTL - 缓存TTL（毫秒），默认使用配置的TTL
     * @returns {Promise} Promise对象
     */
    async request(url, options = {}) {
        const {
            useCache = true,
            cacheTTL = null,
            ...fetchOptions
        } = options;

        const fullURL = url.startsWith('http') ? url : this.getApiBaseUrl(url);
        const method = (fetchOptions.method || 'GET').toUpperCase();

        // GET请求且启用缓存时，先检查缓存
        if (method === 'GET' && useCache && this.enableCache && typeof apiCache !== 'undefined') {
            const cachedData = apiCache.get(fullURL, fetchOptions);
            if (cachedData !== null) {
                return Promise.resolve(cachedData);
            }
        }

        // 合并默认请求头
        const headers = {
            ...this.defaultHeaders,
            ...(fetchOptions.headers || {})
        };

        try {
            const response = await fetch(fullURL, {
                ...fetchOptions,
                headers: headers
            });

            // 处理响应
            let data;
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                data = await response.json();
            } else {
                data = await response.text();
            }

            // 如果请求成功，缓存响应（仅GET请求）
            if (response.ok && method === 'GET' && useCache && this.enableCache && typeof apiCache !== 'undefined') {
                const ttl = cacheTTL || (typeof getTTLForURL === 'function' ? getTTLForURL(fullURL) : apiCache.defaultTTL);
                apiCache.set(fullURL, fetchOptions, data, ttl);
            }

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return data;
        } catch (error) {
            console.error(`API请求失败 (${fullURL}):`, error);
            throw error;
        }
    }

    /**
     * GET请求
     * @param {string} url - API URL
     * @param {Object} options - 请求选项
     * @returns {Promise} Promise对象
     */
    async get(url, options = {}) {
        return this.request(url, {
            ...options,
            method: 'GET'
        });
    }

    /**
     * POST请求
     * @param {string} url - API URL
     * @param {Object} data - 请求数据
     * @param {Object} options - 请求选项
     * @returns {Promise} Promise对象
     */
    async post(url, data, options = {}) {
        return this.request(url, {
            ...options,
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    /**
     * PUT请求
     * @param {string} url - API URL
     * @param {Object} data - 请求数据
     * @param {Object} options - 请求选项
     * @returns {Promise} Promise对象
     */
    async put(url, data, options = {}) {
        return this.request(url, {
            ...options,
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }

    /**
     * DELETE请求
     * @param {string} url - API URL
     * @param {Object} options - 请求选项
     * @returns {Promise} Promise对象
     */
    async delete(url, options = {}) {
        return this.request(url, {
            ...options,
            method: 'DELETE'
        });
    }

    /**
     * 清除指定URL的缓存
     * @param {string} url - API URL
     * @param {Object} options - 请求选项（可选）
     */
    clearCache(url, options = {}) {
        if (typeof apiCache !== 'undefined') {
            const fullURL = url.startsWith('http') ? url : this.getApiBaseUrl(url);
            apiCache.delete(fullURL, options);
        }
    }

    /**
     * 清除所有缓存
     */
    clearAllCache() {
        if (typeof apiCache !== 'undefined') {
            apiCache.clear();
        }
    }
}

// 创建全局API客户端实例
const apiClient = new APIClient();

/**
 * 便捷函数：获取API基础URL
 * @param {string} endpoint - API端点
 * @returns {string} 完整的API URL
 */
function getApiBaseUrl(endpoint = '') {
    return apiClient.getApiBaseUrl(endpoint);
}

/**
 * 便捷函数：执行GET请求
 * @param {string} url - API URL
 * @param {Object} options - 请求选项
 * @returns {Promise} Promise对象
 */
async function apiGet(url, options = {}) {
    return apiClient.get(url, options);
}

/**
 * 便捷函数：执行POST请求
 * @param {string} url - API URL
 * @param {Object} data - 请求数据
 * @param {Object} options - 请求选项
 * @returns {Promise} Promise对象
 */
async function apiPost(url, data, options = {}) {
    return apiClient.post(url, data, options);
}

