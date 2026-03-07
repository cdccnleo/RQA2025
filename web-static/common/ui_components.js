/**
 * 统一的UI组件库
 * 提供加载状态、错误提示、空状态等通用UI组件
 */

/**
 * 显示加载状态
 * @param {string} elementId - 目标元素ID
 * @param {string} message - 加载提示消息，默认'加载中...'
 */
function showLoading(elementId, message = '加载中...') {
    const element = document.getElementById(elementId);
    if (!element) {
        console.warn(`元素 ${elementId} 不存在`);
        return;
    }

    // 保存原始内容（如果还没有保存）
    if (!element.dataset.originalContent) {
        element.dataset.originalContent = element.innerHTML;
    }

    element.innerHTML = `
        <div class="flex flex-col items-center justify-center py-8">
            <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mb-3"></div>
            <div class="text-sm text-gray-600">${message}</div>
        </div>
    `;
}

/**
 * 隐藏加载状态，恢复原始内容
 * @param {string} elementId - 目标元素ID
 */
function hideLoading(elementId) {
    const element = document.getElementById(elementId);
    if (!element) {
        console.warn(`元素 ${elementId} 不存在`);
        return;
    }

    // 恢复原始内容
    if (element.dataset.originalContent) {
        element.innerHTML = element.dataset.originalContent;
        delete element.dataset.originalContent;
    } else {
        element.innerHTML = '';
    }
}

/**
 * 显示错误状态
 * @param {string} elementId - 目标元素ID
 * @param {Error|string} error - 错误对象或错误消息
 * @param {Function} retryCallback - 重试回调函数（可选）
 */
function showError(elementId, error, retryCallback = null) {
    const element = document.getElementById(elementId);
    if (!element) {
        console.warn(`元素 ${elementId} 不存在`);
        return;
    }

    // 保存原始内容（如果还没有保存）
    if (!element.dataset.originalContent) {
        element.dataset.originalContent = element.innerHTML;
    }

    const errorMessage = error instanceof Error ? error.message : error;
    const retryButton = retryCallback ? `
        <button onclick="(${retryCallback.toString()})()" 
                class="mt-3 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition duration-300">
            <i class="fas fa-redo mr-2"></i>重试
        </button>
    ` : '';

    element.innerHTML = `
        <div class="flex flex-col items-center justify-center py-8">
            <i class="fas fa-exclamation-circle text-red-500 text-4xl mb-3"></i>
            <div class="text-red-600 font-medium mb-2">加载失败</div>
            <div class="text-sm text-gray-600 text-center mb-4">${escapeHtml(errorMessage)}</div>
            ${retryButton}
        </div>
    `;
}

/**
 * 显示空状态
 * @param {string} elementId - 目标元素ID
 * @param {string} message - 空状态提示消息，默认'暂无数据'
 * @param {string} icon - 图标类名（Font Awesome），默认'fa-inbox'
 */
function showEmpty(elementId, message = '暂无数据', icon = 'fa-inbox') {
    const element = document.getElementById(elementId);
    if (!element) {
        console.warn(`元素 ${elementId} 不存在`);
        return;
    }

    // 保存原始内容（如果还没有保存）
    if (!element.dataset.originalContent) {
        element.dataset.originalContent = element.innerHTML;
    }

    element.innerHTML = `
        <div class="flex flex-col items-center justify-center py-8">
            <i class="fas ${icon} text-gray-400 text-4xl mb-3"></i>
            <div class="text-sm text-gray-500">${escapeHtml(message)}</div>
        </div>
    `;
}

/**
 * 转义HTML，防止XSS攻击
 * @param {string} text - 要转义的文本
 * @returns {string} 转义后的文本
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * 创建加载状态的包装函数，用于异步操作
 * @param {string} elementId - 目标元素ID
 * @param {Function} asyncFunction - 异步函数
 * @param {string} loadingMessage - 加载提示消息
 * @returns {Promise} Promise对象
 */
async function withLoading(elementId, asyncFunction, loadingMessage = '加载中...') {
    try {
        showLoading(elementId, loadingMessage);
        const result = await asyncFunction();
        hideLoading(elementId);
        return result;
    } catch (error) {
        hideLoading(elementId);
        showError(elementId, error, () => {
            withLoading(elementId, asyncFunction, loadingMessage);
        });
        throw error;
    }
}

/**
 * 创建带有加载状态和错误处理的异步函数包装器
 * @param {Function} asyncFunction - 异步函数
 * @param {Object} options - 配置选项
 * @param {string} options.elementId - 目标元素ID
 * @param {string} options.loadingMessage - 加载提示消息
 * @param {Function} options.onSuccess - 成功回调
 * @param {Function} options.onError - 错误回调
 * @returns {Function} 包装后的函数
 */
function createAsyncHandler(asyncFunction, options = {}) {
    const {
        elementId,
        loadingMessage = '加载中...',
        onSuccess,
        onError
    } = options;

    return async function(...args) {
        try {
            if (elementId) {
                showLoading(elementId, loadingMessage);
            }
            const result = await asyncFunction.apply(this, args);
            if (elementId) {
                hideLoading(elementId);
            }
            if (onSuccess) {
                onSuccess(result);
            }
            return result;
        } catch (error) {
            if (elementId) {
                hideLoading(elementId);
                showError(elementId, error, () => {
                    createAsyncHandler(asyncFunction, options)(...args);
                });
            }
            if (onError) {
                onError(error);
            } else {
                console.error('异步操作失败:', error);
            }
            throw error;
        }
    };
}

