/**
 * Toast通知组件
 * 用于替换alert()，提供更友好的错误提示
 */

// Toast容器（如果不存在则创建）
function getToastContainer() {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'fixed top-4 right-4 z-50 space-y-2';
        document.body.appendChild(container);
    }
    return container;
}

/**
 * 显示Toast通知
 * @param {string} message - 通知消息
 * @param {string} type - 通知类型: 'success', 'error', 'warning', 'info'
 * @param {number} duration - 显示时长（毫秒），默认3000，0表示不自动关闭
 * @param {Function} onClose - 关闭回调函数
 * @returns {Function} 关闭函数
 */
function showToast(message, type = 'info', duration = 3000, onClose = null) {
    const container = getToastContainer();
    const toastId = 'toast-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    
    // 根据类型设置图标和颜色
    const icons = {
        success: { icon: 'fa-check-circle', color: 'bg-green-500', textColor: 'text-green-800', bgColor: 'bg-green-50', borderColor: 'border-green-200' },
        error: { icon: 'fa-exclamation-circle', color: 'bg-red-500', textColor: 'text-red-800', bgColor: 'bg-red-50', borderColor: 'border-red-200' },
        warning: { icon: 'fa-exclamation-triangle', color: 'bg-yellow-500', textColor: 'text-yellow-800', bgColor: 'bg-yellow-50', borderColor: 'border-yellow-200' },
        info: { icon: 'fa-info-circle', color: 'bg-blue-500', textColor: 'text-blue-800', bgColor: 'bg-blue-50', borderColor: 'border-blue-200' }
    };
    
    const config = icons[type] || icons.info;
    
    // 创建Toast元素
    const toast = document.createElement('div');
    toast.id = toastId;
    toast.className = `flex items-center p-4 rounded-lg shadow-lg border ${config.bgColor} ${config.borderColor} border-l-4 ${config.borderColor.replace('border-', 'border-l-')} min-w-[300px] max-w-[500px] transform transition-all duration-300 translate-x-full opacity-0`;
    toast.innerHTML = `
        <div class="flex-shrink-0">
            <i class="fas ${config.icon} ${config.textColor} text-xl"></i>
        </div>
        <div class="ml-3 flex-1">
            <p class="${config.textColor} text-sm font-medium">${escapeHtml(message)}</p>
        </div>
        <button onclick="closeToast('${toastId}')" class="ml-4 flex-shrink-0 ${config.textColor} hover:opacity-70 transition-opacity">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    container.appendChild(toast);
    
    // 触发动画
    setTimeout(() => {
        toast.classList.remove('translate-x-full', 'opacity-0');
        toast.classList.add('translate-x-0', 'opacity-100');
    }, 10);
    
    // 关闭函数
    const closeToastFunc = () => {
        toast.classList.remove('translate-x-0', 'opacity-100');
        toast.classList.add('translate-x-full', 'opacity-0');
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
            if (onClose) {
                onClose();
            }
        }, 300);
    };
    
    // 自动关闭
    let timeoutId = null;
    if (duration > 0) {
        timeoutId = setTimeout(closeToastFunc, duration);
    }
    
    // 保存关闭函数到元素
    toast.closeFunc = closeToastFunc;
    toast.timeoutId = timeoutId;
    
    return closeToastFunc;
}

/**
 * 全局关闭Toast函数（供HTML调用）
 * @param {string} toastId - Toast ID
 */
function closeToast(toastId) {
    const toast = document.getElementById(toastId);
    if (toast && toast.closeFunc) {
        toast.closeFunc();
    }
}

/**
 * 显示成功通知
 * @param {string} message - 消息
 * @param {number} duration - 显示时长
 */
function showSuccess(message, duration = 3000) {
    return showToast(message, 'success', duration);
}

/**
 * 显示错误通知
 * @param {string} message - 消息
 * @param {number} duration - 显示时长，默认5000
 */
function showError(message, duration = 5000) {
    return showToast(message, 'error', duration);
}

/**
 * 显示警告通知
 * @param {string} message - 消息
 * @param {number} duration - 显示时长
 */
function showWarning(message, duration = 4000) {
    return showToast(message, 'warning', duration);
}

/**
 * 显示信息通知
 * @param {string} message - 消息
 * @param {number} duration - 显示时长
 */
function showInfo(message, duration = 3000) {
    return showToast(message, 'info', duration);
}

/**
 * 显示带确认按钮的错误通知
 * @param {string} message - 消息
 * @param {Function} retryCallback - 重试回调函数
 * @param {number} duration - 显示时长，0表示不自动关闭
 */
function showErrorWithRetry(message, retryCallback, duration = 0) {
    const container = getToastContainer();
    const toastId = 'toast-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    
    const toast = document.createElement('div');
    toast.id = toastId;
    toast.className = 'flex items-center p-4 rounded-lg shadow-lg border bg-red-50 border-red-200 border-l-4 border-l-red-500 min-w-[300px] max-w-[500px] transform transition-all duration-300 translate-x-full opacity-0';
    toast.innerHTML = `
        <div class="flex-shrink-0">
            <i class="fas fa-exclamation-circle text-red-800 text-xl"></i>
        </div>
        <div class="ml-3 flex-1">
            <p class="text-red-800 text-sm font-medium">${escapeHtml(message)}</p>
            <div class="mt-2">
                <button onclick="(${retryCallback.toString()})(); closeToast('${toastId}')" 
                        class="px-3 py-1 bg-red-600 text-white text-xs rounded hover:bg-red-700 transition-colors">
                    <i class="fas fa-redo mr-1"></i>重试
                </button>
            </div>
        </div>
        <button onclick="closeToast('${toastId}')" class="ml-4 flex-shrink-0 text-red-800 hover:opacity-70 transition-opacity">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.remove('translate-x-full', 'opacity-0');
        toast.classList.add('translate-x-0', 'opacity-100');
    }, 10);
    
    const closeToastFunc = () => {
        toast.classList.remove('translate-x-0', 'opacity-100');
        toast.classList.add('translate-x-full', 'opacity-0');
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    };
    
    toast.closeFunc = closeToastFunc;
    
    if (duration > 0) {
        setTimeout(closeToastFunc, duration);
    }
    
    return closeToastFunc;
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

// 导出到全局作用域（如果使用模块系统）
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        showToast,
        showSuccess,
        showError,
        showWarning,
        showInfo,
        showErrorWithRetry,
        closeToast
    };
}

