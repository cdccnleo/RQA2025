/**
 * 页面状态管理器
 * 用于在页面间保持和恢复状态
 */

// 防止重复声明
if (typeof window.PageStateManager === 'undefined') {
    class PageStateManager {
        constructor() {
            this.storageKey = 'page_state_' + window.location.pathname;
            this.state = this.loadState();
        }

        /**
         * 加载状态
         */
        loadState() {
            try {
                const saved = sessionStorage.getItem(this.storageKey);
                return saved ? JSON.parse(saved) : {};
            } catch (e) {
                console.warn('加载页面状态失败:', e);
                return {};
            }
        }

        /**
         * 保存状态
         */
        saveState() {
            try {
                sessionStorage.setItem(this.storageKey, JSON.stringify(this.state));
            } catch (e) {
                console.warn('保存页面状态失败:', e);
            }
        }

        /**
         * 设置状态值
         */
        set(key, value) {
            this.state[key] = value;
            this.saveState();
        }

        /**
         * 获取状态值
         */
        get(key, defaultValue = null) {
            return this.state[key] !== undefined ? this.state[key] : defaultValue;
        }

        /**
         * 删除状态值
         */
        remove(key) {
            delete this.state[key];
            this.saveState();
        }

        /**
         * 清除所有状态
         */
        clear() {
            this.state = {};
            sessionStorage.removeItem(this.storageKey);
        }

        /**
         * 保存表单状态
         */
        saveFormState(formId) {
            const form = document.getElementById(formId);
            if (!form) return;

            const formData = new FormData(form);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = value;
            });

            this.set('form_' + formId, data);
        }

        /**
         * 恢复表单状态
         */
        restoreFormState(formId) {
            const data = this.get('form_' + formId);
            if (!data) return;

            const form = document.getElementById(formId);
            if (!form) return;

            Object.keys(data).forEach(key => {
                const field = form.elements[key];
                if (field) {
                    field.value = data[key];
                }
            });
        }

        /**
         * 保存滚动位置
         */
        saveScrollPosition() {
            this.set('scroll_position', {
                x: window.scrollX,
                y: window.scrollY
            });
        }

        /**
         * 恢复滚动位置
         */
        restoreScrollPosition() {
            const pos = this.get('scroll_position');
            if (pos) {
                window.scrollTo(pos.x, pos.y);
            }
        }

        /**
         * 保存表格状态
         */
        saveTableState(tableId, state) {
            this.set('table_' + tableId, state);
        }

        /**
         * 恢复表格状态
         */
        restoreTableState(tableId) {
            return this.get('table_' + tableId);
        }
    }

    // 将类挂载到 window 对象
    window.PageStateManager = PageStateManager;

    // 创建全局实例
    window.pageState = new PageStateManager();

    // 页面加载时恢复状态
    document.addEventListener('DOMContentLoaded', () => {
        // 恢复滚动位置
        window.pageState.restoreScrollPosition();

        // 监听滚动事件，保存位置
        let scrollTimeout;
        window.addEventListener('scroll', () => {
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => {
                window.pageState.saveScrollPosition();
            }, 100);
        });
    });

    // 页面卸载前保存状态
    window.addEventListener('beforeunload', () => {
        window.pageState.saveScrollPosition();
    });

    // 导出（用于模块系统）
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = { PageStateManager: window.PageStateManager, pageState: window.pageState };
    }
}
