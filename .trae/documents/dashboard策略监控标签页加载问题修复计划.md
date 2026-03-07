# Dashboard策略监控标签页加载问题修复计划

## 问题描述

Dashboard页面点击"策略监控"标签页后，持续显示"正在加载策略监控..."，内容无法正常显示。

## 问题分析

### 可能原因

1. **JavaScript执行错误**：`dashboard-tabs.js`中的`loadTabContent`方法可能存在运行时错误
2. **Promise未正确resolve**：`simulateLoad`模拟加载可能未正确完成
3. **DOM操作失败**：内容替换后未正确显示
4. **CSS样式问题**：`hidden`类未被正确移除，或`active`类未正确添加
5. **事件监听问题**：标签页点击事件未正确触发加载逻辑

### 需要检查的点

1. 浏览器控制台是否有JavaScript错误
2. `loadTabContent`方法是否被正确调用
3. `generateStrategyContent`方法是否返回有效内容
4. DOM元素是否正确获取和更新
5. CSS类是否正确切换

## 修复方案

### Phase 1: 诊断问题

#### 1.1 添加详细日志
在`dashboard-tabs.js`的关键位置添加console.log，追踪执行流程：
- `switchTab`方法入口和出口
- `loadTabContent`方法各步骤
- `generateStrategyContent`方法调用
- DOM操作前后状态

#### 1.2 检查错误处理
确保所有Promise都有catch处理，避免静默失败

### Phase 2: 修复加载逻辑

#### 2.1 简化加载流程
移除复杂的模拟加载，直接使用同步内容生成：
```javascript
async loadTabContent(tabId) {
    const contentContainer = document.getElementById(`tab-content-${tabId}`);
    if (!contentContainer) {
        console.error(`容器未找到: tab-content-${tabId}`);
        return;
    }
    
    console.log(`开始加载标签页: ${tabId}`);
    
    try {
        // 直接生成内容，不使用异步模拟
        const content = this.generateTabContent(tabId);
        console.log(`内容生成完成，长度: ${content.length}`);
        
        // 直接替换内容
        contentContainer.innerHTML = content;
        console.log(`DOM更新完成`);
        
        // 确保可见
        contentContainer.classList.remove('hidden');
        contentContainer.classList.add('active');
        console.log(`样式更新完成`);
        
    } catch (error) {
        console.error(`加载失败:`, error);
        contentContainer.innerHTML = `<div class="tab-error">加载失败: ${error.message}</div>`;
    }
}
```

#### 2.2 修复CSS显示问题
检查并确保以下CSS规则正确：
```css
.tab-content.hidden { display: none; }
.tab-content.active { display: block; }
```

### Phase 3: 验证修复

#### 3.1 本地测试
1. 刷新页面
2. 点击策略监控标签页
3. 观察控制台日志
4. 验证内容是否正确显示

#### 3.2 检查所有标签页
确保其他标签页（交易监控、数据监控等）也有相同问题，一并修复

## 文件变更

### 修改文件
1. `web-static/js/dashboard-tabs.js` - 修复加载逻辑和添加日志

## 时间估算

- Phase 1 (诊断): 30分钟
- Phase 2 (修复): 30分钟
- Phase 3 (验证): 30分钟
- **总计**: 1.5小时
