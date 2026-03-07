# 模型管理仪表盘实施计划

## 概述

在模型训练监控页面（model-training-monitor.html）中增加模型管理仪表盘，提供对已训练模型的全面管理功能。

## 现有功能分析

### 后端API（已存在）
- `GET /api/v1/models` - 列出所有模型
- `GET /api/v1/models/{model_id}` - 获取模型详情
- `POST /api/v1/models/{model_id}/load` - 加载模型
- `DELETE /api/v1/models/{model_id}` - 删除模型（软删除）
- `POST /api/v1/models/{model_id}/deploy` - 部署模型

### 前端现状
- 当前页面主要展示训练任务列表和训练监控图表
- 缺少模型管理功能区域
- 已有重新训练功能

## 设计方案

### 1. 页面布局调整

在现有页面中增加"模型管理"标签页，与"训练监控"并列：

```
┌─────────────────────────────────────────────────────────────┐
│  RQA2025                                    [刷新]          │
├─────────────────────────────────────────────────────────────┤
│  模型训练监控                                                │
│  训练任务监控、进度跟踪和模型性能分析                          │
├─────────────────────────────────────────────────────────────┤
│  [训练监控]  [模型管理]  [系统状态]                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  模型管理仪表盘内容区域                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. 模型管理仪表盘功能模块

#### 2.1 模型统计卡片
- 总模型数
- 已部署模型数
- 活跃模型数
- 平均准确率

#### 2.2 模型列表表格
展示所有已训练模型的信息：
- 模型ID
- 模型名称
- 模型类型（LSTM、GRU、Transformer等）
- 准确率
- 损失值
- 训练时间
- 状态（active/archived/deleted）
- 是否已部署
- 操作按钮

#### 2.3 模型操作功能
- **查看详情** - 显示模型完整信息（超参数、特征列、训练配置等）
- **部署/取消部署** - 切换模型部署状态
- **删除** - 软删除模型
- **加载测试** - 测试模型加载

#### 2.4 模型详情弹窗
点击模型行显示详细信息：
- 基本信息（ID、名称、类型、版本）
- 性能指标（准确率、损失、精确率、召回率等）
- 训练信息（训练时间、轮数、样本数）
- 超参数配置
- 特征列信息
- 模型路径

### 3. UI设计细节

#### 3.1 标签页切换
```html
<div class="border-b border-gray-200 mb-6">
    <nav class="-mb-px flex space-x-8">
        <button onclick="switchTab('training')" class="tab-btn active">
            <i class="fas fa-chart-line mr-2"></i>训练监控
        </button>
        <button onclick="switchTab('models')" class="tab-btn">
            <i class="fas fa-cube mr-2"></i>模型管理
        </button>
    </nav>
</div>
```

#### 3.2 模型统计卡片
```html
<div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
    <!-- 总模型数 -->
    <div class="bg-white rounded-lg shadow p-6">
        <i class="fas fa-cubes text-blue-500 text-2xl"></i>
        <div class="ml-4">
            <dt class="text-sm font-medium text-gray-500">总模型数</dt>
            <dd class="text-2xl font-semibold" id="total-models">0</dd>
        </div>
    </div>
    
    <!-- 已部署模型 -->
    <div class="bg-white rounded-lg shadow p-6">
        <i class="fas fa-rocket text-green-500 text-2xl"></i>
        <div class="ml-4">
            <dt class="text-sm font-medium text-gray-500">已部署</dt>
            <dd class="text-2xl font-semibold" id="deployed-models">0</dd>
        </div>
    </div>
    
    <!-- 活跃模型 -->
    <div class="bg-white rounded-lg shadow p-6">
        <i class="fas fa-check-circle text-indigo-500 text-2xl"></i>
        <div class="ml-4">
            <dt class="text-sm font-medium text-gray-500">活跃模型</dt>
            <dd class="text-2xl font-semibold" id="active-models">0</dd>
        </div>
    </div>
    
    <!-- 平均准确率 -->
    <div class="bg-white rounded-lg shadow p-6">
        <i class="fas fa-bullseye text-purple-500 text-2xl"></i>
        <div class="ml-4">
            <dt class="text-sm font-medium text-gray-500">平均准确率</dt>
            <dd class="text-2xl font-semibold" id="avg-model-accuracy">--%</dd>
        </div>
    </div>
</div>
```

#### 3.3 模型列表表格
```html
<div class="bg-white rounded-lg shadow overflow-hidden">
    <div class="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
        <h3 class="text-lg font-semibold text-gray-900">模型列表</h3>
        <div class="flex space-x-2">
            <select id="modelFilter" class="border rounded-lg px-3 py-2">
                <option value="all">全部状态</option>
                <option value="active">活跃</option>
                <option value="archived">已归档</option>
                <option value="deployed">已部署</option>
            </select>
            <button onclick="loadModels()" class="bg-indigo-600 text-white px-4 py-2 rounded-lg">
                <i class="fas fa-sync-alt"></i>
            </button>
        </div>
    </div>
    
    <table class="min-w-full divide-y divide-gray-200">
        <thead class="bg-gray-50">
            <tr>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">模型ID</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">模型名称</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">类型</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">准确率</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">状态</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">部署状态</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">训练时间</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">操作</th>
            </tr>
        </thead>
        <tbody id="modelsTableBody" class="bg-white divide-y divide-gray-200">
            <!-- 动态填充 -->
        </tbody>
    </table>
</div>
```

#### 3.4 模型详情弹窗
```html
<div id="modelDetailModal" class="fixed inset-0 bg-gray-600 bg-opacity-50 hidden overflow-y-auto">
    <div class="flex items-center justify-center min-h-screen px-4">
        <div class="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-screen overflow-y-auto">
            <!-- 弹窗头部 -->
            <div class="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
                <h3 class="text-xl font-semibold text-gray-900">模型详情</h3>
                <button onclick="closeModelDetail()" class="text-gray-400 hover:text-gray-600">
                    <i class="fas fa-times text-xl"></i>
                </button>
            </div>
            
            <!-- 弹窗内容 -->
            <div class="p-6">
                <!-- 基本信息 -->
                <div class="mb-6">
                    <h4 class="text-lg font-medium text-gray-900 mb-3">基本信息</h4>
                    <div class="grid grid-cols-2 gap-4 bg-gray-50 p-4 rounded-lg">
                        <div><span class="text-gray-500">模型ID:</span> <span id="detail-model-id"></span></div>
                        <div><span class="text-gray-500">模型名称:</span> <span id="detail-model-name"></span></div>
                        <div><span class="text-gray-500">模型类型:</span> <span id="detail-model-type"></span></div>
                        <div><span class="text-gray-500">版本:</span> <span id="detail-model-version"></span></div>
                    </div>
                </div>
                
                <!-- 性能指标 -->
                <div class="mb-6">
                    <h4 class="text-lg font-medium text-gray-900 mb-3">性能指标</h4>
                    <div class="grid grid-cols-3 gap-4">
                        <div class="bg-blue-50 p-4 rounded-lg text-center">
                            <div class="text-2xl font-bold text-blue-600" id="detail-accuracy"></div>
                            <div class="text-sm text-gray-600">准确率</div>
                        </div>
                        <div class="bg-green-50 p-4 rounded-lg text-center">
                            <div class="text-2xl font-bold text-green-600" id="detail-precision"></div>
                            <div class="text-sm text-gray-600">精确率</div>
                        </div>
                        <div class="bg-purple-50 p-4 rounded-lg text-center">
                            <div class="text-2xl font-bold text-purple-600" id="detail-recall"></div>
                            <div class="text-sm text-gray-600">召回率</div>
                        </div>
                    </div>
                </div>
                
                <!-- 超参数 -->
                <div class="mb-6">
                    <h4 class="text-lg font-medium text-gray-900 mb-3">超参数配置</h4>
                    <pre id="detail-hyperparameters" class="bg-gray-50 p-4 rounded-lg text-sm overflow-x-auto"></pre>
                </div>
                
                <!-- 特征列 -->
                <div class="mb-6">
                    <h4 class="text-lg font-medium text-gray-900 mb-3">特征列</h4>
                    <div id="detail-feature-columns" class="flex flex-wrap gap-2"></div>
                </div>
            </div>
            
            <!-- 弹窗底部 -->
            <div class="px-6 py-4 border-t border-gray-200 flex justify-end space-x-3">
                <button onclick="closeModelDetail()" class="px-4 py-2 border rounded-lg text-gray-700 hover:bg-gray-50">
                    关闭
                </button>
                <button onclick="deployCurrentModel()" id="btn-deploy" class="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700">
                    部署模型
                </button>
            </div>
        </div>
    </div>
</div>
```

### 4. JavaScript功能实现

#### 4.1 标签页切换
```javascript
function switchTab(tab) {
    // 更新按钮样式
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active', 'border-indigo-500', 'text-indigo-600');
        btn.classList.add('border-transparent', 'text-gray-500');
    });
    event.target.classList.add('active', 'border-indigo-500', 'text-indigo-600');
    event.target.classList.remove('border-transparent', 'text-gray-500');
    
    // 显示/隐藏内容区域
    if (tab === 'training') {
        document.getElementById('trainingSection').classList.remove('hidden');
        document.getElementById('modelsSection').classList.add('hidden');
    } else if (tab === 'models') {
        document.getElementById('trainingSection').classList.add('hidden');
        document.getElementById('modelsSection').classList.remove('hidden');
        loadModels(); // 加载模型数据
    }
}
```

#### 4.2 加载模型列表
```javascript
async function loadModels() {
    try {
        showGlobalLoading(true);
        
        const response = await fetch(getApiBaseUrl('/api/v1/models'));
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            renderModels(data.models);
            updateModelStats(data.models);
        } else {
            throw new Error(data.detail || '加载模型失败');
        }
    } catch (error) {
        console.error('加载模型列表失败:', error);
        alert(`加载模型列表失败: ${error.message}`);
    } finally {
        showGlobalLoading(false);
    }
}
```

#### 4.3 渲染模型列表
```javascript
function renderModels(models) {
    const tbody = document.getElementById('modelsTableBody');
    
    if (!models || models.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" class="px-6 py-8 text-center text-gray-500">
                    <i class="fas fa-inbox text-2xl mb-2"></i>
                    <div>暂无模型</div>
                </td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = models.map(model => {
        const statusColor = {
            'active': 'bg-green-100 text-green-800',
            'archived': 'bg-gray-100 text-gray-800',
            'deleted': 'bg-red-100 text-red-800'
        }[model.status] || 'bg-gray-100 text-gray-800';
        
        const statusText = {
            'active': '活跃',
            'archived': '已归档',
            'deleted': '已删除'
        }[model.status] || model.status;
        
        return `
            <tr class="hover:bg-gray-50">
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    ${model.model_id}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    ${model.model_name}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    ${model.model_type}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    ${model.accuracy ? (model.accuracy * 100).toFixed(2) + '%' : '--'}
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${statusColor}">
                        ${statusText}
                    </span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    ${model.is_deployed ? 
                        '<span class="text-green-600"><i class="fas fa-check-circle mr-1"></i>已部署</span>' : 
                        '<span class="text-gray-400"><i class="fas fa-circle mr-1"></i>未部署</span>'}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    ${model.trained_at ? new Date(model.trained_at).toLocaleString('zh-CN') : '--'}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <button onclick="viewModelDetail('${model.model_id}')" class="text-blue-600 hover:text-blue-900 mr-3" title="查看详情">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button onclick="toggleDeployModel('${model.model_id}', ${!model.is_deployed})" class="${model.is_deployed ? 'text-orange-600 hover:text-orange-900' : 'text-green-600 hover:text-green-900'} mr-3" title="${model.is_deployed ? '取消部署' : '部署'}">
                        <i class="fas fa-${model.is_deployed ? 'pause' : 'rocket'}"></i>
                    </button>
                    <button onclick="deleteModel('${model.model_id}')" class="text-red-600 hover:text-red-900" title="删除">
                        <i class="fas fa-trash"></i>
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}
```

#### 4.4 查看模型详情
```javascript
async function viewModelDetail(modelId) {
    try {
        showGlobalLoading(true);
        
        const response = await fetch(getApiBaseUrl(`/api/v1/models/${modelId}`));
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            const model = data.model;
            
            // 填充详情信息
            document.getElementById('detail-model-id').textContent = model.model_id;
            document.getElementById('detail-model-name').textContent = model.model_name;
            document.getElementById('detail-model-type').textContent = model.model_type;
            document.getElementById('detail-model-version').textContent = model.model_version;
            document.getElementById('detail-accuracy').textContent = model.accuracy ? (model.accuracy * 100).toFixed(2) + '%' : '--';
            document.getElementById('detail-precision').textContent = model.precision ? (model.precision * 100).toFixed(2) + '%' : '--';
            document.getElementById('detail-recall').textContent = model.recall ? (model.recall * 100).toFixed(2) + '%' : '--';
            document.getElementById('detail-hyperparameters').textContent = JSON.stringify(model.hyperparameters, null, 2);
            
            // 特征列标签
            const featureColumnsDiv = document.getElementById('detail-feature-columns');
            if (model.feature_columns && model.feature_columns.length > 0) {
                featureColumnsDiv.innerHTML = model.feature_columns.map(col => 
                    `<span class="px-2 py-1 bg-indigo-100 text-indigo-800 text-xs rounded">${col}</span>`
                ).join('');
            } else {
                featureColumnsDiv.innerHTML = '<span class="text-gray-400">无特征列信息</span>';
            }
            
            // 更新部署按钮
            const deployBtn = document.getElementById('btn-deploy');
            if (model.is_deployed) {
                deployBtn.textContent = '取消部署';
                deployBtn.classList.remove('bg-green-600', 'hover:bg-green-700');
                deployBtn.classList.add('bg-orange-600', 'hover:bg-orange-700');
            } else {
                deployBtn.textContent = '部署模型';
                deployBtn.classList.remove('bg-orange-600', 'hover:bg-orange-700');
                deployBtn.classList.add('bg-green-600', 'hover:bg-green-700');
            }
            deployBtn.onclick = () => toggleDeployModel(modelId, !model.is_deployed);
            
            // 显示弹窗
            document.getElementById('modelDetailModal').classList.remove('hidden');
        } else {
            throw new Error(data.detail || '获取模型详情失败');
        }
    } catch (error) {
        console.error('获取模型详情失败:', error);
        alert(`获取模型详情失败: ${error.message}`);
    } finally {
        showGlobalLoading(false);
    }
}
```

#### 4.5 部署/取消部署模型
```javascript
async function toggleDeployModel(modelId, deploy) {
    const action = deploy ? '部署' : '取消部署';
    if (!confirm(`确定要${action}此模型吗？`)) return;
    
    try {
        showGlobalLoading(true);
        
        const response = await fetch(getApiBaseUrl(`/api/v1/models/${modelId}/deploy`), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            alert(`${action}成功`);
            await loadModels(); // 刷新列表
            
            // 如果弹窗打开，更新按钮状态
            const modal = document.getElementById('modelDetailModal');
            if (!modal.classList.contains('hidden')) {
                closeModelDetail();
            }
        } else {
            throw new Error(data.detail || `${action}失败`);
        }
    } catch (error) {
        console.error(`${action}模型失败:`, error);
        alert(`${action}模型失败: ${error.message}`);
    } finally {
        showGlobalLoading(false);
    }
}
```

#### 4.6 删除模型
```javascript
async function deleteModel(modelId) {
    if (!confirm('确定要删除此模型吗？此操作将软删除模型，数据仍可恢复。')) return;
    
    try {
        showGlobalLoading(true);
        
        const response = await fetch(getApiBaseUrl(`/api/v1/models/${modelId}`), {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            alert('模型已删除');
            await loadModels(); // 刷新列表
        } else {
            throw new Error(data.detail || '删除失败');
        }
    } catch (error) {
        console.error('删除模型失败:', error);
        alert(`删除模型失败: ${error.message}`);
    } finally {
        showGlobalLoading(false);
    }
}
```

## 实施步骤

### 第一阶段：页面结构调整（30分钟）
1. 在页面中添加标签页导航（训练监控/模型管理）
2. 将现有训练监控内容包裹在训练监控标签页中
3. 创建模型管理标签页容器

### 第二阶段：模型统计卡片（20分钟）
1. 添加模型统计卡片HTML
2. 实现 `updateModelStats()` 函数
3. 添加统计数据的计算逻辑

### 第三阶段：模型列表表格（30分钟）
1. 添加模型列表表格HTML
2. 实现 `loadModels()` 函数
3. 实现 `renderModels()` 函数
4. 添加筛选功能

### 第四阶段：模型详情弹窗（30分钟）
1. 添加模型详情弹窗HTML
2. 实现 `viewModelDetail()` 函数
3. 实现 `closeModelDetail()` 函数
4. 添加弹窗内容填充逻辑

### 第五阶段：模型操作功能（30分钟）
1. 实现 `toggleDeployModel()` 函数
2. 实现 `deleteModel()` 函数
3. 添加操作确认对话框
4. 添加操作成功/失败提示

### 第六阶段：样式优化（20分钟）
1. 优化标签页切换样式
2. 优化表格样式
3. 优化弹窗样式
4. 添加响应式布局支持

## 验收标准

### 功能验收
- [ ] 标签页可以正常切换
- [ ] 模型列表正确显示所有模型
- [ ] 模型统计卡片数据正确
- [ ] 可以查看模型详情
- [ ] 可以部署/取消部署模型
- [ ] 可以删除模型
- [ ] 操作后有成功提示

### 性能验收
- [ ] 模型列表加载时间 < 2秒
- [ ] 模型详情加载时间 < 1秒
- [ ] 操作响应时间 < 1秒

### 兼容性验收
- [ ] 在Chrome浏览器正常显示
- [ ] 在Firefox浏览器正常显示
- [ ] 移动端适配正常

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| API返回数据格式不一致 | 页面显示错误 | 添加数据格式验证和错误处理 |
| 模型数量过多导致加载慢 | 用户体验差 | 添加分页或虚拟滚动 |
| 弹窗内容过长 | 显示不完整 | 添加滚动条和内容折叠 |
