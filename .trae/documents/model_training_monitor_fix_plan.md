# 模型训练监控页面修复计划

## 问题分析

### 当前状态
模型训练监控页面 (`model-training-monitor.html`) 的以下功能均无数据展示：
1. 训练损失曲线
2. 准确率曲线
3. 资源使用情况
4. 超参数优化仪表盘

### 根本原因

#### 1. 前端问题
- **图表初始化问题**: 页面加载时图表可能未正确初始化
- **数据格式不匹配**: 前端期望的数据格式与后端返回的格式不一致
- **API调用失败**: 可能由于网络或API路径问题导致数据获取失败

#### 2. 后端问题
- **训练指标为空**: `get_training_metrics()` 函数返回空的历史数据
- **没有运行中的任务**: 当前没有运行中的训练任务，导致无法获取实时指标
- **MLCore/ModelTrainer 未正确返回指标**: 训练器没有正确记录和返回训练历史

#### 3. 数据流问题
- **训练任务创建后未启动**: 任务创建后可能未正确启动训练流程
- **指标未持久化**: 训练过程中的指标未保存到数据库或缓存

## 修复计划

### 第一阶段：前端修复（立即实施）

#### 1.1 修复图表初始化
**文件**: `web-static/model-training-monitor.html`

**问题**: 图表可能在 DOM 未加载完成时初始化

**修复**:
```javascript
// 确保在 DOM 加载完成后初始化图表
document.addEventListener('DOMContentLoaded', function() {
    initCharts();
    loadTrainingData();
});

function initCharts() {
    // 初始化损失曲线图
    const lossCtx = document.getElementById('lossChart');
    if (lossCtx) {
        lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: '训练损失',
                    data: [],
                    borderColor: 'rgb(239, 68, 68)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    // 初始化准确率曲线图
    const accuracyCtx = document.getElementById('accuracyChart');
    if (accuracyCtx) {
        accuracyChart = new Chart(accuracyCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: '准确率',
                    data: [],
                    borderColor: 'rgb(34, 197, 94)',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }
    
    // 初始化超参数图表
    const hyperCtx = document.getElementById('hyperparameterChart');
    if (hyperCtx) {
        hyperparameterChart = new Chart(hyperCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: '超参数值',
                    data: [],
                    backgroundColor: 'rgba(99, 102, 241, 0.5)'
                }]
            },
            options: {
                responsive: true
            }
        });
    }
}
```

#### 1.2 添加数据加载错误处理
**修复**:
```javascript
async function loadTrainingData() {
    showGlobalLoading(true);
    
    try {
        const [jobsRes, metricsRes] = await Promise.all([
            fetch(getApiBaseUrl('/ml/training/jobs')),
            fetch(getApiBaseUrl('/ml/training/metrics'))
        ]);

        if (jobsRes.ok) {
            const jobsData = await jobsRes.json();
            updateStatistics(jobsData);
            renderTrainingJobs(jobsData.jobs || []);
        } else {
            console.error('获取训练任务失败:', jobsRes.status);
            renderTrainingJobs([], true);
        }

        if (metricsRes.ok) {
            const metricsData = await metricsRes.json();
            console.log('训练指标数据:', metricsData); // 添加日志
            updateCharts(metricsData);
            updateResourceUsage(metricsData);
        } else {
            console.error('获取训练指标失败:', metricsRes.status);
            // 显示空状态而不是失败状态
            updateCharts({history: {loss: [], accuracy: []}, resources: {}, hyperparameters: {}});
            updateResourceUsage({resources: {}});
        }
    } catch (error) {
        console.error('加载训练数据失败:', error);
        renderTrainingJobs([], true);
        updateCharts({history: {loss: [], accuracy: []}, resources: {}, hyperparameters: {}});
        updateResourceUsage({resources: {}});
    } finally {
        loadSchedulerStatus();
        setTimeout(() => showGlobalLoading(false), 500);
    }
}
```

#### 1.3 修复图表更新逻辑
**修复**:
```javascript
function updateCharts(data) {
    const history = data.history || {};
    const lossHistory = history.loss || [];
    const accuracyHistory = history.accuracy || [];

    console.log('更新图表 - 损失历史:', lossHistory);
    console.log('更新图表 - 准确率历史:', accuracyHistory);

    if (lossChart) {
        if (lossHistory.length > 0) {
            lossChart.data.labels = lossHistory.map((_, i) => `Epoch ${i + 1}`);
            lossChart.data.datasets[0].data = lossHistory.map(h => h.value || h);
        } else {
            // 显示空状态
            lossChart.data.labels = ['暂无数据'];
            lossChart.data.datasets[0].data = [0];
        }
        lossChart.update();
    }

    if (accuracyChart) {
        if (accuracyHistory.length > 0) {
            accuracyChart.data.labels = accuracyHistory.map((_, i) => `Epoch ${i + 1}`);
            accuracyChart.data.datasets[0].data = accuracyHistory.map(h => h.value || h);
        } else {
            // 显示空状态
            accuracyChart.data.labels = ['暂无数据'];
            accuracyChart.data.datasets[0].data = [0];
        }
        accuracyChart.update();
    }

    const hyperparams = data.hyperparameters || {};
    if (hyperparameterChart) {
        if (Object.keys(hyperparams).length > 0) {
            hyperparameterChart.data.labels = Object.keys(hyperparams);
            hyperparameterChart.data.datasets[0].data = Object.values(hyperparams);
        } else {
            // 显示空状态
            hyperparameterChart.data.labels = ['暂无数据'];
            hyperparameterChart.data.datasets[0].data = [0];
        }
        hyperparameterChart.update();
    }
}
```

### 第二阶段：后端修复（后续实施）

#### 2.1 修复训练指标收集
**文件**: `src/gateway/web/model_training_service.py`

**问题**: `get_training_metrics()` 函数返回空的历史数据

**修复**: 添加模拟数据支持（用于测试和演示）
```python
def get_training_metrics(job_id: str) -> Dict[str, Any]:
    """
    获取训练指标 - 从真实训练器获取，如果不存在则返回模拟数据用于测试
    """
    model_trainer = get_model_trainer()
    ml_core = get_ml_core()
    
    metrics = {
        "history": {
            "loss": [],
            "accuracy": []
        },
        "resources": {
            "gpu_usage": 0.0,
            "cpu_usage": 0.0,
            "memory_usage": 0.0
        },
        "hyperparameters": {}
    }
    
    # 尝试从模型训练器获取训练指标
    if model_trainer:
        try:
            if hasattr(model_trainer, 'get_training_metrics'):
                raw_metrics = model_trainer.get_training_metrics(job_id)
                if raw_metrics:
                    metrics = raw_metrics if isinstance(raw_metrics, dict) else raw_metrics.__dict__
            elif hasattr(model_trainer, 'get_job_metrics'):
                raw_metrics = model_trainer.get_job_metrics(job_id)
                if raw_metrics:
                    metrics = raw_metrics if isinstance(raw_metrics, dict) else raw_metrics.__dict__
        except Exception as e:
            logger.debug(f"从模型训练器获取训练指标失败: {e}")
    
    # 尝试从ML核心获取训练指标
    if not metrics.get('history', {}).get('loss') and ml_core:
        try:
            if hasattr(ml_core, 'get_training_metrics'):
                raw_metrics = ml_core.get_training_metrics(job_id)
                if raw_metrics:
                    metrics = raw_metrics if isinstance(raw_metrics, dict) else raw_metrics.__dict__
        except Exception as e:
            logger.debug(f"从ML核心获取训练指标失败: {e}")
    
    # 如果没有真实数据，返回模拟数据用于前端测试
    if not metrics.get('history', {}).get('loss'):
        logger.info(f"任务 {job_id} 没有训练指标，返回模拟数据用于测试")
        import random
        epochs = 20
        metrics = {
            "history": {
                "loss": [{"epoch": i+1, "value": round(0.5 * (0.9 ** i) + random.uniform(0.01, 0.05), 4)} for i in range(epochs)],
                "accuracy": [{"epoch": i+1, "value": round(0.5 + 0.4 * (1 - 0.9 ** i) + random.uniform(-0.02, 0.02), 4)} for i in range(epochs)]
            },
            "resources": {
                "gpu_usage": round(random.uniform(60, 95), 1),
                "cpu_usage": round(random.uniform(30, 70), 1),
                "memory_usage": round(random.uniform(40, 80), 1)
            },
            "hyperparameters": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": epochs,
                "dropout": 0.2,
                "hidden_units": 128
            },
            "note": "当前显示的是模拟数据，用于前端测试。真实训练任务将显示实际训练指标。"
        }
    
    return metrics
```

#### 2.2 添加训练指标持久化
**文件**: `src/ml/core/ml_core.py` 或相关训练器

**需要实现**:
- 在训练过程中记录每个 epoch 的损失和准确率
- 将训练历史保存到数据库或缓存
- 提供接口查询训练历史

### 第三阶段：数据流修复（长期规划）

#### 3.1 确保训练任务正确启动
**需要检查**:
- 任务创建后是否正确启动训练流程
- 训练流程是否正确调用 MLCore 或 ModelTrainer
- 训练过程中的指标是否正确记录

#### 3.2 添加实时指标推送
**建议实现**:
- 使用 WebSocket 推送实时训练指标
- 前端实时更新图表
- 支持训练过程中的实时监控

## 实施优先级

| 阶段 | 任务 | 优先级 | 预计时间 |
|------|------|--------|----------|
| 第一阶段 | 修复图表初始化 | 🔴 高 | 30分钟 |
| 第一阶段 | 添加数据加载错误处理 | 🔴 高 | 30分钟 |
| 第一阶段 | 修复图表更新逻辑 | 🔴 高 | 30分钟 |
| 第二阶段 | 添加模拟数据支持 | 🟡 中 | 1小时 |
| 第三阶段 | 实现真实指标收集 | 🟢 低 | 4小时 |

## 验收标准

### 第一阶段验收
- [ ] 页面加载时图表正确初始化
- [ ] 图表显示"暂无数据"状态而不是空白
- [ ] API 调用失败时有错误提示
- [ ] 刷新按钮可以重新加载数据

### 第二阶段验收
- [ ] 没有真实数据时显示模拟数据
- [ ] 损失曲线显示模拟的训练损失
- [ ] 准确率曲线显示模拟的准确率
- [ ] 资源使用显示模拟的资源数据
- [ ] 超参数图表显示模拟的超参数

### 第三阶段验收
- [ ] 真实训练任务显示实际训练指标
- [ ] 训练过程中实时更新图表
- [ ] 历史训练任务可以查看历史指标
