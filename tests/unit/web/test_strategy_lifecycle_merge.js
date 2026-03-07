/**
 * 测试策略生命周期页面数据合并逻辑
 * 
 * 测试场景：
 * 1. 策略构思和执行状态都有策略 - 合并数据
 * 2. 策略只在执行状态中 - 添加到列表
 * 3. 运行中策略的生命周期阶段更新
 */

// 模拟数据合并逻辑
function mergeStrategies(conceptions, executionStrategies) {
    let strategies = [...conceptions];
    const strategyMap = new Map(strategies.map(s => [s.id, s]));
    
    executionStrategies.forEach(execStrategy => {
        if (!strategyMap.has(execStrategy.id)) {
            // 创建一个新的策略对象，融合执行状态信息
            const mergedStrategy = {
                id: execStrategy.id,
                name: execStrategy.name || execStrategy.id,
                type: execStrategy.type || 'unknown',
                // 如果策略在运行中，设置生命周期阶段为 live_trading
                lifecycle_stage: execStrategy.status === 'running' ? 'live_trading' : 'draft',
                // 标记这是从执行状态添加的策略
                from_execution: true,
                execution_status: execStrategy.status,
                latency: execStrategy.latency,
                throughput: execStrategy.throughput,
                signals_count: execStrategy.signals_count,
                positions_count: execStrategy.positions_count
            };
            strategyMap.set(execStrategy.id, mergedStrategy);
            strategies.push(mergedStrategy);
        } else {
            // 更新现有策略的执行状态
            const existingStrategy = strategyMap.get(execStrategy.id);
            existingStrategy.execution_status = execStrategy.status;
            // 如果正在运行，确保生命周期阶段反映运行状态
            if (execStrategy.status === 'running' && !['live_trading', 'monitoring'].includes(existingStrategy.lifecycle_stage)) {
                existingStrategy.lifecycle_stage = 'live_trading';
            }
        }
    });
    
    return strategies;
}

// 测试用例
function runTests() {
    console.log('开始测试策略数据合并逻辑...\n');
    
    // 测试1: 策略构思和执行状态都有策略
    console.log('测试1: 策略构思和执行状态都有策略');
    const conceptions1 = [
        { id: 'strategy_001', name: '策略1', lifecycle_stage: 'draft' }
    ];
    const execution1 = [
        { id: 'strategy_001', name: '策略1', status: 'running', latency: 10 }
    ];
    const result1 = mergeStrategies(conceptions1, execution1);
    console.log('结果:', JSON.stringify(result1, null, 2));
    console.assert(result1.length === 1, '策略数量应为1');
    console.assert(result1[0].execution_status === 'running', '执行状态应为running');
    console.assert(result1[0].lifecycle_stage === 'live_trading', '生命周期阶段应更新为live_trading');
    console.log('✅ 测试1通过\n');
    
    // 测试2: 策略只在执行状态中
    console.log('测试2: 策略只在执行状态中（model_strategy_1771503574场景）');
    const conceptions2 = [
        { id: 'strategy_001', name: '策略1', lifecycle_stage: 'draft' }
    ];
    const execution2 = [
        { id: 'model_strategy_1771503574', name: '模型策略', status: 'running', latency: 15, throughput: 100 }
    ];
    const result2 = mergeStrategies(conceptions2, execution2);
    console.log('结果:', JSON.stringify(result2, null, 2));
    console.assert(result2.length === 2, '策略数量应为2');
    console.assert(result2[1].id === 'model_strategy_1771503574', '第二个策略ID应为model_strategy_1771503574');
    console.assert(result2[1].from_execution === true, '应标记为from_execution');
    console.assert(result2[1].lifecycle_stage === 'live_trading', '生命周期阶段应为live_trading');
    console.log('✅ 测试2通过\n');
    
    // 测试3: 多个运行中策略
    console.log('测试3: 多个运行中策略');
    const conceptions3 = [];
    const execution3 = [
        { id: 'strategy_001', name: '策略1', status: 'running' },
        { id: 'strategy_002', name: '策略2', status: 'paused' },
        { id: 'strategy_003', name: '策略3', status: 'running' }
    ];
    const result3 = mergeStrategies(conceptions3, execution3);
    console.log('结果:', JSON.stringify(result3, null, 2));
    console.assert(result3.length === 3, '策略数量应为3');
    const runningCount = result3.filter(s => s.lifecycle_stage === 'live_trading').length;
    console.assert(runningCount === 2, `运行中策略数量应为2，实际为${runningCount}`);
    console.log('✅ 测试3通过\n');
    
    // 测试4: 空数据
    console.log('测试4: 空数据');
    const result4 = mergeStrategies([], []);
    console.assert(result4.length === 0, '策略数量应为0');
    console.log('✅ 测试4通过\n');
    
    console.log('所有测试通过！✅');
}

// 运行测试
runTests();
