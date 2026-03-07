# 并行测试策略文档

## 📋 策略概述

为了提高测试执行效率，除基础设施层外，其他各层级建议启用并行测试。

## 🎯 测试策略

### 基础设施层（不使用并行测试）
- **原因**: 基础设施层测试可能存在资源竞争和状态共享问题，并行执行可能导致不稳定
- **执行方式**: `pytest tests/unit/infrastructure/ -n 0` 或直接运行

### 其他所有层级（使用并行测试）
- **执行方式**: `pytest tests/unit/<layer>/ -n auto`
- **优势**: 
  - 显著减少测试执行时间
  - 提高CI/CD流水线效率
  - 充分利用多核CPU资源

## 📊 各层级并行测试配置

### 核心子系统（8个）

1. **核心服务层**
   ```bash
   pytest --cov=src/core --cov-report=term-missing -k "not e2e" tests/unit/core/ -n auto
   ```

2. **数据管理层**
   ```bash
   pytest --cov=src/data --cov-report=term-missing -k "not e2e" tests/unit/data/ -n auto
   ```

3. **特征分析层**
   ```bash
   pytest --cov=src/features --cov-report=term-missing -k "not e2e" tests/unit/features/ -n auto
   ```

4. **机器学习层**
   ```bash
   pytest --cov=src/ml --cov-report=term-missing -k "not e2e" tests/unit/ml/ -n auto
   ```

5. **策略服务层**
   ```bash
   pytest --cov=src/strategy --cov-report=term-missing -k "not e2e" tests/unit/strategy/ -n auto
   ```

6. **交易层**
   ```bash
   pytest --cov=src/trading --cov-report=term-missing -k "not e2e" tests/unit/trading/ -n auto
   ```

7. **风险控制层**
   ```bash
   pytest --cov=src/risk --cov-report=term-missing -k "not e2e" tests/unit/risk/ -n auto
   ```

### 辅助支撑层（13个）

8. **监控层**
   ```bash
   pytest --cov=src/monitoring --cov-report=term-missing -k "not e2e" tests/unit/monitoring/ -n auto
   ```

9. **流处理层**
   ```bash
   pytest --cov=src/streaming --cov-report=term-missing -k "not e2e" tests/unit/streaming/ -n auto
   ```

10. **网关层**
    ```bash
    pytest --cov=src/gateway --cov-report=term-missing -k "not e2e" tests/unit/gateway/ -n auto
    ```

11. **优化层**
    ```bash
    pytest --cov=src/optimization --cov-report=term-missing -k "not e2e" tests/unit/optimization/ -n auto
    ```

12. **适配器层**
    ```bash
    pytest --cov=src/adapters --cov-report=term-missing -k "not e2e" tests/unit/adapters/ -n auto
    ```

13. **自动化层**
    ```bash
    pytest --cov=src/automation --cov-report=term-missing -k "not e2e" tests/unit/automation/ -n auto
    ```

14. **弹性层**
    ```bash
    pytest --cov=src/resilience --cov-report=term-missing -k "not e2e" tests/unit/resilience/ -n auto
    ```

15. **测试层**
    ```bash
    pytest --cov=src/testing --cov-report=term-missing -k "not e2e" tests/unit/testing/ -n auto
    ```

16. **工具层**
    ```bash
    pytest --cov=src/utils --cov-report=term-missing -k "not e2e" tests/unit/utils/ -n auto
    ```

17. **分布式协调器层**
    ```bash
    pytest --cov=src/coordinator --cov-report=term-missing -k "not e2e" tests/unit/coordinator/ -n auto
    ```

18. **异步处理器层**
    ```bash
    pytest --cov=src/async --cov-report=term-missing -k "not e2e" tests/unit/async/ -n auto
    ```

19. **移动端层**
    ```bash
    pytest --cov=src/mobile --cov-report=term-missing -k "not e2e" tests/unit/mobile/ -n auto
    ```

20. **业务边界层**
    ```bash
    pytest --cov=src/boundary --cov-report=term-missing -k "not e2e" tests/unit/boundary/ -n auto
    ```

## ⚙️ 并行测试配置说明

### pytest-xdist 参数
- `-n auto`: 自动检测CPU核心数并创建相应数量的worker
- `-n 4`: 固定使用4个worker（如果已知最佳worker数量）
- `-n 0`: 禁用并行测试（单进程执行）

### 推荐配置
- **开发环境**: `-n auto`（自动检测）
- **CI/CD环境**: `-n 4`（固定worker数量，更可预测）
- **调试模式**: `-n 0`（单进程，便于调试）

## 📈 性能提升效果

根据实际测试，使用并行测试可以：
- **减少执行时间**: 50-80%（取决于测试数量和CPU核心数）
- **提高资源利用率**: 充分利用多核CPU
- **加快反馈速度**: 更快获得测试结果

## ⚠️ 注意事项

1. **测试隔离**: 确保测试之间没有共享状态
2. **资源竞争**: 注意数据库、文件系统等资源的并发访问
3. **随机性**: 某些测试可能因为执行顺序不同而表现不同
4. **调试**: 遇到问题时，使用`-n 0`单进程执行以便调试

## 🔧 故障排查

如果并行测试出现问题：
1. 先使用`-n 0`验证测试本身是否正确
2. 检查测试是否有共享状态或资源竞争
3. 查看pytest-xdist的日志输出
4. 考虑使用`--dist=loadscope`来改善测试分发策略

---
**创建时间**: 2025-01-28  
**策略版本**: v1.0  
**适用范围**: 除基础设施层外的所有层级

