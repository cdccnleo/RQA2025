# RQA2025 质量门禁与自动回滚机制指南

## 1. 目标
- 在CI/CD流程中设置关键质量门槛，保障主干分支与生产环境稳定
- 支持自动健康检查与异常自动回滚，提升交付安全性

## 2. 质量门禁配置
- CI流程中已设置如下门槛：
  - 测试通过率必须100%
  - 代码覆盖率不低于80%（`--cov-fail-under=80`）
  - 失败时阻止合并并输出提示
- 相关配置见`.github/workflows/python-ci.yml`：

```yaml
- name: Run all tests with coverage (质量门禁)
  run: |
    pytest --cov=src --cov-report=xml --cov-report=term-missing --cov-fail-under=80 --maxfail=1 --disable-warnings --alluredir=allure-results
```

## 3. 自动回滚机制建议
- 生产/预发环境建议配置自动健康检查（如K8s liveness/readiness probe、接口探活等）
- 检查失败时自动回滚到上一个稳定版本
- 可结合Kubernetes、Argo Rollouts、云平台等实现自动回滚
- 建议关键服务部署脚本支持`rollback`命令

## 4. 质量门禁与回滚常见问题
- Q: 覆盖率不达标如何处理？
  A: 优先补测低覆盖模块，CI通过后方可合并。
- Q: 如何自定义门槛？
  A: 修改CI配置中的`--cov-fail-under`参数即可。
- Q: 自动回滚如何实现？
  A: 结合K8s、云平台、Argo Rollouts等，配置健康检查与回滚策略。
- Q: 如何本地模拟CI门禁？
  A: 本地运行`pytest --cov=src --cov-fail-under=80`即可。

---
如需更多质量门禁与自动回滚建议，请联系DevOps负责人。 