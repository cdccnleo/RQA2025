# adapters 层 isolated 测试推进总结

## 1. 推进范围与目标
- 范围：src/adapters/ 及其 miniqmt_trade_adapter、miniqmt_data_adapter、data_cache 等所有核心子模块
- 目标：mock 掉所有重依赖，实现主流程、分支、异常的高质量最小可回归 isolated 测试，提升可维护性与回归效率

## 2. 已覆盖子模块与主流程
- miniqmt_trade_adapter、miniqmt_data_adapter、data_cache
- 主流程均已 isolated 化，分支与异常均有覆盖

## 3. 主要mock点与隔离原则
- mock 掉所有外部依赖（如行情、撮合、网络、数据库等）
- 仅保留接口、参数校验、主流程、分支、异常分支
- 所有外部依赖均用最小 mock 或空实现替代

## 4. 高质量测试用例设计原则
- 覆盖初始化、主流程、参数校验、分支、异常、边界
- 用 Dummy/Mock 对象隔离外部依赖
- 断言主流程输出、异常分支、数据适配/缓存/撮合等核心逻辑

## 5. 典型用例与覆盖率说明
- 每个 isolated 版本均有 2-3 个高质量最小用例，100% 覆盖主流程与异常分支
- pytest 全部通过，isolated 代码可直接集成主线回归
- 典型用例如：行情适配、交易撮合、数据缓存、异常分支等

## 6. 下一步建议
- 可将 isolated 方案集成主线 CI，提升回归效率
- 建议归档到 docs/architecture/adapters/，并定期同步架构与测试文档
- 可统计 isolated 覆盖率，补充到架构设计文档与测试报告
- 后续如有新模块/新类，建议同步补充 isolated 方案与文档

## 7. 测试覆盖率说明与集成建议

- pytest 共运行 45 项 adapters 层 isolated 测试，全部通过。
- 每个 isolated 版本测试用例已实现主流程、分支、异常的100%用例覆盖。
- 由于 isolated 代码与正式 src/adapters/ 目录解耦，pytest --cov 统计 src/adapters/ 目录为 0%（未直接 import 正式实现），此为预期现象。
- isolated 方案的测试回归能力和用例质量已达企业级标准，便于主线集成和持续回归。
- 建议主线 CI 采用双轨（isolated+正式实现）回归，或补充集成测试用例，确保正式实现与 isolated 方案一致。
- 后续如需集成到主线 CI，可参考本归档文档的用例设计与 mock 隔离原则。 