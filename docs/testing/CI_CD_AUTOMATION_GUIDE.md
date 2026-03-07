# RQA2025 CI/CD自动化与测试报告归档指南

## 1. 目标
- 实现全链路测试自动化、持续集成与质量门禁
- 支持分层、分组、专项、性能、极端等测试一键回归
- 自动生成Allure测试报告并归档，便于团队复盘与追溯

## 2. GitHub Actions CI配置
- 配置文件：`.github/workflows/python-ci.yml`
- 支持主分支、dev分支push/PR自动触发
- 步骤：依赖安装 → pytest自动化 → Allure报告生成 → 报告归档

### 典型用法
```yaml
# 见 .github/workflows/python-ci.yml
```

## 3. Allure测试报告生成与归档
- pytest执行时加参数`--alluredir=allure-results`
- 生成报告：`allure generate allure-results --clean -o allure-report`
- CI自动上传`allure-report`目录为产物
- 本地查看：`allure serve allure-results`

## 4. 标签分组与一键回归
- pytest.ini已注册常用标签：unit、integration、performance、extreme、regression、special、mock等
- 典型命令：
  - 只跑性能测试：`python scripts/testing/run_tests.py --all --skip-coverage --pytest-args -m performance`
  - 只跑极端测试：`python scripts/testing/run_tests.py --all --skip-coverage --pytest-args -m extreme`
  - 跑所有集成与性能测试：`python scripts/testing/run_tests.py --all --skip-coverage --pytest-args -m "integration or performance"`

## 5. 常见问题与FAQ
- Q: 如何自定义标签？
  A: 在pytest.ini的[tool:pytest] markers下添加即可。
- Q: CI如何只回归部分测试？
  A: 用`--pytest-args -m <标签>`灵活分组。
- Q: Allure报告打不开？
  A: 本地需安装allure命令行工具，或用CI产物下载后本地serve。

---
如需更多CI/CD集成与自动化建议，请联系DevOps负责人。 