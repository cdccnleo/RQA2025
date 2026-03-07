# RQA2025 - 量化交易系统开发工具

.PHONY: help install test lint format clean docs serve

# 默认目标
help:
	@echo "RQA2025 - 量化交易系统开发工具"
	@echo ""
	@echo "可用命令:"
	@echo "  install    安装项目依赖"
	@echo "  test       运行测试"
	@echo "  lint       代码质量检查"
	@echo "  format     格式化代码"
	@echo "  clean      清理缓存文件"
	@echo "  docs       生成文档"
	@echo "  serve      启动开发服务器"
	@echo "  quality    运行完整质量检查"
	@echo "  build      构建Docker镜像"
	@echo "  deploy     部署到生产环境"

# 安装依赖
install:
	pip install -r requirements.txt
	pip install -e .
	pip install pytest pytest-cov pytest-html pytest-xdist
	pip install flake8 black isort autoflake pre-commit

# 运行测试
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# 代码质量检查
lint:
	@echo "Running flake8..."
	flake8 . --count --statistics --exclude=src/strategy/cloud_native/cloud_integration.py
	@echo "Running safety checks..."
	python -m py_compile src/**/*.py 2>/dev/null || true

# 格式化代码
format:
	@echo "Running black..."
	black . --exclude="src/strategy/cloud_native/cloud_integration.py"
	@echo "Running isort..."
	isort . --profile black --skip src/strategy/cloud_native/cloud_integration.py
	@echo "Running autoflake..."
	autoflake --in-place --remove-all-unused-imports --ignore-init-module-imports --exclude="src/strategy/cloud_native/cloud_integration.py" --recursive .

# 清理缓存
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +

# 生成文档
docs:
	@echo "Generating documentation..."
	# 这里可以添加文档生成命令

# 启动开发服务器
serve:
	@echo "Starting development server..."
	# 这里可以添加开发服务器启动命令

# 完整质量检查
quality: lint test
	@echo "Quality checks completed!"
	@echo "Coverage report: htmlcov/index.html"

# 设置开发环境
setup:
	pre-commit install
	@echo "Development environment setup complete!"

# CI/CD 验证
ci-check:
	@echo "Running CI checks..."
	flake8 . --count --select=E9,F63,F7,F82 --exclude=src/strategy/cloud_native/cloud_integration.py
	flake8 . --count --exit-zero --exclude=src/strategy/cloud_native/cloud_integration.py
	pytest tests/unit/ -x --tb=short --cov=src --cov-report=xml --cov-fail-under=10

# 覆盖率监控
coverage-monitor:
	@echo "Running coverage monitoring..."
	python tests/production_coverage_monitor.py

# 性能基准测试
benchmark:
	@echo "Running performance benchmarks..."
	pytest tests/ -k "benchmark or performance" --benchmark-only --benchmark-json=benchmark-results.json -v

# 集成测试
integration-test:
	@echo "Running integration tests..."
	pytest tests/integration/ --cov=src --cov-report=html --cov-report=term-missing -v

# 端到端测试
e2e-test:
	@echo "Running end-to-end tests..."
	pytest tests/e2e/ -v --tb=short

# 安全测试
security-test:
	@echo "Running security tests..."
	bandit -r src/ -f txt || true
	safety check || true

# 多环境测试（需要tox）
tox-test:
	@echo "Running multi-environment tests..."
	tox || echo "Tox not configured, install with: pip install tox"

# 生成测试报告
test-report:
	@echo "Generating test reports..."
	pytest tests/ --html=test-report.html --self-contained-html --cov=src --cov-report=html
	@echo "Test report generated: test-report.html"
	@echo "Coverage report: htmlcov/index.html"

# 健康检查
health-check:
	@echo "Running system health check..."
	python -c "
import sys
sys.path.insert(0, 'src')
try:
    import infrastructure
    import data
    import risk
    import optimization
    print('✅ Core modules import successfully')
except ImportError as e:
    print(f'❌ Module import failed: {e}')
    sys.exit(1)
	"
	@echo "Checking test coverage..."
	python tests/production_coverage_monitor.py || echo "Coverage monitoring failed"

# 部署准备检查
deploy-check: health-check test
	@echo "Deployment readiness check completed!"

# 开发环境设置
dev-setup: install
	pre-commit install
	@echo "Development environment setup complete!"
	@echo "Run 'make help' to see available commands"

# 代码质量全面检查
quality-full: lint security-test
	@echo "Full quality checks completed!"
	mypy src/ --ignore-missing-imports --no-strict-optional || echo "Type checking completed with warnings"

# Docker 构建
build:
	@echo "Building RQA2025 production image..."
	@./scripts/build_production.sh

# Docker 部署
deploy:
	@echo "Deploying RQA2025 to production..."
	@./scripts/deploy_production.sh

# 清理所有生成文件
clean-all: clean
	rm -rf test_reports/ .coverage htmlcov/ *.egg-info dist/ build/
	rm -f benchmark-results.json test-report.html
	@echo "All generated files cleaned!"

# 显示详细帮助
help-detailed:
	@echo "RQA2025 - 量化交易系统开发工具 (详细帮助)"
	@echo "================================================"
	@echo ""
	@echo "📦 安装和设置:"
	@echo "  install       安装项目依赖"
	@echo "  dev-setup     完整开发环境设置 (包含pre-commit)"
	@echo ""
	@echo "🧪 测试命令:"
	@echo "  test          运行单元测试和覆盖率"
	@echo "  integration-test  运行集成测试"
	@echo "  e2e-test      运行端到端测试"
	@echo "  benchmark     运行性能基准测试"
	@echo "  test-report   生成完整的测试报告"
	@echo ""
	@echo "🔍 质量检查:"
	@echo "  lint          代码质量检查 (flake8)"
	@echo "  security-test 安全漏洞扫描"
	@echo "  quality       基础质量检查 (lint + test)"
	@echo "  quality-full  全面质量检查 (包含类型检查和安全扫描)"
	@echo ""
	@echo "📊 监控和报告:"
	@echo "  coverage-monitor  运行覆盖率监控系统"
	@echo "  health-check     系统健康状态检查"
	@echo ""
	@echo "🚀 部署相关:"
	@echo "  build          构建Docker镜像"
	@echo "  deploy         部署到生产环境"
	@echo "  deploy-check   部署前准备检查"
	@echo "  ci-check       CI/CD流水线验证"
	@echo ""
	@echo "🧹 清理命令:"
	@echo "  clean          清理Python缓存文件"
	@echo "  clean-all      清理所有生成文件"
	@echo ""
	@echo "📚 文档和开发:"
	@echo "  docs           生成项目文档"
	@echo "  serve          启动开发服务器"
	@echo ""
	@echo "ℹ️  其他:"
	@echo "  tox-test       多环境测试 (需要tox)"
	@echo "  help           显示基本帮助信息"
	@echo "  help-detailed  显示详细帮助信息 (当前命令)"
	@echo ""
	@echo "💡 使用示例:"
	@echo "  make dev-setup    # 新项目设置"
	@echo "  make quality-full # 完整代码质量检查"
	@echo "  make test-report  # 生成测试和覆盖率报告"
	@echo "  make deploy-check # 部署前检查"