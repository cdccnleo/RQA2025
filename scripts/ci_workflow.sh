#!/bin/bash

# RQA2025 CI/CD 工作流脚本
# 用于在CI环境中自动执行测试和质量检查

set -e  # 遇到错误立即退出

# 配置变量
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/tests"
export PYTHONPATH

# 日志函数
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $1"
}

# 检查Python环境
check_environment() {
    log_info "检查Python环境..."

    if ! command -v python &> /dev/null; then
        log_error "Python未安装"
        exit 1
    fi

    PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
    log_info "Python版本: ${PYTHON_VERSION}"

    # 检查必需的包
    python -c "import pytest, coverage" || {
        log_error "缺少必需的测试包，请安装: pip install pytest pytest-cov pytest-xdist"
        exit 1
    }

    log_success "环境检查通过"
}

# 安装依赖
install_dependencies() {
    log_info "安装项目依赖..."

    if [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
        pip install -r "${PROJECT_ROOT}/requirements.txt" || {
            log_error "依赖安装失败"
            exit 1
        }
    fi

    # 安装测试依赖
    pip install pytest pytest-cov pytest-xdist pytest-json-report || {
        log_error "测试依赖安装失败"
        exit 1
    }

    log_success "依赖安装完成"
}

# 运行代码质量检查
run_quality_checks() {
    log_info "运行代码质量检查..."

    # 创建报告目录
    mkdir -p "${PROJECT_ROOT}/test_logs"

    # 运行flake8代码质量检查
    if command -v flake8 &> /dev/null; then
        log_info "运行Flake8代码质量检查..."
        flake8 --max-line-length=120 --extend-ignore=E203,W503 "${PROJECT_ROOT}/src" > "${PROJECT_ROOT}/test_logs/flake8_report.txt" 2>&1 || {
            log_warning "Flake8检查发现问题，请查看: ${PROJECT_ROOT}/test_logs/flake8_report.txt"
        }
    else
        log_info "Flake8未安装，跳过代码质量检查"
    fi

    # 运行mypy类型检查
    if command -v mypy &> /dev/null; then
        log_info "运行MyPy类型检查..."
        mypy "${PROJECT_ROOT}/src" > "${PROJECT_ROOT}/test_logs/mypy_report.txt" 2>&1 || {
            log_warning "MyPy检查发现类型问题，请查看: ${PROJECT_ROOT}/test_logs/mypy_report.txt"
        }
    else
        log_info "MyPy未安装，跳过类型检查"
    fi

    log_success "代码质量检查完成"
}

# 运行单元测试
run_unit_tests() {
    log_info "运行单元测试..."

    cd "${PROJECT_ROOT}"

    # 运行CI/CD测试执行器
    python scripts/ci_cd_test_runner.py \
        --parallel \
        --max-workers 4 \
        2>&1 | tee "${PROJECT_ROOT}/test_logs/ci_test_output.log"

    # 检查退出码
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log_error "单元测试失败"
        exit 1
    fi

    log_success "单元测试通过"
}

# 运行集成测试
run_integration_tests() {
    log_info "运行集成测试..."

    cd "${PROJECT_ROOT}"

    # 运行集成测试
    python -m pytest tests/integration/ \
        -v \
        --tb=short \
        --maxfail=3 \
        --json-report \
        --json-report-file="${PROJECT_ROOT}/test_logs/integration_test_report.json" \
        --cov=src \
        --cov-report=json:"${PROJECT_ROOT}/test_logs/integration_coverage.json" \
        2>&1 | tee "${PROJECT_ROOT}/test_logs/integration_test_output.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log_error "集成测试失败"
        exit 1
    fi

    log_success "集成测试通过"
}

# 运行端到端测试
run_e2e_tests() {
    log_info "运行端到端测试..."

    cd "${PROJECT_ROOT}"

    # 只有在CI环境且有特定标记时才运行E2E测试
    if [ "${RUN_E2E_TESTS}" = "true" ]; then
        python -m pytest tests/e2e/ \
            -v \
            --tb=short \
            --maxfail=1 \
            --json-report \
            --json-report-file="${PROJECT_ROOT}/test_logs/e2e_test_report.json" \
            2>&1 | tee "${PROJECT_ROOT}/test_logs/e2e_test_output.log"

        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            log_error "端到端测试失败"
            exit 1
        fi

        log_success "端到端测试通过"
    else
        log_info "跳过端到端测试 (未设置RUN_E2E_TESTS=true)"
    fi
}

# 生成综合报告
generate_reports() {
    log_info "生成综合测试报告..."

    cd "${PROJECT_ROOT}"

    # 运行报告生成脚本
    python scripts/generate_test_report.py || {
        log_warning "报告生成失败，但不影响CI状态"
    }

    log_success "综合报告生成完成"
}

# 检查质量门禁
check_quality_gates() {
    log_info "检查质量门禁..."

    cd "${PROJECT_ROOT}"

    # 使用CI/CD测试执行器的质量门禁检查
    python scripts/ci_cd_test_runner.py --no-coverage | tail -20 | grep -q "部署就绪状态:" || {
        log_error "无法确定部署就绪状态"
        exit 1
    }

    # 检查最后一行是否包含"部署就绪状态: 是"
    if tail -1 "${PROJECT_ROOT}/test_logs/ci_test_output.log" | grep -q "部署就绪状态: 是"; then
        log_success "质量门禁检查通过，系统可以部署"
    else
        log_error "质量门禁检查失败，系统不可部署"
        exit 1
    fi
}

# 清理临时文件
cleanup() {
    log_info "清理临时文件..."

    # 清理pytest缓存
    find "${PROJECT_ROOT}" -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    find "${PROJECT_ROOT}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

    # 清理覆盖率文件
    rm -f "${PROJECT_ROOT}/.coverage" "${PROJECT_ROOT}/.coverage.*"

    log_success "临时文件清理完成"
}

# 主函数
main() {
    local start_time=$(date +%s)

    log_info "开始 RQA2025 CI/CD 工作流"

    # 执行各个阶段
    check_environment
    install_dependencies
    run_quality_checks
    run_unit_tests
    run_integration_tests
    run_e2e_tests
    generate_reports
    check_quality_gates
    cleanup

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_success "CI/CD 工作流执行完成，耗时: ${duration} 秒"

    # 输出最终状态
    echo ""
    echo "🎯 CI/CD 执行结果:"
    echo "   📊 总耗时: ${duration} 秒"
    echo "   ✅ 所有检查通过"
    echo "   🚀 系统可以部署到生产环境"
    echo ""
    echo "📄 详细报告请查看: ${PROJECT_ROOT}/test_logs/"
}

# 参数处理
case "${1:-}" in
    "unit")
        log_info "仅运行单元测试"
        check_environment
        run_unit_tests
        ;;
    "integration")
        log_info "仅运行集成测试"
        check_environment
        install_dependencies
        run_integration_tests
        ;;
    "e2e")
        log_info "仅运行端到端测试"
        check_environment
        install_dependencies
        RUN_E2E_TESTS=true run_e2e_tests
        ;;
    "quality")
        log_info "仅运行质量检查"
        check_environment
        run_quality_checks
        ;;
    *)
        main
        ;;
esac
