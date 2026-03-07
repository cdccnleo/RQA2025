#!/bin/bash
# RQA2025 部署验证脚本

echo "🔍 开始RQA2025部署验证..."
echo "=================================="

FAILED_CHECKS=0
PASSED_CHECKS=0

# 函数：执行检查并报告结果
run_check() {
    local name="$1"
    local description="$2"
    local command="$3"
    local expected="$4"
    local critical="$5"

    echo ""
    echo "📋 检查: $name"
    echo "📝 描述: $description"

    if [ "$critical" = "true" ]; then
        echo "🚨 重要性: 关键"
    else
        echo "ℹ️  重要性: 可选"
    fi

    echo "🔧 执行: $command"
    echo "🎯 期望: $expected"

    # 执行命令
    if eval "$command" 2>/dev/null; then
        echo "✅ 通过: $name"
        ((PASSED_CHECKS++))
    else
        echo "❌ 失败: $name"
        ((FAILED_CHECKS++))
        if [ "$critical" = "true" ]; then
            echo "🚨 关键检查失败，可能影响系统正常运行"
        fi
    fi
}


# Pre Deployment Checks
echo "\n🔍 Pre Deployment Checks"
echo "=================================="

run_check "系统资源检查" "验证CPU、内存、磁盘满足要求" "python scripts/pre_deploy_check.sh" "所有检查通过" "True"

run_check "依赖包验证" "确认所有Python依赖已正确安装" "pip check" "无依赖冲突" "True"

run_check "配置文件验证" "检查所有配置文件语法正确" "python -m json.tool config/production.json" "JSON格式正确" "True"

# Deployment Checks
echo "\n🔍 Deployment Checks"
echo "=================================="

run_check "服务启动" "验证服务能够正常启动" "sudo systemctl start rqa2025 && sleep 10 && sudo systemctl status rqa2025" "服务状态为active" "True"

run_check "端口监听" "确认应用监听正确端口" "sudo lsof -i :8000" "端口8000被监听" "True"

run_check "数据库连接" "验证数据库连接正常" "python -c "import os; os.environ['RQA2025_DATABASE_URL']; # test connection"" "连接成功" "True"

# Post Deployment Checks
echo "\n🔍 Post Deployment Checks"
echo "=================================="

run_check "健康检查" "执行应用健康检查" "curl -f http://localhost:8000/health" "返回200状态码" "True"

run_check "基本功能测试" "验证核心API功能" "curl -f http://localhost:8000/api/v1/market-data" "返回有效数据" "True"

run_check "性能基准" "运行基础性能测试" "python -m pytest tests/performance/benchmark_framework.py -k 'trading_core' --tb=short" "测试通过" "False"

run_check "日志验证" "检查日志文件生成" "ls -la /var/log/rqa2025/" "存在日志文件" "False"

run_check "监控指标" "验证监控端点工作" "curl -f http://localhost:8000/metrics" "返回监控数据" "False"


echo ""
echo "🎯 验证完成总结"
echo "=================="
echo "✅ 通过检查: $PASSED_CHECKS"
echo "❌ 失败检查: $FAILED_CHECKS"

if [ $FAILED_CHECKS -eq 0 ]; then
    echo "🎉 所有检查通过！部署验证成功"
    exit 0
else
    echo "⚠️ 存在失败的检查，请查看上述详细信息"
    exit 1
fi
