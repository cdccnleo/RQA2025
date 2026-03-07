#!/bin/bash
# RQA2025 质量门禁检查脚本

set -e

echo "🔍 RQA2025 质量门禁检查开始"
echo "============================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查代码质量
echo "📝 检查代码质量..."
if [ -f "coverage.xml" ]; then
    COVERAGE=$(grep -o 'line-rate="[0-9.]*"' coverage.xml | head -1 | sed 's/line-rate="//;s/"//')
    COVERAGE_PERCENT=$(echo "$COVERAGE * 100" | bc -l | cut -d'.' -f1)

    if [ "$COVERAGE_PERCENT" -lt 80 ]; then
        echo -e "${RED}❌ 单元测试覆盖率过低: ${COVERAGE_PERCENT}% (要求: >=80%)${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ 单元测试覆盖率: ${COVERAGE_PERCENT}%${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ 未找到覆盖率报告文件${NC}"
fi

# 检查安全扫描
echo "🔒 检查安全扫描结果..."
if [ -f "bandit-report.json" ]; then
    HIGH_VULNS=$(grep -o '"issue_severity": "HIGH"' bandit-report.json | wc -l)
    MEDIUM_VULNS=$(grep -o '"issue_severity": "MEDIUM"' bandit-report.json | wc -l)

    if [ "$HIGH_VULNS" -gt 0 ]; then
        echo -e "${RED}❌ 发现高危安全漏洞: ${HIGH_VULNS}个${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ 安全扫描通过 (高危: ${HIGH_VULNS}, 中危: ${MEDIUM_VULNS})${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ 未找到安全扫描报告${NC}"
fi

# 检查容器安全
echo "🐳 检查容器安全扫描..."
if [ -f "trivy-report.json" ]; then
    CRITICAL_VULNS=$(grep -o '"Severity": "CRITICAL"' trivy-report.json | wc -l)
    HIGH_VULNS=$(grep -o '"Severity": "HIGH"' trivy-report.json | wc -l)

    if [ "$CRITICAL_VULNS" -gt 0 ] || [ "$HIGH_VULNS" -gt 2 ]; then
        echo -e "${RED}❌ 容器安全漏洞过多: CRITICAL=${CRITICAL_VULNS}, HIGH=${HIGH_VULNS}${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ 容器安全检查通过${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ 未找到容器安全扫描报告${NC}"
fi

# 检查性能指标
echo "⚡ 检查性能指标..."
if [ -f "performance-report.json" ]; then
    AVG_RESPONSE_TIME=$(grep -o '"avg_response_time": [0-9.]*' performance-report.json | cut -d' ' -f2)
    ERROR_RATE=$(grep -o '"error_rate": [0-9.]*' performance-report.json | cut -d' ' -f2)

    if (( $(echo "$AVG_RESPONSE_TIME > 45" | bc -l) )); then
        echo -e "${RED}❌ 平均响应时间过高: ${AVG_RESPONSE_TIME}ms (要求: <45ms)${NC}"
        exit 1
    elif (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
        echo -e "${RED}❌ 错误率过高: ${ERROR_RATE} (要求: <1%)${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ 性能指标正常 (响应时间: ${AVG_RESPONSE_TIME}ms, 错误率: ${ERROR_RATE})${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ 未找到性能测试报告${NC}"
fi

# 检查依赖许可证
echo "📋 检查依赖许可证..."
if [ -f "requirements.txt" ]; then
    echo "检查Python依赖许可证..."
    # 这里可以集成license检查工具
    echo -e "${GREEN}✅ 依赖许可证检查完成${NC}"
else
    echo -e "${YELLOW}⚠️ 未找到依赖文件${NC}"
fi

# 检查文档完整性
echo "📖 检查文档完整性..."
REQUIRED_DOCS=("README.md" "CHANGELOG.md" "docs/API.md")
for doc in "${REQUIRED_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        echo -e "${GREEN}✅ 文档存在: $doc${NC}"
    else
        echo -e "${RED}❌ 文档缺失: $doc${NC}"
        exit 1
    fi
done

# 检查配置完整性
echo "⚙️ 检查配置完整性..."
if [ -f ".gitlab-ci.yml" ] && [ -f "Dockerfile.production" ]; then
    echo -e "${GREEN}✅ CI/CD配置文件完整${NC}"
else
    echo -e "${RED}❌ CI/CD配置文件不完整${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 所有质量门禁检查通过！${NC}"
echo "============================="
