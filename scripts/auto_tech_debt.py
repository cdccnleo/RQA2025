import re

# 1. 解析结构文档，获取模块列表


def extract_modules(structure_files):
    modules = set()
    for file in structure_files:
        with open(file, encoding='utf-8') as f:
            for line in f:
                # 假设模块名以“* ”、“- ”、“### ”等开头
                match = re.match(r'^[\*\-\#]{1,3}\s*([\u4e00-\u9fa5A-Za-z0-9_\- ]+)', line)
                if match:
                    modules.add(match.group(1).strip())
    return modules

# 2. 解析覆盖率报告，获取低覆盖模块


def extract_low_coverage_modules(report_files, threshold=80):
    low_coverage = {}
    for file in report_files:
        with open(file, encoding='utf-8') as f:
            for line in f:
                # 假设格式为“模块名 覆盖率%”
                match = re.match(r'(.+?)\s+(\d+)%', line)
                if match:
                    module, coverage = match.group(1).strip(), int(match.group(2))
                    if coverage < threshold:
                        low_coverage[module] = coverage
    return low_coverage

# 3. 写入 technical_debt.md


def append_to_technical_debt(debt_file, low_coverage_modules):
    with open(debt_file, 'a', encoding='utf-8') as f:
        for module, coverage in low_coverage_modules.items():
            f.write(f"\n### [{module}]\n")
            f.write(f"- 问题描述：测试覆盖率仅{coverage}%，需补充测试\n")
            f.write(f"- 优先级：高\n")
            f.write(f"- 发现日期：2024-06-XX\n")

# 技术债务清单（自动生成/更新）
# 更新时间：2025-07-14
#
# 1. 数据版本管理的Series标签一致性与异常处理需完善
# 2. 熔断器CollectorRegistry重复注册问题需修复
# 3. SmartOrderRetry相关依赖未定义或mock不全
# 4. 合规与监管相关接口未实现或未补全
# 5. 集成测试环境依赖不全，部分mock/依赖未隔离
# 6. 性能测试用例需适配高并发与缓存场景
# 7. 部分断言与业务逻辑不符，需review并修正
# 8. sqlite3等外部依赖需mock或环境隔离
# 9. Mock对象未正确调用或未按预期断言
# 10. pandas等依赖API升级适配问题
# 11. 部分测试用例未补全异常捕获与兜底逻辑
# 12. 合规接口、数字签名、报告生成等功能需补全实现

# TODO: 数据版本管理的Series标签一致性与异常处理
# TODO: 熔断器CollectorRegistry重复注册修复
# TODO: SmartOrderRetry依赖与mock补全
# TODO: 合规与监管接口实现与补全
# TODO: 集成测试环境依赖与mock隔离
# TODO: 性能测试用例高并发与缓存适配
# TODO: 断言与业务逻辑review修正
# TODO: sqlite3等外部依赖mock/隔离
# TODO: Mock对象调用与断言修正
# TODO: pandas等依赖API升级适配
# TODO: 测试用例异常捕获与兜底
# TODO: 合规接口、数字签名、报告生成实现


if __name__ == "__main__":
    # 结构文档
    structure_files = [
        "docs/code_structure_guide.md",
        "docs/data_layer_design.md",
        "docs/feature_layer_design.md",
        "docs/model_system_design.md"
    ]
    # 覆盖率报告
    report_files = [
        "docs/coverage_report_latest.md",
        "docs/low_coverage_report.md",
        "docs/data_layer_low_coverage_report.md"
    ]
    debt_file = "docs/technical_debt.md"

    modules = extract_modules(structure_files)
    low_coverage_modules = extract_low_coverage_modules(report_files, threshold=80)
    # 只登记结构中存在的低覆盖模块
    filtered = {m: c for m, c in low_coverage_modules.items() if m in modules}
    append_to_technical_debt(debt_file, filtered)
    print("技术债务清单已自动更新。")
