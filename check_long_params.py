#!/usr/bin/env python3
import json


def check_long_param_details():
    """检查长参数函数的详细信息"""
    with open('phase1_post_refactor_analysis.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 查找最严重的长参数问题
    long_param_ops = []
    for opp in data['opportunities']:
        if 'parameter' in opp['title'].lower() or '参数' in opp['title']:
            long_param_ops.append(opp)

    # 按严重程度排序
    long_param_ops.sort(key=lambda x: x['severity'], reverse=True)

    print("🎯 长参数函数问题详细信息:")
    print("=" * 80)

    for i, opp in enumerate(long_param_ops[:10]):  # 只显示前10个
        print(f"{i+1}. {opp['title']}")
        print(f"   文件: {opp['file_path']}")
        print(f"   行号: {opp['line_number']}")
        print(f"   严重程度: {opp['severity']}")
        print(f"   风险等级: {opp['risk_level']}")
        print(f"   置信度: {opp['confidence']}")
        print(f"   建议: {opp['suggested_fix']}")
        print("   ---")


if __name__ == "__main__":
    check_long_param_details()
