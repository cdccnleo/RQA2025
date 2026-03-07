#!/usr/bin/env python3
"""
智能重复代码检测工具 - 命令行接口
"""

import argparse
import sys
import json
import time

from .core.config import SmartDuplicateConfig
from .analyzers.clone_detector import CloneDetector


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="智能重复代码检测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python -m tools.smart_duplicate_detector src/
  python -m tools.smart_duplicate_detector --preset strict --format html src/
  python -m tools.smart_duplicate_detector --config custom_config.json src/
        """
    )

    parser.add_argument(
        'target_path',
        help='检测目标路径（文件或目录）'
    )

    parser.add_argument(
        '--preset', '-p',
        choices=['strict', 'normal', 'relaxed', 'performance', 'quality'],
        default='normal',
        help='预设配置 (默认: normal)'
    )

    parser.add_argument(
        '--format', '-f',
        choices=['json', 'xml', 'html'],
        default='json',
        help='输出格式 (默认: json)'
    )

    parser.add_argument(
        '--output', '-o',
        help='输出文件路径'
    )

    parser.add_argument(
        '--config', '-c',
        help='自定义配置文件路径 (JSON格式)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )

    parser.add_argument(
        '--compare-files',
        nargs=2,
        metavar=('FILE1', 'FILE2'),
        help='比较两个特定文件'
    )

    args = parser.parse_args()

    try:
        # 创建配置
        config = create_config(args)

        print("🚀 开始智能重复代码检测...")
        print(f"目标路径: {args.target_path}")
        print(f"配置预设: {args.preset}")
        print(f"输出格式: {args.format}")
        print()

        start_time = time.time()

        if args.compare_files:
            # 文件对比模式
            result = compare_files(args.compare_files[0], args.compare_files[1], config)
            output_result(result, args.format, args.output)
        else:
            # 完整检测模式
            result = detect_clones(args.target_path, config)
            output_result(result, args.format, args.output)

        duration = time.time() - start_time
        print(f"检测完成，耗时: {duration:.2f}秒")
    except Exception as e:
        print(f"❌ 检测失败: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def create_config(args) -> SmartDuplicateConfig:
    """
    创建配置对象

    Args:
        args: 命令行参数

    Returns:
        SmartDuplicateConfig: 配置对象
    """
    # 加载自定义配置
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = SmartDuplicateConfig.from_dict(config_dict)
    else:
        config = SmartDuplicateConfig()

    # 应用预设
    config.get_preset_config(args.preset)

    return config


def detect_clones(target_path: str, config: SmartDuplicateConfig) -> dict:
    """
    执行克隆检测

    Args:
        target_path: 目标路径
        config: 配置对象

    Returns:
        dict: 检测结果
    """
    detector = CloneDetector(config)
    result = detector.analyze(target_path)

    # 转换为字典格式用于输出
    result_dict = result.to_dict()
    result_dict['stats'] = detector.get_detection_stats(result)

    return result_dict


def compare_files(file1: str, file2: str, config: SmartDuplicateConfig) -> dict:
    """
    比较两个文件

    Args:
        file1: 文件1路径
        file2: 文件2路径
        config: 配置对象

    Returns:
        dict: 比较结果
    """
    detector = CloneDetector(config)
    result = detector.analyze_file_pair(file1, file2)

    return result


def output_result(result: dict, format: str, output_file: str = None):
    """
    输出结果

    Args:
        result: 结果字典
        format: 输出格式
        output_file: 输出文件路径
    """
    if format == 'json':
        output = json.dumps(result, indent=2, ensure_ascii=False)
    elif format == 'xml':
        # 简化的XML输出
        output = dict_to_xml(result)
    elif format == 'html':
        output = dict_to_html(result)
    else:
        raise ValueError(f"不支持的格式: {format}")

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"📄 结果已保存到: {output_file}")
    else:
        print(output)


def dict_to_xml(data: dict, root_tag: str = 'result') -> str:
    """将字典转换为XML字符串"""
    def _dict_to_xml(data, tag):
        if isinstance(data, dict):
            xml_parts = [f'<{tag}>']
            for key, value in data.items():
                xml_parts.append(_dict_to_xml(value, key))
            xml_parts.append(f'</{tag}>')
            return '\n'.join(xml_parts)
        elif isinstance(data, list):
            xml_parts = []
            for item in data:
                xml_parts.append(_dict_to_xml(item, 'item'))
            return '\n'.join(xml_parts)
        else:
            return f'<{tag}>{str(data)}</{tag}>'

    return f'<?xml version="1.0" encoding="UTF-8"?>\n{_dict_to_xml(data, root_tag)}'


def dict_to_html(data: dict) -> str:
    """将字典转换为HTML字符串"""
    html_parts = [
        '<!DOCTYPE html>',
        '<html><head><title>智能重复代码检测报告</title></head><body>',
        '<h1>智能重复代码检测报告</h1>'
    ]

    # 基本统计信息
    if 'stats' in data:
        stats = data['stats']
        html_parts.extend([
            '<h2>检测统计</h2>',
            '<table border="1">',
            '<tr><th>指标</th><th>值</th></tr>',
            f'<tr><td>克隆组数量</td><td>{stats.get("total_groups", 0)}</td></tr>',
            f'<tr><td>受影响文件数</td><td>{stats.get("files_affected_count", 0)}</td></tr>',
            f'<tr><td>克隆代码行数</td><td>{stats.get("total_lines_cloned", 0)}</td></tr>',
            f'<tr><td>分析耗时</td><td>{stats.get("analysis_time_seconds", 0):.2f}秒</td></tr>',
            '</table>'
        ])

    # 克隆组详情
    if 'clone_groups' in data and data['clone_groups']:
        html_parts.extend([
            '<h2>克隆组详情</h2>',
            '<table border="1">',
            '<tr><th>组ID</th><th>类型</th><th>片段数</th><th>相似度</th><th>受影响文件</th></tr>'
        ])

        for group in data['clone_groups'][:20]:  # 限制显示前20个
            html_parts.append(
                f'<tr><td>{group["group_id"]}</td><td>{group["clone_type"]}</td>'
                f'<td>{group["fragment_count"]}</td><td>{group["similarity_score"]:.2f}</td>'
                f'<td>{", ".join(group["files"])}</td></tr>'
            )

        html_parts.append('</table>')

        if len(data['clone_groups']) > 20:
            html_parts.append(f'<p>... 还有 {len(data["clone_groups"]) - 20} 个克隆组</p>')

    # 重构建议
    if 'refactoring_opportunities' in data and data['refactoring_opportunities']:
        html_parts.extend([
            '<h2>重构建议</h2>',
            '<table border="1">',
            '<tr><th>建议类型</th><th>影响程度</th><th>复杂度</th><th>描述</th></tr>'
        ])

        for opp in data['refactoring_opportunities'][:10]:  # 限制显示前10个
            html_parts.append(
                f'<tr><td>{opp["type"]}</td><td>{opp["impact"]}</td>'
                f'<td>{opp["complexity"]}</td><td>{opp["description"]}</td></tr>'
            )

        html_parts.append('</table>')

    html_parts.extend(['</body></html>'])
    return '\n'.join(html_parts)


if __name__ == '__main__':
    main()
