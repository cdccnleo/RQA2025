#!/usr/bin/env python3
"""
文档同步管理脚本

整合文档同步、质量检查、版本控制和文档生成功能。
"""

from src.infrastructure.docs import (
    DocumentSyncManager,
    DocumentQualityChecker,
    DocumentVersionController,
    DocumentGenerator
)
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def setup_logging(level: str = "INFO") -> logging.Logger:
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/document_sync.log', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)


def sync_documents(args) -> Dict:
    """同步文档"""
    logger = logging.getLogger(__name__)
    logger.info("开始文档同步...")

    sync_manager = DocumentSyncManager()

    # 检查同步状态
    sync_status = sync_manager.check_sync_status()

    # 统计状态
    status_counts = {}
    for status in sync_status.values():
        status_counts[status] = status_counts.get(status, 0) + 1

    logger.info(f"同步状态统计: {status_counts}")

    # 自动更新文档
    if args.auto_update:
        updated_files = sync_manager.auto_update_documents(force=args.force)
        logger.info(f"自动更新了 {len(updated_files)} 个文档")

    # 生成缺失的文档
    if args.generate_missing:
        generated_files = sync_manager.generate_missing_documents()
        logger.info(f"生成了 {len(generated_files)} 个缺失的文档")

    return {
        "sync_status": sync_status,
        "status_counts": status_counts,
        "updated_files": updated_files if args.auto_update else [],
        "generated_files": generated_files if args.generate_missing else []
    }


def check_quality(args) -> Dict:
    """检查文档质量"""
    logger = logging.getLogger(__name__)
    logger.info("开始文档质量检查...")

    quality_checker = DocumentQualityChecker()

    if args.file:
        # 检查单个文件
        report = quality_checker.check_document_quality(args.file)
        reports = {args.file: report}
    else:
        # 检查目录
        directory = args.directory or "docs/features"
        reports = quality_checker.check_directory_quality(directory)

    # 生成质量总结
    summary = quality_checker.generate_quality_summary(reports)

    logger.info(f"质量检查完成: {len(reports)} 个文档")
    logger.info(f"平均评分: {summary.get('average_score', 0):.1f}")

    return {
        "reports": reports,
        "summary": summary
    }


def manage_versions(args) -> Dict:
    """管理文档版本"""
    logger = logging.getLogger(__name__)
    logger.info("开始版本管理...")

    version_controller = DocumentVersionController()

    if args.create_version:
        # 创建版本
        version = version_controller.create_version(
            args.create_version,
            args.author or "system",
            args.message or f"自动版本化: {datetime.now()}"
        )
        logger.info(f"创建版本: {args.create_version} -> {version}")
        return {"created_version": version}

    elif args.list_versions:
        # 列出版本
        versions = version_controller.list_versions(args.list_versions)
        logger.info(f"找到 {len(versions)} 个版本")
        return {"versions": versions}

    elif args.restore_version:
        # 恢复版本
        success = version_controller.restore_version(
            args.restore_version[0],
            args.restore_version[1]
        )
        logger.info(f"版本恢复: {'成功' if success else '失败'}")
        return {"restore_success": success}

    elif args.compare_versions:
        # 比较版本
        comparison = version_controller.compare_versions(
            args.compare_versions[0],
            args.compare_versions[1],
            args.compare_versions[2]
        )
        logger.info(f"版本比较完成: {comparison.get('lines_changed', 0)} 行变更")
        return {"comparison": comparison}

    elif args.auto_version:
        # 自动版本化
        changed_files = version_controller.auto_version_changes()
        logger.info(f"自动版本化: {len(changed_files)} 个文件")
        return {"changed_files": changed_files}

    elif args.cleanup:
        # 清理旧版本
        cleaned_files = version_controller.cleanup_old_versions(args.cleanup)
        logger.info(f"清理完成: {len(cleaned_files)} 个版本文件")
        return {"cleaned_files": cleaned_files}

    return {}


def generate_documents(args) -> Dict:
    """生成文档"""
    logger = logging.getLogger(__name__)
    logger.info("开始文档生成...")

    doc_generator = DocumentGenerator()

    if args.file:
        # 生成单个文件的文档
        if args.all_docs:
            generated_docs = doc_generator.generate_all_documentation(args.file)
            logger.info(f"生成完整文档集: {len(generated_docs)} 个文档")
            return {"generated_docs": generated_docs}
        else:
            # 根据类型生成文档
            if args.api_doc:
                doc_path = doc_generator.generate_api_documentation(args.file)
            elif args.usage_guide:
                doc_path = doc_generator.generate_usage_guide(args.file)
            elif args.architecture_doc:
                doc_path = doc_generator.generate_architecture_documentation(args.file)
            elif args.readme:
                doc_path = doc_generator.generate_readme(args.file)
            else:
                # 默认生成API文档
                doc_path = doc_generator.generate_api_documentation(args.file)

            logger.info(f"文档生成成功: {doc_path}")
            return {"generated_doc": doc_path}

    else:
        # 批量生成文档
        src_path = Path("src/features")
        generated_docs = {}

        for py_file in src_path.rglob("*.py"):
            if py_file.name.startswith('__'):
                continue

            try:
                if args.all_docs:
                    docs = doc_generator.generate_all_documentation(str(py_file))
                    generated_docs[str(py_file)] = docs
                else:
                    doc_path = doc_generator.generate_api_documentation(str(py_file))
                    generated_docs[str(py_file)] = doc_path

            except Exception as e:
                logger.error(f"生成文档失败: {py_file}, 错误: {e}")

        logger.info(f"批量文档生成完成: {len(generated_docs)} 个文件")
        return {"generated_docs": generated_docs}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="文档同步管理工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 同步命令
    sync_parser = subparsers.add_parser('sync', help='同步文档')
    sync_parser.add_argument('--auto-update', action='store_true', help='自动更新文档')
    sync_parser.add_argument('--generate-missing', action='store_true', help='生成缺失的文档')
    sync_parser.add_argument('--force', action='store_true', help='强制更新')

    # 质量检查命令
    quality_parser = subparsers.add_parser('quality', help='检查文档质量')
    quality_parser.add_argument('--file', help='检查单个文件')
    quality_parser.add_argument('--directory', help='检查目录')

    # 版本管理命令
    version_parser = subparsers.add_parser('version', help='管理文档版本')
    version_parser.add_argument('--create-version', help='为文件创建版本')
    version_parser.add_argument('--author', help='版本作者')
    version_parser.add_argument('--message', help='版本消息')
    version_parser.add_argument('--list-versions', help='列出文件的所有版本')
    version_parser.add_argument('--restore-version', nargs=2, help='恢复到指定版本 (文件 版本)')
    version_parser.add_argument('--compare-versions', nargs=3, help='比较两个版本 (文件 版本1 版本2)')
    version_parser.add_argument('--auto-version', action='store_true', help='自动版本化变更')
    version_parser.add_argument('--cleanup', type=int, help='清理旧版本，保留指定数量')

    # 文档生成命令
    generate_parser = subparsers.add_parser('generate', help='生成文档')
    generate_parser.add_argument('--file', help='生成单个文件的文档')
    generate_parser.add_argument('--all-docs', action='store_true', help='生成所有类型的文档')
    generate_parser.add_argument('--api-doc', action='store_true', help='生成API文档')
    generate_parser.add_argument('--usage-guide', action='store_true', help='生成使用指南')
    generate_parser.add_argument('--architecture-doc', action='store_true', help='生成架构文档')
    generate_parser.add_argument('--readme', action='store_true', help='生成README')

    # 通用参数
    parser.add_argument('--log-level', default='INFO', help='日志级别')
    parser.add_argument('--output', help='输出结果到文件')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 设置日志
    logger = setup_logging(args.log_level)

    try:
        # 执行命令
        if args.command == 'sync':
            result = sync_documents(args)
        elif args.command == 'quality':
            result = check_quality(args)
        elif args.command == 'version':
            result = manage_versions(args)
        elif args.command == 'generate':
            result = generate_documents(args)
        else:
            logger.error(f"未知命令: {args.command}")
            return

        # 输出结果
        if args.output:
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"结果已保存到: {args.output}")
        else:
            print(f"执行结果: {result}")

        logger.info("命令执行完成")

    except Exception as e:
        logger.error(f"执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
