#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
版本管理API服务 - 重构版本

重构要点:
1. 将159行的_register_routes函数拆分为多个独立方法
2. 每个路由处理逻辑提取为单独的方法
3. 使用装饰器模式简化路由注册
4. 提高可测试性和可维护性
"""

from flask import Flask, jsonify, request
from typing import Optional, Dict, Any
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.versioning.manager.manager import VersionManager
try:  # 兼容旧命名，测试环境中可能只提供 VersionPolicy
    from src.infrastructure.versioning.manager.policy import VersionPolicyManager
except ImportError:  # pragma: no cover - 仅在缺少管理器时生效
    from src.infrastructure.versioning.manager.policy import VersionPolicy as VersionPolicyManager
from src.infrastructure.versioning.data.data_version_manager import DataVersionManager
from src.infrastructure.versioning.config.config_version_manager import ConfigVersionManager
from src.infrastructure.versioning.core.version import Version, VersionComparator


class VersionAPIService:
    """版本管理API服务 - 重构版本"""
    
    def __init__(
        self,
        version_manager: Optional[VersionManager] = None,
        policy_manager: Optional[VersionPolicyManager] = None,
        data_version_manager: Optional[DataVersionManager] = None,
        config_version_manager: Optional[ConfigVersionManager] = None
    ):
        """初始化API服务"""
        self.app = Flask(__name__)
        self.version_manager = version_manager or VersionManager()
        self.policy_manager = policy_manager or VersionPolicyManager()
        self.data_version_manager = data_version_manager or DataVersionManager()
        self.config_version_manager = config_version_manager or ConfigVersionManager()
        
        # 注册路由
        self._register_routes()
    
    def _register_routes(self):
        """注册API路由 - 重构版本（调用各个路由处理方法）"""
        # 版本管理路由
        self.app.add_url_rule('/api/v1/versions', 'list_versions', 
                             self._handle_list_versions, methods=['GET'])
        self.app.add_url_rule('/api/v1/versions/<name>', 'get_version',
                             self._handle_get_version, methods=['GET'])
        self.app.add_url_rule('/api/v1/versions/<name>', 'create_version',
                             self._handle_create_version, methods=['POST'])
        self.app.add_url_rule('/api/v1/versions/<name>', 'update_version',
                             self._handle_update_version, methods=['PUT'])
        self.app.add_url_rule('/api/v1/versions/<name>', 'delete_version',
                             self._handle_delete_version, methods=['DELETE'])
        
        # 策略管理路由
        self.app.add_url_rule('/api/v1/policies', 'list_policies',
                             self._handle_list_policies, methods=['GET'])
        
        # 验证和比较路由
        self.app.add_url_rule('/api/v1/validate', 'validate_version',
                             self._handle_validate_version, methods=['POST'])
        self.app.add_url_rule('/api/v1/compare', 'compare_versions',
                             self._handle_compare_versions, methods=['POST'])
        
        # 数据版本路由
        self.app.add_url_rule('/api/v1/data/versions', 'list_data_versions',
                             self._handle_list_data_versions, methods=['GET'])
        
        # 配置版本路由
        self.app.add_url_rule('/api/v1/config/versions/<config_name>', 'list_config_versions',
                             self._handle_list_config_versions, methods=['GET'])
        
        # 健康检查路由
        self.app.add_url_rule('/api/v1/health', 'health_check',
                             self._handle_health_check, methods=['GET'])
    
    # ============ 版本管理路由处理器 ============
    
    def _handle_list_versions(self):
        """处理：列出所有版本"""
        versions = self.version_manager.list_versions()
        return jsonify({
            'versions': {name: str(version) for name, version in versions.items()},
            'count': len(versions)
        })
    
    def _handle_get_version(self, name: str):
        """处理：获取指定版本"""
        version = self.version_manager.get_version(name)
        if version:
            return jsonify({
                'name': name,
                'version': str(version),
                'is_stable': version.is_stable(),
                'is_prerelease': version.is_prerelease()
            })
        return jsonify({'error': 'Version not found'}), 404
    
    def _handle_create_version(self, name: str):
        """处理：创建新版本"""
        data = request.get_json()
        if not data or 'version' not in data:
            return jsonify({'error': 'Version string required'}), 400
        
        try:
            version = Version(data['version'])
            self.version_manager.register_version(name, version)
            return jsonify({
                'name': name,
                'version': str(version),
                'status': 'created'
            }), 201
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
    
    def _handle_update_version(self, name: str):
        """处理：更新版本"""
        data = request.get_json()
        if not data or 'version' not in data:
            return jsonify({'error': 'Version string required'}), 400
        
        try:
            version = Version(data['version'])
            success = self.version_manager.update_version(name, version)
            if success:
                return jsonify({
                    'name': name,
                    'version': str(version),
                    'status': 'updated'
                })
            return jsonify({'error': 'Update failed'}), 500
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
    
    def _handle_delete_version(self, name: str):
        """处理：删除版本"""
        success = self.version_manager.remove_version(name)
        if success:
            return jsonify({'status': 'deleted'})
        return jsonify({'error': 'Version not found'}), 404
    
    # ============ 策略管理路由处理器 ============
    
    def _handle_list_policies(self):
        """处理：列出所有策略"""
        policies = self.policy_manager.list_policies()
        return jsonify({
            'policies': policies,
            'count': len(policies)
        })
    
    # ============ 验证和比较路由处理器 ============
    
    def _handle_validate_version(self):
        """处理：验证版本"""
        data = request.get_json()
        if not data or 'version' not in data:
            return jsonify({'error': 'Version string required'}), 400
        
        policy_name = data.get('policy')
        try:
            version = Version(data['version'])
            is_valid = self.policy_manager.validate_version(version, policy_name)
            return jsonify({
                'version': str(version),
                'policy': policy_name or 'default',
                'is_valid': is_valid
            })
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
    
    def _handle_compare_versions(self):
        """处理：比较版本"""
        data = request.get_json()
        if not data or 'version1' not in data or 'version2' not in data:
            return jsonify({'error': 'Two version strings required'}), 400
        
        try:
            v1 = Version(data['version1'])
            v2 = Version(data['version2'])
            result = VersionComparator.compare_versions(v1, v2)
            
            comparison = '='
            if result > 0:
                comparison = '>'
            elif result < 0:
                comparison = '<'
            
            return jsonify({
                'version1': str(v1),
                'version2': str(v2),
                'comparison': comparison,
                'result': result
            })
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
    
    # ============ 数据版本路由处理器 ============
    
    def _handle_list_data_versions(self):
        """处理：列出数据版本"""
        try:
            versions = self.data_version_manager.list_versions()
            return jsonify({
                'versions': [str(v) for v in versions],
                'count': len(versions)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # ============ 配置版本路由处理器 ============
    
    def _handle_list_config_versions(self, config_name: str):
        """处理：列出配置版本"""
        try:
            versions = self.config_version_manager.list_versions(config_name)
            return jsonify({
                'config_name': config_name,
                'versions': [str(v) for v in versions],
                'count': len(versions)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # ============ 健康检查路由处理器 ============
    
    def _handle_health_check(self):
        """处理：健康检查"""
        return jsonify({
            'status': 'healthy',
            'service': 'version-management-api',
            'version': '1.0.0'
        })
    
    # ============ 服务控制 ============
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """运行API服务"""
        self.app.run(host=host, port=port, debug=debug)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='版本管理API服务 (重构版本)')
    parser.add_argument('--host', default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=5000, help='监听端口')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    service = VersionAPIService()
    print(f"🚀 版本管理API服务启动...")
    print(f"   地址: http://{args.host}:{args.port}")
    print(f"   调试模式: {args.debug}")
    service.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()

