"""
版本管理API接口

提供RESTful风格的版本管理API接口。
"""

import argparse
from typing import Optional
from flask import Flask, request, jsonify

from ..core.version import Version, VersionComparator
from ..manager.manager import VersionManager
from ..manager.policy import VersionPolicy
from ..data.data_version_manager import DataVersionManager
from ..config.config_version_manager import ConfigVersionManager


class HTTPConstants:
    """HTTP状态码常量"""
    BAD_REQUEST = 400
    CREATED = 201
    INTERNAL_SERVER_ERROR = 500
    DEFAULT_API_PORT = 8080


class VersionAPI:
    """
    版本管理API

    提供统一的HTTP API接口来管理各种版本资源。
    """

    def __init__(self, app: Optional[Flask] = None):
        """
        初始化版本管理API

        Args:
            app: Flask应用实例，如果为None则创建新实例
        """
        self.app = app or Flask(__name__)
        self.version_manager = VersionManager()
        self.policy_manager = VersionPolicy()
        self.data_version_manager = DataVersionManager()
        self.config_version_manager = ConfigVersionManager()

        # 注册路由
        self._register_routes()

    def _register_routes(self):
        """注册API路由"""

        @self.app.route('/api/v1/versions', methods=['GET'])
        def list_versions():
            """列出所有版本"""
            versions = self.version_manager.list_versions(as_dict=True)
            return jsonify({
                'versions': {name: str(version) for name, version in versions.items()},
                'count': len(versions)
            })

        @self.app.route('/api/v1/versions/<name>', methods=['GET'])
        def get_version(name):
            """获取指定版本"""
            version = self.version_manager.get_version(name)
            if version:
                return jsonify({
                    'name': name,
                    'version': str(version),
                    'is_stable': version.is_stable(),
                    'is_prerelease': version.is_prerelease()
                })
            return jsonify({'error': 'Version not found'}), 404

        @self.app.route('/api/v1/versions/<name>', methods=['POST'])
        def create_version(name):
            """创建新版本"""
            data = request.get_json()
            if not data or 'version' not in data:
                return jsonify({'error': 'Version string required'}), HTTPConstants.BAD_REQUEST

            try:
                version = Version(data['version'])
                self.version_manager.register_version(name, version)
                return jsonify({
                    'name': name,
                    'version': str(version),
                    'status': 'created'
                }), HTTPConstants.CREATED
            except ValueError as e:
                return jsonify({'error': str(e)}), HTTPConstants.BAD_REQUEST

        @self.app.route('/api/v1/versions/<name>', methods=['PUT'])
        def update_version(name):
            """更新版本"""
            data = request.get_json()
            if not data or 'version' not in data:
                return jsonify({'error': 'Version string required'}), HTTPConstants.BAD_REQUEST

            try:
                version = Version(data['version'])
                success = self.version_manager.update_version(name, version)
                if success:
                    return jsonify({
                        'name': name,
                        'version': str(version),
                        'status': 'updated'
                    })
                return jsonify({'error': 'Update failed'}), HTTPConstants.INTERNAL_SERVER_ERROR
            except ValueError as e:
                return jsonify({'error': str(e)}), HTTPConstants.BAD_REQUEST

        @self.app.route('/api/v1/versions/<name>', methods=['DELETE'])
        def delete_version(name):
            """删除版本"""
            success = self.version_manager.remove_version(name)
            if success:
                return jsonify({'status': 'deleted'})
            return jsonify({'error': 'Version not found'}), 404

        @self.app.route('/api/v1/policies', methods=['GET'])
        def list_policies():
            """列出所有策略"""
            policies = self.policy_manager.list_policies()
            return jsonify({
                'policies': policies,
                'count': len(policies)
            })

        @self.app.route('/api/v1/validate', methods=['POST'])
        def validate_version():
            """验证版本"""
            data = request.get_json()
            if not data or 'version' not in data:
                return jsonify({'error': 'Version string required'}), HTTPConstants.BAD_REQUEST

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
                return jsonify({'error': str(e)}), HTTPConstants.BAD_REQUEST

        @self.app.route('/api/v1/compare', methods=['POST'])
        def compare_versions():
            """比较版本"""
            data = request.get_json()
            if not data or 'version1' not in data or 'version2' not in data:
                return jsonify({'error': 'Two version strings required'}), HTTPConstants.BAD_REQUEST

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
                return jsonify({'error': str(e)}), HTTPConstants.BAD_REQUEST

        @self.app.route('/api/v1/data/versions', methods=['GET'])
        def list_data_versions():
            """列出数据版本"""
            try:
                versions = self.data_version_manager.list_versions()
                return jsonify({
                    'versions': [str(v) for v in versions],
                    'count': len(versions)
                })
            except Exception as e:
                return jsonify({'error': str(e)}), HTTPConstants.INTERNAL_SERVER_ERROR

        @self.app.route('/api/v1/config/versions/<config_name>', methods=['GET'])
        def list_config_versions(config_name):
            """列出配置版本"""
            try:
                versions = self.config_version_manager.list_versions(config_name)
                return jsonify({
                    'config_name': config_name,
                    'versions': [str(v) for v in versions],
                    'count': len(versions)
                })
            except Exception as e:
                return jsonify({'error': str(e)}), HTTPConstants.INTERNAL_SERVER_ERROR

        @self.app.route('/api/v1/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                'status': 'healthy',
                'service': 'version-management-api',
                'version': '1.0.0'
            })

    def run(self, host: str = '0.0.0.0', port: int = HTTPConstants.DEFAULT_API_PORT, debug: bool = False):
        """
        启动API服务器

        Args:
            host: 监听主机
            port: 监听端口
            debug: 是否开启调试模式
        """
        print(f"🚀 启动版本管理API服务器 - {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


# 工厂函数
def create_version_api(app: Optional[Flask] = None) -> VersionAPI:
    """
    创建版本管理API实例

    Args:
        app: Flask应用实例

    Returns:
        版本管理API实例
    """
    return VersionAPI(app)


# CLI入口点
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='版本管理API服务器')
    parser.add_argument('--host', default='0.0.0.0', help='监听主机')
    parser.add_argument('--port', type=int, default=HTTPConstants.DEFAULT_API_PORT, help='监听端口')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')

    args = parser.parse_args()

    api = create_version_api()
    api.run(host=args.host, port=args.port, debug=args.debug)
