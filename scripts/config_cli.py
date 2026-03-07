#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理Web服务CLI工具
支持：登录、拉取配置、推送配置、同步、回滚、批量操作、环境切换
"""
import argparse
import requests
import sys
import json
import os

API_BASE = "http://localhost:8080"


class ConfigWebCLI:
    def __init__(self, api_base=API_BASE):
        self.api_base = api_base
        self.session_id = None
        self.config_file = os.path.expanduser("~/.config_cli.json")

    def login(self, username, password):
        url = f"{self.api_base}/api/login"
        resp = requests.post(url, json={"username": username, "password": password})
        data = resp.json()
        if data.get("success"):
            self.session_id = data["session_id"]
            self._save_session()
            print(f"登录成功，session_id: {self.session_id}")
        else:
            print(f"登录失败: {data.get('detail', '未知错误')}")
            sys.exit(1)

    def get_config(self, path=None):
        url = f"{self.api_base}/api/config"
        if path:
            url = f"{self.api_base}/api/config/{path}"

        resp = requests.get(url, headers=self._auth_header())
        data = resp.json()
        if path:
            print(json.dumps(data.get("value", {}), ensure_ascii=False, indent=2))
        else:
            print(json.dumps(data.get("config", {}), ensure_ascii=False, indent=2))

    def push_config(self, path, value):
        url = f"{self.api_base}/api/config/{path}"
        payload = {"path": path, "value": value}
        resp = requests.put(url, headers=self._auth_header(), json=payload)
        data = resp.json()
        if data.get("success"):
            print("配置推送成功！")
        else:
            print(f"推送失败: {data.get('message', data)}")
            sys.exit(1)

    def batch_push(self, config_file):
        """批量推送配置文件"""
        if not os.path.exists(config_file):
            print(f"配置文件不存在: {config_file}")
            sys.exit(1)

        with open(config_file, 'r', encoding='utf-8') as f:
            configs = json.load(f)

        success_count = 0
        total_count = len(configs)

        for path, value in configs.items():
            try:
                self.push_config(path, value)
                success_count += 1
                print(f"✅ {path} = {value}")
            except Exception as e:
                print(f"❌ {path}: {e}")

        print(f"\n批量推送完成: {success_count}/{total_count} 成功")

    def validate_config(self, config_file):
        """验证配置文件"""
        url = f"{self.api_base}/api/config/validate"

        if not os.path.exists(config_file):
            print(f"配置文件不存在: {config_file}")
            sys.exit(1)

        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        resp = requests.post(url, headers=self._auth_header(), json={"config": config})
        data = resp.json()

        if data.get("valid"):
            print("✅ 配置验证通过")
        else:
            print("❌ 配置验证失败:")
            for error in data.get("errors", []):
                print(f"  - {error}")

    def sync(self, target_nodes=None):
        url = f"{self.api_base}/api/sync"
        payload = {"target_nodes": target_nodes} if target_nodes else {}
        resp = requests.post(url, headers=self._auth_header(), json=payload)
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))

    def get_sync_status(self):
        """获取同步状态"""
        url = f"{self.api_base}/api/sync/status"
        resp = requests.get(url, headers=self._auth_header())
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))

    def get_sync_history(self):
        """获取同步历史"""
        url = f"{self.api_base}/api/sync/history"
        resp = requests.get(url, headers=self._auth_header())
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))

    def encrypt_config(self, config_file, output_file=None):
        """加密配置文件"""
        url = f"{self.api_base}/api/config/encrypt"

        if not os.path.exists(config_file):
            print(f"配置文件不存在: {config_file}")
            sys.exit(1)

        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        resp = requests.post(url, headers=self._auth_header(), json={"config": config})
        data = resp.json()

        if data.get("success"):
            encrypted_config = data.get("encrypted_config", {})
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(encrypted_config, f, ensure_ascii=False, indent=2)
                print(f"✅ 加密配置已保存到: {output_file}")
            else:
                print(json.dumps(encrypted_config, ensure_ascii=False, indent=2))
        else:
            print(f"❌ 加密失败: {data.get('message', '未知错误')}")

    def decrypt_config(self, config_file, output_file=None):
        """解密配置文件"""
        url = f"{self.api_base}/api/config/decrypt"

        if not os.path.exists(config_file):
            print(f"配置文件不存在: {config_file}")
            sys.exit(1)

        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        resp = requests.post(url, headers=self._auth_header(), json={"config": config})
        data = resp.json()

        if data.get("success"):
            decrypted_config = data.get("decrypted_config", {})
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(decrypted_config, f, ensure_ascii=False, indent=2)
                print(f"✅ 解密配置已保存到: {output_file}")
            else:
                print(json.dumps(decrypted_config, ensure_ascii=False, indent=2))
        else:
            print(f"❌ 解密失败: {data.get('message', '未知错误')}")

    def _save_session(self):
        """保存session到本地文件"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump({"session_id": self.session_id}, f)
        except Exception:
            pass

    def _load_session(self):
        """从本地文件加载session"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.session_id = data.get("session_id")
        except Exception:
            pass

    def _auth_header(self):
        if not self.session_id:
            self._load_session()

        if not self.session_id:
            print("请先登录！")
            sys.exit(1)
        return {"Authorization": f"Bearer {self.session_id}"}


def main():
    parser = argparse.ArgumentParser(description="配置管理Web服务CLI工具")
    parser.add_argument('--api', default=API_BASE, help='API服务地址，默认http://localhost:8080')
    subparsers = parser.add_subparsers(dest='command')

    # 登录
    login_parser = subparsers.add_parser('login', help='登录')
    login_parser.add_argument('--username', required=True, help='用户名')
    login_parser.add_argument('--password', required=True, help='密码')

    # 拉取配置
    get_parser = subparsers.add_parser('get', help='拉取配置')
    get_parser.add_argument('--path', help='配置路径（可选，不指定则获取全部）')
    get_parser.add_argument('--session', help='session_id（可选，未登录时需先login）')

    # 推送配置
    push_parser = subparsers.add_parser('push', help='推送配置')
    push_parser.add_argument('--session', help='session_id（可选，未登录时需先login）')
    push_parser.add_argument('--path', required=True, help='配置路径，如database.host')
    push_parser.add_argument('--value', required=True, help='配置新值（字符串）')

    # 批量推送
    batch_parser = subparsers.add_parser('batch', help='批量推送配置')
    batch_parser.add_argument('--session', help='session_id（可选，未登录时需先login）')
    batch_parser.add_argument('--file', required=True, help='配置文件路径（JSON格式）')

    # 验证配置
    validate_parser = subparsers.add_parser('validate', help='验证配置文件')
    validate_parser.add_argument('--session', help='session_id（可选，未登录时需先login）')
    validate_parser.add_argument('--file', required=True, help='配置文件路径')

    # 同步
    sync_parser = subparsers.add_parser('sync', help='触发配置同步')
    sync_parser.add_argument('--session', help='session_id（可选，未登录时需先login）')
    sync_parser.add_argument('--nodes', nargs='*', help='目标节点ID列表')

    # 同步状态
    status_parser = subparsers.add_parser('status', help='获取同步状态')
    status_parser.add_argument('--session', help='session_id（可选，未登录时需先login）')

    # 同步历史
    history_parser = subparsers.add_parser('history', help='获取同步历史')
    history_parser.add_argument('--session', help='session_id（可选，未登录时需先login）')

    # 加密配置
    encrypt_parser = subparsers.add_parser('encrypt', help='加密配置文件')
    encrypt_parser.add_argument('--session', help='session_id（可选，未登录时需先login）')
    encrypt_parser.add_argument('--file', required=True, help='配置文件路径')
    encrypt_parser.add_argument('--output', help='输出文件路径（可选）')

    # 解密配置
    decrypt_parser = subparsers.add_parser('decrypt', help='解密配置文件')
    decrypt_parser.add_argument('--session', help='session_id（可选，未登录时需先login）')
    decrypt_parser.add_argument('--file', required=True, help='配置文件路径')
    decrypt_parser.add_argument('--output', help='输出文件路径（可选）')

    args = parser.parse_args()
    cli = ConfigWebCLI(api_base=args.api)

    if args.command == 'login':
        cli.login(args.username, args.password)
    elif args.command == 'get':
        if args.session:
            cli.session_id = args.session
        cli.get_config(args.path)
    elif args.command == 'push':
        if args.session:
            cli.session_id = args.session
        cli.push_config(args.path, args.value)
    elif args.command == 'batch':
        if args.session:
            cli.session_id = args.session
        cli.batch_push(args.file)
    elif args.command == 'validate':
        if args.session:
            cli.session_id = args.session
        cli.validate_config(args.file)
    elif args.command == 'sync':
        if args.session:
            cli.session_id = args.session
        cli.sync(args.nodes)
    elif args.command == 'status':
        if args.session:
            cli.session_id = args.session
        cli.get_sync_status()
    elif args.command == 'history':
        if args.session:
            cli.session_id = args.session
        cli.get_sync_history()
    elif args.command == 'encrypt':
        if args.session:
            cli.session_id = args.session
        cli.encrypt_config(args.file, args.output)
    elif args.command == 'decrypt':
        if args.session:
            cli.session_id = args.session
        cli.decrypt_config(args.file, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
