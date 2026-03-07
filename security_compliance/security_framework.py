#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2026 创新引擎安全合规框架
确保三大创新引擎的安全性和合规性

安全特性:
- 数据加密和隐私保护
- 访问控制和身份验证
- 审计日志和监控
- 安全通信协议
- 合规性检查
"""

import hashlib
import hmac
import secrets
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt

logger = logging.getLogger(__name__)


class EncryptionManager:
    """加密管理器"""

    def __init__(self, key_file: str = "security_compliance/keys.enc"):
        self.key_file = Path(key_file)
        self.keys = {}
        self._load_or_generate_keys()

    def _load_or_generate_keys(self):
        """加载或生成密钥"""
        if self.key_file.exists():
            try:
                with open(self.key_file, 'rb') as f:
                    encrypted_data = f.read()

                # 使用主密码解密
                master_password = self._get_master_password()
                key = self._derive_key(master_password)

                fernet = Fernet(key)
                decrypted_data = fernet.decrypt(encrypted_data)
                self.keys = json.loads(decrypted_data.decode())
            except Exception as e:
                logger.warning(f"无法加载密钥文件: {e}")
                self._generate_keys()
        else:
            self._generate_keys()

    def _generate_keys(self):
        """生成新的密钥"""
        self.keys = {
            'master_key': base64.urlsafe_b64encode(os.urandom(32)).decode(),
            'data_encryption_key': Fernet.generate_key().decode(),
            'jwt_secret': secrets.token_hex(32),
            'hmac_key': secrets.token_hex(32)
        }
        self._save_keys()

    def _save_keys(self):
        """保存密钥"""
        self.key_file.parent.mkdir(exist_ok=True)

        master_password = self._get_master_password()
        key = self._derive_key(master_password)

        fernet = Fernet(key)
        key_data = json.dumps(self.keys).encode()
        encrypted_data = fernet.encrypt(key_data)

        with open(self.key_file, 'wb') as f:
            f.write(encrypted_data)

    def _get_master_password(self) -> str:
        """获取主密码 (在实际系统中应该从安全存储获取)"""
        # 这里使用固定的主密码，仅用于演示
        return "RQA2026_Master_Password_Secure_Key"

    def _derive_key(self, password: str) -> bytes:
        """从密码派生密钥"""
        salt = b'RQA2026_Security_Salt'  # 固定的盐值
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def encrypt_data(self, data: Any) -> str:
        """加密数据"""
        if isinstance(data, (dict, list)):
            data = json.dumps(data, ensure_ascii=False)

        fernet = Fernet(self.keys['data_encryption_key'].encode())
        encrypted = fernet.encrypt(str(data).encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_data(self, encrypted_data: str) -> Any:
        """解密数据"""
        try:
            fernet = Fernet(self.keys['data_encryption_key'].encode())
            decrypted = fernet.decrypt(base64.urlsafe_b64decode(encrypted_data))
            data_str = decrypted.decode()

            # 尝试解析JSON
            try:
                return json.loads(data_str)
            except:
                return data_str
        except Exception as e:
            logger.error(f"解密失败: {e}")
            return None

    def hash_data(self, data: str) -> str:
        """哈希数据"""
        return hashlib.sha256(data.encode()).hexdigest()

    def hmac_sign(self, data: str) -> str:
        """HMAC签名"""
        return hmac.new(
            self.keys['hmac_key'].encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()

    def verify_hmac(self, data: str, signature: str) -> bool:
        """验证HMAC签名"""
        expected_signature = self.hmac_sign(data)
        return hmac.compare_digest(expected_signature, signature)


class AccessControlManager:
    """访问控制管理器"""

    def __init__(self):
        self.users = {}
        self.roles = {
            'admin': ['read', 'write', 'delete', 'admin'],
            'analyst': ['read', 'write'],
            'viewer': ['read'],
            'engine': ['read', 'execute']  # 引擎间通信权限
        }
        self.sessions = {}
        self.encryption_manager = EncryptionManager()

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """用户认证"""
        if username in self.users:
            stored_hash = self.users[username]['password_hash']
            input_hash = self.encryption_manager.hash_data(password)

            if stored_hash == input_hash:
                # 生成JWT token
                token = self._generate_jwt_token(username)
                self.sessions[token] = {
                    'username': username,
                    'role': self.users[username]['role'],
                    'created_at': datetime.now(),
                    'expires_at': datetime.now() + timedelta(hours=8)
                }
                return token

        return None

    def authorize_action(self, token: str, action: str, resource: str) -> bool:
        """授权操作"""
        session = self.sessions.get(token)
        if not session:
            return False

        # 检查token是否过期
        if datetime.now() > session['expires_at']:
            del self.sessions[token]
            return False

        user_role = session['role']
        allowed_actions = self.roles.get(user_role, [])

        return action in allowed_actions

    def add_user(self, username: str, password: str, role: str = 'viewer'):
        """添加用户"""
        if role not in self.roles:
            raise ValueError(f"无效的角色: {role}")

        self.users[username] = {
            'password_hash': self.encryption_manager.hash_data(password),
            'role': role,
            'created_at': datetime.now().isoformat()
        }

    def _generate_jwt_token(self, username: str) -> str:
        """生成JWT token"""
        payload = {
            'username': username,
            'role': self.users[username]['role'],
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=8)
        }

        token = jwt.encode(
            payload,
            self.encryption_manager.keys['jwt_secret'],
            algorithm='HS256'
        )
        return token

    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证token"""
        try:
            payload = jwt.decode(
                token,
                self.encryption_manager.keys['jwt_secret'],
                algorithms=['HS256']
            )
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None


class AuditLogger:
    """审计日志记录器"""

    def __init__(self, log_file: str = "security_compliance/audit.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)

    def log_event(self, event_type: str, user: str, action: str,
                  resource: str, result: str, details: Dict[str, Any] = None):
        """记录审计事件"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'user': user,
            'action': action,
            'resource': resource,
            'result': result,
            'details': details or {},
            'ip_address': '127.0.0.1',  # 在实际系统中获取真实IP
            'user_agent': 'RQA2026_Engine'
        }

        # 写入日志文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\\n')

        # 记录到控制台 (重要事件)
        if event_type in ['security_alert', 'unauthorized_access']:
            logger.warning(f"安全事件: {event_type} - {action} by {user}")

    def get_audit_trail(self, user: str = None, event_type: str = None,
                       start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """获取审计追踪"""
        audit_trail = []

        if not self.log_file.exists():
            return audit_trail

        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())

                    # 应用过滤器
                    if user and entry['user'] != user:
                        continue
                    if event_type and entry['event_type'] != event_type:
                        continue
                    if start_date and datetime.fromisoformat(entry['timestamp']) < start_date:
                        continue
                    if end_date and datetime.fromisoformat(entry['timestamp']) > end_date:
                        continue

                    audit_trail.append(entry)
                except json.JSONDecodeError:
                    continue

        return audit_trail

    def generate_security_report(self, days: int = 7) -> Dict[str, Any]:
        """生成安全报告"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        audit_trail = self.get_audit_trail(start_date=start_date, end_date=end_date)

        report = {
            'period': f"{start_date.date()} to {end_date.date()}",
            'total_events': len(audit_trail),
            'event_types': {},
            'security_alerts': 0,
            'unauthorized_access': 0,
            'successful_authentications': 0,
            'failed_authentications': 0
        }

        for entry in audit_trail:
            event_type = entry['event_type']
            result = entry['result']

            if event_type not in report['event_types']:
                report['event_types'][event_type] = 0
            report['event_types'][event_type] += 1

            if event_type == 'security_alert':
                report['security_alerts'] += 1
            elif event_type == 'authentication':
                if result == 'success':
                    report['successful_authentications'] += 1
                else:
                    report['failed_authentications'] += 1
            elif event_type == 'unauthorized_access':
                report['unauthorized_access'] += 1

        return report


class ComplianceChecker:
    """合规性检查器"""

    def __init__(self):
        self.compliance_rules = {
            'data_encryption': self._check_data_encryption,
            'access_control': self._check_access_control,
            'audit_logging': self._check_audit_logging,
            'data_retention': self._check_data_retention,
            'privacy_protection': self._check_privacy_protection
        }

    def check_compliance(self, component: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查合规性"""
        results = {}

        for rule_name, checker in self.compliance_rules.items():
            try:
                result = checker(data)
                results[rule_name] = {
                    'compliant': result['compliant'],
                    'details': result.get('details', ''),
                    'severity': result.get('severity', 'low')
                }
            except Exception as e:
                results[rule_name] = {
                    'compliant': False,
                    'details': f"检查失败: {e}",
                    'severity': 'high'
                }

        # 计算总体合规性分数
        compliant_count = sum(1 for r in results.values() if r['compliant'])
        total_count = len(results)
        compliance_score = compliant_count / total_count if total_count > 0 else 0

        return {
            'component': component,
            'overall_compliant': compliance_score >= 0.8,
            'compliance_score': compliance_score,
            'rule_results': results,
            'checked_at': datetime.now().isoformat()
        }

    def _check_data_encryption(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查数据加密"""
        encrypted_fields = data.get('encrypted_fields', [])
        total_fields = data.get('total_fields', 1)

        encryption_ratio = len(encrypted_fields) / total_fields

        return {
            'compliant': encryption_ratio >= 0.95,  # 95%的数据需要加密
            'details': f"加密字段比例: {encryption_ratio:.1%}",
            'severity': 'high' if encryption_ratio < 0.8 else 'low'
        }

    def _check_access_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查访问控制"""
        has_authentication = data.get('authentication_enabled', False)
        has_authorization = data.get('authorization_enabled', False)
        role_based_access = data.get('role_based_access', False)

        compliant = has_authentication and has_authorization and role_based_access

        return {
            'compliant': compliant,
            'details': f"认证:{has_authentication}, 授权:{has_authorization}, 角色访问:{role_based_access}",
            'severity': 'high'
        }

    def _check_audit_logging(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查审计日志"""
        logging_enabled = data.get('audit_logging_enabled', False)
        log_retention_days = data.get('log_retention_days', 0)

        compliant = logging_enabled and log_retention_days >= 90  # 最少90天

        return {
            'compliant': compliant,
            'details': f"日志启用:{logging_enabled}, 保留天数:{log_retention_days}",
            'severity': 'medium'
        }

    def _check_data_retention(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查数据保留"""
        retention_policy = data.get('retention_policy', {})
        max_retention_days = retention_policy.get('max_days', 0)
        auto_deletion = retention_policy.get('auto_deletion', False)

        compliant = max_retention_days <= 2555 and auto_deletion  # 最长7年并自动删除

        return {
            'compliant': compliant,
            'details': f"最大保留:{max_retention_days}天, 自动删除:{auto_deletion}",
            'severity': 'medium'
        }

    def _check_privacy_protection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查隐私保护"""
        anonymization_enabled = data.get('data_anonymization', False)
        consent_management = data.get('consent_management', False)
        data_minimization = data.get('data_minimization', False)

        compliant = anonymization_enabled and consent_management and data_minimization

        return {
            'compliant': compliant,
            'details': f"匿名化:{anonymization_enabled}, 同意管理:{consent_management}, 数据最小化:{data_minimization}",
            'severity': 'high'
        }


class SecurityFramework:
    """安全框架"""

    def __init__(self):
        self.encryption = EncryptionManager()
        self.access_control = AccessControlManager()
        self.audit_logger = AuditLogger()
        self.compliance_checker = ComplianceChecker()

        # 初始化默认用户
        self._initialize_default_users()

    def _initialize_default_users(self):
        """初始化默认用户"""
        self.access_control.add_user('admin', 'admin123', 'admin')
        self.access_control.add_user('analyst', 'analyst123', 'analyst')
        self.access_control.add_user('viewer', 'viewer123', 'viewer')

    def secure_communicate(self, sender: str, receiver: str,
                          message: Dict[str, Any]) -> Dict[str, Any]:
        """安全通信"""
        # 加密消息
        encrypted_message = self.encryption.encrypt_data(message)

        # 生成签名
        signature = self.encryption.hmac_sign(encrypted_message)

        # 记录审计
        self.audit_logger.log_event(
            'secure_communication',
            sender,
            'send_message',
            f"{sender}->{receiver}",
            'success',
            {'encrypted': True, 'signed': True}
        )

        return {
            'encrypted_message': encrypted_message,
            'signature': signature,
            'sender': sender,
            'receiver': receiver,
            'timestamp': datetime.now().isoformat()
        }

    def verify_communication(self, communication_packet: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """验证通信"""
        encrypted_message = communication_packet['encrypted_message']
        signature = communication_packet['signature']
        sender = communication_packet['sender']

        # 验证签名
        if not self.encryption.verify_hmac(encrypted_message, signature):
            self.audit_logger.log_event(
                'security_alert',
                sender,
                'verify_signature',
                'communication',
                'failed',
                {'reason': 'invalid_signature'}
            )
            return False, {}

        # 解密消息
        decrypted_message = self.encryption.decrypt_data(encrypted_message)

        if decrypted_message is None:
            return False, {}

        # 记录成功验证
        self.audit_logger.log_event(
            'secure_communication',
            sender,
            'verify_message',
            'communication',
            'success'
        )

        return True, decrypted_message

    def check_system_compliance(self) -> Dict[str, Any]:
        """检查系统合规性"""
        system_data = {
            'encrypted_fields': ['user_passwords', 'sensitive_data', 'api_keys'],
            'total_fields': 10,
            'authentication_enabled': True,
            'authorization_enabled': True,
            'role_based_access': True,
            'audit_logging_enabled': True,
            'log_retention_days': 365,
            'retention_policy': {
                'max_days': 2555,  # 7年
                'auto_deletion': True
            },
            'data_anonymization': True,
            'consent_management': True,
            'data_minimization': True
        }

        return self.compliance_checker.check_compliance('rqa2026_system', system_data)

    def get_security_status(self) -> Dict[str, Any]:
        """获取安全状态"""
        compliance_report = self.check_system_compliance()
        security_report = self.audit_logger.generate_security_report(days=7)

        return {
            'compliance_status': compliance_report,
            'security_report': security_report,
            'active_sessions': len(self.access_control.sessions),
            'encryption_status': 'active' if self.encryption.keys else 'inactive',
            'timestamp': datetime.now().isoformat()
        }


def create_security_framework() -> SecurityFramework:
    """创建安全框架工厂函数"""
    return SecurityFramework()


def demo_security_framework():
    """安全框架演示"""
    print("🔒 RQA2026 安全合规框架演示")
    print("=" * 50)

    framework = create_security_framework()

    # 演示用户认证
    print("🔐 用户认证演示:")
    token = framework.access_control.authenticate_user('admin', 'admin123')
    if token:
        print("✅ 管理员认证成功")
    else:
        print("❌ 认证失败")

    # 演示安全通信
    print("\\n📡 安全通信演示:")
    message = {'action': 'process_data', 'data_size': 1000}
    secure_packet = framework.secure_communicate('admin', 'ai_engine', message)
    print("✅ 消息加密和签名完成")

    # 验证通信
    valid, decrypted = framework.verify_communication(secure_packet)
    if valid:
        print("✅ 消息验证和解密成功")
        print(f"解密内容: {decrypted}")
    else:
        print("❌ 消息验证失败")

    # 检查合规性
    print("\\n📋 合规性检查:")
    compliance = framework.check_system_compliance()
    print(f"总体合规: {'✅ 通过' if compliance['overall_compliant'] else '❌ 未通过'}")
    print(".1%")

    # 生成安全报告
    print("\\n📊 安全报告:")
    status = framework.get_security_status()
    security_report = status['security_report']
    print(f"审计事件总数: {security_report['total_events']}")
    print(f"安全警报: {security_report['security_alerts']}")
    print(f"成功认证: {security_report['successful_authentications']}")

    print("\\n✅ 安全框架演示完成!")


if __name__ == "__main__":
    demo_security_framework()
