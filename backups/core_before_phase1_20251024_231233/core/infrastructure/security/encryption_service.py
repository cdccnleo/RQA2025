import logging
#!/usr/bin/env python3
"""
RQA2025 加密服务

提供数据加密和解密功能，支持多种加密算法和密钥管理
"""

import base64
import hashlib
import hmac
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import threading

# 导入统一基础设施集成层
try:
    from src.integration import get_trading_layer_adapter
    INFRASTRUCTURE_INTEGRATION_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_INTEGRATION_AVAILABLE = False


logger = logging.getLogger(__name__)


class EncryptionAlgorithm:

    """加密算法枚举"""
    AES_256_GCM = "aes_256_gcm"
    AES_128_CBC = "aes_128_cbc"
    FERNET = "fernet"  # 默认使用Fernet


class KeyManager:

    """密钥管理器"""

    def __init__(self):

        self.keys: Dict[str, bytes] = {}
        self.key_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

        # 初始化默认密钥
        self._init_default_keys()

    def _init_default_keys(self):
        """初始化默认密钥"""
        # 生成主密钥
        master_key = self._generate_key("master")
        self.store_key("master", master_key)

        # 生成用于不同用途的密钥
        purposes = ["api", "database", "session", "file"]
        for purpose in purposes:
            key = self._generate_key(purpose)
            self.store_key(purpose, key)

    def _generate_key(self, purpose: str) -> bytes:
        """生成密钥"""
        # 使用密码学安全的随机数生成器
        salt = os.urandom(16)
        password = f"RQA2025_{purpose}_key_{datetime.now().isoformat()}".encode()

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )

        return base64.urlsafe_b64encode(kdf.derive(password))

    def store_key(self, key_id: str, key: bytes, metadata: Dict[str, Any] = None):
        """存储密钥"""
        with self._lock:
            self.keys[key_id] = key
            self.key_metadata[key_id] = {
                "created_at": datetime.now().isoformat(),
                "algorithm": "PBKDF2 - SHA256",
                "key_length": len(key),
                "purpose": metadata.get("purpose", "general") if metadata else "general",
                **(metadata or {})
            }

    def get_key(self, key_id: str) -> Optional[bytes]:
        """获取密钥"""
        with self._lock:
            return self.keys.get(key_id)

    def rotate_key(self, key_id: str) -> bool:
        """轮换密钥"""
        with self._lock:
            if key_id in self.keys:
                # 生成新密钥
                new_key = self._generate_key(key_id)

                # 备份旧密钥（用于解密旧数据）
                old_key = self.keys[key_id]
                backup_id = f"{key_id}_backup_{datetime.now().strftime('%Y % m % d_ % H % M % S')}"
                self.store_key(backup_id, old_key, {"purpose": "backup", "original_key": key_id})

                # 存储新密钥
                self.store_key(key_id, new_key, {"rotated_at": datetime.now().isoformat()})

                logger.info(f"密钥轮换完成: {key_id}")
                return True

        return False

    def list_keys(self) -> Dict[str, Dict[str, Any]]:
        """列出所有密钥"""
        with self._lock:
            return self.key_metadata.copy()


class DataEncryptor:

    """数据加密器"""

    def __init__(self, key_manager: KeyManager):

        self.key_manager = key_manager
        self.algorithm = EncryptionAlgorithm.FERNET

    def encrypt(self, data: str, key_id: str = "master") -> Optional[str]:
        """加密数据"""
        try:
            key = self.key_manager.get_key(key_id)
            if not key:
                logger.error(f"密钥不存在: {key_id}")
                return None

            f = Fernet(key)
            encrypted_data = f.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()

        except Exception as e:
            logger.error(f"数据加密失败: {e}")
            return None

    def decrypt(self, encrypted_data: str, key_id: str = "master") -> Optional[str]:
        """解密数据"""
        try:
            key = self.key_manager.get_key(key_id)
            if not key:
                logger.error(f"密钥不存在: {key_id}")
                return None

            f = Fernet(key)
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = f.decrypt(encrypted_bytes)
            return decrypted_data.decode()

        except Exception as e:
            logger.error(f"数据解密失败: {e}")
            return None

    def encrypt_dict(self, data: Dict[str, Any], key_id: str = "master") -> Optional[str]:
        """加密字典数据"""
        json_data = json.dumps(data, ensure_ascii=False)
        return self.encrypt(json_data, key_id)

    def decrypt_dict(self, encrypted_data: str, key_id: str = "master") -> Optional[Dict[str, Any]]:
        """解密字典数据"""
        decrypted_str = self.decrypt(encrypted_data, key_id)
        if decrypted_str:
            try:
                return json.loads(decrypted_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON解密失败: {e}")
                return None
        return None


class SecureCommunication:

    """安全通信"""

    def __init__(self, key_manager: KeyManager):

        self.key_manager = key_manager

    def generate_signature(self, data: str, key_id: str = "api") -> Optional[str]:
        """生成数据签名"""
        try:
            key = self.key_manager.get_key(key_id)
            if not key:
                return None

            signature = hmac.new(key, data.encode(), hashlib.sha256).digest()
            return base64.urlsafe_b64encode(signature).decode()

        except Exception as e:
            logger.error(f"生成签名失败: {e}")
            return None

    def verify_signature(self, data: str, signature: str, key_id: str = "api") -> bool:
        """验证数据签名"""
        try:
            key = self.key_manager.get_key(key_id)
            if not key:
                return False

            expected_signature = hmac.new(key, data.encode(), hashlib.sha256).digest()
            provided_signature = base64.urlsafe_b64decode(signature.encode())

            return hmac.compare_digest(expected_signature, provided_signature)

        except Exception as e:
            logger.error(f"验证签名失败: {e}")
            return False

    def create_secure_token(self, payload: Dict[str, Any], expiration_minutes: int = 60,


                            key_id: str = "session") -> Optional[str]:
        """创建安全令牌"""
        try:
            # 添加过期时间
            payload["exp"] = (datetime.now() + timedelta(minutes=expiration_minutes)).timestamp()
            payload["iat"] = datetime.now().timestamp()

            # 序列化负载
            payload_str = json.dumps(payload, sort_keys=True)

            # 生成签名
            signature = self.generate_signature(payload_str, key_id)
            if not signature:
                return None

            # 创建令牌
            token_data = {
                "payload": payload_str,
                "signature": signature
            }

            token_json = json.dumps(token_data, sort_keys=True)
            return base64.urlsafe_b64encode(token_json.encode()).decode()

        except Exception as e:
            logger.error(f"创建安全令牌失败: {e}")
            return None

    def verify_secure_token(self, token: str, key_id: str = "session") -> Optional[Dict[str, Any]]:
        """验证安全令牌"""
        try:
            # 解码令牌
            token_json = base64.urlsafe_b64decode(token.encode()).decode()
            token_data = json.loads(token_json)

            payload_str = token_data["payload"]
            signature = token_data["signature"]

            # 验证签名
            if not self.verify_signature(payload_str, signature, key_id):
                return None

            # 解析负载
            payload = json.loads(payload_str)

            # 检查过期时间
            if "exp" in payload:
                if datetime.now().timestamp() > payload["exp"]:
                    return None

            return payload

        except Exception as e:
            logger.error(f"验证安全令牌失败: {e}")
            return None


class EncryptionService:

    """加密服务主类"""

    def __init__(self):

        self.key_manager = KeyManager()
        self.encryptor = DataEncryptor(self.key_manager)
        self.secure_comm = SecureCommunication(self.key_manager)

        # 基础设施集成
        self._infrastructure_adapter = None
        if INFRASTRUCTURE_INTEGRATION_AVAILABLE:
            try:
                self._infrastructure_adapter = get_trading_layer_adapter()
            except Exception as e:
                logger.warning(f"基础设施集成初始化失败: {e}")

        logger.info("加密服务初始化完成")

    def encrypt(self, data: str, key_id: str = "master") -> Optional[str]:
        """加密数据"""
        return self.encryptor.encrypt(data, key_id)

    def decrypt(self, encrypted_data: str, key_id: str = "master") -> Optional[str]:
        """解密数据"""
        return self.encryptor.decrypt(encrypted_data, key_id)

    def encrypt_json(self, data: Dict[str, Any], key_id: str = "master") -> Optional[str]:
        """加密JSON数据"""
        return self.encryptor.encrypt_dict(data, key_id)

    def decrypt_json(self, encrypted_data: str, key_id: str = "master") -> Optional[Dict[str, Any]]:
        """解密JSON数据"""
        return self.encryptor.decrypt_dict(encrypted_data, key_id)

    def generate_signature(self, data: str, key_id: str = "api") -> Optional[str]:
        """生成数字签名"""
        return self.secure_comm.generate_signature(data, key_id)

    def verify_signature(self, data: str, signature: str, key_id: str = "api") -> bool:
        """验证数字签名"""
        return self.secure_comm.verify_signature(data, signature, key_id)

    def create_token(self, payload: Dict[str, Any], expiration_minutes: int = 60,


                     key_id: str = "session") -> Optional[str]:
        """创建安全令牌"""
        return self.secure_comm.create_secure_token(payload, expiration_minutes, key_id)

    def verify_token(self, token: str, key_id: str = "session") -> Optional[Dict[str, Any]]:
        """验证安全令牌"""
        return self.secure_comm.verify_secure_token(token, key_id)

    def rotate_key(self, key_id: str) -> bool:
        """轮换密钥"""
        return self.key_manager.rotate_key(key_id)

    def get_key_info(self, key_id: str) -> Optional[Dict[str, Any]]:
        """获取密钥信息"""
        return self.key_manager.key_metadata.get(key_id)

    def list_keys(self) -> Dict[str, Dict[str, Any]]:
        """列出所有密钥"""
        return self.key_manager.list_keys()

    def encrypt_file(self, input_file: str, output_file: str, key_id: str = "file") -> bool:
        """加密文件"""
        try:
            with open(input_file, 'rb') as f:
                data = f.read()

            # 生成文件哈希
            file_hash = hashlib.sha256(data).hexdigest()

            # 加密数据
            encrypted_data = self.encryptor.encrypt(data.decode('latin - 1'), key_id)
            if not encrypted_data:
                return False

            # 保存加密文件（包含元数据）
            metadata = {
                "original_hash": file_hash,
                "encrypted_at": datetime.now().isoformat(),
                "key_id": key_id
            }

            file_data = {
                "metadata": metadata,
                "data": encrypted_data
            }

            with open(output_file, 'w', encoding='utf - 8') as f:
                json.dump(file_data, f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            logger.error(f"文件加密失败: {e}")
            return False

    def decrypt_file(self, input_file: str, output_file: str, key_id: str = "file") -> bool:
        """解密文件"""
        try:
            with open(input_file, 'r', encoding='utf - 8') as f:
                file_data = json.load(f)

            metadata = file_data["metadata"]
            encrypted_data = file_data["data"]

            # 解密数据
            decrypted_data = self.encryptor.decrypt(encrypted_data, key_id)
            if not decrypted_data:
                return False

            # 验证文件完整性
            data_bytes = decrypted_data.encode('latin - 1')
            current_hash = hashlib.sha256(data_bytes).hexdigest()
            original_hash = metadata["original_hash"]

            if current_hash != original_hash:
                logger.error("文件完整性验证失败")
                return False

            # 保存解密文件
            with open(output_file, 'wb') as f:
                f.write(data_bytes)

            return True

        except Exception as e:
            logger.error(f"文件解密失败: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_info = {
            'component': 'EncryptionService',
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'keys_count': len(self.key_manager.list_keys()),
            'warnings': [],
            'critical_issues': []
        }

        # 检查密钥状态
        keys = self.key_manager.list_keys()
        if not keys:
            health_info['critical_issues'].append("无可用密钥")

        # 检查是否有过期的密钥
        expired_keys = []
        for key_id, metadata in keys.items():
            if "rotated_at" in metadata:
                rotated_at = datetime.fromisoformat(metadata["rotated_at"])
                if datetime.now() - rotated_at > timedelta(days=90):  # 90天后建议轮换
                    expired_keys.append(key_id)

        if expired_keys:
            health_info['warnings'].append(f"以下密钥需要轮换: {', '.join(expired_keys)}")

        # 测试加密解密功能
        try:
            test_data = "test_encryption_functionality"
            encrypted = self.encrypt(test_data)
            if encrypted:
                decrypted = self.decrypt(encrypted)
                if decrypted != test_data:
                    health_info['critical_issues'].append("加密解密功能异常")
            else:
                health_info['critical_issues'].append("加密功能异常")
        except Exception as e:
            health_info['critical_issues'].append(f"加密功能测试失败: {e}")

        # 总体状态评估
        if health_info['critical_issues']:
            health_info['status'] = 'critical'
        elif health_info['warnings']:
            health_info['status'] = 'warning'

        return health_info


# 全局加密服务实例
_encryption_service = None
_encryption_service_lock = threading.Lock()


def get_encryption_service() -> EncryptionService:
    """获取全局加密服务实例"""
    global _encryption_service

    if _encryption_service is None:
        with _encryption_service_lock:
            if _encryption_service is None:
                _encryption_service = EncryptionService()

    return _encryption_service


# 便捷函数

def encrypt_data(data: str, key_id: str = "master") -> Optional[str]:
    """加密数据"""
    service = get_encryption_service()
    return service.encrypt(data, key_id)


def decrypt_data(encrypted_data: str, key_id: str = "master") -> Optional[str]:
    """解密数据"""
    service = get_encryption_service()
    return service.decrypt(encrypted_data, key_id)


def create_secure_token(payload: Dict[str, Any], expiration_minutes: int = 60) -> Optional[str]:
    """创建安全令牌"""
    service = get_encryption_service()
    return service.create_token(payload, expiration_minutes)


def verify_secure_token(token: str) -> Optional[Dict[str, Any]]:
    """验证安全令牌"""
    service = get_encryption_service()
    return service.verify_token(token)
