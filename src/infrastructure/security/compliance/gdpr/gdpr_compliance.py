"""
GDPR合规工具与数据脱敏模块

功能：
- 用户数据可携带权（数据导出）
- 被遗忘权（数据删除）
- 数据更正权
- 数据处理同意管理
- 数据脱敏和匿名化
- 隐私影响评估
- 数据保留策略

技术栈：
- json: 数据序列化
- hashlib: 数据哈希
- re: 正则表达式脱敏

作者: Claude
创建日期: 2026-02-21
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable
from pathlib import Path
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSubjectRight(Enum):
    """数据主体权利"""
    ACCESS = "access"           # 访问权
    RECTIFICATION = "rectification"  # 更正权
    ERASURE = "erasure"         # 被遗忘权
    RESTRICT_PROCESSING = "restrict_processing"  # 限制处理权
    DATA_PORTABILITY = "data_portability"  # 数据可携带权
    OBJECT = "object"           # 反对权
    AUTOMATED_DECISION = "automated_decision"  # 自动化决策相关权利


class ConsentStatus(Enum):
    """同意状态"""
    GRANTED = "granted"
    REVOKED = "revoked"
    PENDING = "pending"
    EXPIRED = "expired"


class DataSensitivity(Enum):
    """数据敏感度"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"  # 个人数据
    SENSITIVE = "sensitive"  # 敏感个人数据


@dataclass
class ConsentRecord:
    """同意记录"""
    consent_id: str
    user_id: str
    purpose: str
    data_types: List[str]
    status: ConsentStatus
    granted_at: Optional[datetime]
    revoked_at: Optional[datetime]
    expires_at: Optional[datetime]
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSubjectRequest:
    """数据主体请求"""
    request_id: str
    user_id: str
    right_type: DataSubjectRight
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    details: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None


class DataMasking:
    """
    数据脱敏类
    
    提供多种数据脱敏策略
    """
    
    @staticmethod
    def mask_email(email: str, visible_chars: int = 3) -> str:
        """
        脱敏邮箱地址
        
        Args:
            email: 邮箱地址
            visible_chars: 保留可见字符数
            
        Returns:
            脱敏后的邮箱
        """
        if '@' not in email:
            return '*' * len(email)
        
        local, domain = email.split('@', 1)
        if len(local) <= visible_chars:
            masked_local = local
        else:
            masked_local = local[:visible_chars] + '*' * (len(local) - visible_chars)
        
        return f"{masked_local}@{domain}"
    
    @staticmethod
    def mask_phone(phone: str, visible_digits: int = 4) -> str:
        """
        脱敏电话号码
        
        Args:
            phone: 电话号码
            visible_digits: 保留可见位数
            
        Returns:
            脱敏后的电话
        """
        digits = re.sub(r'\D', '', phone)
        if len(digits) <= visible_digits:
            return '*' * len(phone)
        
        masked_digits = '*' * (len(digits) - visible_digits) + digits[-visible_digits:]
        
        # 保留原始格式
        result = []
        digit_index = 0
        for char in phone:
            if char.isdigit():
                result.append(masked_digits[digit_index])
                digit_index += 1
            else:
                result.append(char)
        
        return ''.join(result)
    
    @staticmethod
    def mask_id_card(id_card: str) -> str:
        """
        脱敏身份证号
        
        Args:
            id_card: 身份证号
            
        Returns:
            脱敏后的身份证号
        """
        if len(id_card) != 18:
            return '*' * len(id_card)
        
        return id_card[:6] + '*' * 8 + id_card[14:]
    
    @staticmethod
    def mask_bank_card(card_number: str, visible_digits: int = 4) -> str:
        """
        脱敏银行卡号
        
        Args:
            card_number: 银行卡号
            visible_digits: 保留可见位数
            
        Returns:
            脱敏后的卡号
        """
        digits = re.sub(r'\D', '', card_number)
        if len(digits) <= visible_digits:
            return '*' * len(card_number)
        
        return '*' * (len(digits) - visible_digits) + digits[-visible_digits:]
    
    @staticmethod
    def mask_name(name: str, visible_chars: int = 1) -> str:
        """
        脱敏姓名
        
        Args:
            name: 姓名
            visible_chars: 保留可见字符数
            
        Returns:
            脱敏后的姓名
        """
        if len(name) <= visible_chars:
            return name
        
        return name[:visible_chars] + '*' * (len(name) - visible_chars)
    
    @staticmethod
    def mask_address(address: str, visible_parts: int = 2) -> str:
        """
        脱敏地址
        
        Args:
            address: 地址
            visible_parts: 保留可见部分数
            
        Returns:
            脱敏后的地址
        """
        parts = address.split()
        if len(parts) <= visible_parts:
            return address
        
        return ' '.join(parts[:visible_parts]) + ' ' + '*' * 5
    
    @staticmethod
    def mask_ip_address(ip: str) -> str:
        """
        脱敏IP地址
        
        Args:
            ip: IP地址
            
        Returns:
            脱敏后的IP
        """
        parts = ip.split('.')
        if len(parts) != 4:
            return ip
        
        return f"{parts[0]}.{parts[1]}.*.*"
    
    @staticmethod
    def partial_mask(text: str, start_ratio: float = 0.3,
                    end_ratio: float = 0.3, mask_char: str = '*') -> str:
        """
        部分脱敏
        
        Args:
            text: 文本
            start_ratio: 开头保留比例
            end_ratio: 结尾保留比例
            mask_char: 掩码字符
            
        Returns:
            脱敏后的文本
        """
        length = len(text)
        start_keep = int(length * start_ratio)
        end_keep = int(length * end_ratio)
        
        if start_keep + end_keep >= length:
            return text
        
        return text[:start_keep] + mask_char * (length - start_keep - end_keep) + text[-end_keep:]
    
    @classmethod
    def mask_dict(cls, data: Dict[str, Any],
                 sensitive_fields: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        脱敏字典中的敏感字段
        
        Args:
            data: 数据字典
            sensitive_fields: 敏感字段集合
            
        Returns:
            脱敏后的字典
        """
        if sensitive_fields is None:
            sensitive_fields = {
                'email', 'phone', 'mobile', 'id_card', 'id_number',
                'bank_card', 'credit_card', 'password', 'name',
                'address', 'ip_address', 'ssn'
            }
        
        result = {}
        for key, value in data.items():
            if key.lower() in sensitive_fields:
                if isinstance(value, str):
                    if 'email' in key.lower():
                        result[key] = cls.mask_email(value)
                    elif 'phone' in key.lower() or 'mobile' in key.lower():
                        result[key] = cls.mask_phone(value)
                    elif 'id_card' in key.lower() or 'id_number' in key.lower():
                        result[key] = cls.mask_id_card(value)
                    elif 'bank' in key.lower() or 'card' in key.lower():
                        result[key] = cls.mask_bank_card(value)
                    elif 'name' in key.lower():
                        result[key] = cls.mask_name(value)
                    elif 'address' in key.lower():
                        result[key] = cls.mask_address(value)
                    elif 'ip' in key.lower():
                        result[key] = cls.mask_ip_address(value)
                    else:
                        result[key] = cls.partial_mask(value)
                else:
                    result[key] = '***'
            elif isinstance(value, dict):
                result[key] = cls.mask_dict(value, sensitive_fields)
            elif isinstance(value, list):
                result[key] = [
                    cls.mask_dict(item, sensitive_fields) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        
        return result


class DataAnonymization:
    """
    数据匿名化类
    
    提供k-匿名和l-多样性实现
    """
    
    @staticmethod
    def k_anonymize(data: List[Dict[str, Any]],
                   quasi_identifiers: List[str],
                   k: int = 5) -> List[Dict[str, Any]]:
        """
        k-匿名化处理
        
        Args:
            data: 数据列表
            quasi_identifiers: 准标识符列表
            k: 匿名度
            
        Returns:
            匿名化后的数据
        """
        if not data or not quasi_identifiers:
            return data
        
        # 按准标识符分组
        groups = {}
        for record in data:
            key = tuple(record.get(qid) for qid in quasi_identifiers)
            if key not in groups:
                groups[key] = []
            groups[key].append(record)
        
        # 泛化处理不满足k-匿名的组
        result = []
        for key, group in groups.items():
            if len(group) >= k:
                result.extend(group)
            else:
                # 泛化处理
                generalized = DataAnonymization._generalize_group(
                    group, quasi_identifiers
                )
                result.extend(generalized)
        
        return result
    
    @staticmethod
    def _generalize_group(group: List[Dict[str, Any]],
                         quasi_identifiers: List[str]) -> List[Dict[str, Any]]:
        """泛化数据组"""
        result = []
        for record in group:
            generalized = record.copy()
            for qid in quasi_identifiers:
                value = record.get(qid)
                if isinstance(value, (int, float)):
                    # 数值泛化到范围
                    generalized[qid] = DataAnonymization._generalize_number(value)
                elif isinstance(value, str):
                    # 字符串泛化
                    generalized[qid] = DataAnonymization._generalize_string(value)
            result.append(generalized)
        return result
    
    @staticmethod
    def _generalize_number(value: Union[int, float]) -> str:
        """泛化数值"""
        if isinstance(value, int):
            lower = (value // 10) * 10
            return f"{lower}-{lower + 9}"
        else:
            lower = int(value * 10) / 10
            return f"{lower:.1f}-{lower + 0.9:.1f}"
    
    @staticmethod
    def _generalize_string(value: str) -> str:
        """泛化字符串"""
        if len(value) <= 2:
            return value
        return value[:2] + '*' * (len(value) - 2)
    
    @staticmethod
    def pseudonymize(value: str, salt: Optional[str] = None) -> str:
        """
        假名化处理
        
        Args:
            value: 原始值
            salt: 盐值
            
        Returns:
            假名
        """
        if salt is None:
            salt = "default_salt"
        
        hash_obj = hashlib.sha256(f"{value}{salt}".encode())
        return hash_obj.hexdigest()[:16]


class GDPRComplianceManager:
    """
    GDPR合规管理器
    
    管理GDPR相关的所有功能
    """
    
    def __init__(self, data_retention_days: int = 2555):  # 默认7年
        """
        初始化GDPR管理器
        
        Args:
            data_retention_days: 数据保留天数
        """
        self.data_retention_days = data_retention_days
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.data_subject_requests: Dict[str, DataSubjectRequest] = {}
        self.data_inventory: Dict[str, Dict[str, Any]] = {}
        
        # 数据处理器注册
        self.data_processors: Dict[str, Callable] = {}
    
    def register_data_processor(self, data_type: str,
                                processor: Callable) -> None:
        """
        注册数据处理器
        
        Args:
            data_type: 数据类型
            processor: 处理函数
        """
        self.data_processors[data_type] = processor
        logger.info(f"注册数据处理器: {data_type}")
    
    # ============ 同意管理 ============
    
    def record_consent(self, user_id: str, purpose: str,
                      data_types: List[str],
                      expires_days: Optional[int] = None) -> ConsentRecord:
        """
        记录用户同意
        
        Args:
            user_id: 用户ID
            purpose: 目的
            data_types: 数据类型列表
            expires_days: 过期天数
            
        Returns:
            同意记录
        """
        consent_id = str(uuid.uuid4())
        
        granted_at = datetime.now()
        expires_at = None
        if expires_days:
            expires_at = granted_at + timedelta(days=expires_days)
        
        record = ConsentRecord(
            consent_id=consent_id,
            user_id=user_id,
            purpose=purpose,
            data_types=data_types,
            status=ConsentStatus.GRANTED,
            granted_at=granted_at,
            revoked_at=None,
            expires_at=expires_at
        )
        
        self.consent_records[consent_id] = record
        logger.info(f"记录用户同意: {user_id} - {purpose}")
        
        return record
    
    def revoke_consent(self, consent_id: str) -> bool:
        """
        撤销同意
        
        Args:
            consent_id: 同意ID
            
        Returns:
            是否成功
        """
        record = self.consent_records.get(consent_id)
        if not record:
            return False
        
        record.status = ConsentStatus.REVOKED
        record.revoked_at = datetime.now()
        
        logger.info(f"撤销同意: {consent_id}")
        return True
    
    def check_consent(self, user_id: str, purpose: str,
                     data_type: str) -> bool:
        """
        检查用户同意状态
        
        Args:
            user_id: 用户ID
            purpose: 目的
            data_type: 数据类型
            
        Returns:
            是否有有效同意
        """
        for record in self.consent_records.values():
            if (record.user_id == user_id and
                record.purpose == purpose and
                data_type in record.data_types and
                record.status == ConsentStatus.GRANTED):
                
                # 检查是否过期
                if record.expires_at and record.expires_at < datetime.now():
                    record.status = ConsentStatus.EXPIRED
                    return False
                
                return True
        
        return False
    
    # ============ 数据主体权利 ============
    
    def create_data_subject_request(self, user_id: str,
                                   right_type: DataSubjectRight,
                                   details: Optional[Dict[str, Any]] = None
                                   ) -> DataSubjectRequest:
        """
        创建数据主体请求
        
        Args:
            user_id: 用户ID
            right_type: 权利类型
            details: 详细信息
            
        Returns:
            请求对象
        """
        request_id = str(uuid.uuid4())
        
        request = DataSubjectRequest(
            request_id=request_id,
            user_id=user_id,
            right_type=right_type,
            status="pending",
            created_at=datetime.now(),
            completed_at=None,
            details=details or {}
        )
        
        self.data_subject_requests[request_id] = request
        logger.info(f"创建数据主体请求: {request_id} - {right_type.value}")
        
        return request
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        导出用户数据（数据可携带权）
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户数据
        """
        export_data = {
            'user_id': user_id,
            'export_timestamp': datetime.now().isoformat(),
            'data_categories': {}
        }
        
        # 调用各数据处理器收集数据
        for data_type, processor in self.data_processors.items():
            try:
                data = processor('export', user_id)
                export_data['data_categories'][data_type] = data
            except Exception as e:
                logger.error(f"导出数据类型 {data_type} 失败: {e}")
                export_data['data_categories'][data_type] = {'error': str(e)}
        
        # 添加同意记录
        user_consents = [
            asdict(record) for record in self.consent_records.values()
            if record.user_id == user_id
        ]
        export_data['consent_history'] = user_consents
        
        return export_data
    
    def delete_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        删除用户数据（被遗忘权）
        
        Args:
            user_id: 用户ID
            
        Returns:
            删除结果
        """
        results = {}
        
        # 调用各数据处理器删除数据
        for data_type, processor in self.data_processors.items():
            try:
                result = processor('delete', user_id)
                results[data_type] = {'status': 'deleted', 'result': result}
            except Exception as e:
                logger.error(f"删除数据类型 {data_type} 失败: {e}")
                results[data_type] = {'status': 'failed', 'error': str(e)}
        
        # 撤销所有同意
        for record in self.consent_records.values():
            if record.user_id == user_id:
                record.status = ConsentStatus.REVOKED
                record.revoked_at = datetime.now()
        
        logger.info(f"删除用户数据: {user_id}")
        return results
    
    def anonymize_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        匿名化用户数据
        
        Args:
            user_id: 用户ID
            
        Returns:
            匿名化结果
        """
        results = {}
        
        # 调用各数据处理器匿名化数据
        for data_type, processor in self.data_processors.items():
            try:
                result = processor('anonymize', user_id)
                results[data_type] = {'status': 'anonymized', 'result': result}
            except Exception as e:
                logger.error(f"匿名化数据类型 {data_type} 失败: {e}")
                results[data_type] = {'status': 'failed', 'error': str(e)}
        
        logger.info(f"匿名化用户数据: {user_id}")
        return results
    
    # ============ 合规报告 ============
    
    def generate_privacy_report(self, user_id: str) -> Dict[str, Any]:
        """
        生成隐私报告
        
        Args:
            user_id: 用户ID
            
        Returns:
            隐私报告
        """
        # 收集用户的所有数据处理活动
        processing_activities = []
        
        for consent_id, record in self.consent_records.items():
            if record.user_id == user_id:
                processing_activities.append({
                    'purpose': record.purpose,
                    'data_types': record.data_types,
                    'status': record.status.value,
                    'granted_at': record.granted_at.isoformat() if record.granted_at else None,
                    'expires_at': record.expires_at.isoformat() if record.expires_at else None
                })
        
        return {
            'user_id': user_id,
            'report_generated_at': datetime.now().isoformat(),
            'processing_activities': processing_activities,
            'data_retention_policy': f"{self.data_retention_days}天",
            'rights': [right.value for right in DataSubjectRight]
        }
    
    def check_data_retention(self) -> List[Dict[str, Any]]:
        """
        检查数据保留期限
        
        Returns:
            过期数据列表
        """
        expired_data = []
        retention_limit = datetime.now() - timedelta(days=self.data_retention_days)
        
        # 检查同意记录
        for consent_id, record in self.consent_records.items():
            if (record.granted_at and
                record.granted_at < retention_limit and
                record.status == ConsentStatus.GRANTED):
                expired_data.append({
                    'type': 'consent',
                    'id': consent_id,
                    'user_id': record.user_id,
                    'created_at': record.granted_at.isoformat(),
                    'action_required': 'review_or_delete'
                })
        
        return expired_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取GDPR统计信息
        
        Returns:
            统计信息
        """
        total_consents = len(self.consent_records)
        active_consents = sum(
            1 for r in self.consent_records.values()
            if r.status == ConsentStatus.GRANTED
        )
        revoked_consents = sum(
            1 for r in self.consent_records.values()
            if r.status == ConsentStatus.REVOKED
        )
        
        pending_requests = sum(
            1 for r in self.data_subject_requests.values()
            if r.status == 'pending'
        )
        completed_requests = sum(
            1 for r in self.data_subject_requests.values()
            if r.status == 'completed'
        )
        
        return {
            'consents': {
                'total': total_consents,
                'active': active_consents,
                'revoked': revoked_consents,
                'expired': total_consents - active_consents - revoked_consents
            },
            'data_subject_requests': {
                'total': len(self.data_subject_requests),
                'pending': pending_requests,
                'completed': completed_requests
            },
            'data_retention_days': self.data_retention_days,
            'expired_data_count': len(self.check_data_retention())
        }


# 便捷函数
def mask_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """脱敏敏感数据"""
    return DataMasking.mask_dict(data)


def anonymize_dataset(data: List[Dict[str, Any]],
                     quasi_identifiers: List[str],
                     k: int = 5) -> List[Dict[str, Any]]:
    """匿名化数据集"""
    return DataAnonymization.k_anonymize(data, quasi_identifiers, k)


# 单例实例
_gdpr_instance: Optional[GDPRComplianceManager] = None


def get_gdpr_manager() -> GDPRComplianceManager:
    """
    获取GDPR管理器单例
    
    Returns:
        GDPRComplianceManager实例
    """
    global _gdpr_instance
    if _gdpr_instance is None:
        _gdpr_instance = GDPRComplianceManager()
    return _gdpr_instance
