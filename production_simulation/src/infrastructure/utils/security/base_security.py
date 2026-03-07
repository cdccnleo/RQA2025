"""
base_security 模块

提供 base_security 相关功能和接口。
"""

import logging

from enum import Enum
from typing import Dict, List, Any
"""
基础设施层 - 基础安全模块

提供安全相关的枚举和基础类定义。
"""

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """安全级别枚举"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    def __lt__(self, other):
        """支持级别比较"""
        if not isinstance(other, SecurityLevel):
            return NotImplemented
        order = {
            SecurityLevel.LOW: 1,
            SecurityLevel.MEDIUM: 2,
            SecurityLevel.HIGH: 3,
            SecurityLevel.CRITICAL: 4
        }
        return order[self] < order[other]


class ThreatLevel(Enum):
    """威胁级别枚举"""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """安全事件类型"""

    LOGIN_ATTEMPT = "login_attempt"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    PERMISSION_DENIED = "permission_denied"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class SecurityPolicy:
    """安全策略基类"""

    def __init__(self, name: str, level: SecurityLevel = SecurityLevel.MEDIUM, description: str = "", security_level: SecurityLevel = None):
        """初始化安全策略"""
        try:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("策略名称必须是非空字符串")
            
            # 支持security_level参数
            if security_level is not None:
                level = security_level
                
            if not isinstance(level, SecurityLevel):
                raise ValueError("安全级别必须是SecurityLevel枚举值")

            self.name = name.strip()
            self.level = level
            self.security_level = level  # 添加别名以支持测试
            self.description = description
            self.enabled = True
            self.is_active = True  # 添加is_active属性
            self.rules: List[Dict[str, Any]] = []
            
            # 添加时间戳
            from datetime import datetime
            self.created_at = datetime.now()
            self.updated_at = datetime.now()
        except Exception as e:
            logger.error(f"初始化安全策略 '{name}' 时发生错误: {e}")
            raise

    def add_rule(self, rule: Dict[str, Any]) -> None:
        """添加安全规则"""
        try:
            if not isinstance(rule, dict):
                raise ValueError("安全规则必须是字典类型")
            if not rule:
                logger.warning("尝试添加空的规则到策略 '{self.name}'")
                return

            self.rules.append(rule)
            logger.debug(f"成功添加规则到策略 '{self.name}'，当前规则数量: {len(self.rules)}")
        except Exception as e:
            logger.error(f"添加规则到策略 '{self.name}' 时发生错误: {e}")
            raise

    def validate(self, context: Dict[str, Any]) -> bool:
        """验证安全上下文"""
        try:
            if not isinstance(context, dict):
                logger.warning(f"策略 '{self.name}' 收到无效的上下文类型: {type(context)}")
                return False

            # 基础实现，子类可以重写
            logger.debug(f"策略 '{self.name}' 验证上下文，通过基础验证")
            return True
        except Exception as e:
            logger.error(f"策略 '{self.name}' 验证上下文时发生错误: {e}")
            return False

    def get_violations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取违规项"""
        try:
            if not isinstance(context, dict):
                logger.warning(f"策略 '{self.name}' 收到无效的上下文类型: {type(context)}")
                return [{"type": "invalid_context", "message": "上下文必须是字典类型"}]

            # 基础实现，子类可以重写
            logger.debug(f"策略 '{self.name}' 检查违规项，无违规")
            return []
        except Exception as e:
            logger.error(f"策略 '{self.name}' 获取违规项时发生错误: {e}")
            return [{"type": "internal_error", "message": f"内部错误: {str(e)}"}]
    
    def update_security_level(self, new_level: SecurityLevel) -> None:
        """更新安全级别"""
        from datetime import datetime
        self.level = new_level
        self.security_level = new_level  # 更新别名
        self.updated_at = datetime.now()
    
    def is_compliant_with_level(self, required_level: SecurityLevel) -> bool:
        """检查是否符合指定安全级别"""
        # 级别顺序：LOW < MEDIUM < HIGH < CRITICAL
        level_order = {
            SecurityLevel.LOW: 1,
            SecurityLevel.MEDIUM: 2,
            SecurityLevel.HIGH: 3,
            SecurityLevel.CRITICAL: 4
        }
        return level_order.get(self.level, 0) >= level_order.get(required_level, 0)
    
    def get_policy_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        return {
            "name": self.name,
            "security_level": self.level.value if isinstance(self.level, SecurityLevel) else self.level,
            "description": self.description,
            "enabled": self.enabled,
            "is_active": self.is_active,
            "rules_count": len(self.rules),
            "created_at": self.created_at.isoformat() if hasattr(self.created_at, 'isoformat') else str(self.created_at),
            "updated_at": self.updated_at.isoformat() if hasattr(self.updated_at, 'isoformat') else str(self.updated_at)
        }
    
    def activate(self) -> None:
        """激活策略"""
        from datetime import datetime
        self.enabled = True
        self.is_active = True
        self.updated_at = datetime.now()
    
    def deactivate(self) -> None:
        """停用策略"""
        from datetime import datetime
        self.enabled = False
        self.is_active = False
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        info = self.get_policy_info()
        info['is_active'] = self.is_active
        return info
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityPolicy':
        """从字典创建策略"""
        name = data.get("name", "")
        level_value = data.get("security_level", "medium")
        description = data.get("description", "")
        
        # 转换安全级别
        if isinstance(level_value, str):
            level = SecurityLevel(level_value)
        else:
            level = level_value
        
        policy = cls(name=name, level=level, description=description)
        policy.enabled = data.get("enabled", True)
        policy.is_active = data.get("is_active", True)
        
        return policy


class BaseSecurity:
    """基础安全组件"""

    def __init__(self):
        """初始化基础安全组件"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._initialized = False
        self._security_level = SecurityLevel.MEDIUM

    def initialize(self) -> bool:
        """初始化安全组件"""
        try:
            if self._initialized:
                self.logger.warning("安全组件已经初始化")
                return True

            # 执行初始化逻辑
            self._initialized = True
            self.logger.info("安全组件初始化成功")
            return True
        except Exception as e:
            self.logger.error(f"安全组件初始化失败: {e}")
            return False

    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized

    def get_security_level(self) -> SecurityLevel:
        """获取安全级别"""
        return self._security_level

    def set_security_level(self, level: SecurityLevel) -> None:
        """设置安全级别"""
        if not isinstance(level, SecurityLevel):
            raise ValueError("安全级别必须是SecurityLevel枚举值")

        self._security_level = level
        self.logger.info(f"安全级别已设置为: {level.value}")

    def validate_security_context(self, context: Dict[str, Any]) -> bool:
        """验证安全上下文"""
        try:
            if not isinstance(context, dict):
                self.logger.warning("安全上下文必须是字典类型")
                return False

            # 基础安全验证逻辑
            required_fields = ['user_id', 'action', 'resource']
            for field in required_fields:
                if field not in context:
                    self.logger.warning(f"安全上下文中缺少必需字段: {field}")
                    return False

            return True
        except Exception as e:
            self.logger.error(f"验证安全上下文时发生错误: {e}")
            return False

    def log_security_event(self, event_type: SecurityEventType, details: Dict[str, Any]) -> None:
        """记录安全事件"""
        try:
            event = {
                'timestamp': logging.time.time(),
                'event_type': event_type.value,
                'details': details,
                'security_level': self._security_level.value
            }

            self.logger.info(f"安全事件: {event}")
        except Exception as e:
            self.logger.error(f"记录安全事件时发生错误: {e}")

    def cleanup(self) -> None:
        """清理资源"""
        try:
            self._initialized = False
            self.logger.info("安全组件已清理")
        except Exception as e:
            self.logger.error(f"清理安全组件时发生错误: {e}")
