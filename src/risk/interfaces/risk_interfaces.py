"""
RQA2025 风险控制层接口定义

本模块定义风险控制层提供的业务接口，
这些接口描述风险控制层提供的核心风险管理能力。

风险控制层职责：
1. 风险控制器接口 - 实时风险检查和控制
2. 合规检查接口 - 监管要求验证和报告
3. 风险监控接口 - 风险指标计算和监控
4. 风险报告接口 - 风险分析和报告生成
5. 异常处理接口 - 风险事件的检测和响应
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# 风险相关枚举
# =============================================================================

class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAction(Enum):
    """风险响应动作"""
    APPROVE = "approve"          # 批准
    REJECT = "reject"            # 拒绝
    LIMIT = "limit"              # 限制
    DELAY = "delay"              # 延迟
    ALERT = "alert"              # 告警
    FREEZE = "freeze"            # 冻结


class ComplianceStatus(Enum):
    """合规状态"""
    COMPLIANT = "compliant"      # 合规
    NON_COMPLIANT = "non_compliant"  # 不合规
    UNDER_REVIEW = "under_review"     # 审核中
    PENDING = "pending"          # 待处理


# =============================================================================
# 风险数据结构
# =============================================================================

@dataclass
class RiskCheckRequest:
    """风险检查请求"""
    symbol: str
    order_type: str
    quantity: int
    price: float
    portfolio_value: float
    current_position: int = 0
    account_id: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class RiskCheckResponse:
    """风险检查响应"""
    approved: bool
    risk_level: RiskLevel
    action: RiskAction
    message: Optional[str] = None
    limits: Optional[Dict[str, Any]] = None
    conditions: Optional[List[str]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class RiskMetrics:
    """风险指标"""
    value_at_risk: float  # VaR
    expected_shortfall: float  # ES
    max_drawdown: float  # 最大回撤
    sharpe_ratio: float  # 夏普比率
    volatility: float  # 波动率
    beta: float  # 贝塔系数
    correlation_matrix: Dict[str, Dict[str, float]]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ComplianceCheck:
    """合规检查"""
    check_type: str
    status: ComplianceStatus
    description: str
    violations: List[str]
    recommendations: List[str]
    checked_at: datetime = None
    next_check_due: Optional[datetime] = None

    def __post_init__(self):
        if self.checked_at is None:
            self.checked_at = datetime.now()


@dataclass
class RiskAlert:
    """风险告警"""
    alert_id: str
    alert_type: str
    severity: RiskLevel
    message: str
    affected_assets: List[str]
    suggested_actions: List[str]
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    status: str = "active"


# =============================================================================
# 风险控制器接口
# =============================================================================

class IRiskController(Protocol):
    """风险控制器接口 - 实时风险检查和控制"""

    @abstractmethod
    def check_risk(self, request: RiskCheckRequest) -> RiskCheckResponse:
        """执行风险检查"""

    @abstractmethod
    def calculate_position_limit(self, symbol: str, current_position: int,
                                 portfolio_value: float) -> Dict[str, Any]:
        """计算持仓限额"""

    @abstractmethod
    def validate_order_amount(self, symbol: str, order_amount: float,
                              available_balance: float) -> Dict[str, Any]:
        """验证订单金额"""

    @abstractmethod
    def check_market_risk(self, positions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """检查市场风险"""

    @abstractmethod
    def get_risk_limits(self) -> Dict[str, Any]:
        """获取风险限额设置"""

    @abstractmethod
    def update_risk_limits(self, limits: Dict[str, Any]) -> bool:
        """更新风险限额"""


# =============================================================================
# 合规检查接口
# =============================================================================

class IComplianceChecker(Protocol):
    """合规检查器接口 - 监管要求验证和报告"""

    @abstractmethod
    def check_trade_compliance(self, trade_details: Dict[str, Any]) -> ComplianceCheck:
        """检查交易合规性"""

    @abstractmethod
    def validate_position_limits(self, positions: Dict[str, Dict[str, Any]]) -> ComplianceCheck:
        """验证持仓限额合规"""

    @abstractmethod
    def check_regulatory_reporting(self, period: str) -> Dict[str, Any]:
        """检查监管报告要求"""

    @abstractmethod
    def generate_compliance_report(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """生成合规报告"""

    @abstractmethod
    def get_compliance_status(self) -> Dict[str, Any]:
        """获取合规状态"""


# =============================================================================
# 风险监控接口
# =============================================================================

class IRiskMonitor(Protocol):
    """风险监控器接口 - 风险指标计算和监控"""

    @abstractmethod
    def calculate_portfolio_risk(self, positions: Dict[str, Dict[str, Any]],
                                 market_data: Dict[str, Any]) -> RiskMetrics:
        """计算投资组合风险"""

    @abstractmethod
    def monitor_risk_limits(self) -> List[RiskAlert]:
        """监控风险限额"""

    @abstractmethod
    def detect_risk_anomalies(self, risk_metrics: RiskMetrics) -> List[Dict[str, Any]]:
        """检测风险异常"""

    @abstractmethod
    def get_risk_trends(self, days: int = 30) -> Dict[str, List[float]]:
        """获取风险趋势"""

    @abstractmethod
    def predict_risk_scenarios(self, positions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """预测风险情景"""


# =============================================================================
# 风险报告接口
# =============================================================================

class IRiskReporter(Protocol):
    """风险报告器接口 - 风险分析和报告生成"""

    @abstractmethod
    def generate_daily_risk_report(self, date: str) -> Dict[str, Any]:
        """生成日报风险报告"""

    @abstractmethod
    def generate_weekly_risk_report(self, week_start: str) -> Dict[str, Any]:
        """生成周报风险报告"""

    @abstractmethod
    def generate_monthly_risk_report(self, month: str) -> Dict[str, Any]:
        """生成月报风险报告"""

    @abstractmethod
    def generate_stress_test_report(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """生成压力测试报告"""

    @abstractmethod
    def export_risk_data(self, format: str = "json") -> Any:
        """导出风险数据"""


# =============================================================================
# 异常处理接口
# =============================================================================

class IRiskExceptionHandler(Protocol):
    """风险异常处理器接口 - 风险事件的检测和响应"""

    @abstractmethod
    def detect_risk_events(self) -> List[RiskAlert]:
        """检测风险事件"""

    @abstractmethod
    def handle_risk_alert(self, alert: RiskAlert) -> Dict[str, Any]:
        """处理风险告警"""

    @abstractmethod
    def escalate_risk_issue(self, issue: Dict[str, Any]) -> bool:
        """升级风险问题"""

    @abstractmethod
    def resolve_risk_alert(self, alert_id: str, resolution: Dict[str, Any]) -> bool:
        """解决风险告警"""

    @abstractmethod
    def get_alert_history(self, days: int = 7) -> List[RiskAlert]:
        """获取告警历史"""


# =============================================================================
# 风险管理服务提供者接口
# =============================================================================

class IRiskManagementServiceProvider(Protocol):
    """风险管理服务提供者接口 - 风险控制层的统一服务访问点"""

    @property
    def risk_controller(self) -> IRiskController:
        """风险控制器"""

    @property
    def compliance_checker(self) -> IComplianceChecker:
        """合规检查器"""

    @property
    def risk_monitor(self) -> IRiskMonitor:
        """风险监控器"""

    @property
    def risk_reporter(self) -> IRiskReporter:
        """风险报告器"""

    @property
    def exception_handler(self) -> IRiskExceptionHandler:
        """异常处理器"""

    @abstractmethod
    def get_service_status(self) -> str:
        """获取风险管理服务整体状态"""

    @abstractmethod
    def get_risk_overview(self) -> Dict[str, Any]:
        """获取风险总览"""

    @abstractmethod
    def enable_emergency_mode(self) -> bool:
        """启用紧急模式"""

    @abstractmethod
    def disable_emergency_mode(self) -> bool:
        """禁用紧急模式"""
