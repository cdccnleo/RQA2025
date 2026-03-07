#!/usr/bin/env python3
"""
# 跨境交易合规管理误
支持跨国误/ 地区的合规检查、税务申报、外汇管制等功能
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class Country(Enum):

    """国家 / 地区枚举"""
    CHINA = "CN"
    USA = "US"
    JAPAN = "JP"
    UK = "GB"
    GERMANY = "DE"
    FRANCE = "FR"
    HONG_KONG = "HK"
    SINGAPORE = "SG"
    AUSTRALIA = "AU"
    CANADA = "CA"
    SOUTH_KOREA = "KR"
    INDIA = "IN"
    BRAZIL = "BR"
    RUSSIA = "RU"
    UAE = "AE"


class Currency(Enum):

    """货币枚举"""
    CNY = "CNY"  # 人民币
    USD = "USD"  # 美元
    EUR = "EUR"  # 欧元
    JPY = "JPY"  # 日元
    GBP = "GBP"  # 英镑
    HKD = "HKD"  # 港元
    SGD = "SGD"  # 新加坡元
    AUD = "AUD"  # 澳大利亚元
    CAD = "CAD"  # 加拿大元
    KRW = "KRW"  # 韩元
    INR = "INR"  # 印度卢比
    BRL = "BRL"  # 巴西雷亚尔
    RUB = "RUB"  # 俄罗斯卢布
    AED = "AED"  # 阿联酋迪拉姆


class ComplianceType(Enum):

    """合规类型枚举"""
    FX_CONTROL = "fx_control"              # 外汇管制
    TAX_REPORTING = "tax_reporting"        # 税务申报
    TRADE_RESTRICTIONS = "trade_restrictions"  # 交易限制
    CAPITAL_FLOWS = "capital_flows"        # 资本流动
    AML_KYC = "aml_kyc"                    # 反洗误/ KYC
    POSITION_REPORTING = "position_reporting"  # 持仓报告
    DERIVATIVE_REGULATION = "derivative_regulation"  # 衍生品监误


@dataclass
class CrossBorderTransaction:

    """跨境交易"""
    transaction_id: str
    from_country: Country
    to_country: Country
    from_currency: Currency
    to_currency: Currency
    amount: float
    transaction_type: str  # buy, sell, transfer, etc.
    asset_type: str
    asset_symbol: str
    counterparty: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceRule:

    """合规规则"""
    rule_id: str
    country: Country
    compliance_type: ComplianceType
    name: str
    description: str
    conditions: Dict[str, Any]
    actions: List[str]
    severity: str = "medium"  # low, medium, high, critical
    enabled: bool = True
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ComplianceCheckResult:

    """合规检查结果"""
    transaction_id: str
    rule_id: str
    passed: bool
    risk_level: str
    message: str
    required_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaxReporting:

    """税务申报"""
    reporting_id: str
    country: Country
    tax_year: int
    transaction_count: int
    total_amount: float
    taxable_amount: float
    tax_due: float
    currency: Currency
    status: str = "pending"  # pending, submitted, approved, rejected
    due_date: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=90))
    submission_date: Optional[datetime] = None


class CrossBorderComplianceManager:

    """跨境合规管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.lock = threading.RLock()

        # 合规规则存储
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.country_rules: Dict[Country, List[ComplianceRule]] = defaultdict(list)

        # 交易记录
        self.transactions: Dict[str, CrossBorderTransaction] = {}
        self.transaction_history = defaultdict(lambda: deque(maxlen=10000))

        # 检查结果缓存
        self.check_results = defaultdict(lambda: deque(maxlen=5000))

        # 税务申报记录
        self.tax_reports: Dict[str, TaxReporting] = {}

        # 初始化默认合规规则
        self._initialize_default_rules()

        logger.info("跨境合规管理器初始化完成")

    def _initialize_default_rules(self):
        """初始化默认合规规则"""
        try:
            # 中国外汇管制规则
            china_fx_rule = ComplianceRule(
                rule_id="CN_FX_CONTROL_001",
                country=Country.CHINA,
                compliance_type=ComplianceType.FX_CONTROL,
                name="中国外汇管制",
                description="中国外汇管制合规检查",
                conditions={
                    "max_daily_fx_amount": 50000,  # 每日外汇限额（USD）
                    "max_yearly_fx_amount": 500000,  # 年度外汇限额（USD）
                    "restricted_currencies": [],  # 受限货币列表
                    "approval_required_threshold": 10000  # 需要审批的金额阈值
                },
                actions=[
                    "验证外汇额度",
                    "检查外汇用途",
                    "提交外汇申请",
                    "生成外汇备案"
                ],
                severity="high"
            )
            self._add_rule(china_fx_rule)

            # 美国税务申报规则
            us_tax_rule = ComplianceRule(
                rule_id="US_TAX_REPORTING_001",
                country=Country.USA,
                compliance_type=ComplianceType.TAX_REPORTING,
                name="美国税务申报",
                description="FATCA税务申报要求",
                conditions={
                    "reporting_threshold": 10000,  # 申报阈值（USD）
                    "reporting_period": "annual",  # 申报周期
                    "required_forms": ["FATCA Form 8938", "FBAR"],
                    "due_date": "March 15"  # 申报截止日期
                },
                actions=[
                    "识别美国纳税人",
                    "计算应税收入",
                    "生成申报表格",
                    "提交税务申报"
                ],
                severity="critical"
            )
            self._add_rule(us_tax_rule)

            # 欧盟MiFID II合规
            eu_mifid_rule = ComplianceRule(
                rule_id="EU_MIFID_COMPLIANCE_001",
                country=Country.UK,  # 代表欧盟
                compliance_type=ComplianceType.POSITION_REPORTING,
                name="欧盟MiFID II合规",
                description="欧盟金融工具市场指令II合规要求",
                conditions={
                    "reporting_frequency": "daily",
                    "position_threshold": 500000,  # 持仓报告阈值（EUR）
                    "instrument_types": ["shares", "bonds", "derivatives"],
                    "reporting_fields": ["ISIN", "quantity", "price", "timestamp"]
                },
                actions=[
                    "实时持仓监控",
                    "生成交易报告",
                    "提交监管报告",
                    "记录审计日志"
                ],
                severity="high"
            )
            self._add_rule(eu_mifid_rule)

            # 新加坡反洗钱要求
            sg_aml_rule = ComplianceRule(
                rule_id="SG_AML_KYC_001",
                country=Country.SINGAPORE,
                compliance_type=ComplianceType.AML_KYC,
                name="新加坡反洗钱合规",
                description="新加坡MAS反洗钱和KYC要求",
                conditions={
                    "kyc_refresh_period": 365,  # KYC更新周期（天）
                    "pep_check_required": True,  # 需要PEP检查
                    "sanction_check_required": True,  # 需要制裁检查
                    "source_of_funds_check": True,  # 需要资金来源检查
                    "transaction_monitoring_threshold": 10000  # 交易监控阈值（SGD）
                },
                actions=[
                    "客户身份验证",
                    "PEP和制裁筛选",
                    "资金来源审查",
                    "交易行为监控",
                    "可疑活动报告"
                ],
                severity="critical"
            )
            self._add_rule(sg_aml_rule)

            logger.info("默认合规规则初始化完成")

        except Exception as e:
            logger.error(f"初始化默认规则失败 {e}")

    def _add_rule(self, rule: ComplianceRule):
        """添加合规规则"""
        try:
            self.compliance_rules[rule.rule_id] = rule
            self.country_rules[rule.country].append(rule)
            logger.debug(f"添加合规规则: {rule.rule_id}")
        except Exception as e:
            logger.error(f"添加合规规则失败: {e}")

    def register_transaction(self, transaction: Dict[str, Any]) -> str:
        """
        注册跨境交易

        Args:
            transaction: 交易数据字典

        Returns:
            交易ID
        """
        with self.lock:
            try:
                transaction_id = transaction.get('transaction_id',
                                                 f"CBT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(transaction))}")

                # 创建交易对象
                cross_border_tx = CrossBorderTransaction(
                    transaction_id=transaction_id,
                    from_country=Country(transaction['from_country']),
                    to_country=Country(transaction['to_country']),
                    from_currency=Currency(transaction['from_currency']),
                    to_currency=Currency(transaction['to_currency']),
                    amount=transaction['amount'],
                    transaction_type=transaction['transaction_type'],
                    asset_type=transaction.get('asset_type', 'cash'),
                    asset_symbol=transaction.get('asset_symbol', ''),
                    counterparty=transaction.get('counterparty', ''),
                    timestamp=transaction.get('timestamp', datetime.now()),
                    metadata=transaction.get('metadata', {})
                )

                # 存储交易
                self.transactions[transaction_id] = cross_border_tx
                self.transaction_history[cross_border_tx.from_country].append(cross_border_tx)

                logger.info(f"跨境交易已注册 {transaction_id}")
                return transaction_id

            except Exception as e:
                logger.error(f"注册跨境交易失败: {e}")
                raise

    def check_compliance(self, transaction_id: str) -> List[ComplianceCheckResult]:
        """
        检查交易合规性
        Args:
            transaction_id: 交易ID

        Returns:
            合规检查结果列表
        """
        with self.lock:
            try:
                if transaction_id not in self.transactions:
                    raise ValueError(f"交易不存在 {transaction_id}")

                transaction = self.transactions[transaction_id]
                results = []

                # 获取相关的合规规则
                relevant_rules = self._get_relevant_rules(transaction)

                # 对每个规则进行检查
                for rule in relevant_rules:
                    result = self._check_single_rule(transaction, rule)
                    results.append(result)

                    # 缓存检查结果
                    self.check_results[transaction_id].append(result)

                logger.info(f"合规检查完成 {transaction_id}, 检查了 {len(results)} 项规则")
                return results

            except Exception as e:
                logger.error(f"合规检查失败 {e}")
                raise

    def _get_relevant_rules(self, transaction: CrossBorderTransaction) -> List[ComplianceRule]:
        """获取相关合规规则"""
        try:
            relevant_rules = []

            # 获取出发国家的规则
            from_country_rules = self.country_rules.get(transaction.from_country, [])
            relevant_rules.extend(from_country_rules)

            # 获取目的国家的规则
            to_country_rules = self.country_rules.get(transaction.to_country, [])
            relevant_rules.extend(to_country_rules)

            # 根据交易类型过滤规则
            filtered_rules = []
            for rule in relevant_rules:
                if self._is_rule_relevant(rule, transaction):
                    filtered_rules.append(rule)

            return filtered_rules

        except Exception as e:
            logger.error(f"获取相关规则失败: {e}")
            return []

    def _is_rule_relevant(self, rule: ComplianceRule, transaction: CrossBorderTransaction) -> bool:
        """判断规则是否相关"""
        try:
            # 根据合规类型判断相关性
            if rule.compliance_type == ComplianceType.FX_CONTROL:
                # 外汇管制对货币兑换交易相关
                return (transaction.transaction_type in ['fx_exchange', 'transfer'] or
                        transaction.from_currency != transaction.to_currency)

            elif rule.compliance_type == ComplianceType.TAX_REPORTING:
                # 税务申报对大多数跨境交易相关
                return transaction.amount > rule.conditions.get('reporting_threshold', 0)

            elif rule.compliance_type == ComplianceType.AML_KYC:
                # 反洗钱对所有跨境交易相关
                return True

            elif rule.compliance_type == ComplianceType.POSITION_REPORTING:
                # 持仓报告对投资交易相关
                return transaction.transaction_type in ['buy', 'sell', 'position']

            return True

        except Exception as e:
            logger.error(f"判断规则相关性失败 {e}")
            return False

    def _check_single_rule(self, transaction: CrossBorderTransaction,
                           rule: ComplianceRule) -> ComplianceCheckResult:
        """检查单个规则"""
        try:
            passed = True
            risk_level = "low"
            message = "合规检查通过"
            required_actions = []

            # 根据规则类型进行具体检查
            if rule.compliance_type == ComplianceType.FX_CONTROL:
                passed, risk_level, message, required_actions = self._check_fx_control(
                    transaction, rule)

            elif rule.compliance_type == ComplianceType.TAX_REPORTING:
                passed, risk_level, message, required_actions = self._check_tax_reporting(
                    transaction, rule)

            elif rule.compliance_type == ComplianceType.AML_KYC:
                passed, risk_level, message, required_actions = self._check_aml_kyc(
                    transaction, rule)

            elif rule.compliance_type == ComplianceType.POSITION_REPORTING:
                passed, risk_level, message, required_actions = self._check_position_reporting(
                    transaction, rule)

            # 创建检查结果
            result = ComplianceCheckResult(
                transaction_id=transaction.transaction_id,
                rule_id=rule.rule_id,
                passed=passed,
                risk_level=risk_level,
                message=message,
                required_actions=required_actions,
                timestamp=datetime.now(),
                details={
                    'rule_name': rule.name,
                    'rule_description': rule.description,
                    'compliance_type': rule.compliance_type.value,
                    'country': rule.country.value
                }
            )

            return result

        except Exception as e:
            logger.error(f"检查单个规则失败 {e}")
            return ComplianceCheckResult(
                transaction_id=transaction.transaction_id,
                rule_id=rule.rule_id,
                passed=False,
                risk_level="high",
                message=f"规则检查失败 {e}",
                required_actions=["手动审查"],
                timestamp=datetime.now()
            )

    def _check_fx_control(self, transaction: CrossBorderTransaction,
                          rule: ComplianceRule) -> Tuple[bool, str, str, List[str]]:
        """检查外汇管制"""
        try:
            conditions = rule.conditions
            max_daily_amount = conditions.get('max_daily_fx_amount', 50000)
            approval_threshold = conditions.get('approval_required_threshold', 10000)

            # 检查每日限额（简化版）
            if transaction.amount > max_daily_amount:
                return False, "high", f"超过外汇每日限额 {max_daily_amount}", ["申请外汇额度", "提交外汇用途证明"]

            # 检查是否需要审批
            if transaction.amount > approval_threshold:
                return True, "medium", f"需要外汇审批，金额 {transaction.amount}", ["提交外汇审批申请"]

            return True, "low", "外汇管制检查通过", []

        except Exception as e:
            logger.error(f"外汇管制检查失败 {e}")
            return False, "high", f"外汇管制检查失败 {e}", ["手动审查"]

    def _check_tax_reporting(self, transaction: CrossBorderTransaction,
                             rule: ComplianceRule) -> Tuple[bool, str, str, List[str]]:
        """检查税务申报"""
        try:
            conditions = rule.conditions
            reporting_threshold = conditions.get('reporting_threshold', 10000)

            if transaction.amount > reporting_threshold:
                return True, "medium", f"需要税务申报，金额 {transaction.amount}", ["生成税务申报", "提交税务文件"]

            return True, "low", "税务申报检查通过", []

        except Exception as e:
            logger.error(f"税务申报检查失败 {e}")
            return False, "high", f"税务申报检查失败 {e}", ["手动审查"]

    def _check_aml_kyc(self, transaction: CrossBorderTransaction,
                       rule: ComplianceRule) -> Tuple[bool, str, str, List[str]]:
        """检查反洗钱 / KYC"""
        try:
            # 简化版检查逻辑
            counterparty = transaction.counterparty

            # 检查是否是高风险国家
            high_risk_countries = ['North Korea', 'Iran', 'Venezuela']  # 简化列表
            if any(country.lower() in counterparty.lower() for country in high_risk_countries):
                return False, "critical", "涉及高风险国家交易对方", ["加强尽职调查", "上报监管机构"]

            # 检查交易金额是否异常
            monitoring_threshold = rule.conditions.get('transaction_monitoring_threshold', 10000)
            if transaction.amount > monitoring_threshold:
                return True, "medium", f"大额交易监控，金额{transaction.amount}", ["记录交易详情", "监控交易模式"]

            return True, "low", "反洗钱检查通过", []

        except Exception as e:
            logger.error(f"反洗钱检查失败 {e}")
            return False, "high", f"反洗钱检查失败 {e}", ["手动审查"]

    def _check_position_reporting(self, transaction: CrossBorderTransaction,
                                  rule: ComplianceRule) -> Tuple[bool, str, str, List[str]]:
        """检查持仓报告"""
        try:
            conditions = rule.conditions
            position_threshold = conditions.get('position_threshold', 500000)

            if transaction.amount > position_threshold:
                return True, "medium", f"需要持仓报告，金额 {transaction.amount}", ["生成持仓报告", "提交监管机构"]

            return True, "low", "持仓报告检查通过", []

        except Exception as e:
            logger.error(f"持仓报告检查失败 {e}")
            return False, "high", f"持仓报告检查失败 {e}", ["手动审查"]

    def generate_tax_report(self, country: Country, tax_year: int,
                            transactions: List[Dict[str, Any]]) -> TaxReporting:
        """
        生成税务申报

        Args:
            country: 国家
            tax_year: 税务年度
            transactions: 交易列表

        Returns:
            税务申报对象
        """
        with self.lock:
            try:
                reporting_id = f"TAX_{country.value}_{tax_year}_{datetime.now().strftime('%Y%m%d')}"

                # 计算税务数据
                total_amount = sum(tx['amount'] for tx in transactions)
                taxable_amount = self._calculate_taxable_amount(country, transactions)
                tax_due = self._calculate_tax_due(country, taxable_amount)

                # 创建税务申报
                tax_report = TaxReporting(
                    reporting_id=reporting_id,
                    country=country,
                    tax_year=tax_year,
                    transaction_count=len(transactions),
                    total_amount=total_amount,
                    taxable_amount=taxable_amount,
                    tax_due=tax_due,
                    currency=self._get_country_currency(country)
                )

                # 存储税务申报
                self.tax_reports[reporting_id] = tax_report

                logger.info(f"税务申报生成完成: {reporting_id}")
                return tax_report

            except Exception as e:
                logger.error(f"生成税务申报失败: {e}")
                raise

    def _calculate_taxable_amount(self, country: Country, transactions: List[Dict[str, Any]]) -> float:
        """计算应税金额"""
        try:
            # 简化版税务计算逻辑
            if country == Country.USA:
                # 美国FATCA规则
                taxable_amount = sum(tx['amount'] for tx in transactions
                                     if tx.get('us_person', False))
            elif country == Country.CHINA:
                # 中国税务规则
                taxable_amount = sum(tx['amount'] for tx in transactions
                                     if tx['amount'] > 50000)  # 超过5万人民币
            else:
                # 默认规则
                taxable_amount = sum(tx['amount'] for tx in transactions
                                     if tx['amount'] > 10000)

            return taxable_amount

        except Exception as e:
            logger.error(f"计算应税金额失败: {e}")
            return 0.0

    def _calculate_tax_due(self, country: Country, taxable_amount: float) -> float:
        """计算应纳税款"""
        try:
            # 简化版税率计算
            tax_rates = {
                Country.USA: 0.30,      # 美国资本利得税
                Country.CHINA: 0.20,    # 中国资本利得税
                Country.UK: 0.20,       # 英国资本利得税
                Country.JAPAN: 0.20,    # 日本资本利得税
                Country.GERMANY: 0.26,  # 德国资本利得税
                Country.HONG_KONG: 0.0,  # 香港无资本利得税
                Country.SINGAPORE: 0.0  # 新加坡无资本利得税
            }

            tax_rate = tax_rates.get(country, 0.20)  # 默认20%
            return taxable_amount * tax_rate

        except Exception as e:
            logger.error(f"计算应纳税款失败: {e}")
            return 0.0

    def _get_country_currency(self, country: Country) -> Currency:
        """获取国家主要货币"""
        currency_mapping = {
            Country.CHINA: Currency.CNY,
            Country.USA: Currency.USD,
            Country.UK: Currency.GBP,
            Country.JAPAN: Currency.JPY,
            Country.GERMANY: Currency.EUR,
            Country.FRANCE: Currency.EUR,
            Country.HONG_KONG: Currency.HKD,
            Country.SINGAPORE: Currency.SGD,
            Country.AUSTRALIA: Currency.AUD,
            Country.CANADA: Currency.CAD,
            Country.SOUTH_KOREA: Currency.KRW,
            Country.INDIA: Currency.INR,
            Country.BRAZIL: Currency.BRL,
            Country.RUSSIA: Currency.RUB,
            Country.UAE: Currency.AED
        }

        return currency_mapping.get(country, Currency.USD)

    def get_compliance_status(self, transaction_id: str) -> Dict[str, Any]:
        """
        获取交易合规状态
        Args:
            transaction_id: 交易ID

        Returns:
            合规状态信息
        """
        try:
            if transaction_id not in self.check_results:
                return {"status": "not_checked", "message": "尚未进行合规检查"}

            results = list(self.check_results[transaction_id])

            # 汇总检查结果
            total_checks = len(results)
            passed_checks = sum(1 for r in results if r.passed)
            failed_checks = total_checks - passed_checks

            # 确定整体风险级别
            risk_levels = [r.risk_level for r in results if not r.passed]
            overall_risk = "low"
            if "critical" in risk_levels:
                overall_risk = "critical"
            elif "high" in risk_levels:
                overall_risk = "high"
            elif "medium" in risk_levels:
                overall_risk = "medium"

            # 汇总所需行动
            all_actions = []
            for result in results:
                all_actions.extend(result.required_actions)
            unique_actions = list(set(all_actions))

            return {
                "status": "compliant" if failed_checks == 0 else "non_compliant",
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "overall_risk": overall_risk,
                "required_actions": unique_actions,
                "last_check": max(r.timestamp for r in results).isoformat() if results else None
            }

        except Exception as e:
            logger.error(f"获取合规状态失败 {e}")
            return {"status": "error", "message": str(e)}

    def generate_compliance_report(self, country: Country,
                                   start_date: datetime,
                                   end_date: datetime) -> Dict[str, Any]:
        """
        生成合规报告

        Args:
            country: 国家
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            合规报告
        """
        try:
            # 获取时间范围内的交易
            country_transactions = list(self.transaction_history[country])
            period_transactions = [
                tx for tx in country_transactions
                if start_date <= tx.timestamp <= end_date
            ]

            # 获取相关的合规检查结果
            compliance_results = []
            for tx in period_transactions:
                if tx.transaction_id in self.check_results:
                    compliance_results.extend(list(self.check_results[tx.transaction_id]))

            # 生成报告
            report = {
                "report_type": "compliance_report",
                "country": country.value,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "summary": {
                    "total_transactions": len(period_transactions),
                    "total_compliance_checks": len(compliance_results),
                    "passed_checks": sum(1 for r in compliance_results if r.passed),
                    "failed_checks": sum(1 for r in compliance_results if not r.passed)
                },
                "transactions": [
                    {
                        "transaction_id": tx.transaction_id,
                        "amount": tx.amount,
                        "type": tx.transaction_type,
                        "timestamp": tx.timestamp.isoformat()
                    }
                    for tx in period_transactions
                ],
                "compliance_issues": [
                    {
                        "transaction_id": r.transaction_id,
                        "rule_id": r.rule_id,
                        "risk_level": r.risk_level,
                        "message": r.message,
                        "actions": r.required_actions
                    }
                    for r in compliance_results if not r.passed
                ],
                "generated_at": datetime.now().isoformat()
            }

            return report

        except Exception as e:
            logger.error(f"生成合规报告失败: {e}")
            return {"error": str(e)}

    def add_compliance_rule(self, rule_config: Dict[str, Any]):
        """
        添加合规规则

        Args:
            rule_config: 规则配置字典
        """
        with self.lock:
            try:
                rule = ComplianceRule(
                    rule_id=rule_config['rule_id'],
                    country=Country(rule_config['country']),
                    compliance_type=ComplianceType(rule_config['compliance_type']),
                    name=rule_config['name'],
                    description=rule_config['description'],
                    conditions=rule_config['conditions'],
                    actions=rule_config['actions'],
                    severity=rule_config.get('severity', 'medium'),
                    enabled=rule_config.get('enabled', True)
                )

                self._add_rule(rule)
                logger.info(f"合规规则添加成功: {rule.rule_id}")

            except Exception as e:
                logger.error(f"添加合规规则失败: {e}")
                raise

    def get_supported_countries(self) -> List[str]:
        """
        获取支持的国家列表
        Returns:
            支持的国家代码列表
        """
        return [country.value for country in Country]

    def get_country_rules(self, country: Country) -> List[Dict[str, Any]]:
        """
        获取国家的合规规则
        Args:
            country: 国家

        Returns:
            规则列表
        """
        try:
            rules = self.country_rules.get(country, [])
            return [
                {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "description": rule.description,
                    "compliance_type": rule.compliance_type.value,
                    "severity": rule.severity,
                    "enabled": rule.enabled
                }
                for rule in rules
            ]

        except Exception as e:
            logger.error(f"获取国家规则失败: {e}")
            return []
