#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 数据生态系统管理器

构建完整的数据生态体系：
- 数据目录和元数据管理
- 数据血缘追踪
- 数据质量门户
- 数据治理框架
- 数据共享和协作平台
- 数据市场和交易平台
"""

import uuid
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import time

from ..infrastructure_integration_manager import (
    get_data_integration_manager,
    log_data_operation, record_data_metric, publish_data_event
)
from ..interfaces.standard_interfaces import DataSourceType


@dataclass
class DataAsset:

    """数据资产"""
    asset_id: str
    name: str
    description: str
    data_type: DataSourceType
    owner: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    version: str = "1.0.0"


@dataclass
class DataLineage:

    """数据血缘"""
    lineage_id: str
    source_asset_id: str
    target_asset_id: str
    transformation_type: str
    transformation_details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    execution_time: Optional[float] = None
    success: bool = True


@dataclass
class DataContract:

    """数据契约"""
    contract_id: str
    provider_asset_id: str
    consumer_id: str
    sla_requirements: Dict[str, Any] = field(default_factory=dict)
    quality_requirements: Dict[str, Any] = field(default_factory=dict)
    access_permissions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    status: str = "active"  # active, expired, terminated


@dataclass
class DataMarketItem:

    """数据市场商品"""
    item_id: str
    asset_id: str
    title: str
    description: str
    price: float
    currency: str = "CNY"
    category: str = "financial_data"
    tags: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    download_count: int = 0
    rating: float = 0.0
    reviews_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    published: bool = False


@dataclass
class EcosystemConfig:

    """生态系统配置"""
    enable_data_catalog: bool = True
    enable_lineage_tracking: bool = True
    enable_data_contracts: bool = True
    enable_data_marketplace: bool = True
    enable_collaboration: bool = True
    enable_data_governance: bool = True
    catalog_update_interval: int = 3600  # 1小时
    lineage_retention_days: int = 90
    contract_monitoring_interval: int = 1800  # 30分钟
    marketplace_commission_rate: float = 0.05  # 5 % 佣金


class DataEcosystemManager:

    """
    数据生态系统管理器

    构建完整的数据生态体系：
    - 数据目录管理：资产发现、分类、搜索
    - 数据血缘追踪：数据流转路径追踪
    - 数据契约管理：服务级别协议和质量保证
    - 数据市场：数据商品化交易平台
    - 数据协作：团队协作和知识共享
    - 数据治理：合规管理和质量控制
    """

    def __init__(self, config: Optional[EcosystemConfig] = None):
        """
        初始化数据生态系统管理器

        Args:
            config: 生态系统配置
        """
        # 使用基础设施集成管理器获取配置
        self.config_obj = config or EcosystemConfig()
        merged_config = self._load_config_from_integration_manager()
        self.config = EcosystemConfig(**merged_config)

        # 初始化基础设施集成管理器
        self.integration_manager = get_data_integration_manager()
        if not self.integration_manager._initialized:
            self.integration_manager.initialize()

        # 数据资产目录
        self.data_assets: Dict[str, DataAsset] = {}

        # 数据血缘图
        self.data_lineage: Dict[str, List[DataLineage]] = defaultdict(list)

        # 数据契约
        self.data_contracts: Dict[str, DataContract] = {}

        # 数据市场
        self.marketplace_items: Dict[str, DataMarketItem] = {}

        # 用户会话和权限
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.user_permissions: Dict[str, Set[str]] = defaultdict(set)

        # 统计信息
        self.stats = {
            'total_assets': 0,
            'total_lineages': 0,
            'total_contracts': 0,
            'total_market_items': 0,
            'active_users': 0,
            'data_access_events': 0
        }

        # 监控线程
        self.monitoring_thread = None
        self._stop_monitoring = False

        # 初始化生态系统
        self._initialize_ecosystem()

        # 启动监控
        if self.config.enable_data_governance:
            self._start_monitoring()

        # 注册健康检查
        self._register_health_checks()

        log_data_operation("ecosystem_manager_init", DataSourceType.STOCK,
                           {"config": self.config.__dict__}, "info")

    def _load_config_from_integration_manager(self) -> Dict[str, Any]:
        """从基础设施集成管理器加载配置"""
        try:
            merged_config = self.config_obj.__dict__.copy()

            # 从基础设施集成管理器获取配置
            if hasattr(self.integration_manager, '_integration_config'):
                infra_config = self.integration_manager._integration_config
                merged_config.update({
                    'enable_data_catalog': infra_config.get('enable_data_catalog', self.config_obj.enable_data_catalog),
                    'enable_data_marketplace': infra_config.get('enable_data_marketplace', self.config_obj.enable_data_marketplace),
                    'catalog_update_interval': infra_config.get('catalog_update_interval', self.config_obj.catalog_update_interval)
                })

            return merged_config

        except Exception as e:
            return self.config_obj.__dict__.copy()

    def _register_health_checks(self) -> None:
        """注册健康检查"""
        try:
            health_bridge = self.integration_manager.get_health_check_bridge()
            if health_bridge:
                health_bridge.register_data_health_check(
                    "data_ecosystem",
                    self._ecosystem_health_check,
                    DataSourceType.STOCK
                )

        except Exception as e:
            log_data_operation("ecosystem_health_check_registration_error", DataSourceType.STOCK,
                               {"error": str(e)}, "warning")

    def _initialize_ecosystem(self) -> None:
        """初始化生态系统"""
        try:
            # 创建默认分类和标签
            self.categories = {
                'financial_data': '金融数据',
                'market_data': '市场数据',
                'alternative_data': '另类数据',
                'research_data': '研究数据',
                'analytics': '分析结果'
            }

            self.default_tags = [
                'high_quality', 'verified', 'real_time', 'historical',
                'structured', 'unstructured', 'public', 'private'
            ]

            log_data_operation("ecosystem_initialized", DataSourceType.STOCK, {}, "info")

        except Exception as e:
            log_data_operation("ecosystem_initialization_error", DataSourceType.STOCK,
                               {"error": str(e)}, "error")

    def _start_monitoring(self) -> None:
        """启动监控"""
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name="EcosystemMonitor"
        )
        self.monitoring_thread.start()

        log_data_operation("ecosystem_monitoring_started", DataSourceType.STOCK, {}, "info")

    def register_data_asset(self, name: str, description: str, data_type: DataSourceType,


                            owner: str, tags: Optional[List[str]] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        注册数据资产

        Args:
            name: 资产名称
            description: 资产描述
            data_type: 数据类型
            owner: 所有者
            tags: 标签列表
            metadata: 元数据

        Returns:
            资产ID
        """
        try:
            asset_id = str(uuid.uuid4())

            asset = DataAsset(
                asset_id=asset_id,
                name=name,
                description=description,
                data_type=data_type,
                owner=owner,
                tags=tags or [],
                metadata=metadata or {},
                quality_score=0.0  # 初始质量分数
            )

            self.data_assets[asset_id] = asset
            self.stats['total_assets'] += 1

            log_data_operation("data_asset_registered", data_type,
                               {
                                   "asset_id": asset_id,
                                   "name": name,
                                   "owner": owner
                               }, "info")

            # 发布资产注册事件
            publish_data_event("data_asset_registered", {
                "asset_id": asset_id,
                "name": name,
                "data_type": data_type.value,
                "owner": owner
            }, data_type, "normal")

            return asset_id

        except Exception as e:
            log_data_operation("data_asset_registration_error", data_type,
                               {"error": str(e)}, "error")
            raise

    def update_data_quality(self, asset_id: str, quality_score: float,


                            quality_metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        更新数据质量

        Args:
            asset_id: 资产ID
            quality_score: 质量分数
            quality_metrics: 质量指标

        Returns:
            是否更新成功
        """
        try:
            if asset_id not in self.data_assets:
                raise ValueError(f"数据资产不存在: {asset_id}")

            asset = self.data_assets[asset_id]
            asset.quality_score = quality_score
            asset.last_updated = datetime.now()

            if quality_metrics:
                asset.metadata['quality_metrics'] = quality_metrics

            log_data_operation("data_quality_updated", asset.data_type,
                               {
                                   "asset_id": asset_id,
                                   "quality_score": quality_score
                               }, "info")

            # 记录质量指标
            record_data_metric("data_quality_score", quality_score, asset.data_type,
                               {"asset_id": asset_id})

            return True

        except Exception as e:
            log_data_operation("data_quality_update_error", DataSourceType.STOCK,
                               {"asset_id": asset_id, "error": str(e)}, "error")
            return False

    def track_data_lineage(self, source_asset_id: str, target_asset_id: str,


                           transformation_type: str, transformation_details: Optional[Dict[str, Any]] = None,
                           execution_time: Optional[float] = None, success: bool = True) -> str:
        """
        追踪数据血缘

        Args:
            source_asset_id: 源资产ID
            target_asset_id: 目标资产ID
            transformation_type: 转换类型
            transformation_details: 转换详情
            execution_time: 执行时间
            success: 是否成功

        Returns:
            血缘ID
        """
        try:
            lineage_id = str(uuid.uuid4())

            lineage = DataLineage(
                lineage_id=lineage_id,
                source_asset_id=source_asset_id,
                target_asset_id=target_asset_id,
                transformation_type=transformation_type,
                transformation_details=transformation_details or {},
                execution_time=execution_time,
                success=success
            )

            self.data_lineage[target_asset_id].append(lineage)
            self.stats['total_lineages'] += 1

            log_data_operation("data_lineage_tracked", DataSourceType.STOCK,
                               {
                                   "lineage_id": lineage_id,
                                   "source": source_asset_id,
                                   "target": target_asset_id,
                                   "transformation": transformation_type
                               }, "info")

            return lineage_id

        except Exception as e:
            log_data_operation("data_lineage_tracking_error", DataSourceType.STOCK,
                               {"error": str(e)}, "error")
            raise

    def create_data_contract(self, provider_asset_id: str, consumer_id: str,


                             sla_requirements: Optional[Dict[str, Any]] = None,
                             quality_requirements: Optional[Dict[str, Any]] = None,
                             access_permissions: Optional[Dict[str, Any]] = None,
                             expires_at: Optional[datetime] = None) -> str:
        """
        创建数据契约

        Args:
            provider_asset_id: 提供者资产ID
            consumer_id: 消费者ID
            sla_requirements: SLA要求
            quality_requirements: 质量要求
            access_permissions: 访问权限
            expires_at: 过期时间

        Returns:
            契约ID
        """
        try:
            if provider_asset_id not in self.data_assets:
                raise ValueError(f"数据资产不存在: {provider_asset_id}")

            contract_id = str(uuid.uuid4())

            contract = DataContract(
                contract_id=contract_id,
                provider_asset_id=provider_asset_id,
                consumer_id=consumer_id,
                sla_requirements=sla_requirements or {},
                quality_requirements=quality_requirements or {},
                access_permissions=access_permissions or {},
                expires_at=expires_at
            )

            self.data_contracts[contract_id] = contract
            self.stats['total_contracts'] += 1

            log_data_operation("data_contract_created", self.data_assets[provider_asset_id].data_type,
                               {
                "contract_id": contract_id,
                "provider": provider_asset_id,
                "consumer": consumer_id
            }, "info")

            return contract_id

        except Exception as e:
            log_data_operation("data_contract_creation_error", DataSourceType.STOCK,
                               {"error": str(e)}, "error")
            raise

    def publish_to_marketplace(self, asset_id: str, title: str, description: str,


                               price: float, category: str = "financial_data",
                               tags: Optional[List[str]] = None) -> str:
        """
        发布到数据市场

        Args:
            asset_id: 资产ID
            title: 标题
            description: 描述
            price: 价格
            category: 分类
            tags: 标签

        Returns:
            商品ID
        """
        try:
            if asset_id not in self.data_assets:
                raise ValueError(f"数据资产不存在: {asset_id}")

            asset = self.data_assets[asset_id]

            item_id = str(uuid.uuid4())

            item = DataMarketItem(
                item_id=item_id,
                asset_id=asset_id,
                title=title,
                description=description,
                price=price,
                category=category,
                tags=tags or asset.tags,
                quality_score=asset.quality_score
            )

            self.marketplace_items[item_id] = item
            self.stats['total_market_items'] += 1

            log_data_operation("marketplace_item_published", asset.data_type,
                               {
                                   "item_id": item_id,
                                   "asset_id": asset_id,
                                   "title": title,
                                   "price": price
                               }, "info")

            return item_id

        except Exception as e:
            log_data_operation("marketplace_publish_error", DataSourceType.STOCK,
                               {"asset_id": asset_id, "error": str(e)}, "error")
            raise

    def search_data_assets(self, query: str, data_type: Optional[DataSourceType] = None,


                           tags: Optional[List[str]] = None, owner: Optional[str] = None,
                           min_quality: float = 0.0, limit: int = 50) -> List[Dict[str, Any]]:
        """
        搜索数据资产

        Args:
            query: 搜索查询
            data_type: 数据类型过滤
            tags: 标签过滤
            owner: 所有者过滤
            min_quality: 最低质量分数
            limit: 返回结果数量限制

        Returns:
            搜索结果
        """
        try:
            results = []

            for asset in self.data_assets.values():
                # 基本过滤
                if data_type and asset.data_type != data_type:
                    continue
                if owner and asset.owner != owner:
                    continue
                if asset.quality_score < min_quality:
                    continue

                # 标签过滤
                if tags:
                    if not set(tags).issubset(set(asset.tags)):
                        continue

                # 文本搜索
                search_text = f"{asset.name} {asset.description} {' '.join(asset.tags)}"
                if query.lower() not in search_text.lower():
                    continue

                # 添加到结果
                results.append({
                    'asset_id': asset.asset_id,
                    'name': asset.name,
                    'description': asset.description,
                    'data_type': asset.data_type.value,
                    'owner': asset.owner,
                    'tags': asset.tags,
                    'quality_score': asset.quality_score,
                    'last_updated': asset.last_updated.isoformat(),
                    'access_count': asset.access_count
                })

                if len(results) >= limit:
                    break

            log_data_operation("data_asset_search", DataSourceType.STOCK,
                               {
                                   "query": query,
                                   "results_count": len(results),
                                   "filters": {
                                       "data_type": data_type.value if data_type else None,
                                       "tags": tags,
                                       "owner": owner,
                                       "min_quality": min_quality
                                   }
                               }, "info")

            return results

        except Exception as e:
            log_data_operation("data_asset_search_error", DataSourceType.STOCK,
                               {"query": query, "error": str(e)}, "error")
            return []

    def get_data_lineage(self, asset_id: str, depth: int = 3) -> Dict[str, Any]:
        """
        获取数据血缘

        Args:
            asset_id: 资产ID
            depth: 追踪深度

        Returns:
            血缘信息
        """
        try:
            if asset_id not in self.data_assets:
                raise ValueError(f"数据资产不存在: {asset_id}")

            lineage_graph = self._build_lineage_graph(asset_id, depth)

            return {
                'asset_id': asset_id,
                'lineage_graph': lineage_graph,
                'depth': depth,
                'total_lineages': len(lineage_graph)
            }

        except Exception as e:
            log_data_operation("data_lineage_retrieval_error", DataSourceType.STOCK,
                               {"asset_id": asset_id, "error": str(e)}, "error")
            return {
                'asset_id': asset_id,
                'error': str(e)
            }

    def _build_lineage_graph(self, asset_id: str, depth: int, current_depth: int = 0,


                             visited: Optional[Set[str]] = None) -> Dict[str, Any]:
        """构建血缘图"""
        if visited is None:
            visited = set()

        if current_depth >= depth or asset_id in visited:
            return {}

        visited.add(asset_id)

        graph = {}

        # 获取指向此资产的血缘
        if asset_id in self.data_lineage:
            for lineage in self.data_lineage[asset_id]:
                if lineage.source_asset_id not in graph:
                    graph[lineage.source_asset_id] = {
                        'asset_info': self._get_asset_info(lineage.source_asset_id),
                        'transformation': {
                            'type': lineage.transformation_type,
                            'details': lineage.transformation_details,
                            'execution_time': lineage.execution_time,
                            'success': lineage.success,
                            'timestamp': lineage.created_at.isoformat()
                        },
                        'upstream': self._build_lineage_graph(
                            lineage.source_asset_id, depth, current_depth + 1, visited
                        )
                    }

        return graph

    def _get_asset_info(self, asset_id: str) -> Dict[str, Any]:
        """获取资产信息"""
        if asset_id in self.data_assets:
            asset = self.data_assets[asset_id]
            return {
                'name': asset.name,
                'data_type': asset.data_type.value,
                'owner': asset.owner,
                'quality_score': asset.quality_score
            }
        return {'name': 'Unknown', 'data_type': 'unknown'}

    def get_marketplace_items(self, category: Optional[str] = None,


                              min_price: float = 0.0, max_price: Optional[float] = None,
                              min_rating: float = 0.0, tags: Optional[List[str]] = None,
                              limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取市场商品

        Args:
            category: 分类过滤
            min_price: 最低价格
            max_price: 最高价格
            min_rating: 最低评分
            tags: 标签过滤
            limit: 返回数量限制

        Returns:
            商品列表
        """
        try:
            items = []

            for item in self.marketplace_items.values():
                if not item.published:
                    continue

                # 基本过滤
                if category and item.category != category:
                    continue
                if item.price < min_price:
                    continue
                if max_price and item.price > max_price:
                    continue
                if item.rating < min_rating:
                    continue

                # 标签过滤
                if tags:
                    if not set(tags).issubset(set(item.tags)):
                        continue

                # 添加到结果
                items.append({
                    'item_id': item.item_id,
                    'asset_id': item.asset_id,
                    'title': item.title,
                    'description': item.description,
                    'price': item.price,
                    'currency': item.currency,
                    'category': item.category,
                    'tags': item.tags,
                    'quality_score': item.quality_score,
                    'rating': item.rating,
                    'reviews_count': item.reviews_count,
                    'download_count': item.download_count,
                    'created_at': item.created_at.isoformat()
                })

            # 按评分和下载量排序
            items.sort(key=lambda x: (x['rating'], x['download_count']), reverse=True)

            return items[:limit]

        except Exception as e:
            log_data_operation("marketplace_items_retrieval_error", DataSourceType.STOCK,
                               {"error": str(e)}, "error")
            return []

    def _monitoring_worker(self) -> None:
        """监控工作线程"""
        while not self._stop_monitoring:
            try:
                # 检查契约状态
                self._check_contracts_status()

                # 更新数据质量
                self._update_data_quality_scores()

                # 清理过期数据
                self._cleanup_expired_data()

                time.sleep(3600)  # 每小时检查一次

            except Exception as e:
                log_data_operation("ecosystem_monitoring_error", DataSourceType.STOCK,
                                   {"error": str(e)}, "error")
                time.sleep(60)

    def _check_contracts_status(self) -> None:
        """检查契约状态"""
        try:
            current_time = datetime.now()
            expired_contracts = []

            for contract_id, contract in self.data_contracts.items():
                if contract.expires_at and current_time > contract.expires_at:
                    contract.status = "expired"
                    expired_contracts.append(contract_id)

            if expired_contracts:
                log_data_operation("contracts_expired", DataSourceType.STOCK,
                                   {"expired_count": len(expired_contracts)}, "warning")

                publish_data_event("contracts_expired", {
                    "expired_contracts": expired_contracts,
                    "count": len(expired_contracts)
                }, DataSourceType.STOCK, "medium")

        except Exception as e:
            log_data_operation("contract_status_check_error", DataSourceType.STOCK,
                               {"error": str(e)}, "error")

    def _update_data_quality_scores(self) -> None:
        """更新数据质量分数"""
        try:
            # 这里可以实现质量分数的定期更新逻辑
            # 基于最近的访问模式和质量检查结果

            updated_count = 0
            for asset_id, asset in self.data_assets.items():
                # 简单的质量衰减逻辑
                time_since_update = (datetime.now() - asset.last_updated).days
                if time_since_update > 30:  # 超过30天未更新
                    decay_factor = 0.95 ** (time_since_update // 30)
                    asset.quality_score *= decay_factor
                    updated_count += 1

            if updated_count > 0:
                log_data_operation("quality_scores_updated", DataSourceType.STOCK,
                                   {"updated_count": updated_count}, "info")

        except Exception as e:
            log_data_operation("quality_scores_update_error", DataSourceType.STOCK,
                               {"error": str(e)}, "error")

    def _cleanup_expired_data(self) -> None:
        """清理过期数据"""
        try:
            current_time = datetime.now()
            cleanup_threshold = current_time - timedelta(days=self.config.lineage_retention_days)

            # 清理过期的血缘数据
            cleaned_lineages = 0
            for asset_id in list(self.data_lineage.keys()):
                original_count = len(self.data_lineage[asset_id])
                self.data_lineage[asset_id] = [
                    lineage for lineage in self.data_lineage[asset_id]
                    if lineage.created_at > cleanup_threshold
                ]
                cleaned_lineages += original_count - len(self.data_lineage[asset_id])

            if cleaned_lineages > 0:
                log_data_operation("expired_data_cleaned", DataSourceType.STOCK,
                                   {"cleaned_lineages": cleaned_lineages}, "info")

        except Exception as e:
            log_data_operation("expired_data_cleanup_error", DataSourceType.STOCK,
                               {"error": str(e)}, "error")

    def _ecosystem_health_check(self) -> Dict[str, Any]:
        """生态系统健康检查"""
        try:
            health_status = {
                'component': 'DataEcosystemManager',
                'status': 'healthy',
                'total_assets': len(self.data_assets),
                'total_lineages': sum(len(lineages) for lineages in self.data_lineage.values()),
                'total_contracts': len(self.data_contracts),
                'total_market_items': len(self.marketplace_items),
                'active_users': len(self.user_sessions),
                'data_access_events': self.stats['data_access_events'],
                'monitoring_active': self.monitoring_thread is not None and self.monitoring_thread.is_alive(),
                'timestamp': datetime.now().isoformat()
            }

            # 检查关键指标
            # 确保有足够的活跃资产
            if len(self.data_assets) < 10:
                health_status['status'] = 'warning'
                health_status['message'] = f'数据资产数量偏少: {len(self.data_assets)}'

            # 检查契约合规性
            expired_contracts = sum(1 for c in self.data_contracts.values()
                                    if c.status == "expired")
            if expired_contracts > 0:
                health_status['status'] = 'warning'
                health_status['message'] = f'存在过期契约: {expired_contracts}个'

            return health_status

        except Exception as e:
            return {
                'component': 'DataEcosystemManager',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_ecosystem_stats(self) -> Dict[str, Any]:
        """获取生态系统统计信息"""
        try:
            # 资产统计
            assets_by_type = defaultdict(int)
            assets_by_owner = defaultdict(int)
            for asset in self.data_assets.values():
                assets_by_type[asset.data_type.value] += 1
                assets_by_owner[asset.owner] += 1

            # 契约统计
            contracts_by_status = defaultdict(int)
            for contract in self.data_contracts.values():
                contracts_by_status[contract.status] += 1

            # 市场统计
            market_by_category = defaultdict(int)
            for item in self.marketplace_items.values():
                if item.published:
                    market_by_category[item.category] += 1

            return {
                'assets': {
                    'total': len(self.data_assets),
                    'by_type': dict(assets_by_type),
                    'by_owner': dict(assets_by_owner),
                    'avg_quality_score': sum(a.quality_score for a in self.data_assets.values()) / max(len(self.data_assets), 1)
                },
                'lineages': {
                    'total': sum(len(lineages) for lineages in self.data_lineage.values()),
                    'by_target_asset': {asset_id: len(lineages) for asset_id, lineages in self.data_lineage.items()}
                },
                'contracts': {
                    'total': len(self.data_contracts),
                    'by_status': dict(contracts_by_status)
                },
                'marketplace': {
                    'total_items': len([i for i in self.marketplace_items.values() if i.published]),
                    'by_category': dict(market_by_category),
                    'avg_price': sum(i.price for i in self.marketplace_items.values() if i.published) / max(len([i for i in self.marketplace_items.values() if i.published]), 1)
                },
                'activity': {
                    'active_users': len(self.user_sessions),
                    'total_access_events': self.stats['data_access_events']
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            log_data_operation("ecosystem_stats_error", DataSourceType.STOCK,
                               {"error": str(e)}, "error")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def shutdown(self) -> None:
        """关闭生态系统管理器"""
        try:
            self._stop_monitoring = True

            # 等待监控线程结束
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)

            log_data_operation("ecosystem_manager_shutdown", DataSourceType.STOCK,
                               {"final_stats": self.get_ecosystem_stats()}, "info")

        except Exception as e:
            log_data_operation("ecosystem_manager_shutdown_error", DataSourceType.STOCK,
                               {"error": str(e)}, "error")


# 全局单例实例
_data_ecosystem_manager = None


def get_data_ecosystem_manager() -> DataEcosystemManager:
    """
    获取数据生态系统管理器单例实例

    Returns:
        数据生态系统管理器实例
    """
    global _data_ecosystem_manager
    if _data_ecosystem_manager is None:
        _data_ecosystem_manager = DataEcosystemManager()
    return _data_ecosystem_manager


__all__ = [
    'DataAsset',
    'DataLineage',
    'DataContract',
    'DataMarketItem',
    'EcosystemConfig',
    'DataEcosystemManager',
    'get_data_ecosystem_manager'
]
