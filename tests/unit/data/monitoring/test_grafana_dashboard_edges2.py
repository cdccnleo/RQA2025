"""
Grafana仪表板管理模块的边界测试
"""
import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.data.monitoring.grafana_dashboard import (
    PanelType,
    MetricType,
    DashboardPanel,
    DashboardTemplate,
    DataGrafanaDashboard,
)


class TestPanelType:
    """测试 PanelType 枚举"""

    def test_panel_type_graph(self):
        """测试 GRAPH 类型"""
        assert PanelType.GRAPH.value == "graph"

    def test_panel_type_singlestat(self):
        """测试 SINGLSTAT 类型"""
        assert PanelType.SINGLSTAT.value == "singlestat"

    def test_panel_type_table(self):
        """测试 TABLE 类型"""
        assert PanelType.TABLE.value == "table"

    def test_panel_type_heatmap(self):
        """测试 HEATMAP 类型"""
        assert PanelType.HEATMAP.value == "heatmap"

    def test_panel_type_bargauge(self):
        """测试 BARGAUGE 类型"""
        assert PanelType.BARGAUGE.value == "bargauge"

    def test_panel_type_gauge(self):
        """测试 GAUGE 类型"""
        assert PanelType.GAUGE.value == "gauge"

    def test_panel_type_text(self):
        """测试 TEXT 类型"""
        assert PanelType.TEXT.value == "text"

    def test_panel_type_all_values(self):
        """测试所有类型值"""
        types = [
            PanelType.GRAPH, PanelType.SINGLSTAT, PanelType.TABLE,
            PanelType.HEATMAP, PanelType.BARGAUGE, PanelType.GAUGE, PanelType.TEXT
        ]
        assert len(types) == 7
        assert all(isinstance(t.value, str) for t in types)


class TestMetricType:
    """测试 MetricType 枚举"""

    def test_metric_type_prometheus(self):
        """测试 PROMETHEUS 类型"""
        assert MetricType.PROMETHEUS.value == "prometheus"

    def test_metric_type_elasticsearch(self):
        """测试 ELASTICSEARCH 类型"""
        assert MetricType.ELASTICSEARCH.value == "elasticsearch"

    def test_metric_type_influxdb(self):
        """测试 INFLUXDB 类型"""
        assert MetricType.INFLUXDB.value == "influxdb"

    def test_metric_type_all_values(self):
        """测试所有类型值"""
        types = [MetricType.PROMETHEUS, MetricType.ELASTICSEARCH, MetricType.INFLUXDB]
        assert len(types) == 3
        assert all(isinstance(t.value, str) for t in types)


class TestDashboardPanel:
    """测试 DashboardPanel 数据类"""

    def test_dashboard_panel_init_required(self):
        """测试必需参数初始化"""
        panel = DashboardPanel(
            id=1,
            title="Test Panel",
            type=PanelType.GRAPH,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 0},
            targets=[]
        )
        assert panel.id == 1
        assert panel.title == "Test Panel"
        assert panel.type == PanelType.GRAPH
        assert panel.grid_pos == {"h": 8, "w": 12, "x": 0, "y": 0}
        assert panel.targets == []

    def test_dashboard_panel_init_all_params(self):
        """测试所有参数初始化"""
        panel = DashboardPanel(
            id=2,
            title="Test Panel 2",
            type=PanelType.TABLE,
            grid_pos={"h": 4, "w": 6, "x": 0, "y": 0},
            targets=[{"expr": "test"}],
            options={"key": "value"},
            field_config={"defaults": {}},
            description="Test description",
            repeat="variable",
            repeat_options={"max": 10}
        )
        assert panel.options == {"key": "value"}
        assert panel.field_config == {"defaults": {}}
        assert panel.description == "Test description"
        assert panel.repeat == "variable"
        assert panel.repeat_options == {"max": 10}

    def test_dashboard_panel_to_dict_basic(self):
        """测试基本 to_dict"""
        panel = DashboardPanel(
            id=1,
            title="Test",
            type=PanelType.GRAPH,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 0},
            targets=[]
        )
        result = panel.to_dict()
        assert result["id"] == 1
        assert result["title"] == "Test"
        assert result["type"] == "graph"
        assert result["gridPos"] == {"h": 8, "w": 12, "x": 0, "y": 0}

    def test_dashboard_panel_to_dict_with_options(self):
        """测试带选项的 to_dict"""
        panel = DashboardPanel(
            id=1,
            title="Test",
            type=PanelType.GRAPH,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 0},
            targets=[],
            options={"key": "value"}
        )
        result = panel.to_dict()
        assert result["options"] == {"key": "value"}

    def test_dashboard_panel_to_dict_with_repeat(self):
        """测试带重复的 to_dict"""
        panel = DashboardPanel(
            id=1,
            title="Test",
            type=PanelType.GRAPH,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 0},
            targets=[],
            repeat="variable",
            repeat_options={"max": 10}
        )
        result = panel.to_dict()
        assert result["repeat"] == "variable"
        assert result["repeatOptions"] == {"max": 10}

    def test_dashboard_panel_empty_title(self):
        """测试空标题"""
        panel = DashboardPanel(
            id=1,
            title="",
            type=PanelType.GRAPH,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 0},
            targets=[]
        )
        assert panel.title == ""

    def test_dashboard_panel_zero_id(self):
        """测试零ID"""
        panel = DashboardPanel(
            id=0,
            title="Test",
            type=PanelType.GRAPH,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 0},
            targets=[]
        )
        assert panel.id == 0

    def test_dashboard_panel_negative_id(self):
        """测试负ID"""
        panel = DashboardPanel(
            id=-1,
            title="Test",
            type=PanelType.GRAPH,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 0},
            targets=[]
        )
        assert panel.id == -1


class TestDashboardTemplate:
    """测试 DashboardTemplate 数据类"""

    def test_dashboard_template_init_required(self):
        """测试必需参数初始化"""
        template = DashboardTemplate(
            name="Test Template",
            description="Test Description",
            panels=[]
        )
        assert template.name == "Test Template"
        assert template.description == "Test Description"
        assert template.panels == []

    def test_dashboard_template_init_all_params(self):
        """测试所有参数初始化"""
        template = DashboardTemplate(
            name="Test",
            description="Desc",
            panels=[],
            variables=[{"name": "var1"}],
            time_settings={"from": "now-1h", "to": "now"},
            refresh="60s",
            tags=["tag1", "tag2"]
        )
        assert template.variables == [{"name": "var1"}]
        assert template.time_settings == {"from": "now-1h", "to": "now"}
        assert template.refresh == "60s"
        assert template.tags == ["tag1", "tag2"]

    def test_dashboard_template_default_values(self):
        """测试默认值"""
        template = DashboardTemplate(
            name="Test",
            description="Desc",
            panels=[]
        )
        assert template.variables == []
        assert template.time_settings == {"from": "now - 1h", "to": "now"}
        assert template.refresh == "30s"
        assert "rqa2025" in template.tags

    def test_dashboard_template_to_dict(self):
        """测试 to_dict"""
        panel = DashboardPanel(
            id=1,
            title="Test",
            type=PanelType.GRAPH,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 0},
            targets=[]
        )
        template = DashboardTemplate(
            name="Test Template",
            description="Test Description",
            panels=[panel]
        )
        result = template.to_dict()
        assert "dashboard" in result
        assert result["dashboard"]["title"] == "RQA2025 数据层监控 - Test Template"
        assert len(result["dashboard"]["panels"]) == 1

    def test_dashboard_template_empty_panels(self):
        """测试空面板列表"""
        template = DashboardTemplate(
            name="Test",
            description="Desc",
            panels=[]
        )
        result = template.to_dict()
        assert result["dashboard"]["panels"] == []

    def test_dashboard_template_multiple_panels(self):
        """测试多个面板"""
        panels = [
            DashboardPanel(
                id=i,
                title=f"Panel {i}",
                type=PanelType.GRAPH,
                grid_pos={"h": 8, "w": 12, "x": 0, "y": i * 8},
                targets=[]
            )
            for i in range(5)
        ]
        template = DashboardTemplate(
            name="Test",
            description="Desc",
            panels=panels
        )
        result = template.to_dict()
        assert len(result["dashboard"]["panels"]) == 5


class TestDataGrafanaDashboard:
    """测试 DataGrafanaDashboard 类"""

    def test_init_default(self):
        """测试默认初始化"""
        dashboard = DataGrafanaDashboard()
        assert dashboard.data_source_type == MetricType.PROMETHEUS
        assert dashboard.data_source_name == "Prometheus"
        assert isinstance(dashboard.templates, dict)
        assert isinstance(dashboard.deployed_dashboards, dict)

    def test_init_custom_params(self):
        """测试自定义参数初始化"""
        dashboard = DataGrafanaDashboard(
            data_source_type=MetricType.INFLUXDB,
            data_source_name="Custom InfluxDB"
        )
        assert dashboard.data_source_type == MetricType.INFLUXDB
        assert dashboard.data_source_name == "Custom InfluxDB"

    def test_init_all_metric_types(self):
        """测试所有指标类型初始化"""
        for metric_type in MetricType:
            dashboard = DataGrafanaDashboard(data_source_type=metric_type)
            assert dashboard.data_source_type == metric_type

    def test_init_empty_data_source_name(self):
        """测试空数据源名称"""
        dashboard = DataGrafanaDashboard(data_source_name="")
        assert dashboard.data_source_name == ""

    def test_templates_initialized(self):
        """测试模板已初始化"""
        dashboard = DataGrafanaDashboard()
        # 应该至少有一些标准模板
        assert len(dashboard.templates) > 0

    def test_deployed_dashboards_empty(self):
        """测试已部署仪表板为空"""
        dashboard = DataGrafanaDashboard()
        assert dashboard.deployed_dashboards == {}


class TestEdgeCases:
    """测试边界情况"""

    def test_dashboard_panel_all_panel_types(self):
        """测试所有面板类型"""
        for panel_type in PanelType:
            panel = DashboardPanel(
                id=1,
                title="Test",
                type=panel_type,
                grid_pos={"h": 8, "w": 12, "x": 0, "y": 0},
                targets=[]
            )
            assert panel.type == panel_type
            result = panel.to_dict()
            assert result["type"] == panel_type.value

    def test_dashboard_template_very_long_name(self):
        """测试非常长的名称"""
        long_name = "A" * 1000
        template = DashboardTemplate(
            name=long_name,
            description="Desc",
            panels=[]
        )
        assert template.name == long_name

    def test_dashboard_panel_very_large_grid_pos(self):
        """测试非常大的网格位置"""
        panel = DashboardPanel(
            id=1,
            title="Test",
            type=PanelType.GRAPH,
            grid_pos={"h": 10000, "w": 10000, "x": 10000, "y": 10000},
            targets=[]
        )
        assert panel.grid_pos["h"] == 10000

    def test_dashboard_template_empty_name(self):
        """测试空名称"""
        template = DashboardTemplate(
            name="",
            description="Desc",
            panels=[]
        )
        assert template.name == ""

    def test_dashboard_panel_negative_grid_pos(self):
        """测试负网格位置"""
        panel = DashboardPanel(
            id=1,
            title="Test",
            type=PanelType.GRAPH,
            grid_pos={"h": -1, "w": -1, "x": -1, "y": -1},
            targets=[]
        )
        assert panel.grid_pos["h"] == -1

    def test_dashboard_template_custom_tags(self):
        """测试自定义标签"""
        template = DashboardTemplate(
            name="Test",
            description="Desc",
            panels=[],
            tags=["custom1", "custom2"]
        )
        assert "custom1" in template.tags
        assert "custom2" in template.tags

    def test_dashboard_panel_multiple_targets(self):
        """测试多个目标"""
        panel = DashboardPanel(
            id=1,
            title="Test",
            type=PanelType.GRAPH,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 0},
            targets=[
                {"expr": "query1", "refId": "A"},
                {"expr": "query2", "refId": "B"},
                {"expr": "query3", "refId": "C"}
            ]
        )
        assert len(panel.targets) == 3

    def test_dashboard_template_custom_time_settings(self):
        """测试自定义时间设置"""
        template = DashboardTemplate(
            name="Test",
            description="Desc",
            panels=[],
            time_settings={"from": "2024-01-01", "to": "2024-01-31"}
        )
        assert template.time_settings["from"] == "2024-01-01"
        assert template.time_settings["to"] == "2024-01-31"

    def test_data_grafana_dashboard_multiple_instances(self):
        """测试多个实例"""
        dashboard1 = DataGrafanaDashboard()
        dashboard2 = DataGrafanaDashboard()
        assert dashboard1 is not dashboard2
        # 修改一个实例不应该影响另一个
        dashboard1.data_source_name = "Custom1"
        assert dashboard2.data_source_name == "Prometheus"

    def test_create_dashboard_success(self):
        """测试成功创建仪表板"""
        dashboard = DataGrafanaDashboard()
        result = dashboard.create_dashboard("main")
        assert "dashboard" in result
        assert result["dashboard"]["title"] is not None

    def test_create_dashboard_invalid_template(self):
        """测试无效模板名称"""
        dashboard = DataGrafanaDashboard()
        with pytest.raises(ValueError, match="模板不存在"):
            dashboard.create_dashboard("nonexistent")

    def test_create_dashboard_with_custom_config(self):
        """测试带自定义配置创建仪表板"""
        dashboard = DataGrafanaDashboard()
        custom_config = {"dashboard": {"title": "Custom Title"}}
        result = dashboard.create_dashboard("main", custom_config)
        assert "dashboard" in result

    def test_create_custom_dashboard(self):
        """测试创建自定义仪表板"""
        dashboard = DataGrafanaDashboard()
        panel = DashboardPanel(
            id=1,
            title="Custom Panel",
            type=PanelType.GRAPH,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 0},
            targets=[]
        )
        result = dashboard.create_custom_dashboard(
            title="Custom Dashboard",
            description="Custom Description",
            panels=[panel]
        )
        assert "dashboard" in result
        assert result["dashboard"]["title"] == "RQA2025 数据层监控 - Custom Dashboard"

    def test_create_custom_dashboard_with_variables(self):
        """测试带变量的自定义仪表板"""
        dashboard = DataGrafanaDashboard()
        panel = DashboardPanel(
            id=1,
            title="Panel",
            type=PanelType.GRAPH,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 0},
            targets=[]
        )
        variables = [{"name": "var1", "type": "custom"}]
        result = dashboard.create_custom_dashboard(
            title="Custom",
            description="Desc",
            panels=[panel],
            variables=variables
        )
        assert len(result["dashboard"]["templating"]["list"]) == 1

    def test_deploy_dashboard_success(self):
        """测试成功部署仪表板"""
        dashboard = DataGrafanaDashboard()
        config = {
            "dashboard": {
                "title": "Test Dashboard"
            }
        }
        result = dashboard.deploy_dashboard(config)
        assert result is True
        assert "Test Dashboard" in dashboard.deployed_dashboards

    def test_deploy_dashboard_custom_folder(self):
        """测试自定义文件夹部署"""
        dashboard = DataGrafanaDashboard()
        config = {
            "dashboard": {
                "title": "Test Dashboard 2"
            }
        }
        result = dashboard.deploy_dashboard(config, folder_name="Custom Folder")
        assert result is True
        assert dashboard.deployed_dashboards["Test Dashboard 2"]["folder"] == "Custom Folder"

    def test_deploy_dashboard_exception(self):
        """测试部署时抛出异常"""
        dashboard = DataGrafanaDashboard()
        # 无效的配置应该导致异常
        invalid_config = {}
        result = dashboard.deploy_dashboard(invalid_config)
        # 应该返回 False 或抛出异常
        assert isinstance(result, bool)

    def test_update_dashboard_success(self):
        """测试成功更新仪表板"""
        dashboard = DataGrafanaDashboard()
        config = {
            "dashboard": {
                "title": "Test Dashboard"
            }
        }
        dashboard.deploy_dashboard(config)
        new_config = {
            "dashboard": {
                "title": "Test Dashboard",
                "description": "Updated"
            }
        }
        result = dashboard.update_dashboard("Test Dashboard", new_config)
        assert result is True

    def test_update_dashboard_nonexistent(self):
        """测试更新不存在的仪表板"""
        dashboard = DataGrafanaDashboard()
        new_config = {"dashboard": {"title": "Test"}}
        result = dashboard.update_dashboard("Nonexistent", new_config)
        assert result is False

    def test_delete_dashboard_success(self):
        """测试成功删除仪表板"""
        dashboard = DataGrafanaDashboard()
        config = {
            "dashboard": {
                "title": "Test Dashboard"
            }
        }
        dashboard.deploy_dashboard(config)
        result = dashboard.delete_dashboard("Test Dashboard")
        assert result is True
        assert "Test Dashboard" not in dashboard.deployed_dashboards

    def test_delete_dashboard_nonexistent(self):
        """测试删除不存在的仪表板"""
        dashboard = DataGrafanaDashboard()
        result = dashboard.delete_dashboard("Nonexistent")
        assert result is False

    def test_add_panel_to_template_success(self):
        """测试成功添加面板到模板"""
        dashboard = DataGrafanaDashboard()
        panel = DashboardPanel(
            id=999,
            title="New Panel",
            type=PanelType.GRAPH,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 0},
            targets=[]
        )
        result = dashboard.add_panel_to_template("main", panel)
        assert result is True

    def test_add_panel_to_template_invalid(self):
        """测试添加到无效模板"""
        dashboard = DataGrafanaDashboard()
        panel = DashboardPanel(
            id=1,
            title="Panel",
            type=PanelType.GRAPH,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 0},
            targets=[]
        )
        result = dashboard.add_panel_to_template("nonexistent", panel)
        assert result is False

    def test_create_metric_panel(self):
        """测试创建指标面板"""
        dashboard = DataGrafanaDashboard()
        panel = dashboard.create_metric_panel(
            title="Metric Panel",
            metric_expr="test_metric",
            panel_type=PanelType.GRAPH
        )
        assert panel.title == "Metric Panel"
        assert len(panel.targets) > 0

    def test_get_available_templates(self):
        """测试获取可用模板"""
        dashboard = DataGrafanaDashboard()
        templates = dashboard.get_available_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0
        assert "main" in templates

    def test_get_deployed_dashboards(self):
        """测试获取已部署仪表板"""
        dashboard = DataGrafanaDashboard()
        config = {
            "dashboard": {
                "title": "Test Dashboard"
            }
        }
        dashboard.deploy_dashboard(config)
        deployed = dashboard.get_deployed_dashboards()
        assert isinstance(deployed, list)
        assert "Test Dashboard" in deployed

    def test_get_dashboard_stats(self):
        """测试获取仪表板统计信息"""
        dashboard = DataGrafanaDashboard()
        stats = dashboard.get_dashboard_stats()
        assert isinstance(stats, dict)
        assert "available_templates" in stats
        assert "deployed_dashboards" in stats
        assert "total_panels" in stats
        assert stats["available_templates"] > 0

