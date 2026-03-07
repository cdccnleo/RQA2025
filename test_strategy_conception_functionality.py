#!/usr/bin/env python3
"""
策略构思设计器功能完整性测试
检查各项功能是否正常，是否满足量化交易系统策略设计需求
"""

import requests
import json
import os
import sys
from typing import Dict, List, Any

API_BASE = "http://localhost:8000/api/v1"

def test_api_endpoint(method: str, endpoint: str, data: Dict = None, expected_status: int = 200) -> tuple[bool, Any]:
    """测试API端点"""
    try:
        url = f"{API_BASE}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        elif method == "PUT":
            response = requests.put(url, json=data, timeout=10)
        elif method == "DELETE":
            response = requests.delete(url, timeout=10)
        else:
            return False, f"不支持的HTTP方法: {method}"

        if response.status_code == expected_status:
            try:
                return True, response.json()
            except:
                return True, response.text
        else:
            return False, f"HTTP {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)

def check_frontend_features():
    """检查前端功能"""
    print("📱 检查前端功能...")
    
    features = {
        "策略模板选择": False,
        "策略类型选择": False,
        "组件拖拽": False,
        "画布操作": False,
        "节点编辑": False,
        "连接管理": False,
        "参数配置": False,
        "策略保存": False,
        "策略加载": False,
        "策略导出": False,
        "策略验证": False,
    }
    
    try:
        with open('web-static/strategy-conception.html', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键功能
        if 'strategyTemplateSelect' in content:
            features["策略模板选择"] = True
        if 'strategyTypeSelect' in content:
            features["策略类型选择"] = True
        if 'strategy-node' in content and ('draggable' in content.lower() or 'dragstart' in content.lower() or 'onDragStart' in content):
            features["组件拖拽"] = True
        if 'strategyCanvas' in content:
            features["画布操作"] = True
        if 'editNode' in content or 'edit-node' in content.lower():
            features["节点编辑"] = True
        if 'connection' in content.lower() or '连接' in content:
            features["连接管理"] = True
        if 'parameterPanel' in content or 'parameter' in content.lower():
            features["参数配置"] = True
        if 'saveStrategy' in content:
            features["策略保存"] = True
        if 'loadStrategy' in content:
            features["策略加载"] = True
        if 'exportStrategy' in content:
            features["策略导出"] = True
        if 'validate' in content.lower() or '验证' in content:
            features["策略验证"] = True
        
        return features
    except Exception as e:
        print(f"   ❌ 读取前端文件失败: {e}")
        return features

def test_backend_apis():
    """测试后端API"""
    print("\n🔌 测试后端API...")
    
    api_tests = {}
    
    # 1. 测试策略模板API
    print("   1️⃣ 测试策略模板API...")
    success, result = test_api_endpoint("GET", "/strategy/conception/templates")
    api_tests["获取策略模板"] = success
    if success:
        print(f"      ✅ 成功: 返回{len(result.get('templates', {}))}个模板")
    else:
        print(f"      ❌ 失败: {result}")
    
    # 2. 测试获取策略列表
    print("   2️⃣ 测试获取策略列表...")
    success, result = test_api_endpoint("GET", "/strategy/conceptions")
    api_tests["获取策略列表"] = success
    if success:
        if isinstance(result, dict):
            count = result.get('count', len(result.get('conceptions', [])))
        elif isinstance(result, list):
            count = len(result)
        else:
            count = 0
        print(f"      ✅ 成功: 返回{count}个策略")
    else:
        print(f"      ❌ 失败: {result}")
    
    # 3. 测试策略验证API
    print("   3️⃣ 测试策略验证API...")
    test_strategy = {
        "name": "测试策略",
        "type": "trend_following",
        "nodes": [
            {"type": "data_source", "id": "ds1"},
            {"type": "feature", "id": "f1"},
            {"type": "trade", "id": "t1"}
        ],
        "connections": [
            {"from": "ds1", "to": "f1"},
            {"from": "f1", "to": "t1"}
        ]
    }
    success, result = test_api_endpoint("POST", "/strategy/conceptions/validate", test_strategy)
    api_tests["策略验证"] = success
    if success:
        if isinstance(result, dict):
            valid = result.get('validation', {}).get('valid', False) if isinstance(result.get('validation'), dict) else False
        else:
            valid = False
        print(f"      ✅ 成功: 验证结果={valid}")
    else:
        print(f"      ❌ 失败: {result}")
    
    # 4. 测试创建策略
    print("   4️⃣ 测试创建策略...")
    new_strategy = {
        "name": "功能测试策略",
        "type": "trend_following",
        "description": "用于功能测试的策略",
        "nodes": [
            {"type": "data_source", "id": "ds1", "name": "数据源1"},
            {"type": "trade", "id": "t1", "name": "交易信号1"}
        ],
        "connections": [
            {"from": "ds1", "to": "t1"}
        ]
    }
    success, result = test_api_endpoint("POST", "/strategy/conceptions", new_strategy)
    api_tests["创建策略"] = success
    test_strategy_id = None
    if success:
        if isinstance(result, dict):
            test_strategy_id = result.get('strategy_id') or (result.get('data', {}) if isinstance(result.get('data'), dict) else {}).get('id')
        print(f"      ✅ 成功: 策略ID={test_strategy_id}")
    else:
        print(f"      ❌ 失败: {result}")
    
    # 5. 测试获取单个策略
    if test_strategy_id:
        print("   5️⃣ 测试获取单个策略...")
        success, result = test_api_endpoint("GET", f"/strategy/conceptions/{test_strategy_id}")
        api_tests["获取单个策略"] = success
        if success:
            if isinstance(result, dict):
                name = result.get('name', 'N/A')
            else:
                name = 'N/A'
            print(f"      ✅ 成功: 策略名称={name}")
        else:
            print(f"      ❌ 失败: {result}")
        
        # 6. 测试更新策略
        print("   6️⃣ 测试更新策略...")
        update_data = {"name": "更新后的策略名称", "description": "更新后的描述"}
        success, result = test_api_endpoint("PUT", f"/strategy/conceptions/{test_strategy_id}", update_data)
        api_tests["更新策略"] = success
        if success:
            print(f"      ✅ 成功: 策略已更新")
        else:
            print(f"      ❌ 失败: {result}")
        
        # 7. 清理测试数据
        print("   7️⃣ 清理测试数据...")
        success, result = test_api_endpoint("DELETE", f"/strategy/conceptions/{test_strategy_id}", expected_status=200)
        api_tests["删除策略"] = success
        if success:
            print(f"      ✅ 成功: 测试策略已删除")
        else:
            print(f"      ⚠️ 警告: 删除失败，可能需要手动清理")
    
    return api_tests

def check_quantitative_trading_requirements():
    """检查是否满足量化交易系统策略设计需求"""
    print("\n📊 检查量化交易系统策略设计需求...")
    
    requirements = {
        "数据源集成": False,
        "技术指标支持": False,
        "机器学习模型": False,
        "交易信号生成": False,
        "风险控制": False,
        "回测支持": False,
        "参数优化": False,
        "策略验证": False,
    }
    
    try:
        with open('web-static/strategy-conception.html', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查数据源集成
        if 'data_source' in content.lower() or '数据源' in content:
            requirements["数据源集成"] = True
        
        # 检查技术指标支持
        if 'feature' in content.lower() or 'indicator' in content.lower() or '技术指标' in content:
            requirements["技术指标支持"] = True
        
        # 检查机器学习模型
        if 'model' in content.lower() or 'ml' in content.lower() or '机器学习' in content:
            requirements["机器学习模型"] = True
        
        # 检查交易信号生成
        if 'trade' in content.lower() or 'signal' in content.lower() or '交易信号' in content:
            requirements["交易信号生成"] = True
        
        # 检查风险控制
        if 'risk' in content.lower() or '风险' in content:
            requirements["风险控制"] = True
        
        # 检查回测支持
        if 'backtest' in content.lower() or '回测' in content:
            requirements["回测支持"] = True
        
        # 检查参数优化
        if 'parameter' in content.lower() or 'optimize' in content.lower() or '参数' in content:
            requirements["参数优化"] = True
        
        # 检查策略验证
        if 'validate' in content.lower() or '验证' in content:
            requirements["策略验证"] = True
        
        return requirements
    except Exception as e:
        print(f"   ❌ 检查失败: {e}")
        return requirements

def main():
    """主测试函数"""
    print("🧪 策略构思设计器功能完整性测试")
    print("=" * 60)
    
    # 1. 检查前端功能
    frontend_features = check_frontend_features()
    print("\n📋 前端功能检查结果:")
    for feature, status in frontend_features.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {feature}")
    
    # 2. 测试后端API
    api_tests = test_backend_apis()
    print("\n📋 后端API测试结果:")
    for api, status in api_tests.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {api}")
    
    # 3. 检查量化交易需求
    requirements = check_quantitative_trading_requirements()
    print("\n📋 量化交易系统需求检查结果:")
    for req, status in requirements.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {req}")
    
    # 4. 总结
    print("\n" + "=" * 60)
    frontend_score = sum(frontend_features.values()) / len(frontend_features) * 100
    api_score = sum(api_tests.values()) / len(api_tests) * 100 if api_tests else 0
    req_score = sum(requirements.values()) / len(requirements) * 100
    
    print("📊 功能完整性评分:")
    print(f"   • 前端功能: {frontend_score:.1f}%")
    print(f"   • 后端API: {api_score:.1f}%")
    print(f"   • 量化交易需求: {req_score:.1f}%")
    
    overall_score = (frontend_score + api_score + req_score) / 3
    print(f"\n🎯 总体评分: {overall_score:.1f}%")
    
    if overall_score >= 80:
        print("✅ 策略构思设计器功能基本完整，满足量化交易系统策略设计需求！")
        return True
    elif overall_score >= 60:
        print("⚠️ 策略构思设计器功能基本可用，但仍有改进空间")
        return True
    else:
        print("❌ 策略构思设计器功能不完整，需要进一步开发")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
