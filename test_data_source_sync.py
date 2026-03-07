#!/usr/bin/env python3
"""
测试数据源配置同步机制
验证修复后的同步功能是否正常工作
"""

import os
import json
import time
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

# 测试函数
def test_postgresql_sync():
    """测试从PostgreSQL加载配置时的同步机制"""
    logger.info("=== 测试从PostgreSQL加载配置时的同步机制 ===")
    
    try:
        # 导入配置管理模块
        from src.gateway.web.config_manager import load_data_sources
        from src.gateway.web.data_source_config_manager import get_data_source_config_manager
        
        # 步骤1: 加载数据源配置
        logger.info("步骤1: 加载数据源配置")
        sources = load_data_sources()
        logger.info(f"成功加载 {len(sources)} 个数据源")
        
        # 步骤2: 检查本地文件是否存在
        logger.info("步骤2: 检查本地文件是否存在")
        data_dir = os.path.join(project_root, "data")
        config_file = os.path.join(data_dir, "data_sources_config.json")
        
        if os.path.exists(config_file):
            logger.info(f"本地配置文件存在: {config_file}")
            # 读取本地文件内容
            with open(config_file, 'r', encoding='utf-8') as f:
                local_data = json.load(f)
            
            # 检查本地文件格式
            if isinstance(local_data, list):
                logger.info(f"本地文件为列表格式，包含 {len(local_data)} 个数据源")
            elif isinstance(local_data, dict):
                data_sources = local_data.get('data_sources', [])
                logger.info(f"本地文件为字典格式，包含 {len(data_sources)} 个数据源")
            else:
                logger.warning(f"本地文件格式未知: {type(local_data)}")
        else:
            logger.warning(f"本地配置文件不存在: {config_file}")
        
        # 步骤3: 通过数据源配置管理器加载配置
        logger.info("步骤3: 通过数据源配置管理器加载配置")
        manager = get_data_source_config_manager()
        if manager:
            manager.load_config()
            manager_sources = manager.get_data_sources()
            logger.info(f"管理器加载了 {len(manager_sources)} 个数据源")
        else:
            logger.warning("无法获取数据源配置管理器实例")
        
        logger.info("=== PostgreSQL同步测试完成 ===")
        return True
        
    except Exception as e:
        logger.error(f"PostgreSQL同步测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_change_sync():
    """测试数据源配置变更时的同步机制"""
    logger.info("=== 测试数据源配置变更时的同步机制 ===")
    
    try:
        # 导入配置管理模块
        from src.gateway.web.config_manager import load_data_sources, save_data_sources
        
        # 步骤1: 加载当前数据源
        logger.info("步骤1: 加载当前数据源")
        sources = load_data_sources()
        original_count = len(sources)
        logger.info(f"当前数据源数量: {original_count}")
        
        # 步骤2: 添加一个测试数据源
        logger.info("步骤2: 添加一个测试数据源")
        test_source = {
            "id": f"test_source_{int(time.time())}",
            "name": "测试数据源",
            "type": "测试数据",
            "url": "http://test.com",
            "enabled": True,
            "rate_limit": "100次/分钟",
            "last_test": None,
            "status": "未测试"
        }
        
        sources.append(test_source)
        logger.info(f"添加测试数据源后数量: {len(sources)}")
        
        # 步骤3: 保存配置
        logger.info("步骤3: 保存配置")
        save_data_sources(sources)
        logger.info("配置保存成功")
        
        # 步骤4: 检查本地文件是否更新
        logger.info("步骤4: 检查本地文件是否更新")
        data_dir = os.path.join(project_root, "data")
        config_file = os.path.join(data_dir, "data_sources_config.json")
        
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                local_data = json.load(f)
            
            if isinstance(local_data, list):
                local_count = len(local_data)
            elif isinstance(local_data, dict):
                local_count = len(local_data.get('data_sources', []))
            else:
                local_count = 0
            
            logger.info(f"本地文件数据源数量: {local_count}")
            if local_count == original_count + 1:
                logger.info("本地文件同步更新成功")
            else:
                logger.warning("本地文件同步更新失败")
        else:
            logger.warning(f"本地配置文件不存在: {config_file}")
        
        # 步骤5: 移除测试数据源
        logger.info("步骤5: 移除测试数据源")
        sources = [s for s in sources if s['id'] != test_source['id']]
        save_data_sources(sources)
        logger.info(f"移除测试数据源后数量: {len(sources)}")
        
        logger.info("=== 配置变更同步测试完成 ===")
        return True
        
    except Exception as e:
        logger.error(f"配置变更同步测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_frontend_data_consistency():
    """测试前端获取的数据源配置与PostgreSQL的一致性"""
    logger.info("=== 测试前端数据一致性 ===")
    
    try:
        # 导入配置管理模块
        from src.gateway.web.config_manager import load_data_sources
        from src.gateway.web.data_source_config_manager import get_data_source_config_manager
        
        # 步骤1: 模拟前端获取数据
        logger.info("步骤1: 模拟前端获取数据")
        frontend_sources = load_data_sources()
        logger.info(f"前端获取了 {len(frontend_sources)} 个数据源")
        
        # 步骤2: 通过数据源配置管理器获取数据
        logger.info("步骤2: 通过数据源配置管理器获取数据")
        manager = get_data_source_config_manager()
        if manager:
            manager_sources = manager.get_data_sources()
            logger.info(f"管理器获取了 {len(manager_sources)} 个数据源")
            
            # 比较两者数量
            if len(frontend_sources) == len(manager_sources):
                logger.info("前端数据与管理器数据数量一致")
            else:
                logger.warning("前端数据与管理器数据数量不一致")
                
            # 比较前3个数据源的ID
            for i, (fs, ms) in enumerate(zip(frontend_sources[:3], manager_sources[:3])):
                if fs['id'] == ms['id']:
                    logger.info(f"数据源 {i} ID一致: {fs['id']}")
                else:
                    logger.warning(f"数据源 {i} ID不一致: 前端={fs['id']}, 管理器={ms['id']}")
        else:
            logger.warning("无法获取数据源配置管理器实例")
        
        logger.info("=== 前端数据一致性测试完成 ===")
        return True
        
    except Exception as e:
        logger.error(f"前端数据一致性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    logger.info("开始数据源配置同步机制测试")
    
    # 运行所有测试
    tests = [
        ("PostgreSQL同步测试", test_postgresql_sync),
        ("配置变更同步测试", test_config_change_sync),
        ("前端数据一致性测试", test_frontend_data_consistency)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n运行测试: {test_name}")
        result = test_func()
        results.append((test_name, result))
        logger.info(f"测试 {test_name} {'通过' if result else '失败'}")
    
    # 打印测试结果
    logger.info("\n=== 测试结果汇总 ===")
    for test_name, result in results:
        status = "通过" if result else "失败"
        logger.info(f"{test_name}: {status}")
    
    # 检查是否所有测试都通过
    all_passed = all(result for _, result in results)
    if all_passed:
        logger.info("\n✅ 所有测试通过！数据源配置同步机制工作正常。")
    else:
        logger.warning("\n❌ 部分测试失败，需要进一步检查。")

if __name__ == "__main__":
    main()
