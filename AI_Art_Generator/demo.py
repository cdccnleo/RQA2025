#!/usr/bin/env python3
"""
AI Art Generator 演示脚本

展示完整的AI艺术生成平台功能
"""

import requests
import time
import sys

# API配置
API_BASE_URL = "http://localhost:8001"


def test_api_health():
    """测试API健康状态"""
    print("🏥 测试API健康状态...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ API健康检查通过")
            print(f"   状态: {data['status']}")
            print(f"   GPU可用: {data['gpu_available']}")
            return True
        else:
            print(f"❌ API健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API连接失败: {e}")
        return False


def test_model_status():
    """测试模型状态"""
    print("\n🎨 测试模型状态...")
    try:
        response = requests.get(f"{API_BASE_URL}/model/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ 模型状态获取成功")
            print(f"   模型类型: {data['model_type']}")
            print(f"   设备: {data['device']}")
            print(f"   潜在维度: {data['latent_dim']}")
            return True
        else:
            print(f"❌ 模型状态获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 模型状态测试失败: {e}")
        return False


def test_art_generation():
    """测试艺术生成"""
    print("\n🎨 测试艺术生成...")
    try:
        payload = {
            "num_images": 2,
            "style": "random",
            "quality": "standard",
            "seed": 42
        }

        response = requests.post(f"{API_BASE_URL}/generate", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("✅ 艺术生成成功")
            print(f"   生成ID: {data['generation_id']}")
            print(f"   生成时间: {data['timestamp']}")
            print(f"   图像数量: {len(data['images'])}")
            print(f"   元数据: {data['metadata']}")
            return True
        else:
            print(f"❌ 艺术生成失败: {response.status_code}")
            print(f"   响应: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 艺术生成测试失败: {e}")
        return False


def test_random_seed():
    """测试随机种子生成"""
    print("\n🎲 测试随机种子生成...")
    try:
        response = requests.get(f"{API_BASE_URL}/generate/random-seed")
        if response.status_code == 200:
            data = response.json()
            print("✅ 随机种子生成成功")
            print(f"   种子值: {data['seed']}")
            return True
        else:
            print(f"❌ 随机种子生成失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 随机种子测试失败: {e}")
        return False


def test_styles():
    """测试艺术风格列表"""
    print("\n🎭 测试艺术风格列表...")
    try:
        response = requests.get(f"{API_BASE_URL}/styles")
        if response.status_code == 200:
            data = response.json()
            print("✅ 艺术风格列表获取成功")
            print(f"   风格数量: {len(data['styles'])}")
            for style in data['styles'][:3]:  # 显示前3个
                print(f"   - {style['name']}: {style['description']}")
            return True
        else:
            print(f"❌ 艺术风格列表获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 艺术风格测试失败: {e}")
        return False


def test_interpolation():
    """测试潜在空间插值"""
    print("\n🔄 测试潜在空间插值...")
    try:
        payload = {
            "seed1": 123,
            "seed2": 456,
            "steps": 5
        }

        response = requests.post(f"{API_BASE_URL}/generate/interpolate", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("✅ 潜在空间插值成功")
            print(f"   插值ID: {data['interpolation_id']}")
            print(f"   插值步数: {len(data['images'])}")
            print(f"   种子1: {data['metadata']['seed1']}")
            print(f"   种子2: {data['metadata']['seed2']}")
            return True
        else:
            print(f"❌ 潜在空间插值失败: {response.status_code}")
            print(f"   响应: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 潜在空间插值测试失败: {e}")
        return False


def wait_for_model_loading():
    """等待模型加载完成"""
    print("⏳ 等待模型加载...")
    max_attempts = 30
    for i in range(max_attempts):
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                data = response.json()
                if data.get('model_status') == 'loaded':
                    print("✅ 模型已加载完成")
                    return True
        except:
            pass

        time.sleep(2)
        print(f"   等待中... ({i+1}/{max_attempts})")

    print("❌ 模型加载超时")
    return False


def main():
    """主演示函数"""
    print("🎨 AI Art Generator 功能演示")
    print("=" * 60)

    # 等待API启动
    print("🔌 连接到API服务器...")
    if not wait_for_model_loading():
        print("❌ 无法连接到API服务器")
        print("请确保后端服务正在运行: cd backend && python main.py")
        return False

    # 运行所有测试
    tests = [
        ("API健康检查", test_api_health),
        ("模型状态", test_model_status),
        ("艺术生成", test_art_generation),
        ("随机种子", test_random_seed),
        ("艺术风格", test_styles),
        ("潜在插值", test_interpolation),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} 测试失败")

    print("\n" + "=" * 60)
    print(f"📊 演示结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！AI艺术生成平台运行正常！")
        print("\n🚀 前端访问地址: http://localhost:3000")
        print("📚 API文档地址: http://localhost:8001/docs")
        return True
    else:
        print("⚠️  部分测试失败，请检查服务状态")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
