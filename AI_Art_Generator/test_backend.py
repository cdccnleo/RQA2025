#!/usr/bin/env python3
"""
AI Art Generator 后端测试脚本
"""

import torch
import numpy as np
from PIL import Image
import io
import base64
import sys
import os

print("🎨 AI Art Generator 后端测试套件")
print("=" * 60)

# 测试PyTorch
print("🔥 测试PyTorch可用性...")
try:
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    # 测试基本张量操作
    x = torch.randn(3, 64, 64)
    y = torch.randn(3, 64, 64)
    z = x + y
    assert z.shape == (3, 64, 64)
    print("✅ PyTorch测试通过")
except Exception as e:
    print(f"❌ PyTorch测试失败: {e}")

# 测试图像编码
print("\n🖼️  测试图像编码...")
try:
    test_image = np.random.rand(64, 64, 3)
    test_image = (test_image * 255).astype(np.uint8)
    pil_image = Image.fromarray(test_image)

    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    assert img_base64.startswith('iVBORw0KGgo')
    print("✅ 图像编码测试通过")
except Exception as e:
    print(f"❌ 图像编码测试失败: {e}")

# 测试DCGAN导入
print("\n🎨 测试DCGAN模型...")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
    from models.dcgan import DCGAN, create_dcgan_model

    dcgan = create_dcgan_model()
    print("✅ DCGAN模型创建成功")
    print(f"   设备: {dcgan.device}")
    print(f"   潜在维度: {dcgan.latent_dim}")

    # 测试生成
    images = dcgan.generate(1)
    print("✅ 图像生成成功")
    print(f"   图像形状: {images.shape}")

except Exception as e:
    print(f"❌ DCGAN模型测试失败: {e}")

print("\n🎊 后端测试完成！")
print("如需启动服务，请运行: cd backend && python main.py")