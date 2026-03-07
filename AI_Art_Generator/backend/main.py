#!/usr/bin/env python3
"""
AI Art Generator - FastAPI后端服务

提供艺术生成、模型管理、用户交互等API接口
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import numpy as np
from PIL import Image
import io
import base64
import uuid
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from models.dcgan import create_dcgan_model

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="AI Art Generator API",
    description="基于深度学习的智能艺术创作平台",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React开发服务器
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
dcgan_model = None
model_loaded = False
executor = ThreadPoolExecutor(max_workers=4)


# 数据模型
class GenerateRequest(BaseModel):
    """生成请求模型"""
    num_images: int = 1
    seed: Optional[int] = None
    style: Optional[str] = "random"
    quality: Optional[str] = "standard"  # standard, high, ultra


class GenerateResponse(BaseModel):
    """生成响应模型"""
    images: List[str]  # Base64编码的图像列表
    generation_id: str
    timestamp: str
    metadata: Dict[str, Any]


class InterpolationRequest(BaseModel):
    """插值请求模型"""
    seed1: int
    seed2: int
    steps: int = 10


class ModelStatus(BaseModel):
    """模型状态"""
    loaded: bool
    model_type: str
    device: str
    latent_dim: int


# 全局变量
dcgan_model = None
model_loaded = False


def initialize_model():
    """初始化模型"""
    global dcgan_model, model_loaded

    if not model_loaded:
        try:
            logger.info("🔄 Loading DCGAN model...")
            dcgan_model = create_dcgan_model()
            model_loaded = True
            logger.info("✅ DCGAN model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load DCGAN model: {e}")
            model_loaded = False


# 在模块导入时初始化模型
initialize_model()


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "🎨 AI Art Generator API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_loaded
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": "loaded" if model_loaded else "not_loaded",
        "gpu_available": torch.cuda.is_available()
    }


@app.get("/model/status")
async def get_model_status():
    """获取模型状态"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelStatus(
        loaded=True,
        model_type="DCGAN",
        device=str(dcgan_model.device),
        latent_dim=dcgan_model.latent_dim
    )


def generate_art_sync(num_images: int, seed: Optional[int] = None) -> torch.Tensor:
    """同步生成艺术（在线程池中执行）"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if seed is not None:
            torch.manual_seed(seed)

        images = dcgan_model.generate(num_images)
        return images
    except Exception as e:
        logger.error(f"Art generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate", response_model=GenerateResponse)
async def generate_art(request: GenerateRequest, background_tasks: BackgroundTasks):
    """生成艺术作品"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.num_images < 1 or request.num_images > 10:
        raise HTTPException(status_code=400, detail="Number of images must be between 1 and 10")

    try:
        # 在线程池中执行生成任务
        loop = asyncio.get_event_loop()
        images = await loop.run_in_executor(
            executor,
            generate_art_sync,
            request.num_images,
            request.seed
        )

        # 转换为PIL图像并编码为base64
        image_list = []
        for i in range(images.shape[0]):
            # 从tensor转换为PIL图像
            img_tensor = images[i]
            img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array)

            # 转换为base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            image_list.append(f"data:image/png;base64,{img_base64}")

        generation_id = str(uuid.uuid4())

        return GenerateResponse(
            images=image_list,
            generation_id=generation_id,
            timestamp=datetime.now().isoformat(),
            metadata={
                "num_images": request.num_images,
                "seed": request.seed,
                "style": request.style,
                "quality": request.quality,
                "model": "DCGAN",
                "image_size": "64x64"
            }
        )

    except Exception as e:
        logger.error(f"Art generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate/interpolate")
async def generate_interpolation(request: InterpolationRequest):
    """生成潜在空间插值"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.steps < 2 or request.steps > 20:
        raise HTTPException(status_code=400, detail="Steps must be between 2 and 20")

    try:
        # 生成两个随机向量
        torch.manual_seed(request.seed1)
        z1 = torch.randn(1, dcgan_model.latent_dim, 1, 1, device=dcgan_model.device)

        torch.manual_seed(request.seed2)
        z2 = torch.randn(1, dcgan_model.latent_dim, 1, 1, device=dcgan_model.device)

        # 生成插值
        loop = asyncio.get_event_loop()
        interpolated_images = await loop.run_in_executor(
            executor,
            dcgan_model.interpolate_latent,
            z1.squeeze(),
            z2.squeeze(),
            request.steps
        )

        # 转换为base64
        image_list = []
        for img_tensor in interpolated_images:
            img_array = (img_tensor.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array)

            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            image_list.append(f"data:image/png;base64,{img_base64}")

        return {
            "images": image_list,
            "interpolation_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "seed1": request.seed1,
                "seed2": request.seed2,
                "steps": request.steps,
                "model": "DCGAN"
            }
        }

    except Exception as e:
        logger.error(f"Interpolation generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Interpolation failed: {str(e)}")


@app.get("/generate/random-seed")
async def get_random_seed():
    """获取随机种子"""
    import random
    return {"seed": random.randint(0, 2**31 - 1)}


@app.get("/styles")
async def get_available_styles():
    """获取可用艺术风格"""
    styles = [
        {"id": "random", "name": "随机生成", "description": "基于噪声的自由创作"},
        {"id": "abstract", "name": "抽象艺术", "description": "现代抽象艺术风格"},
        {"id": "impressionist", "name": "印象派", "description": "莫奈风格的印象派"},
        {"id": "cubist", "name": "立体派", "description": "毕加索风格的立体主义"},
        {"id": "surrealist", "name": "超现实主义", "description": "达利风格的超现实"},
        {"id": "minimalist", "name": "极简主义", "description": "现代极简艺术风格"}
    ]

    return {"styles": styles}


@app.get("/stats")
async def get_generation_stats():
    """获取生成统计信息"""
    if not model_loaded:
        return {"error": "Model not loaded"}

    stats = dcgan_model.get_training_stats()
    return {
        "model_stats": stats,
        "gpu_memory": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
        "gpu_memory_cached": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
    }


@app.post("/feedback")
async def submit_feedback(feedback: Dict[str, Any]):
    """提交用户反馈"""
    # 这里可以保存到数据库或日志
    logger.info(f"User feedback received: {feedback}")

    return {
        "status": "received",
        "feedback_id": str(uuid.uuid4()),
        "message": "感谢您的反馈！"
    }


if __name__ == "__main__":
    import uvicorn

    print("🎨 Starting AI Art Generator API")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
