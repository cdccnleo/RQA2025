# -*- coding: utf-8 -*-
"""
多模态数据处理器
支持图像、文本、音频等多模态数据的预处理和特征提取
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging

# 可选导入，在使用时检查
try:
    from PIL import Image
    import torch
    from transformers import (
        AutoTokenizer, AutoModel, AutoFeatureExtractor,
        CLIPProcessor, CLIPModel,
        Wav2Vec2Processor, Wav2Vec2Model
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    print("⚠️  部分依赖未安装，某些功能将不可用")


class MultimodalProcessor:
    """多模态数据处理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化多模态处理器

        Args:
            config: 配置字典
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)

        if not IMPORTS_AVAILABLE:
            self.logger.warning("多模态处理依赖不完整，请安装相关包")
            return

        # 初始化模型
        self._init_models()

    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            "text_model": "bert-base-chinese",
            "vision_model": "google/vit-base-patch16-224",
            "audio_model": "facebook/wav2vec2-base-960h",
            "multimodal_model": "openai/clip-vit-base-patch32",
            "max_text_length": 512,
            "image_size": 224,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

    def _init_models(self):
        """初始化各个模态的模型"""
        try:
            self.device = torch.device(self.config["device"])

            # 文本模型
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.config["text_model"])
            self.text_model = AutoModel.from_pretrained(self.config["text_model"]).to(self.device)

            # 视觉模型
            self.vision_processor = AutoFeatureExtractor.from_pretrained(self.config["vision_model"])
            self.vision_model = AutoModel.from_pretrained(self.config["vision_model"]).to(self.device)

            # 多模态模型 (CLIP)
            self.clip_processor = CLIPProcessor.from_pretrained(self.config["multimodal_model"])
            self.clip_model = CLIPModel.from_pretrained(self.config["multimodal_model"]).to(self.device)

            # 音频模型 (可选)
            try:
                self.audio_processor = Wav2Vec2Processor.from_pretrained(self.config["audio_model"])
                self.audio_model = Wav2Vec2Model.from_pretrained(self.config["audio_model"]).to(self.device)
                self.audio_available = True
            except:
                self.logger.warning("音频模型加载失败")
                self.audio_available = False

            self.logger.info("多模态模型初始化完成")

        except Exception as e:
            self.logger.error(f"模型初始化失败: {e}")
            raise

    def process_text(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        处理文本数据

        Args:
            text: 输入文本

        Returns:
            处理结果字典
        """
        if not IMPORTS_AVAILABLE:
            return {"error": "依赖未安装"}

        try:
            # 确保输入是列表
            if isinstance(text, str):
                texts = [text]
            else:
                texts = text

            # 分词
            inputs = self.text_tokenizer(
                texts,
                max_length=self.config["max_text_length"],
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            # 提取特征
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # 平均池化

            return {
                "embeddings": embeddings.cpu().numpy(),
                "attention_mask": inputs["attention_mask"].cpu().numpy(),
                "input_ids": inputs["input_ids"].cpu().numpy(),
                "texts": texts
            }

        except Exception as e:
            self.logger.error(f"文本处理失败: {e}")
            return {"error": str(e)}

    def process_image(self, image_path: Union[str, Path, List[Union[str, Path]]]) -> Dict[str, Any]:
        """
        处理图像数据

        Args:
            image_path: 图像路径

        Returns:
            处理结果字典
        """
        if not IMPORTS_AVAILABLE:
            return {"error": "依赖未安装"}

        try:
            # 确保输入是列表
            if isinstance(image_path, (str, Path)):
                image_paths = [image_path]
            else:
                image_paths = image_path

            images = []
            for path in image_paths:
                img = Image.open(path).convert('RGB')
                images.append(img)

            # 预处理
            inputs = self.vision_processor(images=images, return_tensors="pt").to(self.device)

            # 提取特征
            with torch.no_grad():
                outputs = self.vision_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # 平均池化

            return {
                "embeddings": embeddings.cpu().numpy(),
                "pixel_values": inputs["pixel_values"].cpu().numpy(),
                "image_paths": [str(p) for p in image_paths]
            }

        except Exception as e:
            self.logger.error(f"图像处理失败: {e}")
            return {"error": str(e)}

    def process_multimodal(self, text: str, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        处理多模态数据 (文本+图像)

        Args:
            text: 输入文本
            image_path: 图像路径

        Returns:
            多模态处理结果
        """
        if not IMPORTS_AVAILABLE:
            return {"error": "依赖未安装"}

        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')

            # 多模态预处理
            inputs = self.clip_processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            # 提取特征
            with torch.no_grad():
                outputs = self.clip_model(**inputs)

                # 文本和图像的特征
                text_embeds = outputs.text_embeds
                image_embeds = outputs.image_embeds

                # 计算相似度
                similarity = torch.nn.functional.cosine_similarity(text_embeds, image_embeds)

            return {
                "text_embedding": text_embeds.cpu().numpy(),
                "image_embedding": image_embeds.cpu().numpy(),
                "similarity": similarity.cpu().numpy(),
                "text": text,
                "image_path": str(image_path)
            }

        except Exception as e:
            self.logger.error(f"多模态处理失败: {e}")
            return {"error": str(e)}

    def compute_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
        """
        计算两个嵌入向量的相似度

        Args:
            embeddings1: 第一个嵌入向量
            embeddings2: 第二个嵌入向量

        Returns:
            相似度分数
        """
        if not IMPORTS_AVAILABLE:
            return 0.0

        try:
            # 归一化
            norm1 = embeddings1 / np.linalg.norm(embeddings1, axis=-1, keepdims=True)
            norm2 = embeddings2 / np.linalg.norm(embeddings2, axis=-1, keepdims=True)

            # 计算余弦相似度
            similarity = np.dot(norm1, norm2.T)
            return float(similarity[0, 0])

        except Exception as e:
            self.logger.error(f"相似度计算失败: {e}")
            return 0.0

    def batch_process(self, data: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        批量处理多模态数据

        Args:
            data: 包含不同模态数据的字典

        Returns:
            批量处理结果
        """
        results = {}

        # 处理文本
        if "texts" in data:
            results["text_results"] = self.process_text(data["texts"])

        # 处理图像
        if "images" in data:
            results["image_results"] = self.process_image(data["images"])

        # 处理多模态组合
        if "text_image_pairs" in data:
            multimodal_results = []
            for text, image_path in data["text_image_pairs"]:
                result = self.process_multimodal(text, image_path)
                multimodal_results.append(result)
            results["multimodal_results"] = multimodal_results

        return results


def create_sample_data():
    """创建示例数据用于测试"""
    print("📦 创建多模态AI示例数据...")

    sample_dir = Path(__file__).parent.parent / "multimodal_ai" / "data"

    # 示例文本
    sample_texts = [
        "量化交易策略优化",
        "金融市场风险评估",
        "股票价格预测模型",
        "投资组合再平衡",
        "算法交易系统"
    ]

    # 保存示例文本
    text_file = sample_dir / "sample_texts.json"
    with open(text_file, 'w', encoding='utf-8') as f:
        json.dump({"texts": sample_texts}, f, ensure_ascii=False, indent=2)

    print(f"✅ 示例文本已保存: {text_file}")

    # 创建示例配置文件
    config = {
        "multimodal_processor": {
            "text_model": "bert-base-chinese",
            "vision_model": "google/vit-base-patch16-224",
            "multimodal_model": "openai/clip-vit-base-patch32",
            "device": "cpu",  # 默认使用CPU
            "max_text_length": 128
        },
        "sample_data": {
            "text_count": len(sample_texts),
            "image_count": 0,  # 暂时没有示例图片
            "multimodal_pairs": 0
        }
    }

    config_file = sample_dir / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ 配置文件已保存: {config_file}")


if __name__ == "__main__":
    # 创建示例数据
    create_sample_data()

    print("\n💡 使用示例:")
    print("```python")
    print("from multimodal_processor import MultimodalProcessor")
    print("")
    print("# 初始化处理器")
    print("processor = MultimodalProcessor()")
    print("")
    print("# 处理文本")
    print("text_result = processor.process_text('量化交易策略优化')")
    print("")
    print("# 处理图像")
    print("image_result = processor.process_image('path/to/image.jpg')")
    print("")
    print("# 多模态处理")
    print("multimodal_result = processor.process_multimodal('股票图表', 'chart.jpg')")
    print("```")
