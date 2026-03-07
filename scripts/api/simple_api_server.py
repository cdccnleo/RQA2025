#!/usr/bin/env python3
"""
简单的API服务器，用于端到端测试
提供健康检查、模型推理等基本接口
"""

from src.models.api.monitoring import APIMonitor
from src.models.pretrained_models import PretrainedModelManager
import sys
import time
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
CORS(app)

# 初始化模型管理器和监控器
model_manager = PretrainedModelManager()
api_monitor = APIMonitor()


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        # 检查模型管理器状态
        available_models = model_manager.list_available_models()

        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "available_models": available_models,
            "message": "API服务运行正常"
        }), 200
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }), 500


@app.route('/api/v1/analyze', methods=['POST'])
def analyze_text():
    """文本分析接口"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "缺少text参数"}), 400

        text = data['text']
        model_name = data.get('model_name', 'finbert')
        task_type = data.get('task_type', 'sentiment')

        # 记录API请求开始时间
        start_time = time.time()

        # 调用模型分析
        if model_name in model_manager.list_available_models():
            result = model_manager.analyze_text(text, model_name)

            # 记录API请求
            response_time = time.time() - start_time
            api_monitor.api_metrics.update(response_time, True)

            return jsonify({
                "status": "success",
                "result": result,
                "model": model_name,
                "task_type": task_type,
                "response_time": response_time
            }), 200
        else:
            return jsonify({"error": f"模型 {model_name} 不可用"}), 400

    except Exception as e:
        logger.error(f"文本分析失败: {e}")
        response_time = time.time() - start_time if 'start_time' in locals() else 0
        api_monitor.api_metrics.update(response_time, False)

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/api/v1/finetune', methods=['POST'])
def finetune_model():
    """模型微调接口"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "缺少text参数"}), 400

        text = data['text']
        label = data.get('label', 'test')

        # 模拟微调过程
        start_time = time.time()
        time.sleep(0.5)  # 模拟处理时间

        response_time = time.time() - start_time
        api_monitor.api_metrics.update(response_time, True)

        return jsonify({
            "status": "success",
            "message": "模型微调完成",
            "text": text,
            "label": label,
            "response_time": response_time
        }), 200

    except Exception as e:
        logger.error(f"模型微调失败: {e}")
        response_time = time.time() - start_time if 'start_time' in locals() else 0
        api_monitor.api_metrics.update(response_time, False)

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/api/v1/models', methods=['GET'])
def list_models():
    """列出可用模型接口"""
    try:
        available_models = model_manager.list_available_models()

        return jsonify({
            "status": "success",
            "models": available_models,
            "count": len(available_models)
        }), 200

    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/monitoring/metrics', methods=['GET'])
def get_metrics():
    """获取监控指标接口"""
    try:
        api_metrics = api_monitor.get_api_metrics()
        system_metrics = api_monitor.get_system_metrics()

        return jsonify({
            "status": "success",
            "api_metrics": api_metrics,
            "system_metrics": system_metrics,
            "timestamp": time.time()
        }), 200

    except Exception as e:
        logger.error(f"获取监控指标失败: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/ws', methods=['GET'])
def websocket_endpoint():
    """WebSocket端点（模拟）"""
    return jsonify({
        "status": "ok",
        "message": "WebSocket endpoint available",
        "timestamp": time.time()
    }), 200


def main():
    """启动API服务器"""
    logger.info("启动简单API服务器...")

    # 启动监控
    api_monitor.start_monitoring()

    # 启动Flask应用
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )


if __name__ == "__main__":
    main()
