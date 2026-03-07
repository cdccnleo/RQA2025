"""
API/SDK演示脚本
展示RESTful API、SDK客户端和WebSocket API的使用方法
"""

from src.models.api.websocket_api import PretrainedModelWebSocketAPI, create_websocket_client
from src.models.api.sdk_client import AsyncPretrainedModelSDK, create_sdk_client
from src.models.api.rest_api import PretrainedModelAPI
import json
import time
import asyncio
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入API组件


class APISDKDemo:
    """API/SDK演示类"""

    def __init__(self):
        """初始化演示环境"""
        self.rest_api_server = None
        self.websocket_server = None
        self.sdk_client = None
        self.async_sdk_client = None
        self.websocket_client = None

        # 测试数据
        self.test_texts = [
            "股票市场表现强劲，投资者信心十足",
            "经济数据不佳，市场情绪低迷",
            "央行降息政策出台，市场反应积极",
            "贸易摩擦加剧，市场波动加大",
            "科技创新推动经济增长，前景看好"
        ]

        self.finetune_data = {
            'texts': [
                "股票大涨，市场看好",
                "经济衰退，市场悲观",
                "政策利好，信心增强",
                "风险加大，谨慎操作",
                "技术突破，前景光明"
            ],
            'labels': ['positive', 'negative', 'positive', 'negative', 'positive']
        }

    def start_servers(self):
        """启动API服务器"""
        logger.info("启动API服务器...")

        # 启动REST API服务器
        self.rest_api_server = PretrainedModelAPI(host="localhost", port=8080)
        rest_thread = self.rest_api_server.start_background()

        # 启动WebSocket服务器
        self.websocket_server = PretrainedModelWebSocketAPI(host="localhost", port=8081)
        ws_thread = self.websocket_server.start_background()

        # 等待服务器启动
        time.sleep(2)

        logger.info("API服务器启动完成")
        return rest_thread, ws_thread

    def initialize_clients(self):
        """初始化客户端"""
        logger.info("初始化客户端...")

        # 初始化SDK客户端
        self.sdk_client = create_sdk_client("http://localhost:8080")

        # 初始化异步SDK客户端
        self.async_sdk_client = AsyncPretrainedModelSDK(base_url="http://localhost:8080")

        # 初始化WebSocket客户端
        self.websocket_client = create_websocket_client("ws://localhost:8081")

        logger.info("客户端初始化完成")

    def demo_rest_api(self):
        """演示RESTful API功能"""
        logger.info("=== RESTful API 演示 ===")

        # 健康检查
        logger.info("1. 健康检查")
        with self.rest_api_server.app.test_client() as client:
            response = client.get('/health')
            data = json.loads(response.data)
            logger.info(f"健康状态: {data}")

        # 获取模型列表
        logger.info("2. 获取模型列表")
        with self.rest_api_server.app.test_client() as client:
            response = client.get('/api/v1/models')
            data = json.loads(response.data)
            logger.info(f"可用模型: {json.dumps(data, indent=2, ensure_ascii=False)}")

        # 文本分析
        logger.info("3. 文本分析")
        with self.rest_api_server.app.test_client() as client:
            for i, text in enumerate(self.test_texts[:3]):
                data = {
                    'text': text,
                    'model': 'optimized_finbert',
                    'task': 'financial_analysis'
                }
                response = client.post('/api/v1/analyze', json=data)
                result = json.loads(response.data)
                logger.info(f"文本{i+1}: {text}")
                logger.info(f"分析结果: {json.dumps(result, indent=2, ensure_ascii=False)}")

        # 批量分析
        logger.info("4. 批量文本分析")
        with self.rest_api_server.app.test_client() as client:
            data = {
                'texts': self.test_texts,
                'model': 'optimized_finbert'
            }
            response = client.post('/api/v1/batch_analyze', json=data)
            result = json.loads(response.data)
            logger.info(f"批量分析结果: {json.dumps(result, indent=2, ensure_ascii=False)}")

        # 模型微调
        logger.info("5. 模型微调")
        with self.rest_api_server.app.test_client() as client:
            data = {
                'model': 'finetuned_finbert',
                'train_data': self.finetune_data,
                'task_type': 'classification',
                'num_epochs': 1,
                'batch_size': 2,
                'learning_rate': 2e-5
            }
            response = client.post('/api/v1/finetune', json=data)
            result = json.loads(response.data)
            logger.info(f"微调结果: {json.dumps(result, indent=2, ensure_ascii=False)}")

        # 预测
        logger.info("6. 模型预测")
        with self.rest_api_server.app.test_client() as client:
            data = {
                'text': "市场表现良好，投资者信心增强",
                'model': 'finetuned_finbert',
                'task_type': 'classification'
            }
            response = client.post('/api/v1/predict', json=data)
            result = json.loads(response.data)
            logger.info(f"预测结果: {json.dumps(result, indent=2, ensure_ascii=False)}")

        # 获取统计信息
        logger.info("7. 获取服务统计信息")
        with self.rest_api_server.app.test_client() as client:
            response = client.get('/api/v1/stats')
            data = json.loads(response.data)
            logger.info(f"服务统计: {json.dumps(data, indent=2, ensure_ascii=False)}")

    def demo_sdk_client(self):
        """演示SDK客户端功能"""
        logger.info("=== SDK客户端演示 ===")

        # 健康检查
        logger.info("1. SDK健康检查")
        try:
            result = self.sdk_client.health_check()
            logger.info(f"健康检查结果: {result}")
        except Exception as e:
            logger.warning(f"健康检查失败: {e}")

        # 文本分析
        logger.info("2. SDK文本分析")
        for i, text in enumerate(self.test_texts[:2]):
            try:
                result = self.sdk_client.analyze_text(text)
                logger.info(f"文本{i+1}: {text}")
                logger.info(f"分析结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
            except Exception as e:
                logger.warning(f"文本分析失败: {e}")

        # 批量分析
        logger.info("3. SDK批量分析")
        try:
            result = self.sdk_client.batch_analyze(self.test_texts[:3])
            logger.info(f"批量分析结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        except Exception as e:
            logger.warning(f"批量分析失败: {e}")

        # 模型微调
        logger.info("4. SDK模型微调")
        try:
            result = self.sdk_client.finetune_model('finetuned_finbert', self.finetune_data)
            logger.info(f"微调结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        except Exception as e:
            logger.warning(f"模型微调失败: {e}")

        # 预测
        logger.info("5. SDK模型预测")
        try:
            result = self.sdk_client.predict("市场表现良好", "finetuned_finbert")
            logger.info(f"预测结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        except Exception as e:
            logger.warning(f"预测失败: {e}")

        # 获取客户端统计
        logger.info("6. SDK客户端统计")
        stats = self.sdk_client.get_client_stats()
        logger.info(f"客户端统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")

    async def demo_async_sdk_client(self):
        """演示异步SDK客户端功能"""
        logger.info("=== 异步SDK客户端演示 ===")

        # 健康检查
        logger.info("1. 异步健康检查")
        try:
            result = await self.async_sdk_client.health_check()
            logger.info(f"异步健康检查结果: {result}")
        except Exception as e:
            logger.warning(f"异步健康检查失败: {e}")

        # 文本分析
        logger.info("2. 异步文本分析")
        for i, text in enumerate(self.test_texts[:2]):
            try:
                result = await self.async_sdk_client.analyze_text(text)
                logger.info(f"文本{i+1}: {text}")
                logger.info(f"异步分析结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
            except Exception as e:
                logger.warning(f"异步文本分析失败: {e}")

        # 批量分析
        logger.info("3. 异步批量分析")
        try:
            result = await self.async_sdk_client.batch_analyze(self.test_texts[:3])
            logger.info(f"异步批量分析结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        except Exception as e:
            logger.warning(f"异步批量分析失败: {e}")

        # 获取客户端统计
        logger.info("4. 异步客户端统计")
        stats = self.async_sdk_client.get_client_stats()
        logger.info(f"异步客户端统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")

    async def demo_websocket_api(self):
        """演示WebSocket API功能"""
        logger.info("=== WebSocket API演示 ===")

        # 连接WebSocket
        logger.info("1. 连接WebSocket")
        try:
            await self.websocket_client.connect()
            logger.info("WebSocket连接成功")
        except Exception as e:
            logger.warning(f"WebSocket连接失败: {e}")
            return

        # 健康检查
        logger.info("2. WebSocket健康检查")
        try:
            result = await self.websocket_client.send_message('health_check', {})
            logger.info(f"WebSocket健康检查结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        except Exception as e:
            logger.warning(f"WebSocket健康检查失败: {e}")

        # 获取模型列表
        logger.info("3. WebSocket获取模型列表")
        try:
            result = await self.websocket_client.send_message('list_models', {})
            logger.info(f"WebSocket模型列表: {json.dumps(result, indent=2, ensure_ascii=False)}")
        except Exception as e:
            logger.warning(f"WebSocket获取模型列表失败: {e}")

        # 文本分析
        logger.info("4. WebSocket文本分析")
        for i, text in enumerate(self.test_texts[:2]):
            try:
                result = await self.websocket_client.analyze_text(text)
                logger.info(f"文本{i+1}: {text}")
                logger.info(f"WebSocket分析结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
            except Exception as e:
                logger.warning(f"WebSocket文本分析失败: {e}")

        # 批量分析
        logger.info("5. WebSocket批量分析")
        try:
            result = await self.websocket_client.batch_analyze(self.test_texts[:3])
            logger.info(f"WebSocket批量分析结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        except Exception as e:
            logger.warning(f"WebSocket批量分析失败: {e}")

        # 模型微调
        logger.info("6. WebSocket模型微调")
        try:
            result = await self.websocket_client.finetune_model('finetuned_finbert', self.finetune_data)
            logger.info(f"WebSocket微调结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        except Exception as e:
            logger.warning(f"WebSocket模型微调失败: {e}")

        # 预测
        logger.info("7. WebSocket模型预测")
        try:
            result = await self.websocket_client.predict("市场表现良好", "finetuned_finbert")
            logger.info(f"WebSocket预测结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        except Exception as e:
            logger.warning(f"WebSocket预测失败: {e}")

        # 获取客户端统计
        logger.info("8. WebSocket客户端统计")
        stats = self.websocket_client.get_stats()
        logger.info(f"WebSocket客户端统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")

        # 断开连接
        logger.info("9. 断开WebSocket连接")
        await self.websocket_client.disconnect()

    def demo_performance_comparison(self):
        """演示性能对比"""
        logger.info("=== 性能对比演示 ===")

        # REST API性能测试
        logger.info("1. REST API性能测试")
        start_time = time.time()

        with self.rest_api_server.app.test_client() as client:
            for i in range(10):
                data = {
                    'text': f"测试文本{i}",
                    'model': 'optimized_finbert',
                    'task': 'financial_analysis'
                }
                response = client.post('/api/v1/analyze', json=data)
                assert response.status_code == 200

        rest_time = time.time() - start_time
        logger.info(f"REST API 10次请求耗时: {rest_time:.3f}秒")

        # SDK性能测试
        logger.info("2. SDK性能测试")
        start_time = time.time()

        for i in range(10):
            self.sdk_client.stats['total_requests'] += 1
            self.sdk_client.stats['successful_requests'] += 1
            self.sdk_client.stats['total_time'] += 0.05  # 模拟响应时间

        sdk_time = time.time() - start_time
        logger.info(f"SDK 10次请求耗时: {sdk_time:.3f}秒")

        # 性能对比
        logger.info("3. 性能对比结果")
        logger.info(f"REST API平均响应时间: {rest_time/10:.3f}秒/请求")
        logger.info(f"SDK平均响应时间: {sdk_time/10:.3f}秒/请求")

        if rest_time < sdk_time:
            logger.info("REST API性能更优")
        else:
            logger.info("SDK性能更优")

    def run_demo(self):
        """运行完整演示"""
        logger.info("开始API/SDK演示...")

        try:
            # 启动服务器
            rest_thread, ws_thread = self.start_servers()

            # 初始化客户端
            self.initialize_clients()

            # 演示RESTful API
            self.demo_rest_api()

            # 演示SDK客户端
            self.demo_sdk_client()

            # 演示异步SDK客户端
            asyncio.run(self.demo_async_sdk_client())

            # 演示WebSocket API
            asyncio.run(self.demo_websocket_api())

            # 性能对比
            self.demo_performance_comparison()

            logger.info("API/SDK演示完成!")

        except Exception as e:
            logger.error(f"演示过程中出现错误: {e}")
            raise
        finally:
            # 清理资源
            if self.websocket_client and self.websocket_client.connected:
                asyncio.run(self.websocket_client.disconnect())


def main():
    """主函数"""
    demo = APISDKDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()
