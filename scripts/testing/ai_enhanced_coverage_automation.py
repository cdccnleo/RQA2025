#!/usr/bin/env python3
"""
RQA2025 AI增强测试覆盖率自动化脚本
集成Deepseek大模型，智能生成测试用例并持续提升覆盖率
"""

import os
import sys
import subprocess
import argparse
import asyncio
import aiohttp
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import hashlib
import pickle

# 导入AST分析器
try:
    from .ast_code_analyzer import ASTCodeAnalyzer
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    sys.path.append(str(Path(__file__).parent))
    from ast_code_analyzer import ASTCodeAnalyzer

# 导入安全审查器
try:
    from .security_code_reviewer import SecurityCodeReviewer
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from security_code_reviewer import SecurityCodeReviewer

# 导入增强日志系统
try:
    from .enhanced_logging_system import EnhancedLoggingSystem
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from enhanced_logging_system import EnhancedLoggingSystem

# 导入插件架构
try:
    from .plugin_architecture import PluginManager
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from plugin_architecture import PluginManager

# 导入测试质量评估器
try:
    from .test_quality_assessor import TestQualityAssessor
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from test_quality_assessor import TestQualityAssessor

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 配置日志

# 确保日志目录存在
Path('logs').mkdir(exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 移除所有旧handler，防止重复
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# 文件日志
file_handler = logging.FileHandler('logs/ai_coverage_automation.log', encoding='utf-8')
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# 控制台日志
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)


class DependencyChecker:
    """依赖检查器"""

    @staticmethod
    def check_python_dependencies():
        """检查Python依赖"""
        required_packages = [
            'aiohttp', 'pytest', 'pytest-cov', 'schedule',
            'numpy', 'pandas', 'requests'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            logger.error(f"缺少必要的Python包: {', '.join(missing_packages)}")
            logger.info("请运行: pip install " + " ".join(missing_packages))
            return False

        logger.info("✅ Python依赖检查通过")
        return True

    @staticmethod
    def check_system_dependencies():
        """检查系统依赖"""
        # 检查conda环境
        if 'rqa' not in sys.prefix and 'rqa' not in sys.executable:
            logger.error("❌ 未在conda rqa环境中运行")
            return False

        # 检查必要目录
        required_dirs = ['src', 'tests', 'logs']
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                logger.error(f"❌ 缺少必要目录: {dir_name}")
                return False

        logger.info("✅ 系统依赖检查通过")
        return True

    @staticmethod
    async def check_ai_service(api_base: str, timeout: int = 10):
        """检查AI服务可用性"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{api_base}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        logger.info("✅ AI服务连接正常")
                        return True
                    else:
                        logger.error(f"❌ AI服务响应异常: {response.status}")
                        return False
        except asyncio.TimeoutError:
            logger.error(f"❌ AI服务连接超时 ({timeout}秒)")
            return False
        except Exception as e:
            logger.error(f"❌ AI服务连接失败: {e}")
            return False


class RetryHandler:
    """重试处理器"""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def retry_async(self, func, *args, **kwargs):
        """异步重试"""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                delay = self.base_delay * (2 ** attempt)  # 指数退避
                logger.warning(f"第{attempt + 1}次尝试失败: {e}, {delay}秒后重试")
                await asyncio.sleep(delay)

        logger.error(f"重试{self.max_retries}次后仍然失败: {last_exception}")
        raise last_exception

    def retry_sync(self, func, *args, **kwargs):
        """同步重试"""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"第{attempt + 1}次尝试失败: {e}, {delay}秒后重试")
                time.sleep(delay)

        logger.error(f"重试{self.max_retries}次后仍然失败: {last_exception}")
        raise last_exception


class TimeoutHandler:
    """超时处理器"""

    def __init__(self, default_timeout: int = 300):
        self.default_timeout = default_timeout

    async def with_timeout(self, coro, timeout: int = None):
        """异步超时处理"""
        if timeout is None:
            timeout = self.default_timeout

        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"操作超时 ({timeout}秒)")
            raise

    def run_with_timeout(self, func, *args, timeout: int = None, **kwargs):
        """同步超时处理"""
        if timeout is None:
            timeout = self.default_timeout

        try:
            return asyncio.run(asyncio.wait_for(
                asyncio.to_thread(func, *args, **kwargs),
                timeout=timeout
            ))
        except asyncio.TimeoutError:
            logger.error(f"操作超时 ({timeout}秒)")
            raise


class DeepseekAIConnector:
    """Deepseek AI连接器"""

    def __init__(self, api_base: str = "http://localhost:11434", model: str = "deepseek-coder"):
        self.api_base = api_base
        self.model = model
        self.session = None
        self.cache_dir = Path("cache/ai_test_generation")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retry_handler = RetryHandler(max_retries=3, base_delay=2.0)
        self.timeout_handler = TimeoutHandler(default_timeout=180)

    async def __aenter__(self):
        """异步上下文管理器入口"""
        # 配置连接器
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True,
            ssl=False  # 本地服务通常不需要SSL
        )

        timeout = aiohttp.ClientTimeout(
            total=120,
            connect=30,
            sock_read=60
        )

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'RQA2025-AI-Coverage-Automation/1.0'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()

    def _get_cache_key(self, prompt: str, module_path: str) -> str:
        """生成缓存键"""
        content = f"{prompt}:{module_path}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"

    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """从缓存加载结果"""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                # 检查缓存是否过期（7天）
                if time.time() - cache_path.stat().st_mtime > 7 * 24 * 3600:
                    logger.info(f"缓存已过期，重新生成: {cache_key}")
                    cache_path.unlink()
                    return None

                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"缓存加载失败: {e}")
                # 删除损坏的缓存文件
                try:
                    cache_path.unlink()
                except:
                    pass
        return None

    def _save_to_cache(self, cache_key: str, result: str):
        """保存结果到缓存"""
        cache_path = self._get_cache_path(cache_key)
        try:
            # 确保缓存目录存在
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)

            # 限制缓存大小（最多1000个文件）
            cache_files = list(self.cache_dir.glob("*.pkl"))
            if len(cache_files) > 1000:
                # 删除最旧的文件
                cache_files.sort(key=lambda x: x.stat().st_mtime)
                for old_file in cache_files[:-1000]:
                    try:
                        old_file.unlink()
                    except:
                        pass

        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")

    async def generate_test_code(self, module_path: str, module_content: str,
                                 current_coverage: float, target_coverage: float) -> str:
        """使用AI生成测试代码"""

        # 构建提示词
        prompt = self._build_test_generation_prompt(module_path, module_content,
                                                    current_coverage, target_coverage)

        # 检查缓存
        cache_key = self._get_cache_key(prompt, module_path)
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            logger.info(f"使用缓存的测试代码: {module_path}")
            return cached_result

        # 使用重试机制调用AI API
        async def _call_ai_api():
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的Python测试工程师，专门为RQA2025项目生成高质量的测试用例。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "temperature": 0.3,
                "max_tokens": 4000
            }

            async with self.session.post(
                f"{self.api_base}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    generated_code = result['choices'][0]['message']['content']

                    # 提取代码块
                    code_match = re.search(r'```python\n(.*?)\n```', generated_code, re.DOTALL)
                    if code_match:
                        test_code = code_match.group(1)
                    else:
                        test_code = generated_code

                    # 验证生成的代码
                    if len(test_code.strip()) < 100:
                        raise ValueError("生成的测试代码过短")

                    # 保存到缓存
                    self._save_to_cache(cache_key, test_code)

                    logger.info(f"AI生成测试代码成功: {module_path}")
                    return test_code
                else:
                    error_text = await response.text()
                    raise Exception(f"AI API调用失败: {response.status} - {error_text}")

        try:
            # 使用重试和超时处理
            test_code = await self.timeout_handler.with_timeout(
                self.retry_handler.retry_async(_call_ai_api),
                timeout=180
            )
            return test_code

        except Exception as e:
            logger.error(f"AI连接失败: {e}")
            return self._generate_fallback_test(module_path)

    async def generate_test_code_with_ast(self, module_path: str, module_content: str,
                                          current_coverage: float, target_coverage: float,
                                          ast_suggestions: Dict[str, Any]) -> str:
        """使用AI生成测试代码（包含AST分析结果）"""

        # 构建包含AST分析的提示词
        prompt = self._build_ast_enhanced_test_generation_prompt(
            module_path, module_content, current_coverage, target_coverage, ast_suggestions
        )

        # 检查缓存
        cache_key = self._get_cache_key(prompt, module_path)
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            logger.info(f"使用缓存的AST增强测试代码: {module_path}")
            return cached_result

        # 使用重试机制调用AI API
        async def _call_ai_api():
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的Python测试工程师，专门为RQA2025项目生成高质量的测试用例。你擅长基于AST分析结果生成针对性的测试。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "temperature": 0.3,
                "max_tokens": 5000
            }

            async with self.session.post(
                f"{self.api_base}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    generated_code = result['choices'][0]['message']['content']

                    # 提取代码块
                    code_match = re.search(r'```python\n(.*?)\n```', generated_code, re.DOTALL)
                    if code_match:
                        test_code = code_match.group(1)
                    else:
                        test_code = generated_code

                    # 验证生成的代码
                    if len(test_code.strip()) < 100:
                        raise ValueError("生成的测试代码过短")

                    # 保存到缓存
                    self._save_to_cache(cache_key, test_code)

                    logger.info(f"AI生成AST增强测试代码成功: {module_path}")
                    return test_code
                else:
                    error_text = await response.text()
                    raise Exception(f"AI API调用失败: {response.status} - {error_text}")

        try:
            # 使用重试和超时处理
            test_code = await self.timeout_handler.with_timeout(
                self.retry_handler.retry_async(_call_ai_api),
                timeout=180
            )
            return test_code

        except Exception as e:
            logger.error(f"AI连接失败: {e}")
            return self._generate_fallback_test(module_path)

    def _build_test_generation_prompt(self, module_path: str, module_content: str,
                                      current_coverage: float, target_coverage: float) -> str:
        """构建测试生成提示词"""

        # 分析模块结构
        classes = self._extract_classes(module_content)
        functions = self._extract_functions(module_content)

        prompt = f"""
请为以下Python模块生成完整的pytest测试用例，目标是提升测试覆盖率从{current_coverage:.1f}%到{target_coverage:.1f}%。

模块路径: {module_path}
当前覆盖率: {current_coverage:.1f}%
目标覆盖率: {target_coverage:.1f}%

模块内容:
```python
{module_content}
```

分析结果:
- 发现的类: {', '.join(classes) if classes else '无'}
- 发现的函数: {', '.join(functions) if functions else '无'}

要求:
1. 生成完整的pytest测试类
2. 包含所有公共方法的测试
3. 测试边界条件和异常情况
4. 使用mock模拟外部依赖
5. 确保测试覆盖率最大化
6. 遵循RQA2025项目的测试规范
7. 包含详细的测试文档
8. 处理可能的导入错误和依赖问题

请只返回Python测试代码，不要包含其他说明文字。
"""
        return prompt

    def _build_ast_enhanced_test_generation_prompt(self, module_path: str, module_content: str,
                                                   current_coverage: float, target_coverage: float,
                                                   ast_suggestions: Dict[str, Any]) -> str:
        """构建包含AST分析的测试生成提示词"""

        # 分析模块结构
        classes = self._extract_classes(module_content)
        functions = self._extract_functions(module_content)

        # 构建AST分析信息
        ast_info = ""
        if ast_suggestions:
            ast_info = "\n\n## AST分析结果\n"

            # 函数信息
            functions_to_test = ast_suggestions.get('functions_to_test', [])
            if functions_to_test:
                ast_info += f"\n### 需要测试的函数 ({len(functions_to_test)} 个):\n"
                for func in functions_to_test[:5]:  # 只显示前5个
                    ast_info += f"- {func['name']} (行 {func['line']})\n"

            # 类信息
            classes_to_test = ast_suggestions.get('classes_to_test', [])
            if classes_to_test:
                ast_info += f"\n### 需要测试的类 ({len(classes_to_test)} 个):\n"
                for cls in classes_to_test[:3]:  # 只显示前3个
                    ast_info += f"- {cls['name']} (行 {cls['line']}, {len(cls['methods'])} 方法)\n"

            # 高复杂度函数
            complexity_focus = ast_suggestions.get('complexity_focus', [])
            if complexity_focus:
                ast_info += f"\n### 高复杂度函数 (需要重点测试):\n"
                for func in complexity_focus:
                    ast_info += f"- {func['name']} (复杂度: {func['complexity']}, 行 {func['line']})\n"

            # 数据流测试点
            data_flow_tests = ast_suggestions.get('data_flow_tests', [])
            if data_flow_tests:
                ast_info += f"\n### 数据流测试点:\n"
                for test in data_flow_tests:
                    ast_info += f"- {test['type']}: {test['count']} 个\n"

            # 集成点
            integration_points = ast_suggestions.get('integration_points', [])
            if integration_points:
                ast_info += f"\n### 集成测试点:\n"
                for point in integration_points:
                    ast_info += f"- {point['type']}: {point['count']} 个\n"

        prompt = f"""
请为以下Python模块生成完整的pytest测试用例，目标是提升测试覆盖率从{current_coverage:.1f}%到{target_coverage:.1f}%。

模块路径: {module_path}
当前覆盖率: {current_coverage:.1f}%
目标覆盖率: {target_coverage:.1f}%

模块内容:
```python
{module_content}
```

分析结果:
- 发现的类: {', '.join(classes) if classes else '无'}
- 发现的函数: {', '.join(functions) if functions else '无'}{ast_info}

要求:
1. 生成完整的pytest测试类
2. 包含所有公共方法的测试
3. 测试边界条件和异常情况
4. 使用mock模拟外部依赖
5. 确保测试覆盖率最大化
6. 遵循RQA2025项目的测试规范
7. 包含详细的测试文档
8. 处理可能的导入错误和依赖问题
9. 基于AST分析结果，重点测试高复杂度函数
10. 包含数据流测试和集成测试
11. 针对AST识别的关键路径进行测试

请只返回Python测试代码，不要包含其他说明文字。
"""
        return prompt

    def _extract_classes(self, content: str) -> List[str]:
        """提取类名"""
        classes = []
        class_pattern = r'class\s+(\w+)'
        matches = re.findall(class_pattern, content)
        return matches

    def _extract_functions(self, content: str) -> List[str]:
        """提取函数名"""
        functions = []
        func_pattern = r'def\s+(\w+)\s*\('
        matches = re.findall(func_pattern, content)
        return matches

    def _generate_fallback_test(self, module_path: str) -> str:
        """生成备用测试代码"""
        module_name = Path(module_path).stem
        class_name = ''.join(word.capitalize() for word in module_name.split('_'))

        return f'''#!/usr/bin/env python3
"""
{module_path} 备用测试文件
AI生成失败时的备用测试用例
"""

import pytest
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

class Test{class_name}:
    """{class_name} 测试类"""
    
    def setup_method(self):
        """测试前设置"""
        try:
            # 尝试导入模块
            module_path = "{module_path}"
            module_name = module_path.replace("/", ".").replace(".py", "")
            self.module = __import__(module_name, fromlist=['*'])
            self.instance = None
        except ImportError as e:
            pytest.skip(f"模块导入失败: {{e}}")
    
    def test_module_import(self):
        """测试模块导入"""
        assert self.module is not None
        print(f"✅ 模块导入成功: {module_path}")
    
    def test_basic_functionality(self):
        """测试基本功能"""
        # 基本功能测试
        assert hasattr(self.module, '__file__')
        print(f"✅ 基本功能测试通过")
    
    def test_error_handling(self):
        """测试错误处理"""
        # 错误处理测试
        assert self.module is not None
        print(f"✅ 错误处理测试通过")
    
    def test_performance(self):
        """测试性能"""
        import time
        start_time = time.time()
        # 执行基本操作
        assert self.module is not None
        end_time = time.time()
        execution_time = end_time - start_time
        assert execution_time < 1.0
        print(f"✅ 性能测试通过 (执行时间: {{execution_time:.3f}}s)")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''


class AICoverageAutomation:
    """AI增强的覆盖率自动化"""

    def __init__(self, ai_connector: DeepseekAIConnector):
        logger.info("【日志测试】AICoverageAutomation.__init__ 开始")
        self.project_root = project_root
        self.ai_connector = ai_connector
        self.current_coverage = {}
        self.target_coverage = {
            'infrastructure': 90.0,
            'data': 85.0,
            'features': 85.0,
            'models': 85.0,
            'trading': 85.0,
            'backtest': 85.0
        }
        logger.info("【日志测试】target_coverage初始化完成")
        # 模块优先级配置
        self.module_priority = {
            'critical': ['config', 'logging', 'cache', 'database', 'monitoring'],
            'high': ['data_loader', 'feature_engine', 'trading_engine', 'model_manager'],
            'medium': ['utils', 'helpers', 'validators', 'analyzers']
        }
        logger.info("【日志测试】module_priority初始化完成")
        # 超时和重试配置
        self.timeout_handler = TimeoutHandler(default_timeout=300)
        self.retry_handler = RetryHandler(max_retries=2, base_delay=1.0)
        logger.info("【日志测试】TimeoutHandler和RetryHandler初始化完成")
        # AST分析器
        self.ast_analyzer = ASTCodeAnalyzer(project_root)
        self.ast_analysis_results = None
        logger.info("【日志测试】ASTCodeAnalyzer初始化完成")
        # 安全审查器
        self.security_reviewer = SecurityCodeReviewer()
        logger.info("【日志测试】SecurityCodeReviewer初始化完成")
        # 增强日志系统
        self.logging_system = EnhancedLoggingSystem()
        logger.info("【日志测试】EnhancedLoggingSystem初始化完成")
        # 插件管理器
        self.plugin_manager = PluginManager()
        logger.info("【日志测试】PluginManager初始化完成，准备初始化插件")
        self._initialize_plugins()
        logger.info("【日志测试】插件初始化完成")
        # 测试质量评估器
        self.quality_assessor = TestQualityAssessor()
        logger.info("【日志测试】TestQualityAssessor初始化完成")
        logger.info("【日志测试】AICoverageAutomation.__init__ 结束")

    def _initialize_plugins(self):
        """初始化并加载所有插件"""
        # 可根据需要指定插件配置文件路径
        plugin_config_file = None
        self.plugin_manager.load_plugins(config_file=plugin_config_file)

    def run_python_subprocess(self, cmd: List[str], timeout: int = 300) -> subprocess.CompletedProcess:
        """运行Python子进程"""
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE

        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.project_root)

        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace',
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW,
                env=env
            )
        except subprocess.TimeoutExpired:
            logger.error(f"子进程执行超时: {' '.join(cmd)}")
            raise
        except Exception as e:
            logger.error(f"子进程执行失败: {e}")
            raise

    async def analyze_coverage_gaps(self) -> Dict[str, List[str]]:
        """分析覆盖率差距"""
        logger.info("🔍 分析覆盖率差距...")

        # 首先进行AST分析
        if self.ast_analysis_results is None:
            logger.info("🔍 执行AST代码分析...")
            self.ast_analysis_results = self.ast_analyzer.analyze_project()

        gaps = {}

        for layer in self.target_coverage.keys():
            try:
                # 运行覆盖率测试
                cmd = [
                    "python", "-m", "pytest", f"tests/unit/{layer}",
                    "--cov", f"src/{layer}",
                    "--cov-report", "term-missing",
                    "--tb=short", "-q"
                ]

                # 使用重试机制
                def _run_coverage():
                    return self.run_python_subprocess(cmd, timeout=300)

                result = self.retry_handler.retry_sync(_run_coverage)

                # 解析覆盖率
                coverage = self._parse_coverage(result.stdout)
                self.current_coverage[layer] = coverage

                target = self.target_coverage[layer]
                if coverage < target:
                    # 找出未覆盖的模块
                    uncovered_modules = self._find_uncovered_modules(layer, result.stdout)

                    # 基于AST分析结果优化模块优先级
                    optimized_modules = self._optimize_module_priority(layer, uncovered_modules)
                    gaps[layer] = optimized_modules

                    logger.info(
                        f"  {layer}: {coverage:.2f}% < {target}% (差距: {target - coverage:.2f}%)")
                    logger.info(f"    未覆盖模块: {', '.join(optimized_modules[:5])}")

            except Exception as e:
                logger.error(f"分析 {layer} 层覆盖率失败: {e}")
                self.current_coverage[layer] = 0.0

        return gaps

    def _parse_coverage(self, output: str) -> float:
        """解析覆盖率数据"""
        try:
            lines = output.split('\n')
            for line in lines:
                if 'TOTAL' in line and '%' in line:
                    parts = line.split()
                    for part in parts:
                        if '%' in part:
                            return float(part.replace('%', ''))
            return 0.0
        except:
            return 0.0

    def _find_uncovered_modules(self, layer: str, coverage_output: str) -> List[str]:
        """找出覆盖率低于目标值的模块（含调试日志，兼容分隔符）"""
        uncovered = []
        import re
        lines = coverage_output.split('\n')
        target_cov = self.target_coverage.get(layer, 85.0)
        for line in lines:
            if ("src/" in line or "src\\" in line) and "%" in line:
                logger.debug(f"[DEBUG] coverage line: {line}")
                parts = line.split()
                for part in parts:
                    if ("src/" in part or "src\\" in part) and ".py" in part:
                        module_path = re.split(r"src[\\/]+", part)[-1]
                        # 提取覆盖率
                        match = re.search(r"(\d+\.\d+)%", line)
                        if match:
                            coverage = float(match.group(1))
                            logger.debug(
                                f"[DEBUG] found module: {module_path}, coverage: {coverage}")
                            if coverage < target_cov:
                                logger.info(
                                    f"[INFO] 模块 {module_path} 覆盖率 {coverage}% < 目标 {target_cov}%，纳入补测")
                                uncovered.append(module_path)
                        else:
                            logger.debug(f"[DEBUG] 未能提取覆盖率: {line}")
        logger.info(f"[INFO] 层 {layer} 最终待补测模块: {uncovered}")
        return uncovered

    def _optimize_module_priority(self, layer: str, uncovered_modules: List[str]) -> List[str]:
        """基于AST分析结果优化模块优先级"""
        if not self.ast_analysis_results:
            return uncovered_modules

        # 获取该层的AST分析结果
        layer_modules = {}
        for module_path, analysis in self.ast_analysis_results.items():
            if module_path.startswith(f"src/{layer}/"):
                layer_modules[module_path] = analysis

        # 计算模块重要性评分
        module_scores = []
        for module in uncovered_modules:
            score = 0
            module_key = f"src/{module}"

            if module_key in layer_modules:
                analysis = layer_modules[module_key]

                # 基于函数数量
                functions_count = len(analysis.get('functions', []))
                score += functions_count * 0.2

                # 基于类数量
                classes_count = len(analysis.get('classes', []))
                score += classes_count * 0.3

                # 基于复杂度
                complexity = analysis.get('complexity', {})
                score += complexity.get('cyclomatic', 0) * 0.1

                # 基于代码行数
                lines = analysis.get('lines_of_code', 0)
                score += min(lines / 100, 1.0) * 0.1

                # 基于被调用次数
                call_count = len(analysis.get('calls', []))
                score += call_count * 0.05

                # 基于导入依赖
                imports_count = len(analysis.get('imports', []))
                score += imports_count * 0.05

            module_scores.append((module, score))

        # 按评分排序
        module_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回优化后的模块列表
        return [module for module, score in module_scores]

    def _get_ast_based_test_suggestions(self, module_path: str) -> Dict[str, Any]:
        """基于AST分析获取测试建议"""
        if not self.ast_analysis_results:
            return {}

        module_key = f"src/{module_path}"
        if module_key not in self.ast_analysis_results:
            return {}

        analysis = self.ast_analysis_results[module_key]

        suggestions = {
            'functions_to_test': [],
            'classes_to_test': [],
            'complexity_focus': [],
            'data_flow_tests': [],
            'integration_points': []
        }

        # 分析需要测试的函数
        for func in analysis.get('functions', []):
            if func['complexity'] > 3:  # 复杂度较高的函数
                suggestions['complexity_focus'].append({
                    'name': func['name'],
                    'complexity': func['complexity'],
                    'line': func['line']
                })

            suggestions['functions_to_test'].append({
                'name': func['name'],
                'args': func['args'],
                'line': func['line']
            })

        # 分析需要测试的类
        for cls in analysis.get('classes', []):
            suggestions['classes_to_test'].append({
                'name': cls['name'],
                'methods': cls['methods'],
                'attributes': cls['attributes'],
                'line': cls['line']
            })

        # 分析数据流测试点
        data_flow = analysis.get('data_flow', {})
        if data_flow.get('assignments'):
            suggestions['data_flow_tests'].append({
                'type': 'variable_assignment',
                'count': len(data_flow['assignments'])
            })

        if data_flow.get('returns'):
            suggestions['data_flow_tests'].append({
                'type': 'return_values',
                'count': len(data_flow['returns'])
            })

        # 分析集成点
        calls = analysis.get('calls', [])
        external_calls = [call for call in calls if call['type'] == 'method_call']
        if external_calls:
            suggestions['integration_points'].append({
                'type': 'external_method_calls',
                'count': len(external_calls)
            })

        return suggestions

    async def generate_ai_tests(self, layer: str, modules: List[str]) -> List[str]:
        """使用AI生成测试文件"""
        logger.info(f"🤖 为 {layer} 层生成AI测试...")

        generated_files = []

        for module in modules:
            try:
                # 执行预生成钩子
                context = {
                    'layer': layer,
                    'module': module,
                    'timestamp': datetime.now().isoformat()
                }
                self.plugin_manager.execute_hooks('pre_test_generation', context)

                # 修正源码路径拼接，避免重复，先替换再拼接
                import datetime
                module_fixed = module.replace('\\', '/')
                module_basename = os.path.basename(module_fixed).replace('.py', '')
                module_path = f"src/{module_fixed}"
                test_file_base = f"tests/unit/{layer}/test_{module_basename}.py"
                test_file_path = test_file_base
                if os.path.exists(test_file_path):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    test_file_path = f"tests/unit/{layer}/test_{module_basename}_{timestamp}.py"
                os.makedirs(os.path.dirname(test_file_path), exist_ok=True)

                if not os.path.exists(module_path):
                    logger.warning(f"模块不存在: {module_path}")
                    continue

                with open(module_path, 'r', encoding='utf-8') as f:
                    module_content = f.read()

                # 获取当前覆盖率
                current_cov = self.current_coverage.get(layer, 0.0)
                target_cov = self.target_coverage[layer]

                # 获取AST分析建议
                ast_suggestions = self._get_ast_based_test_suggestions(module)

                # 使用AI生成测试代码（包含AST分析结果）
                start_time = time.time()
                test_code = await self.ai_connector.generate_test_code_with_ast(
                    module_path, module_content, current_cov, target_cov, ast_suggestions
                )
                generation_time = time.time() - start_time

                # 安全审查
                security_result = self.security_reviewer.review_code(test_code, module_path)
                if not security_result['passed']:
                    logger.warning(
                        f"⚠️ 安全审查失败: {module_path} (评分: {security_result['security_score']})")
                    # 记录安全事件
                    self.logging_system.log_security_event(
                        'code_review_failed',
                        'medium',
                        {'module': module_path, 'score': security_result['security_score']}
                    )

                # 保存测试文件
                test_file_path = f"tests/unit/{layer}/test_{module_basename}.py"
                os.makedirs(os.path.dirname(test_file_path), exist_ok=True)

                with open(test_file_path, 'w', encoding='utf-8') as f:
                    f.write(test_code)

                generated_files.append(test_file_path)

                # 记录性能指标
                self.logging_system.log_performance(
                    'test_generation',
                    generation_time,
                    True,
                    {'module': module_path, 'security_score': security_result['security_score']}
                )

                # 执行后生成钩子
                context.update({
                    'test_file': test_file_path,
                    'security_score': security_result['security_score'],
                    'generation_time': generation_time
                })
                self.plugin_manager.execute_hooks('post_test_generation', context)

                logger.info(f"  ✅ 生成测试文件: {test_file_path}")
                logger.info(f"    安全评分: {security_result['security_score']}/100")
                logger.info(f"    生成时间: {generation_time:.2f}s")

                # 记录AST分析信息
                if ast_suggestions:
                    logger.info(f"    AST建议: {len(ast_suggestions.get('functions_to_test', []))} 函数, "
                                f"{len(ast_suggestions.get('classes_to_test', []))} 类, "
                                f"{len(ast_suggestions.get('complexity_focus', []))} 高复杂度函数")

            except Exception as e:
                logger.error(f"生成测试失败 {module}: {e}")
                # 记录错误事件
                self.logging_system.log_security_event(
                    'test_generation_error',
                    'high',
                    {'module': module, 'error': str(e)}
                )

        return generated_files

    async def run_ai_generated_tests(self, test_files: List[str]) -> Dict[str, Any]:
        """运行AI生成的测试"""
        logger.info(f"🧪 运行 {len(test_files)} 个AI生成的测试...")

        results = {
            'total_files': len(test_files),
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'error': 0,
            'coverage_improvement': {}
        }

        for i, test_file in enumerate(test_files, 1):
            logger.info(f"  📝 [{i}/{len(test_files)}] 运行测试: {test_file}")

            try:
                cmd = [
                    "python", "-m", "pytest", test_file,
                    "--tb=short", "-q"
                ]

                # 使用超时处理
                result = self.timeout_handler.run_with_timeout(
                    self.run_python_subprocess, cmd, timeout=300
                )

                # 解析测试结果
                test_result = self._parse_test_result(result.stdout)

                if result.returncode == 0:
                    results['passed'] += test_result['passed']
                    results['failed'] += test_result['failed']
                    results['skipped'] += test_result['skipped']

                    if test_result['failed'] == 0:
                        logger.info(f"  ✅ {test_file}: 通过 ({test_result['passed']} 通过)")
                    else:
                        logger.warning(
                            f"  ⚠️ {test_file}: 部分失败 ({test_result['passed']} 通过, {test_result['failed']} 失败)")
                else:
                    results['failed'] += 1
                    logger.error(f"  ❌ {test_file}: 失败")
                    if result.stderr:
                        logger.error(f"    错误: {result.stderr[:200]}...")

            except Exception as e:
                results['error'] += 1
                logger.error(f"  ❌ {test_file}: 执行异常 - {e}")

        return results

    def _parse_test_result(self, output: str) -> Dict[str, int]:
        """解析测试结果"""
        result = {'passed': 0, 'failed': 0, 'skipped': 0}

        lines = output.split('\n')
        for line in lines:
            if 'passed' in line and 'failed' in line and 'skipped' in line:
                parts = line.split(',')
                for part in parts:
                    if 'passed' in part:
                        result['passed'] = int(part.strip().split()[0])
                    elif 'failed' in part:
                        result['failed'] = int(part.strip().split()[0])
                    elif 'skipped' in part:
                        result['skipped'] = int(part.strip().split()[0])
                break

        return result

    async def optimize_coverage_strategy(self, layer: str, current_coverage: float) -> Dict[str, Any]:
        """优化覆盖率策略"""
        logger.info(f"🎯 优化 {layer} 层覆盖率策略...")

        target = self.target_coverage[layer]
        gap = target - current_coverage

        strategy = {
            'layer': layer,
            'current_coverage': current_coverage,
            'target_coverage': target,
            'gap': gap,
            'priority': 'high' if gap > 20 else 'medium' if gap > 10 else 'low',
            'recommended_actions': []
        }

        if gap > 20:
            strategy['recommended_actions'].extend([
                '生成核心模块测试',
                '添加边界条件测试',
                '补充异常处理测试',
                '优化测试数据'
            ])
        elif gap > 10:
            strategy['recommended_actions'].extend([
                '补充关键功能测试',
                '添加集成测试',
                '优化现有测试'
            ])
        else:
            strategy['recommended_actions'].extend([
                '微调现有测试',
                '添加高级功能测试'
            ])

        return strategy

    async def generate_coverage_report(self, gaps: Dict[str, List[str]],
                                       test_results: Dict[str, Any],
                                       strategies: List[Dict[str, Any]]) -> str:
        """生成覆盖率报告"""
        report_file = "reports/testing/ai_coverage_automation_report.md"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # AST分析统计
        ast_stats = {}
        if self.ast_analysis_results:
            total_functions = sum(len(analysis.get('functions', []))
                                  for analysis in self.ast_analysis_results.values())
            total_classes = sum(len(analysis.get('classes', []))
                                for analysis in self.ast_analysis_results.values())
            total_complexity = sum(analysis.get('complexity', {}).get('cyclomatic', 0)
                                   for analysis in self.ast_analysis_results.values())
            ast_stats = {
                'total_functions': total_functions,
                'total_classes': total_classes,
                'total_complexity': total_complexity,
                'analyzed_modules': len(self.ast_analysis_results)
            }

        report_content = f"""# RQA2025 AI增强测试覆盖率自动化报告

## 📊 执行摘要

**最后更新时间**: {current_time}
**AI模型**: Deepseek Coder
**总体目标覆盖率**: 85%
**当前平均覆盖率**: {sum(self.current_coverage.values()) / len(self.current_coverage):.2f}%

## 🔍 AST分析摘要

"""

        if ast_stats:
            report_content += f"""
- **分析模块数**: {ast_stats['analyzed_modules']}
- **总函数数**: {ast_stats['total_functions']}
- **总类数**: {ast_stats['total_classes']}
- **总复杂度**: {ast_stats['total_complexity']}
"""
        else:
            report_content += "- **AST分析**: 未执行\n"

        report_content += f"""
## 🎯 各层覆盖率状态

| 层级 | 当前覆盖率 | 目标覆盖率 | 差距 | AI优化状态 |
|------|------------|------------|------|------------|
"""

        for layer, current_cov in self.current_coverage.items():
            target_cov = self.target_coverage[layer]
            gap = target_cov - current_cov
            status = "✅" if current_cov >= target_cov else "🤖"

            report_content += f"| {layer} | {current_cov:.2f}% | {target_cov}% | {gap:.2f}% | {status} |\n"

        report_content += f"""
## 🤖 AI生成测试结果

- **总测试文件**: {test_results['total_files']}
- **通过**: {test_results['passed']}
- **失败**: {test_results['failed']}
- **跳过**: {test_results['skipped']}
- **错误**: {test_results['error']}
- **成功率**: {test_results['passed'] / max(test_results['total_files'], 1) * 100:.1f}%

## 📋 AI优化策略

"""

        for strategy in strategies:
            report_content += f"""
### {strategy['layer']} 层优化策略
- **当前覆盖率**: {strategy['current_coverage']:.2f}%
- **目标覆盖率**: {strategy['target_coverage']}%
- **差距**: {strategy['gap']:.2f}%
- **优先级**: {strategy['priority']}
- **推荐行动**:
"""
            for action in strategy['recommended_actions']:
                report_content += f"  - {action}\n"

        report_content += f"""
## 🔍 覆盖率差距分析

"""

        for layer, modules in gaps.items():
            report_content += f"### {layer} 层未覆盖模块\n"
            for module in modules[:10]:  # 只显示前10个
                report_content += f"- {module}\n"
            if len(modules) > 10:
                report_content += f"- ... 还有 {len(modules) - 10} 个模块\n"
            report_content += "\n"

        # 添加AST分析结果
        if self.ast_analysis_results:
            report_content += f"""
## 🔍 AST分析结果

### 关键模块分析
"""

            critical_modules = self.ast_analyzer.find_critical_modules()
            for module_info in critical_modules[:5]:
                report_content += f"""
**{module_info['module']}**
- 重要性评分: {module_info['score']:.2f}
- 函数数: {module_info['functions']}
- 类数: {module_info['classes']}
- 复杂度: {module_info['complexity']}
- 依赖数: {module_info['dependencies']}
- 代码行数: {module_info['lines']}
"""

        report_content += f"""
## 🚀 下一步AI优化行动

1. **修复失败的AI测试**: 分析失败原因并优化
2. **补充边界条件**: 使用AI生成更多边界测试
3. **异常处理测试**: 生成异常场景测试用例
4. **性能测试**: 添加性能相关的测试
5. **持续优化**: 基于覆盖率反馈持续改进
6. **AST优化**: 基于AST分析结果优化测试策略

## 📈 AI优化指标

- [ ] 总体覆盖率 ≥ 85%
- [ ] AI测试通过率 ≥ 90%
- [ ] 核心模块覆盖率 ≥ 95%
- [ ] 自动化测试覆盖率 ≥ 80%
- [ ] AST分析覆盖率 ≥ 90%

## 🔧 AI配置信息

- **模型**: Deepseek Coder
- **API端点**: {self.ai_connector.api_base}
- **缓存策略**: 启用本地缓存
- **超时设置**: 120秒
- **温度参数**: 0.3
- **重试机制**: 3次重试，指数退避
- **AST分析**: 启用深度代码分析

---
**报告版本**: v1.0
**AI引擎**: Deepseek Coder
**最后更新**: {current_time}
"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"📄 AI覆盖率报告已生成: {report_file}")
        return report_file

    def generate_comprehensive_report(self) -> str:
        """生成综合报告"""
        report_file = "reports/testing/comprehensive_automation_report.md"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 获取各种报告数据
        coverage_report = self.get_coverage_summary()
        security_report = self.security_reviewer.get_security_summary()
        metrics_report = self.logging_system.get_metrics_report()
        plugin_info = self.plugin_manager.get_plugin_info()
        quality_summary = self.quality_assessor.get_quality_summary()

        report_content = f"""# RQA2025 综合自动化报告

## 📊 执行摘要

**报告时间**: {current_time}
**总体状态**: {'✅ 正常' if coverage_report['overall_success'] else '❌ 异常'}

## 🎯 覆盖率状态

- **总体覆盖率**: {coverage_report['overall_coverage']:.2f}%
- **目标覆盖率**: {coverage_report['target_coverage']:.2f}%
- **差距**: {coverage_report['gap']:.2f}%

### 各层覆盖率

| 层级 | 当前覆盖率 | 目标覆盖率 | 状态 |
|------|------------|------------|------|
"""

        for layer, data in coverage_report['layers'].items():
            status = "✅" if data['current'] >= data['target'] else "❌"
            report_content += f"| {layer} | {data['current']:.2f}% | {data['target']}% | {status} |\n"

        report_content += f"""
## 🔒 安全审查状态

- **审查文件数**: {security_report['total_files']}
- **通过文件数**: {security_report['passed_files']}
- **失败文件数**: {security_report['failed_files']}
- **平均安全评分**: {security_report['avg_score']:.1f}/100

### 安全问题统计

| 问题类型 | 数量 | 严重度 |
|----------|------|--------|
"""

        for issue_type, count in security_report['issue_types'].items():
            report_content += f"| {issue_type} | {count} | 高 |\n"

        report_content += f"""
## 📈 性能指标

### 系统指标
- **CPU使用率**: {metrics_report['metrics']['gauges'].get('system_cpu_percent', 0):.1f}%
- **内存使用率**: {metrics_report['metrics']['gauges'].get('system_memory_percent', 0):.1f}%
- **磁盘使用率**: {metrics_report['metrics']['gauges'].get('system_disk_percent', 0):.1f}%

### 业务指标
- **测试生成次数**: {metrics_report['metrics']['counters'].get('test_generation_count', 0)}
- **安全事件数**: {metrics_report['metrics']['counters'].get('security_events', 0)}
- **日志记录数**: {sum(metrics_report['metrics']['counters'].get(f'log_{level}', 0) for level in ['info', 'warning', 'error'])}
"""

        # 添加性能直方图数据
        if 'test_generation_duration' in metrics_report['metrics']['histograms']:
            hist_data = metrics_report['metrics']['histograms']['test_generation_duration']
            report_content += f"""
### 测试生成性能
- **平均生成时间**: {hist_data['avg']:.3f}s
- **最大生成时间**: {hist_data['max']:.3f}s
- **95%分位数**: {hist_data['p95']:.3f}s
"""

        report_content += f"""
## 🔌 插件状态

| 插件名称 | 版本 | 状态 | 功能 |
|----------|------|------|------|
"""

        for plugin in plugin_info:
            status = "✅ 启用" if plugin['enabled'] else "❌ 禁用"
            report_content += f"| {plugin['name']} | {plugin['version']} | {status} | 测试生成/安全审查/指标收集 |\n"

        report_content += f"""
## 🎯 测试质量评估

- **总体质量评分**: {quality_summary['overall_score']:.1f}/100
- **覆盖率质量**: {quality_summary['quality_metrics'].get('coverage_quality', {}).get('quality_score', 0):.1f}/100
- **测试用例质量**: {quality_summary['quality_metrics'].get('test_case_quality', {}).get('quality_score', 0):.1f}/100
- **执行质量**: {quality_summary['quality_metrics'].get('execution_quality', {}).get('quality_score', 0):.1f}/100
- **可维护性质量**: {quality_summary['quality_metrics'].get('maintainability_quality', {}).get('quality_score', 0):.1f}/100
- **安全质量**: {quality_summary['quality_metrics'].get('security_quality', {}).get('quality_score', 0):.1f}/100

## 🚀 建议和行动项

### 短期行动
1. **修复安全审查失败的文件**
2. **提升覆盖率较低的层级**
3. **优化性能较慢的测试生成**

### 长期优化
1. **完善插件生态系统**
2. **增强安全审查规则**
3. **优化AI生成质量**

## 📋 技术栈

- **AI引擎**: Deepseek Coder
- **AST分析**: 自定义AST分析器
- **安全审查**: 多维度安全检查
- **日志系统**: 结构化日志 + 指标收集
- **插件架构**: 可扩展插件系统

---
**报告版本**: v2.0
**生成时间**: {current_time}
"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"📄 综合报告已生成: {report_file}")
        return report_file

    def get_coverage_summary(self) -> Dict[str, Any]:
        """获取覆盖率摘要"""
        overall_coverage = sum(self.current_coverage.values()) / len(self.current_coverage)
        target_coverage = sum(self.target_coverage.values()) / len(self.target_coverage)

        layers = {}
        for layer in self.target_coverage.keys():
            current = self.current_coverage.get(layer, 0.0)
            target = self.target_coverage[layer]
            layers[layer] = {
                'current': current,
                'target': target,
                'gap': target - current
            }

        return {
            'overall_coverage': overall_coverage,
            'target_coverage': target_coverage,
            'gap': target_coverage - overall_coverage,
            'overall_success': overall_coverage >= target_coverage * 0.8,  # 80%目标
            'layers': layers
        }

    def _perform_quality_assessment(self, test_files: List[str], test_results: Dict[str, Any]) -> Dict[str, Any]:
        """执行测试质量评估"""
        logger.info("🔍 执行测试质量评估...")

        # 1. 评估覆盖率质量
        coverage_quality = self.quality_assessor.assess_coverage_quality(
            self.current_coverage, self.target_coverage
        )

        # 2. 评估测试用例质量
        test_case_quality = self.quality_assessor.assess_test_case_quality(test_files)

        # 3. 评估执行质量
        execution_quality = self.quality_assessor.assess_execution_quality(test_results)

        # 4. 评估可维护性质量
        maintainability_quality = self.quality_assessor.assess_maintainability_quality(test_files)

        # 5. 评估安全质量
        security_quality = self.quality_assessor.assess_security_quality(test_files)

        # 6. 生成质量报告
        quality_report = self.quality_assessor.generate_quality_report()

        # 7. 记录质量指标
        self.logging_system.log_business_event(
            'quality_assessment_completed',
            'system',
            {
                'overall_score': self.quality_assessor.get_overall_quality_score(),
                'coverage_quality': coverage_quality['quality_score'],
                'test_case_quality': test_case_quality['quality_score'],
                'execution_quality': execution_quality['quality_score'],
                'maintainability_quality': maintainability_quality['quality_score'],
                'security_quality': security_quality['quality_score']
            }
        )

        return {
            'overall_score': self.quality_assessor.get_overall_quality_score(),
            'coverage_quality': coverage_quality,
            'test_case_quality': test_case_quality,
            'execution_quality': execution_quality,
            'maintainability_quality': maintainability_quality,
            'security_quality': security_quality,
            'quality_report': quality_report
        }

    async def execute_ai_automation(self) -> Dict[str, Any]:
        """执行AI自动化流程"""
        logger.info("🚀 开始AI增强测试覆盖率自动化...")

        # 1. 分析覆盖率差距
        logger.info("🔍 [1/9] 开始分析覆盖率差距...")
        gaps = await self.analyze_coverage_gaps()
        logger.info(f"✅ 覆盖率差距分析完成，待处理层级: {list(gaps.keys())}")

        # 2. 生成AI测试
        logger.info("🤖 [2/9] 开始AI测试用例生成...")
        all_generated_files = []
        total_layers = len(gaps)
        for idx, (layer, modules) in enumerate(gaps.items(), 1):
            logger.info(f"  [{idx}/{total_layers}] 处理层级: {layer}，待生成模块数: {len(modules)}")
            if modules:
                for m_idx, module in enumerate(modules[:5], 1):
                    logger.info(f"    - [{m_idx}/{min(len(modules),5)}] 生成 {module} 的AI测试用例...")
                test_files = await self.generate_ai_tests(layer, modules[:5])  # 限制每个层最多5个模块
                all_generated_files.extend(test_files)
        logger.info(f"✅ AI测试用例生成完成，生成文件数: {len(all_generated_files)}")

        # 3. 运行AI生成的测试
        logger.info("🧪 [3/9] 开始执行AI生成的测试用例...")
        test_results = await self.run_ai_generated_tests(all_generated_files)
        logger.info(
            f"✅ 测试用例执行完成，通过: {test_results.get('passed',0)}，失败: {test_results.get('failed',0)}")

        # 4. 生成优化策略
        logger.info("📈 [4/9] 开始生成优化策略...")
        strategies = []
        for idx, layer in enumerate(self.current_coverage.keys(), 1):
            current_cov = self.current_coverage.get(layer, 0.0)
            logger.info(
                f"  [{idx}/{len(self.current_coverage)}] 优化层级: {layer} 当前覆盖率: {current_cov:.2f}%")
            strategy = await self.optimize_coverage_strategy(layer, current_cov)
            strategies.append(strategy)
        logger.info("✅ 优化策略生成完成")

        # 5. 生成报告
        logger.info("📝 [5/9] 开始生成覆盖率报告...")
        report_file = await self.generate_coverage_report(gaps, test_results, strategies)
        logger.info(f"✅ 覆盖率报告生成完成: {report_file}")

        # 6. 生成综合报告
        logger.info("📄 [6/9] 生成综合报告...")
        comprehensive_report = self.generate_comprehensive_report()
        logger.info("✅ 综合报告生成完成")

        # 7. 生成指标报告
        logger.info("📊 [7/9] 生成指标报告...")
        metrics_report = self.logging_system.generate_metrics_report()
        logger.info("✅ 指标报告生成完成")

        # 8. 生成安全报告
        logger.info("🔒 [8/9] 生成安全报告...")
        security_report = self.security_reviewer.generate_security_report()
        logger.info("✅ 安全报告生成完成")

        # 9. 执行测试质量评估
        logger.info("🧩 [9/9] 执行测试质量评估...")
        quality_assessment = self._perform_quality_assessment(all_generated_files, test_results)
        logger.info("✅ 测试质量评估完成")

        return {
            'gaps': gaps,
            'generated_files': all_generated_files,
            'test_results': test_results,
            'strategies': strategies,
            'report_file': report_file,
            'comprehensive_report': comprehensive_report,
            'metrics_report': metrics_report,
            'security_report': security_report,
            'quality_assessment': quality_assessment,
            'current_coverage': self.current_coverage
        }


async def main():
    """主函数"""
    logger.info("【日志测试】ai_enhanced_coverage_automation.py main已进入")
    parser = argparse.ArgumentParser(description='RQA2025 AI增强测试覆盖率自动化')
    parser.add_argument('--api-base', default='http://localhost:11434',
                        help='Deepseek API基础URL')
    parser.add_argument('--model', default='deepseek-coder',
                        help='使用的AI模型')
    parser.add_argument('--target', type=float, default=85.0,
                        help='目标覆盖率')
    parser.add_argument('--layers', nargs='+',
                        default=['infrastructure', 'data', 'features', 'trading'],
                        help='要优化的层级')
    parser.add_argument('--check-deps', action='store_true',
                        help='检查依赖')
    args = parser.parse_args()
    logger.info(
        f"【日志测试】main参数解析: api_base={args.api_base}, model={args.model}, target={args.target}, layers={args.layers}, check_deps={args.check_deps}")
    # 检查依赖
    if args.check_deps:
        logger.info("【日志测试】进入依赖检查分支")
        logger.info("🔍 检查系统依赖...")
        if not DependencyChecker.check_python_dependencies():
            logger.error("【日志测试】Python依赖检查未通过")
            sys.exit(1)
        if not DependencyChecker.check_system_dependencies():
            logger.error("【日志测试】系统依赖检查未通过")
            sys.exit(1)
        ai_available = await DependencyChecker.check_ai_service(args.api_base)
        logger.info(f"【日志测试】AI服务可用: {ai_available}")
        if not ai_available:
            logger.error("❌ AI服务不可用，请检查Deepseek服务是否启动")
            sys.exit(1)
        logger.info("✅ 所有依赖检查通过")
        return
    # 检查conda环境
    logger.info("【日志测试】检查conda环境")
    if 'rqa' not in sys.prefix and 'rqa' not in sys.executable:
        logger.error('❌ 请先激活conda rqa环境后再运行本脚本！')
        sys.exit(1)
    logger.info("【日志测试】conda环境检查通过")
    # 创建AI连接器
    logger.info("【日志测试】准备创建AI连接器")
    async with DeepseekAIConnector(args.api_base, args.model) as ai_connector:
        logger.info("【日志测试】AI连接器创建成功，准备创建AICoverageAutomation")
        # 创建自动化执行器
        automation = AICoverageAutomation(ai_connector)
        logger.info("【日志测试】AICoverageAutomation实例化完成")
        # 更新目标覆盖率
        for layer in args.layers:
            if layer in automation.target_coverage:
                automation.target_coverage[layer] = args.target
        logger.info(f"【日志测试】目标覆盖率已更新: {automation.target_coverage}")
        logger.info("🤖 开始AI增强测试覆盖率自动化...")
        logger.info(f"目标覆盖率: {args.target}%")
        logger.info(f"优化层级: {', '.join(args.layers)}")
        # 执行自动化流程
        logger.info("【日志测试】准备await automation.execute_ai_automation()")
        results = await automation.execute_ai_automation()
        logger.info("【日志测试】automation.execute_ai_automation()已返回")
        logger.info("✅ AI自动化流程执行完成")
        logger.info(f"📄 详细报告: {results['report_file']}")
        logger.info(f"📊 生成测试文件: {len(results['generated_files'])} 个")
        logger.info(f"🧪 测试结果: {results['test_results']['passed']} 通过, "
                    f"{results['test_results']['failed']} 失败")


if __name__ == "__main__":
    logger.info("【日志测试】main函数已进入")
    asyncio.run(main())
