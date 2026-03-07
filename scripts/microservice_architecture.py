#!/usr/bin/env python3
"""
微服务架构改造工具

考虑基础设施层的微服务化改造
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class MicroserviceArchitecture:
    """微服务架构工具"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.infrastructure_dir = self.project_root / "src" / "infrastructure"
        self.microservices_dir = self.project_root / "microservices"

        # 微服务架构配置
        self.config = {
            "service_mapping": {
                "config": "config-service",
                "cache": "cache-service",
                "logging": "logging-service",
                "security": "security-service",
                "error": "error-handling-service",
                "resource": "resource-management-service",
                "health": "health-monitoring-service",
                "utils": "utility-service"
            },
            "service_dependencies": {
                "config-service": [],
                "cache-service": ["config-service"],
                "logging-service": ["config-service"],
                "security-service": ["config-service", "logging-service"],
                "error-handling-service": ["config-service", "logging-service"],
                "resource-management-service": ["config-service", "logging-service"],
                "health-monitoring-service": ["config-service", "logging-service"],
                "utility-service": []
            },
            "service_ports": {
                "config-service": 8001,
                "cache-service": 8002,
                "logging-service": 8003,
                "security-service": 8004,
                "error-handling-service": 8005,
                "resource-management-service": 8006,
                "health-monitoring-service": 8007,
                "utility-service": 8008
            },
            "enable_api_gateway": True,
            "enable_service_discovery": True,
            "enable_circuit_breaker": True
        }

    def analyze_microservice_feasibility(self) -> Dict[str, Any]:
        """分析微服务化可行性"""
        print("🔍 分析微服务化可行性...")

        analysis = {
            "feasibility_score": 0,
            "service_candidates": {},
            "dependencies_analysis": {},
            "complexity_assessment": {},
            "recommendations": []
        }

        # 分析每个分类作为微服务的可行性
        for category, service_name in self.config["service_mapping"].items():
            category_dir = self.infrastructure_dir / category
            if category_dir.exists():
                service_analysis = self._analyze_service_candidate(
                    category_dir, category, service_name)
                analysis["service_candidates"][category] = service_analysis

        # 分析服务间依赖关系
        analysis["dependencies_analysis"] = self._analyze_service_dependencies()

        # 评估复杂度
        analysis["complexity_assessment"] = self._assess_complexity()

        # 计算可行性分数
        analysis["feasibility_score"] = self._calculate_feasibility_score(analysis)

        # 生成建议
        analysis["recommendations"] = self._generate_recommendations(analysis)

        print(f"✅ 微服务化可行性分析完成，得分: {analysis['feasibility_score']}/100")
        return analysis

    def _analyze_service_candidate(self, category_dir: Path, category: str, service_name: str) -> Dict[str, Any]:
        """分析单个服务候选者"""
        analysis = {
            "category": category,
            "service_name": service_name,
            "file_count": 0,
            "interface_count": 0,
            "dependency_count": 0,
            "complexity_score": 0,
            "cohesion_score": 0,
            "coupling_score": 0,
            "microservice_feasibility": 0
        }

        # 统计文件数量
        py_files = list(category_dir.glob("*.py"))
        analysis["file_count"] = len(py_files)

        # 统计接口数量
        interface_count = 0
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                interface_count += len(re.findall(r'class I[A-Z]\w*Component', content))
            except:
                pass
        analysis["interface_count"] = interface_count

        # 分析依赖关系
        analysis["dependency_count"] = self._count_dependencies(category)

        # 计算复杂度分数
        analysis["complexity_score"] = self._calculate_service_complexity(analysis)

        # 计算内聚性分数
        analysis["cohesion_score"] = self._calculate_service_cohesion(category_dir)

        # 计算耦合性分数
        analysis["coupling_score"] = self._calculate_service_coupling(category)

        # 计算微服务化可行性
        analysis["microservice_feasibility"] = self._calculate_microservice_feasibility(analysis)

        return analysis

    def _count_dependencies(self, category: str) -> int:
        """统计依赖数量"""
        dependencies = self.config["service_dependencies"].get(f"{category}-service", [])
        return len(dependencies)

    def _calculate_service_complexity(self, analysis: Dict[str, Any]) -> float:
        """计算服务复杂度"""
        # 基于文件数量、接口数量、依赖数量计算复杂度
        complexity = (analysis["file_count"] * 0.3 +
                      analysis["interface_count"] * 0.4 +
                      analysis["dependency_count"] * 0.3)
        return min(100, complexity)

    def _calculate_service_cohesion(self, category_dir: Path) -> float:
        """计算服务内聚性"""
        try:
            # 分析文件间的关系和共同主题
            py_files = list(category_dir.glob("*.py"))
            if not py_files:
                return 0

            # 提取所有文件的关键词
            all_keywords = set()
            file_keywords = []

            for py_file in py_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()

                    # 提取关键词
                    keywords = set(re.findall(r'\b[a-zA-Z]{3,}\b', content))
                    file_keywords.append(keywords)
                    all_keywords.update(keywords)
                except:
                    pass

            if not file_keywords:
                return 0

            # 计算内聚性：文件间共同关键词的比例
            total_keywords = len(all_keywords)
            if total_keywords == 0:
                return 0

            shared_keywords = set()
            for keywords in file_keywords:
                shared_keywords.update(keywords)

            cohesion = (len(shared_keywords) / total_keywords) * 100
            return cohesion

        except Exception as e:
            print(f"❌ 计算内聚性失败: {e}")
            return 0

    def _calculate_service_coupling(self, category: str) -> float:
        """计算服务耦合性"""
        # 基于依赖数量和依赖深度计算耦合性
        dependencies = self.config["service_dependencies"].get(f"{category}-service", [])
        coupling = len(dependencies) * 20  # 每个依赖增加20%的耦合度
        return min(100, coupling)

    def _calculate_microservice_feasibility(self, analysis: Dict[str, Any]) -> float:
        """计算微服务化可行性"""
        # 基于多个因素计算可行性
        cohesion_weight = 0.4
        coupling_weight = 0.3
        complexity_weight = 0.2
        interface_weight = 0.1

        # 理想的内聚性高，耦合性低，复杂度适中，接口数量合理
        cohesion_score = analysis["cohesion_score"]
        coupling_score = 100 - analysis["coupling_score"]  # 反转，因为低耦合更好
        complexity_score = 100 - abs(50 - analysis["complexity_score"])  # 复杂度适中最好
        interface_score = min(100, analysis["interface_count"] * 10)  # 接口数量适中

        feasibility = (cohesion_score * cohesion_weight +
                       coupling_score * coupling_weight +
                       complexity_score * complexity_weight +
                       interface_score * interface_weight)

        return feasibility

    def _analyze_service_dependencies(self) -> Dict[str, Any]:
        """分析服务间依赖关系"""
        dependencies_analysis = {
            "dependency_graph": self.config["service_dependencies"],
            "circular_dependencies": [],
            "shared_dependencies": [],
            "isolated_services": []
        }

        # 检查循环依赖
        for service, deps in self.config["service_dependencies"].items():
            for dep in deps:
                if dep in self.config["service_dependencies"] and service in self.config["service_dependencies"][dep]:
                    dependencies_analysis["circular_dependencies"].append((service, dep))

        # 查找共享依赖
        all_dependencies = []
        for deps in self.config["service_dependencies"].values():
            all_dependencies.extend(deps)

        dependency_counts = {}
        for dep in all_dependencies:
            dependency_counts[dep] = dependency_counts.get(dep, 0) + 1

        dependencies_analysis["shared_dependencies"] = [
            dep for dep, count in dependency_counts.items() if count > 1
        ]

        # 查找孤立服务
        dependencies_analysis["isolated_services"] = [
            service for service, deps in self.config["service_dependencies"].items()
            if not deps
        ]

        return dependencies_analysis

    def _assess_complexity(self) -> Dict[str, Any]:
        """评估微服务化复杂度"""
        return {
            "communication_complexity": "medium",  # 服务间通信复杂度
            "deployment_complexity": "high",       # 部署复杂度
            "monitoring_complexity": "high",       # 监控复杂度
            "testing_complexity": "high",          # 测试复杂度
            "development_complexity": "medium"     # 开发复杂度
        }

    def _calculate_feasibility_score(self, analysis: Dict[str, Any]) -> float:
        """计算可行性总分"""
        if not analysis["service_candidates"]:
            return 0

        # 计算平均可行性分数
        feasibility_scores = [
            candidate["microservice_feasibility"]
            for candidate in analysis["service_candidates"].values()
        ]

        avg_feasibility = sum(feasibility_scores) / len(feasibility_scores)

        # 根据依赖关系调整分数
        dependency_penalty = len(analysis["dependencies_analysis"]["circular_dependencies"]) * 10
        shared_bonus = len(analysis["dependencies_analysis"]["shared_dependencies"]) * 5

        final_score = avg_feasibility - dependency_penalty + shared_bonus
        return max(0, min(100, final_score))

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成建议"""
        recommendations = []

        # 根据可行性分数生成建议
        if analysis["feasibility_score"] > 80:
            recommendations.append({
                "priority": "high",
                "type": "implementation",
                "description": "可以直接开始微服务化改造",
                "implementation_plan": [
                    "1. 创建服务模板",
                    "2. 按分类拆分服务",
                    "3. 实现服务间通信",
                    "4. 添加服务发现",
                    "5. 配置API网关"
                ]
            })
        elif analysis["feasibility_score"] > 60:
            recommendations.append({
                "priority": "medium",
                "type": "optimization",
                "description": "需要先优化架构再进行微服务化",
                "implementation_plan": [
                    "1. 优化依赖关系",
                    "2. 提高服务内聚性",
                    "3. 降低服务耦合性",
                    "4. 完善接口设计",
                    "5. 然后进行微服务化"
                ]
            })
        else:
            recommendations.append({
                "priority": "low",
                "type": "assessment",
                "description": "不建议立即进行微服务化",
                "implementation_plan": [
                    "1. 继续完善单体架构",
                    "2. 优化模块化设计",
                    "3. 积累微服务经验",
                    "4. 考虑部分模块微服务化"
                ]
            })

        return recommendations

    def create_microservice_template(self, service_name: str) -> Dict[str, Any]:
        """创建微服务模板"""
        print(f"🏗️ 创建微服务模板: {service_name}")

        service_dir = self.microservices_dir / service_name
        service_dir.mkdir(parents=True, exist_ok=True)

        # 创建服务目录结构
        directories = [
            "src",
            "tests",
            "config",
            "docs",
            "scripts"
        ]

        for dir_name in directories:
            (service_dir / dir_name).mkdir(exist_ok=True)

        # 创建基础文件
        files_to_create = {
            "src/__init__.py": "",
            "src/main.py": self._generate_main_py(service_name),
            "src/service.py": self._generate_service_py(service_name),
            "src/models.py": self._generate_models_py(),
            "src/api.py": self._generate_api_py(service_name),
            "tests/__init__.py": "",
            "tests/test_service.py": self._generate_test_service_py(service_name),
            "config/config.yaml": self._generate_config_yaml(service_name),
            "requirements.txt": self._generate_requirements_txt(),
            "Dockerfile": self._generate_dockerfile(service_name),
            "docker-compose.yml": self._generate_docker_compose_yml(service_name)
        }

        for file_path, content in files_to_create.items():
            full_path = service_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)

        print(f"✅ 微服务模板创建完成: {service_name}")
        return {
            "success": True,
            "service_name": service_name,
            "service_dir": str(service_dir),
            "files_created": len(files_to_create)
        }

    def _generate_main_py(self, service_name: str) -> str:
        """生成主程序文件"""
        return f'''#!/usr/bin/env python3
"""
{service_name} 微服务主程序
"""

import uvicorn
import asyncio
from fastapi import FastAPI
from src.service import {service_name.title().replace('-', '')}Service
from src.api import create_app

def main():
    """主函数"""
    # 创建服务实例
    service = {service_name.title().replace('-', '')}Service()

    # 创建FastAPI应用
    app = create_app(service)

    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port={self.config["service_ports"][service_name]},
        reload=True
    )

if __name__ == "__main__":
    main()
'''

    def _generate_service_py(self, service_name: str) -> str:
        """生成服务实现文件"""
        return f'''#!/usr/bin/env python3
"""
{service_name} 微服务实现
"""

from typing import Dict, Any, List
from datetime import datetime

class {service_name.title().replace('-', '')}Service:
    """{service_name} 服务类"""

    def __init__(self):
        """初始化服务"""
        self.service_name = "{service_name}"
        self.start_time = datetime.now()

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {{
            "service": self.service_name,
            "status": "running",
            "uptime": str(datetime.now() - self.start_time),
            "timestamp": datetime.now().isoformat()
        }}

    def get_health(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {{
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }}
'''

    def _generate_models_py(self) -> str:
        """生成数据模型文件"""
        return '''#!/usr/bin/env python3
"""
数据模型定义
"""

from typing import Optional
from pydantic import BaseModel
from datetime import datetime

class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool
    message: str
    timestamp: datetime = datetime.now()

class HealthResponse(BaseResponse):
    """健康检查响应"""
    status: str
    uptime: Optional[str] = None
'''

    def _generate_api_py(self, service_name: str) -> str:
        """生成API文件"""
        return f'''#!/usr/bin/env python3
"""
{service_name} API接口
"""

from fastapi import FastAPI, HTTPException
from typing import Dict, Any

def create_app(service) -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title=f"{service_name} Service",
        description=f"{service_name} 微服务API",
        version="1.0.0"
    )

    @app.get("/")
    async def root():
        """根路径"""
        return {{"message": f"Welcome to {{service.service_name}}"}}

    @app.get("/status")
    async def get_status():
        """获取服务状态"""
        try:
            return service.get_status()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def get_health():
        """健康检查"""
        try:
            return service.get_health()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
'''

    def _generate_test_service_py(self, service_name: str) -> str:
        """生成测试文件"""
        return f'''#!/usr/bin/env python3
"""
{service_name} 服务测试
"""

import pytest
from src.service import {service_name.title().replace('-', '')}Service

class Test{service_name.title().replace('-', '')}Service:
    """服务测试类"""

    def setup_method(self):
        """测试前置"""
        self.service = {service_name.title().replace('-', '')}Service()

    def test_get_status(self):
        """测试获取状态"""
        status = self.service.get_status()
        assert status["service"] == "{service_name}"
        assert status["status"] == "running"
        assert "timestamp" in status

    def test_get_health(self):
        """测试健康检查"""
        health = self.service.get_health()
        assert health["status"] == "healthy"
        assert "timestamp" in health
'''

    def _generate_config_yaml(self, service_name: str) -> str:
        """生成配置文件"""
        port = self.config["service_ports"][service_name]
        return f'''# {service_name} 微服务配置

service:
  name: {service_name}
  port: {port}
  host: "0.0.0.0"

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

dependencies:
  config_service: "http://config-service:8001"
  logging_service: "http://logging-service:8003"
'''

    def _generate_requirements_txt(self) -> str:
        """生成依赖文件"""
        return '''fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
pytest==7.4.3
requests==2.31.0
pyyaml==6.0.1
'''

    def _generate_dockerfile(self, service_name: str) -> str:
        """生成Dockerfile"""
        return f'''FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE {self.config["service_ports"][service_name]}

CMD ["python", "src/main.py"]
'''

    def _generate_docker_compose_yml(self, service_name: str) -> str:
        """生成docker-compose文件"""
        port = self.config["service_ports"][service_name]
        return f'''version: '3.8'

services:
  {service_name}:
    build: .
    ports:
      - "{port}:{port}"
    environment:
      - SERVICE_NAME={service_name}
    depends_on:
      - config-service
    networks:
      - microservice-network

networks:
  microservice-network:
    driver: bridge
'''

    def generate_microservice_architecture_report(self) -> Dict[str, Any]:
        """生成微服务架构报告"""
        # 先进行可行性分析
        analysis = self.analyze_microservice_feasibility()

        report_data = {
            "timestamp": datetime.now(),
            "analysis": analysis,
            "config": self.config,
            "recommendations": analysis["recommendations"],
            "implementation_plan": self._generate_implementation_plan(analysis)
        }

        # 保存报告
        report_path = self.project_root / "reports" / \
            f"microservice_architecture_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

        return {
            "success": True,
            "report_path": str(report_path),
            "data": report_data
        }

    def _generate_implementation_plan(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成实施计划"""
        implementation_plan = []

        if analysis["feasibility_score"] > 80:
            # 高可行性实施计划
            implementation_plan = [
                {
                    "phase": 1,
                    "name": "准备阶段",
                    "duration": "1周",
                    "tasks": [
                        "创建微服务架构文档",
                        "设置微服务开发环境",
                        "准备Docker和Kubernetes环境",
                        "设计服务间通信协议"
                    ]
                },
                {
                    "phase": 2,
                    "name": "核心服务开发",
                    "duration": "2周",
                    "tasks": [
                        "开发配置服务",
                        "开发日志服务",
                        "开发缓存服务",
                        "开发健康检查服务"
                    ]
                },
                {
                    "phase": 3,
                    "name": "业务服务开发",
                    "duration": "3周",
                    "tasks": [
                        "开发安全服务",
                        "开发错误处理服务",
                        "开发资源管理服务",
                        "开发工具服务"
                    ]
                },
                {
                    "phase": 4,
                    "name": "集成和部署",
                    "duration": "2周",
                    "tasks": [
                        "实现API网关",
                        "实现服务发现",
                        "配置微服务监控",
                        "部署和测试"
                    ]
                }
            ]
        elif analysis["feasibility_score"] > 60:
            # 中可行性实施计划
            implementation_plan = [
                {
                    "phase": 1,
                    "name": "架构优化",
                    "duration": "2周",
                    "tasks": [
                        "优化依赖关系",
                        "提高服务内聚性",
                        "降低服务耦合性",
                        "完善接口设计"
                    ]
                },
                {
                    "phase": 2,
                    "name": "试点服务",
                    "duration": "2周",
                    "tasks": [
                        "选择合适的服务进行试点",
                        "开发试点微服务",
                        "测试和验证",
                        "总结经验教训"
                    ]
                },
                {
                    "phase": 3,
                    "name": "逐步迁移",
                    "duration": "4周",
                    "tasks": [
                        "迁移其他服务",
                        "完善微服务基础设施",
                        "优化服务间通信",
                        "完善监控和部署"
                    ]
                }
            ]
        else:
            # 低可行性实施计划
            implementation_plan = [
                {
                    "phase": 1,
                    "name": "架构评估",
                    "duration": "1周",
                    "tasks": [
                        "深入分析架构问题",
                        "评估微服务化成本",
                        "制定改进计划",
                        "确定实施策略"
                    ]
                },
                {
                    "phase": 2,
                    "name": "模块化改进",
                    "duration": "3周",
                    "tasks": [
                        "优化现有架构",
                        "提高模块化程度",
                        "完善接口设计",
                        "加强测试覆盖"
                    ]
                },
                {
                    "phase": 3,
                    "name": "微服务准备",
                    "duration": "2周",
                    "tasks": [
                        "学习微服务技术",
                        "准备基础设施",
                        "制定微服务化计划",
                        "选择试点项目"
                    ]
                }
            ]

        return implementation_plan


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='微服务架构改造工具')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--analyze', action='store_true', help='分析微服务化可行性')
    parser.add_argument('--create-template', help='创建微服务模板')
    parser.add_argument('--report', action='store_true', help='生成微服务架构报告')

    args = parser.parse_args()

    ms_arch = MicroserviceArchitecture(args.project)

    if args.analyze:
        result = ms_arch.analyze_microservice_feasibility()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.create_template:
        result = ms_arch.create_microservice_template(args.create_template)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.report:
        result = ms_arch.generate_microservice_architecture_report()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    else:
        print("🏗️ 微服务架构改造工具")
        print("使用 --help 查看可用命令")


if __name__ == "__main__":
    main()
