#!/usr/bin/env python3
"""
GPU状态检查脚本
检查当前环境的GPU是否正常工作
"""

import sys
import subprocess
import logging
from typing import Dict, List

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GPUStatusChecker:
    """GPU状态检查器"""

    def __init__(self):
        self.gpu_count = 0
        self.gpu_stats = []

    def check_nvidia_smi(self) -> bool:
        """检查nvidia-smi是否可用"""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("✅ nvidia-smi 命令可用")
                return True
            else:
                logger.error(f"❌ nvidia-smi 命令失败: {result.stderr}")
                return False
        except FileNotFoundError:
            logger.error("❌ nvidia-smi 命令未找到，可能未安装NVIDIA驱动")
            return False
        except subprocess.TimeoutExpired:
            logger.error("❌ nvidia-smi 命令超时")
            return False
        except Exception as e:
            logger.error(f"❌ nvidia-smi 命令异常: {e}")
            return False

    def check_pytorch_cuda(self) -> bool:
        """检查PyTorch CUDA支持"""
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_count = torch.cuda.device_count()
                logger.info(f"✅ PyTorch CUDA 可用，检测到 {self.gpu_count} 个GPU")
                return True
            else:
                logger.warning("⚠️ PyTorch CUDA 不可用")
                return False
        except ImportError:
            logger.error("❌ PyTorch 未安装")
            return False
        except Exception as e:
            logger.error(f"❌ PyTorch CUDA 检查异常: {e}")
            return False

    def get_gpu_detailed_info(self) -> List[Dict]:
        """获取详细的GPU信息"""
        gpu_info = []

        try:
            # 使用nvidia-smi获取详细信息
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 8:
                            gpu_info.append({
                                'index': int(parts[0]),
                                'name': parts[1],
                                'memory_total': int(parts[2]),
                                'memory_used': int(parts[3]),
                                'memory_free': int(parts[4]),
                                'utilization': int(parts[5]),
                                'temperature': int(parts[6]),
                                'power_draw': float(parts[7]) if parts[7] != 'N/A' else None
                            })
        except Exception as e:
            logger.error(f"获取GPU详细信息失败: {e}")

        return gpu_info

    def check_gpu_health(self, gpu_info: List[Dict]) -> Dict:
        """检查GPU健康状态"""
        health_status = {
            'total_gpus': len(gpu_info),
            'healthy_gpus': 0,
            'issues': []
        }

        for gpu in gpu_info:
            issues = []

            # 检查温度
            if gpu['temperature'] > 85:
                issues.append(f"GPU {gpu['index']} 温度过高: {gpu['temperature']}°C")

            # 检查内存使用率
            memory_usage = (gpu['memory_used'] / gpu['memory_total']) * 100
            if memory_usage > 90:
                issues.append(f"GPU {gpu['index']} 内存使用率过高: {memory_usage:.1f}%")

            # 检查利用率
            if gpu['utilization'] > 95:
                issues.append(f"GPU {gpu['index']} 利用率过高: {gpu['utilization']}%")

            if not issues:
                health_status['healthy_gpus'] += 1
            else:
                health_status['issues'].extend(issues)

        return health_status

    def test_gpu_computation(self) -> bool:
        """测试GPU计算能力"""
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("⚠️ CUDA不可用，跳过GPU计算测试")
                return False

            # 创建测试张量
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)

            # 执行矩阵乘法
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()
            z = torch.mm(x, y)
            end_time.record()

            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)

            logger.info(f"✅ GPU计算测试通过，矩阵乘法耗时: {elapsed_time:.2f}ms")
            return True

        except Exception as e:
            logger.error(f"❌ GPU计算测试失败: {e}")
            return False

    def run_comprehensive_check(self) -> Dict:
        """运行综合GPU检查"""
        logger.info("🔍 开始GPU状态检查...")

        check_results = {
            'nvidia_smi_available': False,
            'pytorch_cuda_available': False,
            'gpu_count': 0,
            'gpu_info': [],
            'health_status': {},
            'computation_test': False,
            'overall_status': 'UNKNOWN'
        }

        # 1. 检查nvidia-smi
        check_results['nvidia_smi_available'] = self.check_nvidia_smi()

        # 2. 检查PyTorch CUDA
        check_results['pytorch_cuda_available'] = self.check_pytorch_cuda()
        check_results['gpu_count'] = self.gpu_count

        # 3. 获取GPU详细信息
        if check_results['nvidia_smi_available']:
            check_results['gpu_info'] = self.get_gpu_detailed_info()

            # 4. 检查GPU健康状态
            if check_results['gpu_info']:
                check_results['health_status'] = self.check_gpu_health(check_results['gpu_info'])

        # 5. 测试GPU计算能力
        check_results['computation_test'] = self.test_gpu_computation()

        # 6. 确定整体状态
        if (check_results['nvidia_smi_available'] and
            check_results['pytorch_cuda_available'] and
            check_results['gpu_count'] > 0 and
                check_results['computation_test']):
            check_results['overall_status'] = 'HEALTHY'
        elif check_results['gpu_count'] > 0:
            check_results['overall_status'] = 'PARTIAL'
        else:
            check_results['overall_status'] = 'UNHEALTHY'

        return check_results

    def print_results(self, results: Dict):
        """打印检查结果"""
        print("\n" + "="*60)
        print("🎯 GPU状态检查结果")
        print("="*60)

        print(f"📊 整体状态: {results['overall_status']}")
        print(f"🔧 nvidia-smi: {'✅ 可用' if results['nvidia_smi_available'] else '❌ 不可用'}")
        print(f"⚡ PyTorch CUDA: {'✅ 可用' if results['pytorch_cuda_available'] else '❌ 不可用'}")
        print(f"🎮 GPU数量: {results['gpu_count']}")
        print(f"🧮 计算测试: {'✅ 通过' if results['computation_test'] else '❌ 失败'}")

        if results['gpu_info']:
            print(f"\n📋 GPU详细信息:")
            for gpu in results['gpu_info']:
                memory_usage = (gpu['memory_used'] / gpu['memory_total']) * 100
                print(f"  GPU {gpu['index']}: {gpu['name']}")
                print(
                    f"    内存: {gpu['memory_used']}MB/{gpu['memory_total']}MB ({memory_usage:.1f}%)")
                print(f"    利用率: {gpu['utilization']}%")
                print(f"    温度: {gpu['temperature']}°C")
                if gpu['power_draw']:
                    print(f"    功耗: {gpu['power_draw']}W")
                print()

        if results['health_status']:
            health = results['health_status']
            print(f"🏥 健康状态:")
            print(f"  总GPU数: {health['total_gpus']}")
            print(f"  健康GPU数: {health['healthy_gpus']}")
            if health['issues']:
                print(f"  问题:")
                for issue in health['issues']:
                    print(f"    ⚠️ {issue}")

        print("="*60)


def main():
    """主函数"""
    checker = GPUStatusChecker()
    results = checker.run_comprehensive_check()
    checker.print_results(results)

    # 返回适当的退出码
    if results['overall_status'] == 'HEALTHY':
        sys.exit(0)
    elif results['overall_status'] == 'PARTIAL':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
