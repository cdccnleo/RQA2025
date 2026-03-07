#!/usr/bin/env python3
"""
实时架构监控工具

提供实时的架构监控功能，持续监控架构合规性
"""

import time
import threading
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
from collections import defaultdict


class RealtimeArchitectureMonitor:
    """实时架构监控器"""

    def __init__(self, check_interval: int = 300):
        self.check_interval = check_interval  # 检查间隔(秒)
        self.is_running = False
        self.monitor_thread = None
        self.violations_history = []
        self.layer_mapping = self._load_layer_mapping()
        self.last_check_time = None
        self.check_count = 0

    def _load_layer_mapping(self) -> Dict[str, str]:
        """加载层级映射"""
        return {
            'src/core': 'core',
            'src/infrastructure': 'infrastructure',
            'src/data': 'data',
            'src/gateway': 'gateway',
            'src/features': 'features',
            'src/ml': 'ml',
            'src/backtest': 'backtest',
            'src/risk': 'risk',
            'src/trading': 'trading',
            'src/engine': 'engine'
        }

    def start_monitoring(self):
        """开始监控"""
        if self.is_running:
            print("⚠️ 监控已在运行中")
            return

        print("🚀 启动实时架构监控...")
        print(f"📊 检查间隔: {self.check_interval}秒")

        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        print("✅ 实时监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        if not self.is_running:
            print("⚠️ 监控未在运行")
            return

        print("🛑 停止实时架构监控...")
        self.is_running = False

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        print("✅ 实时监控已停止")

    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                current_time = datetime.now()
                print(f"\n🔍 执行监控检查 #{self.check_count + 1} - {current_time.strftime('%H:%M:%S')}")

                violations = self.perform_comprehensive_check()
                self.violations_history.append({
                    'timestamp': current_time.isoformat(),
                    'violations': violations
                })

                # 保持历史记录在合理范围内
                if len(self.violations_history) > 100:
                    self.violations_history = self.violations_history[-100:]

                self.check_count += 1
                self.last_check_time = current_time

                if violations:
                    print(f"   ❌ 发现 {len(violations)} 个违规")
                    self._handle_violations(violations)
                else:
                    print("   ✅ 无违规发现")

                # 生成监控报告
                self._generate_monitoring_report()

                # 等待下一个检查周期
                time.sleep(self.check_interval)

            except Exception as e:
                print(f"❌ 监控循环出错: {e}")
                time.sleep(60)  # 出错后等待1分钟再试

    def perform_comprehensive_check(self) -> List[Dict]:
        """执行综合检查"""
        violations = []

        # 1. 检查架构违规
        architecture_violations = self.check_architecture_violations()
        violations.extend(architecture_violations)

        # 2. 检查依赖关系
        dependency_violations = self.check_dependency_violations()
        violations.extend(dependency_violations)

        # 3. 检查组件工厂
        component_violations = self.check_component_factories()
        violations.extend(component_violations)

        # 4. 检查代码质量
        quality_violations = self.check_code_quality()
        violations.extend(quality_violations)

        return violations

    def check_architecture_violations(self) -> List[Dict]:
        """检查架构违规"""
        violations = []

        for root_path, layer in self.layer_mapping.items():
            layer_dir = Path(root_path)
            if not layer_dir.exists():
                continue

            for file_path in layer_dir.rglob('*.py'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查业务概念使用
                    layer_violations = self._check_business_concepts_in_file(
                        str(file_path), content, layer)
                    violations.extend(layer_violations)

                except Exception as e:
                    violations.append({
                        'type': 'file_error',
                        'severity': 'low',
                        'file': str(file_path),
                        'description': f"无法读取文件: {e}",
                        'timestamp': datetime.now().isoformat()
                    })

        return violations

    def _check_business_concepts_in_file(self, file_path: str, content: str, layer: str) -> List[Dict]:
        """检查文件中的业务概念"""
        violations = []

        forbidden_concepts = {
            'data': ['trading', 'strategy', 'execution', 'model', 'risk', 'order'],
            'features': ['trading', 'order', 'execution'],
            'ml': ['trading', 'order', 'execution'],
            'core': ['trading', 'strategy', 'execution', 'model', 'risk', 'order'],
            'infrastructure': ['trading', 'strategy', 'execution']
        }

        forbidden_in_layer = forbidden_concepts.get(layer, [])

        for concept in forbidden_in_layer:
            import re
            if re.search(r'\b' + re.escape(concept) + r'\b', content, re.IGNORECASE):
                violations.append({
                    'type': 'business_concept',
                    'severity': 'high',
                    'file': file_path,
                    'layer': layer,
                    'concept': concept,
                    'description': f"禁止在{layer}层使用业务概念: {concept}",
                    'timestamp': datetime.now().isoformat()
                })

        return violations

    def check_dependency_violations(self) -> List[Dict]:
        """检查依赖关系违规"""
        violations = []

        # 定义允许的依赖关系
        allowed_dependencies = {
            'core': [],
            'infrastructure': ['core'],
            'data': ['infrastructure', 'core'],
            'gateway': ['infrastructure', 'core'],
            'features': ['data', 'infrastructure', 'core'],
            'ml': ['features', 'infrastructure', 'core'],
            'backtest': ['ml', 'features', 'data', 'infrastructure', 'core'],
            'risk': ['backtest', 'infrastructure', 'core'],
            'trading': ['risk', 'backtest', 'infrastructure', 'core'],
            'engine': ['trading', 'risk', 'backtest', 'ml', 'features', 'data', 'infrastructure', 'core']
        }

        for root_path, current_layer in self.layer_mapping.items():
            layer_dir = Path(root_path)
            if not layer_dir.exists():
                continue

            for file_path in layer_dir.rglob('*.py'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查导入
                    import_violations = self._check_imports_in_file(
                        str(file_path), content, current_layer, allowed_dependencies
                    )
                    violations.extend(import_violations)

                except Exception as e:
                    continue

        return violations

    def _check_imports_in_file(self, file_path: str, content: str, current_layer: str, allowed_dependencies: Dict) -> List[Dict]:
        """检查文件中的导入"""
        violations = []
        allowed_layers = allowed_dependencies.get(current_layer, [])

        import re

        # 检查from导入
        from_imports = re.findall(r'from\s+src\.(\w+)\s+import', content)
        for imported_layer in from_imports:
            if imported_layer not in allowed_layers and imported_layer != current_layer:
                violations.append({
                    'type': 'dependency_violation',
                    'severity': 'high',
                    'file': file_path,
                    'from_layer': current_layer,
                    'to_layer': imported_layer,
                    'import_type': 'from_import',
                    'description': f"禁止的from导入: {current_layer} -> {imported_layer}",
                    'timestamp': datetime.now().isoformat()
                })

        # 检查直接导入
        direct_imports = re.findall(r'import\s+src\.(\w+)', content)
        for imported_layer in direct_imports:
            if imported_layer not in allowed_layers and imported_layer != current_layer:
                violations.append({
                    'type': 'dependency_violation',
                    'severity': 'high',
                    'file': file_path,
                    'from_layer': current_layer,
                    'to_layer': imported_layer,
                    'import_type': 'direct_import',
                    'description': f"禁止的直接导入: {current_layer} -> {imported_layer}",
                    'timestamp': datetime.now().isoformat()
                })

        return violations

    def check_component_factories(self) -> List[Dict]:
        """检查组件工厂"""
        violations = []

        for root_path, layer in self.layer_mapping.items():
            layer_dir = Path(root_path)
            if not layer_dir.exists():
                continue

            for file_path in layer_dir.rglob('*_components.py'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查组件工厂合规性
                    component_violations = self._check_component_factory_compliance(
                        str(file_path), content)
                    violations.extend(component_violations)

                except Exception as e:
                    continue

        return violations

    def _check_component_factory_compliance(self, file_path: str, content: str) -> List[Dict]:
        """检查组件工厂合规性"""
        violations = []

        checks = [
            ('class IComponent', '缺少标准IComponent接口定义'),
            ('class ComponentFactory', '缺少标准ComponentFactory类定义'),
            ('def create_component', '缺少create_component方法'),
            ('try:', '缺少错误处理机制'),
            ('except', '缺少异常捕获')
        ]

        for pattern, description in checks:
            if pattern not in content:
                violations.append({
                    'type': 'component_factory',
                    'severity': 'medium',
                    'file': file_path,
                    'issue': pattern,
                    'description': description,
                    'timestamp': datetime.now().isoformat()
                })

        return violations

    def check_code_quality(self) -> List[Dict]:
        """检查代码质量"""
        violations = []

        for root_path, layer in self.layer_mapping.items():
            layer_dir = Path(root_path)
            if not layer_dir.exists():
                continue

            for file_path in layer_dir.rglob('*.py'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查代码质量问题
                    quality_violations = self._check_file_quality(str(file_path), content)
                    violations.extend(quality_violations)

                except Exception as e:
                    continue

        return violations

    def _check_file_quality(self, file_path: str, content: str) -> List[Dict]:
        """检查文件质量"""
        violations = []
        lines = content.split('\n')

        # 检查行长度
        for i, line in enumerate(lines, 1):
            if len(line) > 120:  # 更严格的行长度限制
                violations.append({
                    'type': 'code_quality',
                    'severity': 'low',
                    'file': file_path,
                    'line': i,
                    'issue': 'line_too_long',
                    'description': f"行过长: {len(line)}字符 (建议不超过120)",
                    'timestamp': datetime.now().isoformat()
                })

        # 检查函数长度
        import ast
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    start_line = node.lineno
                    end_line = node.body[-1].lineno if node.body else start_line
                    func_lines = end_line - start_line + 1

                    if func_lines > 30:  # 更严格的函数长度限制
                        violations.append({
                            'type': 'code_quality',
                            'severity': 'low',
                            'file': file_path,
                            'function': node.name,
                            'issue': 'function_too_long',
                            'description': f"函数过长: {node.name} ({func_lines}行，建议不超过30)",
                            'timestamp': datetime.now().isoformat()
                        })
        except:
            pass

        return violations

    def _handle_violations(self, violations: List[Dict]):
        """处理违规"""
        # 按严重程度分组
        by_severity = defaultdict(list)
        for violation in violations:
            by_severity[violation['severity']].append(violation)

        # 输出高严重度违规
        if 'high' in by_severity:
            print("   🚨 高严重度违规:")
            for violation in by_severity['high'][:5]:  # 最多显示5个
                print(f"      - {violation['description']}")

        # 输出中严重度违规
        if 'medium' in by_severity:
            print("   ⚠️ 中严重度违规:")
            for violation in by_severity['medium'][:3]:  # 最多显示3个
                print(f"      - {violation['description']}")

    def _generate_monitoring_report(self):
        """生成监控报告"""
        report = []

        report.append("# 实时架构监控报告")
        report.append("")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"监控运行时间: {self._get_uptime()}")
        report.append(f"检查次数: {self.check_count}")
        report.append("")

        # 当前状态
        report.append("## 📊 当前状态")
        if self.last_check_time:
            report.append(f"- 最后检查时间: {self.last_check_time.strftime('%H:%M:%S')}")
        report.append(f"- 监控状态: {'运行中' if self.is_running else '已停止'}")
        report.append("")

        # 最新检查结果
        if self.violations_history:
            latest = self.violations_history[-1]
            violations = latest['violations']

            report.append("## 🔍 最新检查结果")
            report.append(
                f"- 检查时间: {datetime.fromisoformat(latest['timestamp']).strftime('%H:%M:%S')}")
            report.append(f"- 违规数量: {len(violations)}")
            report.append("")

            if violations:
                # 按类型分组
                by_type = defaultdict(list)
                for violation in violations:
                    by_type[violation['type']].append(violation)

                for violation_type, type_violations in by_type.items():
                    report.append(f"### {violation_type} ({len(type_violations)}个)")
                    for violation in type_violations[:3]:  # 最多显示3个
                        report.append(
                            f"- {violation.get('description', violation.get('issue', '未知问题'))}")
                    report.append("")

        # 历史趋势
        if len(self.violations_history) > 1:
            report.append("## 📈 历史趋势")
            recent_checks = self.violations_history[-10:]  # 最近10次检查

            for i, check in enumerate(recent_checks):
                check_time = datetime.fromisoformat(check['timestamp'])
                violation_count = len(check['violations'])
                report.append(f"- {check_time.strftime('%H:%M:%S')}: {violation_count} 个违规")

            report.append("")

        # 保存报告
        with open('reports/REALTIME_MONITORING_REPORT.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

    def _get_uptime(self) -> str:
        """获取运行时间"""
        if not self.last_check_time:
            return "未开始"

        uptime = datetime.now() - (self.last_check_time - timedelta(seconds=self.check_interval * self.check_count))
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}小时{minutes}分钟"
        elif minutes > 0:
            return f"{minutes}分钟{seconds}秒"
        else:
            return f"{seconds}秒"

    def get_status(self) -> Dict:
        """获取监控状态"""
        return {
            'is_running': self.is_running,
            'check_interval': self.check_interval,
            'check_count': self.check_count,
            'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
            'uptime': self._get_uptime(),
            'total_violations': sum(len(check['violations']) for check in self.violations_history),
            'latest_violations': len(self.violations_history[-1]['violations']) if self.violations_history else 0
        }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='实时架构监控工具')
    parser.add_argument('action', choices=['start', 'stop', 'status', 'check'],
                        help='监控操作')
    parser.add_argument('--interval', type=int, default=300,
                        help='检查间隔(秒，默认300)')

    args = parser.parse_args()

    monitor = RealtimeArchitectureMonitor(check_interval=args.interval)

    if args.action == 'start':
        monitor.start_monitoring()

        try:
            # 保持主线程运行
            while monitor.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 收到中断信号，正在停止监控...")
            monitor.stop_monitoring()

    elif args.action == 'stop':
        monitor.stop_monitoring()

    elif args.action == 'status':
        status = monitor.get_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))

    elif args.action == 'check':
        violations = monitor.perform_comprehensive_check()
        if violations:
            print(f"❌ 发现 {len(violations)} 个违规:")
            for violation in violations[:10]:  # 最多显示10个
                print(f"   - {violation['description']}")
        else:
            print("✅ 无违规发现")


if __name__ == "__main__":
    main()
