
class APIFlowDiagramGenerator:
    """API流程图生成器"""

    def __init__(self):
        self.diagrams = {}

    def generate_diagram(self, api_spec):
        return {"type": "flow_diagram", "data": api_spec}

    def add_diagram(self, name, diagram):
        self.diagrams[name] = diagram

    def get_diagram(self, name):
        """获取指定名称的流程图"""
        return self.diagrams.get(name)

    def remove_diagram(self, name):
        """移除指定名称的流程图"""
        if name in self.diagrams:
            del self.diagrams[name]
            return True
        return False

    def list_diagrams(self):
        """列出所有流程图名称"""
        return list(self.diagrams.keys())

    def clear_diagrams(self):
        """清空所有流程图"""
        self.diagrams.clear()

    def get_stats(self):
        """获取统计信息"""
        return {
            "total_diagrams": len(self.diagrams),
            "diagram_names": list(self.diagrams.keys())
        }