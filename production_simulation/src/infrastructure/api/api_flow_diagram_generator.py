
class APIFlowDiagramGenerator:
    """API流程图生成器"""

    def __init__(self):
        self.diagrams = {}

    def generate_diagram(self, api_spec):
        return {"type": "flow_diagram", "data": api_spec}

    def add_diagram(self, name, diagram):
        self.diagrams[name] = diagram
