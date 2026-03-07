
class HealthCheckerCore:
    """健康检查器核心"""

    def __init__(self):
        self.checkers = {}

    def add_checker(self, name, checker):
        self.checkers[name] = checker

    def check_all(self):
        results = {}
        for name, checker in self.checkers.items():
            try:
                results[name] = checker.check_health()
            except Exception as e:
                results[name] = {"status": "error", "message": str(e)}
        return results
