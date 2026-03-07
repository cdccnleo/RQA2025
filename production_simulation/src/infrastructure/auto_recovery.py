
class AutoRecoveryManager:
    """自动恢复管理器"""

    def __init__(self):
        self.recovery_actions = {}

    def register_recovery_action(self, name, action):
        self.recovery_actions[name] = action

    def execute_recovery(self, name):
        action = self.recovery_actions.get(name)
        if action:
            return action()
        return False
