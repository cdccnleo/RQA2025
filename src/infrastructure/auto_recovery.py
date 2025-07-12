class AutoRecovery:
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        self.current_retries = 0

    def execute(self, operation):
        while True:
            try:
                result = operation()
                return result
            except Exception:
                self.current_retries += 1
                if self.current_retries >= self.max_retries:
                    raise
