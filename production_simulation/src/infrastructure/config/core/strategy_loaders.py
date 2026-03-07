
class StrategyLoaders:
    """策略加载器"""

    def __init__(self):
        self.loaders = {}

    def register_loader(self, name, loader):
        self.loaders[name] = loader

    def load_strategy(self, name):
        loader = self.loaders.get(name)
        return loader() if loader else None
