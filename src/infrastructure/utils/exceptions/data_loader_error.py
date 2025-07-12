class DataLoaderError(Exception):
    """Exception raised for errors in the DataLoader component."""

    def __init__(self, message="Data loading error occurred"):
        self.message = message
        super().__init__(self.message)
