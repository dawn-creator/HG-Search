from abc import ABC

class BaseDataset(ABC):
    def __init__(self, *args, **kwargs):
        super(BaseDataset, self).__init__()
        self.logger = kwargs['logger']
        self.g = None
        self.meta_paths = None
        self.meta_paths_dict = None