import torch
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data


@DATASET_REGISTRY.register()
class IndexDataset(data.Dataset):
    def __init__(self, opt):
        self.size = torch.load(opt['text_features']).size(0)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return item
