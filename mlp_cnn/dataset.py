import numpy as np
import fens
from fens.dataset import load_dataset
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
import seaborn as sns


class MLP_Dataset(Dataset):
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size
        self.dataLoaders = load_dataset(path=self.data_path, batch_size=self.batch_size)
        # self.batch_size = batch
    @property
    def train_loader(self):
        return self.dataLoaders["train"]

    @property
    def validate_loader(self):
        return self.dataLoaders["validation"]

    @property
    def test_loader(self):
        return self.dataLoaders["test"]


def get_MEI(MEI_path, index):
    return np.load(MEI_path + str(index) + ".npy")
