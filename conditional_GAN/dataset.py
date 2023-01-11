import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class G_Dataset(Dataset):
    def __init__(self, img_path, neu_path, transform=None):
        self.image_files = os.listdir(img_path)
        self.neu_files = os.listdir(neu_path)

        self.list_img_files = []
        self.list_neu_files = []

        self.transform = transform

        for file in self.image_files:
            self.list_img_files.append(os.path.join(img_path, file))
        for file in self.neu_files:
            self.list_neu_files.append(os.path.join(neu_path, file))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = np.load(self.list_img_files[index])
        cond = np.load(self.list_neu_files[index])
        if self.transform is None:
            return image, cond
        else:
            image = self.transform(image)
            return image, cond

    @property
    def training_loader(self):
        return DataLoader(self, batch_size=32, shuffle=True)

    def get_image(self, index):
        image = np.load(self.list_img_files[index])
        return image

    def get_cond(self, index):
        cond = np.load(self.list_neu_files[index])
        return cond
# G_Dataset("neural/Lurz2020/static20457-5-9-preproc0/data/images", "neural/Lurz2020/static20457-5-9-preproc0/data/responses", transform)
