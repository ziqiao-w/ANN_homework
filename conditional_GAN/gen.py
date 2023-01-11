import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
# from main import dataset
import GAN
import dataset

dataset = dataset.G_Dataset("neural/Lurz2020/static20457-5-9-preproc0/data/images", "neural/Lurz2020/static20457-5-9-preproc0/data/responses")
for j in range(0, 5992):

    plt.imshow(dataset.get_image(j).squeeze(), cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.imsave("samples/sample_{}.png".format(j), dataset.get_image(j).squeeze(), cmap='gray',
               vmin=0, vmax=255)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netG = GAN.get_generator(1, 32, 300, device)
restore_ckpt_path = os.path.join('results', str(max(int(step) for step in os.listdir('results'))))
netG.restore(restore_ckpt_path)
print("start generating")
for i in range(0, 5992):
    condition = torch.tensor(dataset.get_cond(i))
    condition = condition.view(-1, 5335, 1, 1).to(device)
    latent = torch.randn((1, 32, 1, 1)).to(device)
    immg = netG(latent, condition).squeeze().to("cpu").detach().numpy()
    plt.imsave('Gen/{}.png'.format(i), immg, cmap='gray', vmin=0,
               vmax=255)