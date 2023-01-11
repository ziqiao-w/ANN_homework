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
import MLP
import dataset

testset = dataset.MLP_Dataset("neural/Lurz2020/static20457-5-9-preproc0/", batch_size=1)
testLoader = testset.test_loader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netMLP = MLP.get_MLP(5335, 5335, device)
netPRE = MLP.get_PRE(device)
restore_ckpt_path = os.path.join('results', str(max(int(step) for step in os.listdir('results'))))
netMLP.restore(restore_ckpt_path)
netPRE.restore(restore_ckpt_path)
netPRE.eval()
netMLP.eval()
MEI = torch.tensor(np.load("MEI/meis.npy")).to(device).float()
idx = 0
for test in testLoader:
    testimg = test[0]
    testres = test[1]
    plt.imshow(testimg.squeeze().squeeze(), cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.imsave("samples/sample_{}.png".format(idx), testimg.squeeze().squeeze(), cmap='gray',
               vmin=0, vmax=255)
    print("start generating")

    latent = netMLP(testres)
    pre_MEI = netPRE(MEI.view(5335, 1, 36, 64))
    fake_img = latent.view(-1, 5335, 1, 1) * pre_MEI.view(1, 5335, 36, 64)
    fake_img = fake_img.sum(axis=1).view(36, 64)
    fake_img = fake_img.to("cpu").detach().numpy()
    plt.imshow(fake_img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.imsave('Gen/{}.png'.format(idx), fake_img, cmap='gray', vmin=0,
               vmax=255)
    idx += 1

