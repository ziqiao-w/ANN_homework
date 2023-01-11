import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
from dataset import get_MEI


class Trainer(object):
    def __init__(self, device, netMLP, netPRE, optimMLP, optimPRE, dataset, ckpt_dir, tb_writer):
        self._device = device
        self._netMLP = netMLP
        self._netPRE = netPRE
        self._optimMLP = optimMLP
        self._optimPRE = optimPRE
        self._dataset = dataset
        self._ckpt_dir = ckpt_dir
        self._tb_writer = tb_writer
        os.makedirs(ckpt_dir, exist_ok=True)
        self._netMLP.restore(ckpt_dir)
        self.MEI = torch.tensor(np.load("MEI/meis.npy"))\
            # .to(self._device).float()

    def validation_step(self, test_img, latent):
        L2loss = nn.MSELoss()
        pre_MEI = self._netPRE(self.MEI.view(5335, 1, 36, 64))
        fake_img = latent.view(-1, 5335, 1, 1) * pre_MEI.view(1, 5335, 36, 64)
        fake_img = fake_img.sum(axis=1).view(-1, 1, 36, 64)
        loss = L2loss(fake_img, test_img)
        return loss

    def train_step(self, real_img, latent):
        L2loss = nn.MSELoss()
        self._netMLP.zero_grad()
        # loss = torch.tensor(0).to(self._device).float()
        # batch_size = 0
        # for item in latent:
        #     fake_img = torch.zeros((1, 36, 64)).to(self._device)
        #     for index, value in enumerate(item):
        #         MEI_i = get_MEI("MEI/", index)
        #         img_i = torch.tensor(MEI_i).float().to(self._device) * value
        #         fake_img += img_i
        #
        #     real_img_t = real_img[batch_size].to(self._device)
        #     loss_ = L2loss(fake_img, real_img_t)
        #     loss += loss_
        #     batch_size += 1
        # loss /= batch_size
        pre_MEI = self._netPRE(self.MEI.view(5335, 1, 36, 64))
        fake_img = latent.view(-1, 5335, 1, 1) * pre_MEI.view(1, 5335, 36, 64)
        fake_img = fake_img.sum(axis=1).view(-1, 1, 36, 64)
        loss = L2loss(fake_img, real_img)
        loss.backward()
        self._optimMLP.step()
        self._optimPRE.step()

        return loss

    def train(self, num_epochs, logging_steps, saving_steps):
        dataloader = self._dataset.train_loader
        i = 0
        for epoch in range(num_epochs):
            for data in dataloader:

                # inp, _ = next(iterator)
                self._netMLP.train()  # 开启train模式，让batchnorm层可以工作
                real_imgs = data[0].to(self._device)
                real_responses = data[1].to(self._device)
                latent = self._netMLP(real_responses)
                Loss = self.train_step(real_imgs, latent)

                if (i + 1) % logging_steps == 0:
                    self._tb_writer.add_scalar("MSELoss", Loss, global_step=i)
                    print("MSELoss", Loss)

                if (i + 1) % saving_steps == 0:
                    dirname = self._netMLP.save(self._ckpt_dir, i)
                    dirname = self._netPRE.save(self._ckpt_dir, i)
                    self._netMLP.eval()
                    test_loader = self._dataset.validate_loader
                    for test_data in test_loader:
                        test_img = test_data[0].to(self._device)
                        test_response = test_data[1].to(self._device)
                        test_latent = self._netMLP(test_response)
                        test_lost = self.validation_step(test_img, test_latent)
                        print("validation loss : ", test_lost)
                        break

                i = i + 1
