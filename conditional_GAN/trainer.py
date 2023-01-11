import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class Trainer(object):
    def __init__(self, device, netG, netD, optimG, optimD, dataset, ckpt_dir, tb_writer):
        self._device = device
        self._netG = netG
        self._netD = netD
        self._optimG = optimG
        self._optimD = optimD
        self._dataset = dataset
        self._ckpt_dir = ckpt_dir
        self._tb_writer = tb_writer
        os.makedirs(ckpt_dir, exist_ok=True)
        self._netG.restore(ckpt_dir)
        self._netD.restore(ckpt_dir)

    def train_step(self, real_cond, real_img, fake_img, unmatch_cond, unmatch_img, BCE_criterion):
        """DO NOT FORGET TO ZERO_GRAD netD and netG
		*   Returns:
			*   loss of netD (scalar)
			*   loss of netG (scalar)
			*   average D(real_imgs) before updating netD
			*   average D(fake_imgs) before updating netD
			*   average D(fake_imgs) after updating netD
		"""
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        #########################
        # clear gradient
        self._netD.zero_grad()

        # compute the gradients of binary_cross_entropy(netD(real_imgs), 1) w.r.t. netD
        # record average D(real_imgs)
        # TODO START
        outp1 = self._netD(real_img, real_cond)
        loss_D_real = BCE_criterion(outp1, torch.ones_like(outp1))

        D_x = self._netD(real_img, real_cond).mean()

        outp2 = self._netD(unmatch_img, unmatch_cond)
        loss_D_mismatch = BCE_criterion(outp2, torch.zeros_like(outp2))
        loss_D1 = loss_D_mismatch + loss_D_real

        loss_D1.backward()
        # TODO END

        # ** accumulate ** the gradients of binary_cross_entropy(netD(fake_imgs), 0) w.r.t. netD
        # record average D(fake_imgs)
        # TODO START
        outp3 = self._netD(fake_img, real_cond)
        loss_D_fake = BCE_criterion(outp3, torch.zeros_like(outp3))
        D_G_z1 = self._netD(fake_img, real_cond).mean()
        loss_D_fake.backward(retain_graph=True)
        # TODO END

        # update netD
        self._optimD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # clear gradients
        self._netG.zero_grad()

        # compute the gradients of binary_cross_entropy(netD(fake_imgs), 1) w.r.t. netG
        # record average D(fake_imgs)
        # TODO START
        l1loss = nn.L1Loss()
        Loss_G = l1loss(fake_img, real_img)
        outp3 = self._netD(fake_img, real_cond)
        loss_G = BCE_criterion(outp3, torch.ones_like(outp3)) + 0.15*Loss_G
        D_G_z2 = self._netD(fake_img, real_cond).mean()
        loss_G.backward()
        # TODO END

        # update netG
        self._optimG.step()

        # return what are specified in the docstring
        return loss_D1 + loss_D_fake, loss_G, D_x, D_G_z1, D_G_z2

    def train(self, num_training_updates, logging_steps, saving_steps):
        fixed_noise = torch.randn(32, self._netG.latent_dim, 1, 1, device=self._device)
        fixed_cond = []
        samples = []
        for j in range(1, 33):
            fixed_cond.append(torch.tensor(self._dataset.get_cond(j)))
            samples.append(torch.tensor(self._dataset.get_image(j)))
            # plt.imshow(self._dataset.get_image(j).squeeze(), cmap='gray', vmin=0, vmax=255)
            # plt.axis('off')
            # plt.show()
        fixed_cond = torch.stack(tuple(fixed_cond)).to(self._device)
        samples = torch.stack(tuple(samples)).to(self._device)
        samples = make_grid(samples)
        # save_image(samples, os.path.join("sample.png"))
        criterion = nn.BCELoss()
        # iterator = iter(cycle(self._dataset.training_loader))
        dataloader = self._dataset.training_loader
        # for i in tqdm(range(num_training_updates), desc='Training'):
        i = 0
        for epoch in range(0, 100):
            for data in dataloader:

                # inp, _ = next(iterator)
                self._netD.train()  # 开启train模式，让batchnorm层可以工作
                self._netG.train()
                real_imgs = data[0].to(self._device)
                real_conds = data[1].to(self._device)

                # .to(self._device)
                x = torch.randint(0, 5000, (1, 8))
                x = x.squeeze(dim=0).numpy().tolist()
                unmatch_imgs = []
                unmatch_conds = []
                for idx in x[0: 4]:
                    unmatch_imgs.append(torch.tensor(self._dataset.get_image(idx)))
                for idx in x[4: 8]:
                    unmatch_conds.append(torch.tensor(self._dataset.get_cond(idx)))
                unmatch_imgs = torch.stack(tuple(unmatch_imgs)).to(self._device)
                unmatch_conds = torch.stack(tuple(unmatch_conds)).to(self._device)

                fake_imgs = self._netG(torch.randn(real_imgs.size(0), self._netG.latent_dim, 1, 1, device=self._device),
                                       real_conds.view((-1, 5335, 1, 1)))

                errD, errG, D_x, D_G_z1, D_G_z2 = self.train_step(real_conds, real_imgs, fake_imgs, unmatch_conds,
                                                                  unmatch_imgs, criterion)

                if (i + 1) % logging_steps == 0:
                    self._tb_writer.add_scalar("discriminator_loss", errD, global_step=i)
                    self._tb_writer.add_scalar("generator_loss", errG, global_step=i)
                    self._tb_writer.add_scalar("D(x)", D_x, global_step=i)
                    self._tb_writer.add_scalar("D(G(z1))", D_G_z1, global_step=i)
                    self._tb_writer.add_scalar("D(G(z2))", D_G_z2, global_step=i)
                    print("errD", errD)
                    print("errG", errG)
                if (i + 1) % 500 == 0:
                    dirname = self._netD.save(self._ckpt_dir, i)
                    dirname = self._netG.save(self._ckpt_dir, i)
                    self._netG.eval()
                    imgs = self._netG(fixed_noise, fixed_cond.view(-1, 5335, 1, 1))
                    for index, immmg in enumerate(imgs):

                        # save_image(immmg, "{}{}.png".format(i, index))
                        plt.imsave("{}_{}_{}.png".format(epoch, i, index), immmg.squeeze().to("cpu").detach().numpy(), cmap='gray', vmin=0, vmax=255)
                        plt.imshow(immmg.squeeze().to("cpu").detach().numpy(), cmap='gray', vmin=0, vmax=255)
                        plt.axis('off')
                        plt.show()
                    imgs = make_grid(imgs, range=(0, 256))
                    self._tb_writer.add_image('samples', imgs, global_step=i)
                    save_image(imgs, os.path.join(dirname, "samples.png"))

                i = i + 1
