import GAN
from trainer import Trainer
# from dataset import Dataset
import dataset
from tensorboardX import SummaryWriter

# from pytorch_fid import fid_score

import torch
import torch.optim as optim
import os
import argparse

if __name__ == "__main__":
    tb_writer = SummaryWriter(log_dir='./runs')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    dataset = dataset.G_Dataset("neural/Lurz2020/static20457-5-9-preproc0/data/images", "neural/Lurz2020/static20457-5-9-preproc0/data/responses")
    netG = GAN.get_generator(1, latent_dim=32, hidden_dim=300, device=device)
    netD = GAN.get_discriminator(1, hidden_dim=300, device=device)
    optimG = optim.Adam(netG.parameters(), 0.0002, betas=(0.5, 0.999))
    optimD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    trainer = Trainer(device, netG, netD, optimG, optimD, dataset, ckpt_dir='results', tb_writer=tb_writer)
    trainer.train(num_training_updates=5000, logging_steps=10, saving_steps=1000)







