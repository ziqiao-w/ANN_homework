import MLP
from trainer import Trainer
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
    dataset = dataset.MLP_Dataset(data_path="neural/Lurz2020/static20457-5-9-preproc0", batch_size=2)
    netMLP = MLP.get_MLP(response_dim=5335, latent_dim=5335, device=device)
    netPRE = MLP.get_PRE(device=device)
    optimMLP = optim.Adam(netMLP.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimPRE = optim.Adam(netPRE.parameters(), lr=0.0002, betas=(0.5, 0.999))
    trainer = Trainer(device=device, netMLP=netMLP, netPRE=netPRE, optimPRE=optimPRE, optimMLP=optimMLP, dataset=dataset, ckpt_dir='results', tb_writer=tb_writer)
    trainer.train(num_epochs=20, logging_steps=10, saving_steps=20)







