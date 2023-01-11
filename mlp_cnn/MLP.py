import torch.nn as nn
import torch
import os


def weights_init(m):
    torch.nn.init.normal_(m.weight, 0.0, 0.02)
    torch.nn.init.zeros_(m.bias)


def get_MLP(response_dim, latent_dim, device):
    model = MLP(response_dim, latent_dim).to(device)
    # model.apply(weights_init)
    return model


def get_PRE(device):
    model = preCnn().to(device)
    return model


class MLP(nn.Module):
    def __init__(self, response_dim, latent_dim):
        super(MLP, self).__init__()
        self.response_dim = response_dim
        self.latent_dim = latent_dim
        self.linear = nn.Sequential(
            nn.Linear(5335, 8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.ReLU(),
            # nn.Linear(8192, 8192),
            # nn.ReLU(),
            # nn.Linear(16384, 8192),
            # nn.ReLU(),
            # nn.Linear(8192, 8192),
            # nn.ReLU(),
            # nn.Linear(8192, 8192),
            # nn.ReLU(),
            # nn.Linear(8192, 8192),
            # nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.Linear(8192, 5335)
        )

    def forward(self, x):
        x = x.float()
        x = x.to(next(self.parameters()).device)
        latent = self.linear(x)
        return latent

    def restore(self, ckpt_dir):
        try:
            if os.path.exists(os.path.join(ckpt_dir, 'MLP.bin')):
                path = os.path.join(ckpt_dir, 'MLP.bin')
            else:
                path = os.path.join(ckpt_dir, str(max(int(name) for name in os.listdir(ckpt_dir))), 'MLP.bin')
        except:
            return None
        self.load_state_dict(torch.load(path))
        return os.path.split(path)[0]

    def save(self, ckpt_dir, global_step):
        os.makedirs(os.path.join(ckpt_dir, str(global_step)), exist_ok=True)
        path = os.path.join(ckpt_dir, str(global_step), 'MLP.bin')
        torch.save(self.state_dict(), path)
        return os.path.split(path)[0]


class preCnn(nn.Module):
    def __init__(self):
        super(preCnn, self).__init__()
        self.convt1 = nn.ConvTranspose2d(in_channels=1, out_channels=2, kernel_size=4,
                                        stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=2)
        self.relu1 = nn.ReLU()

        self.convt2 = nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=4,
                                        stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=1)
        self.relu2 = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=1)
        self.poo1 = nn.MaxPool2d(kernel_size=2)
        self.relu3 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(num_features=1)
        self.poo2 = nn.MaxPool2d(kernel_size=2)
        self.relu4 = nn.ReLU()

    def forward(self, x):

        x = x.float()
        x = x.to(next(self.parameters()).device)

        x = self.convt1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.convt2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv1(x)
        x = self.bn3(x)
        x = self.poo1(x)
        x = self.relu3(x)

        x = self.conv2(x)
        x = self.bn4(x)
        x = self.poo2(x)
        x = self.relu4(x)

        return x

    def restore(self, ckpt_dir):
        try:
            if os.path.exists(os.path.join(ckpt_dir, 'pre.bin')):
                path = os.path.join(ckpt_dir, 'pre.bin')
            else:
                path = os.path.join(ckpt_dir, str(max(int(name) for name in os.listdir(ckpt_dir))), 'pre.bin')
        except:
            return None
        self.load_state_dict(torch.load(path))
        return os.path.split(path)[0]

    def save(self, ckpt_dir, global_step):
        os.makedirs(os.path.join(ckpt_dir, str(global_step)), exist_ok=True)
        path = os.path.join(ckpt_dir, str(global_step), 'pre.bin')
        torch.save(self.state_dict(), path)
        return os.path.split(path)[0]

