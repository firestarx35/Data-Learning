# 1 Convolutional autoencoder

import torch.nn as nn
import torch.nn.functional as F
import torch

types = {1: 'convolutional1', 2: 'variational1', 3: 'variational2'}


def select_model(type=1, latent_dim=None, param2=None):
    if (type == 1):
        model = ConvolutionalAE1()
        loss_function = nn.MSELoss()
    elif (type == 2):
        model = ConvolutionalAE2(latent_dim=latent_dim)
        loss_function = nn.MSELoss()
    elif (type == 3):
        model = VariationalAE1(latent_dim=latent_dim)
        loss_function = loss_function_variational
    elif (type == 4):
        model = VariationalAE2(latent_dim=latent_dim)
        loss_function = loss_function_variational
    return model, loss_function


class ConvolutionalAE2(nn.Module):
    def __init__(self, latent_dim):
        super(ConvolutionalAE2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=3, padding=0),
            nn.ReLU(),
        )
        self.flatten_enc = nn.Sequential(
            nn.Linear(32 * 11 * 11, 750),
            nn.Linear(750, latent_dim)
        )
        self.deflatten_dec = nn.Sequential(
            nn.Linear(latent_dim, 750),
            nn.Linear(750, 32 * 11 * 11)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=3),
            # nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(16, 8, kernel_size=1, stride=1),
            # nn.BatchNorm2d(8), 
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(8, 4, kernel_size=1, stride=2),
            # nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=2, stride=1),
            # nn.Softmax2d()
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), 32 * 11 * 11)
        x = self.flatten_enc(x)
        return x

    def decode(self, x):
        x = self.deflatten_dec(x)
        x = x.view(x.size(0), 32, 11, 11)
        x = self.decoder(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class ConvolutionalAE1(nn.Module):
    def __init__(self):
        super(ConvolutionalAE1, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3,
                               stride=2, output_padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Variational autoencoders

class Encoder1(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc_mu = nn.Linear(32 * 180 * 250, latent_dim)
        self.fc_logvar = nn.Linear(32 * 180 * 250, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 180 * 250)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 32 * 90 * 125)
        self.conv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.ConvTranspose2d(
            16, 1, kernel_size=2, stride=1, padding=1)

    def forward(self, x):
        batch_size = x.shape[0]
        # print(batch_size)
        x = self.fc(x)
        x = x.view(batch_size, 32, 90, 125)
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x


class VariationalAE1(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAE1, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder1(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        # print(z.shape)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


def loss_function_variational(x_hat, x, mu, logvar):
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# model = VariationalAE(latent_dim=15)
# model = model.float().to(device)

# dats = iter(dataloader)
# input1 = next(dats)
# print("Input shape: ", input1.shape)

# a, b, c = model(input1.to(device))
# print("Ouput shape: ", a.shape)


class Encoder2(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder2, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            # batch_size, 16, 180, 250
            nn.MaxPool2d(kernel_size=2, stride=2),
            # batch_size, 16, 90, 125
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # batch_size, 32, 90, 125
            nn.MaxPool2d(kernel_size=2, stride=2),
            # batch_size 32, 45, 62
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32 * 45 * 62, latent_dim)
        self.fc_logvar = nn.Linear(32 * 45 * 62, latent_dim)

    def forward(self, x):
        x = self.encoder1(x)
        x = x.view(-1, 32 * 45 * 62)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class VariationalAE2(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAE2, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder2(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        # print(z.shape)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
