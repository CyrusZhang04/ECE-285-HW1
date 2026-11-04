"""Convolutional VAE for 128x128 grayscale chest X-rays."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def down(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 4, 2, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.2, inplace=True),
    )


def up(cin, cout):
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.2, inplace=True),
    )


class VAE(nn.Module):
    def __init__(self, latent_dim=128, img_size=128, ch=32, in_ch=1):
        super().__init__()
        self.latent_dim = latent_dim
        n_down = int(math.log2(img_size // 4))  # down to a 4x4 feature map
        chs = [in_ch] + [min(ch * 2 ** i, 512) for i in range(n_down)]
        self.enc = nn.Sequential(*[down(chs[i], chs[i + 1]) for i in range(n_down)])
        self.feat_ch = chs[-1]
        feat = self.feat_ch * 4 * 4
        self.fc_mu = nn.Linear(feat, latent_dim)
        self.fc_logvar = nn.Linear(feat, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, feat)
        rchs = chs[::-1]  # e.g. 512,256,128,64,32,1
        self.dec = nn.Sequential(*[up(rchs[i], rchs[i + 1]) for i in range(n_down - 1)])
        self.out = nn.ConvTranspose2d(rchs[-2], in_ch, 4, 2, 1)

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self.feat_ch, 4, 4)
        h = F.leaky_relu(h, 0.2)
        return torch.sigmoid(self.out(self.dec(h)))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device, temperature=1.0):
        z = torch.randn(n, self.latent_dim, device=device) * temperature
        return self.decode(z)


def vae_loss(x, x_hat, mu, logvar, recon="sigma", beta=1.0):
    """Negative ELBO per image (summed over pixels / latent dims, averaged over batch).

    recon="bce":   Bernoulli likelihood (binary cross-entropy).
    recon="mse":   Gaussian likelihood with fixed sigma^2 = 1/2 (i.e. summed squared error).
    recon="sigma": Gaussian likelihood with a shared variance set to its analytic optimum
                   sigma^2 = MSE (Rybkin et al., "Simple and Effective VAE Training with
                   Calibrated Decoders", 2021). This balances recon and KL automatically.
    """
    b = x.shape[0]
    if recon == "bce":
        rec = F.binary_cross_entropy(x_hat, x, reduction="sum") / b
    elif recon == "mse":
        rec = F.mse_loss(x_hat, x, reduction="sum") / b
    elif recon == "sigma":
        mse = F.mse_loss(x_hat, x, reduction="none")
        log_sigma = 0.5 * torch.log(mse.mean().detach().clamp_min(1e-6))
        rec = (0.5 * mse * torch.exp(-2 * log_sigma) + log_sigma + 0.5 * math.log(2 * math.pi)).sum() / b
    else:
        raise ValueError(recon)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
    return rec + beta * kl, rec, kl
