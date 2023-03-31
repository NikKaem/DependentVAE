import torch
from models.base import BaseVAE
from torch import nn, tensor
from torch.nn import functional as F
from typing import List


class DVAE(BaseVAE):

    def __init__(self,
                 cov,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dim: int = 512) -> None:
        super(DVAE, self).__init__()

        self.latent_dim = latent_dim

        self.cov = torch.from_numpy(cov).type(torch.FloatTensor).to('cuda')

        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim))
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(nn.Linear(latent_dim, input_dim))


    def encode(self, input: tensor) -> List[tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: tensor) -> tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        return result

    def reparameterize(self, mu: tensor, logvar: tensor) -> tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        epsilon = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(len(mu)*self.latent_dim).type(torch.FloatTensor).to('cuda'),
            torch.kron(self.cov, torch.diag(torch.ones(self.latent_dim)).type(torch.FloatTensor).to('cuda'))
        )

        z = mu.flatten() + epsilon.sample() * torch.exp(logvar).flatten()

        return z.reshape([len(mu), self.latent_dim])

    def forward(self, input: tensor, **kwargs) -> List[tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        logvar = args[3]

        """epsilon = torch.diag(torch.diag(logvar.flatten())[:self.latent_dim, :self.latent_dim]/self.cov[0,0])"""

        recons_loss = F.mse_loss(recons, input)

        kld_loss = 0.5*(torch.neg(
            torch.sum(
                torch.log(
                    torch.square(
                        logvar.flatten()
                    )
                )
            )
        ) - len(mu) + self.latent_dim*torch.sum(
            torch.square(
                logvar.flatten()
            )
        ) + torch.matmul(
            torch.matmul(
                mu.flatten(), torch.inverse(
                    torch.kron(
                        self.cov,
                        torch.diag(
                            torch.ones(
                                self.latent_dim
                            ).type(torch.FloatTensor).to('cuda')
                        )
                    )
                )
            ),
            mu.flatten()
        ))

        kld_weight = kwargs['M_N']
        loss = recons_loss + kld_weight * kld_loss

        print(recons_loss.detach())
        print(kld_loss.detach())

        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': kld_loss.detach()}

    def sample(self,
               num_samples: int, **kwargs) -> tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        samples = self.decode(z)
        return samples
