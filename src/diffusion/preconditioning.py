from abc import abstractmethod

import torch
import torch.nn as nn


class Preconditiong(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @abstractmethod
    def c_skip(self, sigma):
        pass

    @abstractmethod
    def c_out(self, sigma):
        pass

    @abstractmethod
    def c_in(self, sigma):
        pass

    @abstractmethod
    def c_noise(self, sigma):
        pass

    def forward(self, x, sigma, heights, *args, **kwargs):
        if sigma.dim() == 1:
            sigma = sigma[:, None, None, None, None]
        assert sigma.dim() == 5

        c_skip = self.c_skip(sigma)
        c_out = self.c_out(sigma)
        c_in = self.c_in(sigma)
        c_noise = self.c_noise(sigma)

        F_x = self.model(c_in * x,
                         c_noise.flatten(),
                         heights.flatten() if heights is not None else None,
                         *args, **kwargs)
        D_x = c_skip * x + c_out * F_x
        return D_x


class InitialPreconditioning(Preconditiong):
    def __init__(self, model):
        super().__init__(model)

    def c_skip(self, sigma):
        return torch.zeros_like(sigma)

    def c_out(self, sigma):
        return torch.ones_like(sigma)

    def c_in(self, sigma):
        return torch.ones_like(sigma)

    def c_noise(self, sigma):
        return sigma.log()


class OldNoisePreconditioning(Preconditiong):
    def __init__(self, model):
        super().__init__(model)

    def c_skip(self, sigma):
        return torch.ones_like(sigma)

    def c_out(self, sigma):
        return -sigma

    def c_in(self, sigma):
        return 1

    def c_noise(self, sigma):
        return sigma.log()


class NoisePreconditioning(Preconditiong):
    def __init__(self, model):
        super().__init__(model)

    def c_skip(self, sigma):
        return torch.ones_like(sigma)

    def c_out(self, sigma):
        return -sigma

    def c_in(self, sigma):
        return 1 / torch.sqrt(sigma ** 2 + 1)

    def c_noise(self, sigma):
        return sigma.log()


class EDMPreconditioning(Preconditiong):
    def __init__(self, model, sigma_data):
        super().__init__(model)

        self.sigma_data = sigma_data

    def c_skip(self, sigma):
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    def c_in(self, sigma):
        return 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()

    def c_noise(self, sigma):
        return sigma.log() / 4
