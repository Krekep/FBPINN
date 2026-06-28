import math

import numpy as np
import torch

from .phys_losses import PhysLoss


class LH_ODE_1(PhysLoss):
    def __init__(
        self,
        description: str = "",
        x_min: float = 0,
        x_max: float = math.pi * 2,
        device: str = "cpu",
        **kwargs,
    ):
        description = f"y'' + 100 y = 0, y(0) = 0, y'(0) = 10"
        super().__init__(description)
        self.device = device
        self.val_input = torch.tensor(
            np.linspace([x_min], [x_max], 100_000, dtype=np.float32),
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )

        self.full_losses = [self.phys_loss]
        # nn(x) -> y
        self.full_losses = [self.phys_loss]
        self.sub_losses = [
            (self.boundary_loss_1, [(0.0)]),  # u(0)
            (self.boundary_loss_2, [(0.0)]),  # u(0)
        ]

    def phys_loss(self, model: torch.nn.Module, x, active_models, **kwargs):
        """y'' + 100 y = 0"""
        u = model(x, active_indices=active_models)
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_model = u_xx + 100.0 * u
        phys_loss = torch.mean(torch.square(u_model))

        return phys_loss

    def boundary_loss_1(self, model: torch.nn.Module, x, active_models, **kwargs):
        """y(0) = 0"""
        x = torch.zeros_like(
            x, dtype=torch.float32, device=self.device, requires_grad=True
        )
        u_model = model(x, active_indices=active_models)
        phys_loss = torch.mean(torch.square(u_model))

        return phys_loss

    def boundary_loss_2(self, model: torch.nn.Module, x, active_models, **kwargs):
        """y'(0) = 10"""
        x = torch.zeros_like(
            x, dtype=torch.float32, device=self.device, requires_grad=True
        )

        u = model(x, active_indices=active_models)
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

        u_true = (
            torch.ones_like(
                x, dtype=torch.float32, device=self.device, requires_grad=False
            )
            * 10
        )
        phys_loss = torch.mean(torch.square(u_true - u_x))

        return phys_loss

    def update(self):
        pass

    def solution(self, x):
        return torch.sin(10 * x)
