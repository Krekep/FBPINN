import math

import numpy as np
import torch

from .phys_losses import PhysLoss


class LF_ODE_1(PhysLoss):
    def __init__(
        self,
        description: str = "",
        omega: float = 10,
        x_min: float = 0,
        x_max: float = math.pi * 2,
        device: str = "cpu",
        **kwargs,
    ):
        description = f"du/dx = {omega} * cos({omega}*x), y(0) = 0"
        self.omega = omega
        self.device = device
        super().__init__(description)
        self.val_input = torch.tensor(
            np.linspace([x_min], [x_max], 100_000, dtype=np.float32),
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )

        self.full_losses = [self.phys_loss]
        self.sub_losses = [
            (self.boundary_loss_1, [(0.0)]),
        ]

    def phys_loss(self, model: torch.nn.Module, x, active_models, **kwargs):
        """du/dx = omega * cos(omega*x)"""
        u = model(x, active_indices=active_models)
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_model = u_x - self.omega * torch.cos(self.omega * x)
        u_true = torch.zeros_like(u_model)
        phys_loss = torch.mean(torch.square(u_true - u_model))

        return phys_loss

    def boundary_loss_1(self, model: torch.nn.Module, x, active_models, **kwargs):
        """y(0) = 0"""
        x = torch.zeros_like(
            x, dtype=torch.float32, device=self.device, requires_grad=True
        )
        u_model = model(x, active_indices=active_models)
        u_true = torch.zeros_like(u_model)
        phys_loss = torch.mean(torch.square(u_true - u_model))

        return phys_loss

    def update(self):
        pass

    def solution(self, x):
        return torch.sin(self.omega * x)
