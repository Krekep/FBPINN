import math

import numpy as np
import torch

from .phys_losses import PhysLoss


class LH_PDE1(PhysLoss):
    def __init__(
            self,
            description: str = "",
            a: float = 1,
            end_time: float = 1,
            device="cpu",
            **kwargs
    ):
        description = "d^2u/dt^2 - 4 * d^2u/dx^2 = 0, u(0, x) = sin(pi * x) + 1/2 * sin(4 * pi * x), u_t(0, x) = 0, u(t, 0) = 0, u(t, 1) = 0"
        self.equation_class = "Hyperbolic PDE"
        self.a = torch.tensor(a, dtype=torch.float32)
        self.t_min = 0.0
        self.t_max = end_time
        self.x_min = 0.0
        self.x_max = 1.0
        self.device = device
        super().__init__(description)

        # nn(t, x) -> y
        self.full_losses = [self.phys_loss]
        self.sub_losses = [
            (self.boundary_loss_1, [(0.0, 0.0), (0.0, 1.0)]),  # u(0, x)
            (self.boundary_loss_2, [(0.0, 0.0), (0.0, 1.0)]),  # u(0, x)
            (self.boundary_loss_3, [(0.0, 0.0), (end_time, 0.0)]),  # u(t, 0)
            (self.boundary_loss_4, [(0.0, 1.0), (end_time, 1.0)]),  # u(t, 1)
        ]

        self._build_data(nt=200, nx=200)

    def phys_loss(self, model: torch.nn.Module, x_in, active_models, **kwargs):
        """d^2u/dt^2 - 4 * d^2u/dx^2 = 0"""
        u = model(x_in, active_models=active_models)

        u_grad = torch.autograd.grad(u.sum(), x_in, create_graph=True)[0]
        u_t, u_x = u_grad[:, 0], u_grad[:, 1]
        u_tt = torch.autograd.grad(u_t.sum(), x_in, create_graph=True)[0][:, 0]
        u_xx = torch.autograd.grad(u_x.sum(), x_in, create_graph=True)[0][:, 1]

        u_model = u_tt - 4.0 * u_xx
        phys_loss = torch.mean(torch.square(u_model))
        return phys_loss

    def boundary_loss_1(self, model: torch.nn.Module, x_in, active_models, **kwargs):
        """u(0, x) = sin(pi * x) + 1/2 * sin(4 * pi * x)"""
        x_wout_t = x_in[:, 1:2]
        t = torch.zeros_like(x_wout_t)
        x = torch.cat([t, x_wout_t], dim=1)

        u_model = model(x, active_models=active_models)

        u_true = torch.sin(math.pi * x_wout_t) + 1 / 2 * torch.sin(4 * math.pi * x_wout_t)
        diff = u_true - u_model
        phys_loss = torch.mean(torch.square(diff))

        return phys_loss

    def boundary_loss_2(self, model: torch.nn.Module, x_in, active_models, **kwargs):
        """u_t(0, x) = 0"""
        x_wout_t = x_in[:, 1]
        t = torch.zeros_like(x_wout_t)
        x = torch.stack([t, x_wout_t], dim=1)

        u_model = model(x, active_models=active_models)
        u_t = torch.autograd.grad(u_model.sum(), x, create_graph=True)[0][:, 0]

        return torch.mean(torch.square(u_t))

    def boundary_loss_3(self, model: torch.nn.Module, x_in, active_models, **kwargs):
        """u(t, 0) = 0"""
        t_wout_x = x_in[:, 0]
        x_zeros = torch.zeros_like(t_wout_x)
        x = torch.stack([t_wout_x, x_zeros], dim=1)

        u_model = model(x, active_models=active_models)

        return torch.mean(torch.square(u_model))

    def boundary_loss_4(self, model: torch.nn.Module, x_in, active_models, **kwargs):
        """u(t, 1) = 0"""
        t_wout_x = x_in[:, 0]
        x_ones = torch.ones_like(t_wout_x)
        x = torch.stack([t_wout_x, x_ones], dim=1)

        u_model = model(x, active_models=active_models)
        return torch.mean(torch.square(u_model))

    def _build_data(self, nt: int, nx: int):
        ts = np.linspace(self.t_min, self.t_max, nt, dtype=np.float32)
        xs = np.linspace(self.x_min, self.x_max, nx, dtype=np.float32)
        tt, xx = np.meshgrid(ts, xs)
        tt = torch.tensor(tt.ravel(), device=self.device, requires_grad=False)
        xx = torch.tensor(xx.ravel(), device=self.device, requires_grad=False)
        self.val_input = torch.stack([tt, xx], dim=1)

    def solution(self, x_in):
        t = x_in[:, 0:1]
        x = x_in[:, 1:2]
        res = torch.sin(math.pi * x) * torch.cos(2 * math.pi * t) + 1 / 2 * torch.sin(
            4 * math.pi * x
        ) * torch.cos(8 * math.pi * t)
        return res

    def first_der(self, var, t, x):
        if var == "x":
            return self.first_der_x(t, x)
        elif var == "t":
            return self.first_der_t(t, x)
        else:
            raise NotImplementedError()

    def first_der_x(self, t, x):
        return 2 * x

    def first_der_t(self, t, x):
        return 2 * t
