import numpy as np
import torch

from .phys_losses import PhysLoss


def mse_zero(p):
    return torch.mean(torch.square(p))


class Poisson3D(PhysLoss):
    """
    3D Poisson equation on the unit cube [0,1]^3 with homogeneous Dirichlet BCs.

    Equation:
        -Δu = f(x, y, z),   (x,y,z) ∈ [0,1]^3
         u  = 0             on ∂Ω

    Solution:
        u*(x,y,z) = sin(π x) sin(π y) sin(π z)

    Right-hand side (substituted analytically):
        f(x,y,z) = 3π² sin(π x) sin(π y) sin(π z)

    The model should return u of shape (n, 1) or (n,).
    Input x_in has shape (n, 3): columns are [x, y, z].
    """

    def __init__(self, description: str = "", device=None, **kwargs):
        description = "3D Poisson: -Δu = 3π²sin(πx)sin(πy)sin(πz) on [0,1]^3, u|∂Ω=0"
        super().__init__(description)
        self.equation_class = "Poisson 3D"
        self.time_input = False
        self.device = device

        self.x_min, self.x_max = 0.0, 1.0
        self.y_min, self.y_max = 0.0, 1.0
        self.z_min, self.z_max = 0.0, 1.0

        self.u_scale = 1.0
        self.pde_scale = 3.0 * (torch.pi**2)

        self.loss_func = mse_zero

        self.full_losses = [self.phys_loss]
        self.sub_losses = [
            (
                self.boundary_loss_x0,
                [
                    (self.x_min, self.y_min, self.z_min),
                    (self.x_min, self.y_max, self.z_max),
                ],
            ),
            (
                self.boundary_loss_x1,
                [
                    (self.x_max, self.y_min, self.z_min),
                    (self.x_max, self.y_max, self.z_max),
                ],
            ),
            (
                self.boundary_loss_y0,
                [
                    (self.x_min, self.y_min, self.z_min),
                    (self.x_max, self.y_min, self.z_max),
                ],
            ),
            (
                self.boundary_loss_y1,
                [
                    (self.x_min, self.y_max, self.z_min),
                    (self.x_max, self.y_max, self.z_max),
                ],
            ),
            (
                self.boundary_loss_z0,
                [
                    (self.x_min, self.y_min, self.z_min),
                    (self.x_max, self.y_max, self.z_min),
                ],
            ),
            (
                self.boundary_loss_z1,
                [
                    (self.x_min, self.y_min, self.z_max),
                    (self.x_max, self.y_max, self.z_max),
                ],
            ),
        ]
        xs = np.linspace(self.x_min, self.x_max, 80, dtype=np.float32)
        ys = np.linspace(self.y_min, self.y_max, 80, dtype=np.float32)
        zs = np.linspace(self.z_min, self.z_max, 80, dtype=np.float32)
        xx, yy, zz = np.meshgrid(xs, ys, zs)
        self.val_input = np.stack(
            [xx.ravel(), yy.ravel(), zz.ravel()], axis=1
        )  # (N, 3)

    def phys_loss(self, model: torch.nn.Module, x_in, active_models, **kwargs):
        preds = model(x_in, active_indices=active_models)
        u = preds[:, 0]  # (n,)

        du = torch.autograd.grad(u.sum(), x_in, create_graph=True, retain_graph=True)[0]
        du_dx, du_dy, du_dz = du[:, 0], du[:, 1], du[:, 2]

        d2u_dx2 = torch.autograd.grad(
            du_dx.sum(), x_in, create_graph=True, retain_graph=True
        )[0][:, 0]
        d2u_dy2 = torch.autograd.grad(
            du_dy.sum(), x_in, create_graph=True, retain_graph=True
        )[0][:, 1]
        d2u_dz2 = torch.autograd.grad(du_dz.sum(), x_in, create_graph=True)[0][:, 2]

        x, y, z = x_in[:, 0], x_in[:, 1], x_in[:, 2]
        f = (
            self.pde_scale
            * torch.sin(torch.pi * x)
            * torch.sin(torch.pi * y)
            * torch.sin(torch.pi * z)
        )

        residual = -(d2u_dx2 + d2u_dy2 + d2u_dz2) - f

        loss = self.loss_func(residual)
        return loss

    def _bc_loss(self, model, x_in, active_models, fixed_axis: int, fixed_val: float):
        """
        Replace `fixed_axis` with `fixed_val`, calculate u and ensure that u = 0.
        fixed_axis: 0=x, 1=y, 2=z
        """
        coords = [x_in[:, i] for i in range(3)]
        coords[fixed_axis] = torch.full_like(coords[fixed_axis], fixed_val)
        x = torch.stack(coords, dim=1)

        preds = model(x, active_indices=active_models)
        u = preds[:, 0]

        return self.loss_func(u / self.u_scale)

    def boundary_loss_x0(self, model, x_in, active_models, **kwargs):
        """u(0, y, z) = 0"""
        return self._bc_loss(
            model, x_in, active_models, fixed_axis=0, fixed_val=self.x_min
        )

    def boundary_loss_x1(self, model, x_in, active_models, **kwargs):
        """u(1, y, z) = 0"""
        return self._bc_loss(
            model, x_in, active_models, fixed_axis=0, fixed_val=self.x_max
        )

    def boundary_loss_y0(self, model, x_in, active_models, **kwargs):
        """u(x, 0, z) = 0"""
        return self._bc_loss(
            model, x_in, active_models, fixed_axis=1, fixed_val=self.y_min
        )

    def boundary_loss_y1(self, model, x_in, active_models, **kwargs):
        """u(x, 1, z) = 0"""
        return self._bc_loss(
            model, x_in, active_models, fixed_axis=1, fixed_val=self.y_max
        )

    def boundary_loss_z0(self, model, x_in, active_models, **kwargs):
        """u(x, y, 0) = 0"""
        return self._bc_loss(
            model, x_in, active_models, fixed_axis=2, fixed_val=self.z_min
        )

    def boundary_loss_z1(self, model, x_in, active_models, **kwargs):
        """u(x, y, 1) = 0"""
        return self._bc_loss(
            model, x_in, active_models, fixed_axis=2, fixed_val=self.z_max
        )

    def update(self):
        pass

    def solution(self, inp):
        if isinstance(inp, np.ndarray):
            inp = torch.tensor(inp, dtype=torch.float32, device=self.device)
        x, y, z = inp[:, 0], inp[:, 1], inp[:, 2]
        u = torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.sin(torch.pi * z)
        return u.unsqueeze(1)  # (n, 1)
