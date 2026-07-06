import numpy as np
import torch

from .cylinder_inviscid import mse_zero
from .phys_losses import PhysLoss


class NoObstacleInviscid(PhysLoss):
    """
    Ideal fluid flow without a cylinder. Stationary inviscid 2D case, rho = const.
    The model should return (p, vx, vy)

    Equations:
    continuity: dvx/dx + dvy/dy = 0
    x-momentum: vx*dvx/dx + vy*dvx/dy + (1/rho) * dp/dx = 0
    y-momentum: vx*dvy/dx + vy*dvy/dy + (1/rho) * dp/dy = 0.

    Exact solution:
        vx(x, y) = v_inf = const
        vy(x, y) = 0
        p(x, y) = 0
    """

    def __init__(
        self,
        scale: float = 0.001,
        rho: float = 1.225,
        v_inf: float = 0.0075,
        L: float = 0.1,
        device=None,
        **kwargs
    ):
        super().__init__("Steady 2D Euler: no obstacle, uniform flow")
        self.equation_class = "Euler (inviscid) 2D — no obstacle"
        self.time_input = False

        self.rho = torch.tensor(rho, dtype=torch.float32)
        self.v_inf = v_inf
        self.L = L
        self.scale = scale
        self.device = device

        self.continuity = v_inf / L
        self.momentum = v_inf**2 / L
        self.p_scale = float(self.rho) * v_inf**2

        self.x_min = 0.0
        self.x_max = 2000 * scale
        self.y_min = 0.0
        self.y_max = 1200 * scale

        self._build_val_input(nx=100, ny=60)

        self.sub_losses = [
            (
                self.boundary_loss_1,
                [(self.x_min, self.y_min), (self.x_min, self.y_max)],
            ),
            (
                self.boundary_loss_2,
                [(self.x_max, self.y_min), (self.x_max, self.y_max)],
            ),
            (
                self.boundary_loss_3,
                [(self.x_min, self.y_max), (self.x_max, self.y_max)],
            ),
            (
                self.boundary_loss_4,
                [(self.x_min, self.y_min), (self.x_max, self.y_min)],
            ),
        ]
        self.loss_func = mse_zero

    def phys_loss(self, model, x_in, active_models, **kwargs):
        preds = model(x_in, active_indices=active_models)
        p = torch.squeeze(preds[:, 0:1], dim=-1)
        vx = torch.squeeze(preds[:, 1:2], dim=-1)
        vy = torch.squeeze(preds[:, 2:3], dim=-1)

        dp = torch.autograd.grad(p.sum(), x_in, create_graph=True)[0]
        dvx = torch.autograd.grad(vx.sum(), x_in, create_graph=True)[0]
        dvy = torch.autograd.grad(vy.sum(), x_in, create_graph=True)[0]

        dp_dx, dp_dy = dp[:, 0], dp[:, 1]
        dvx_dx, dvx_dy = dvx[:, 0], dvx[:, 1]
        dvy_dx, dvy_dy = dvy[:, 0], dvy[:, 1]

        vorticity = dvy_dx - dvx_dy
        continuity = dvx_dx + dvy_dy
        mom_x = vx * dvx_dx + vy * dvx_dy + (1.0 / self.rho) * dp_dx
        mom_y = vx * dvy_dx + vy * dvy_dy + (1.0 / self.rho) * dp_dy

        return (
            self.loss_func(continuity / self.continuity)
            + self.loss_func(vorticity / self.continuity)
            + self.loss_func(mom_x / self.momentum)
            + self.loss_func(mom_y / self.momentum)
        )

    def boundary_loss_1(self, model, x_in, active_models, **kwargs):
        """(x=0): vx = 0.0075, vy = 0, p' = 0"""
        y = x_in[:, 1]
        x = torch.zeros_like(y)
        pts = torch.stack([x, y], dim=1)

        preds = model(pts, active_indices=active_models)
        p = preds[:, 0:1]
        vx = preds[:, 1:2]
        vy = preds[:, 2:3]

        return (
            self.loss_func((vx - self.v_inf) / self.v_inf)
            + self.loss_func(vy / self.v_inf)
            + self.loss_func(p / self.p_scale)
        )

    def boundary_loss_2(self, model, x_in, active_models, **kwargs):
        """(x=x_max): p' = 0"""
        y = x_in[:, 1]
        x = torch.ones_like(y) * self.x_max
        pts = torch.stack([x, y], dim=1)

        preds = model(pts, active_indices=active_models)
        p = torch.squeeze(preds[:, 0:1], dim=-1)

        return self.loss_func(p / self.p_scale)

    def boundary_loss_3(self, model, x_in, active_models, **kwargs):
        """(y=y_max): vy=0, dvx/dy=0, dp/dy=0"""
        x = x_in[:, 0].detach()
        y = torch.ones_like(x) * self.y_max
        pts = torch.stack([x, y], dim=1)
        pts.requires_grad_(True)

        preds = model(pts, active_indices=active_models)
        vx = preds[:, 1:2]
        vy = preds[:, 2:3]
        p = preds[:, 0:1]

        dvx_dy = torch.autograd.grad(vx, pts, torch.ones_like(vx), create_graph=True)[
            0
        ][:, 1:2]
        dp_dy = torch.autograd.grad(p, pts, torch.ones_like(p), create_graph=True)[0][
            :, 1:2
        ]

        return (
            self.loss_func(vy / self.v_inf)
            + self.loss_func(dvx_dy / self.continuity)
            + self.loss_func(dp_dy / (self.p_scale / self.L))
        )

    def boundary_loss_4(self, model, x_in, active_models, **kwargs):
        """(y=0): vy=0, dvx/dy=0, dp/dy=0"""
        x = x_in[:, 0].detach()
        y = torch.zeros_like(x)
        pts = torch.stack([x, y], dim=1)
        pts.requires_grad_(True)

        preds = model(pts, active_indices=active_models)
        vx = preds[:, 1:2]
        vy = preds[:, 2:3]
        p = preds[:, 0:1]

        dvx_dy = torch.autograd.grad(vx, pts, torch.ones_like(vx), create_graph=True)[
            0
        ][:, 1:2]
        dp_dy = torch.autograd.grad(p, pts, torch.ones_like(p), create_graph=True)[0][
            :, 1:2
        ]

        return (
            self.loss_func(vy / self.v_inf)
            + self.loss_func(dvx_dy / self.continuity)
            + self.loss_func(dp_dy / (self.p_scale / self.L))
        )

    def _build_val_input(self, nx: int, ny: int):
        xs = np.linspace(self.x_min, self.x_max, nx, dtype=np.float32)
        ys = np.linspace(self.y_min, self.y_max, ny, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        self.val_input = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (N, 2)

    def update(self):
        pass

    def solution(self, x_in: np.ndarray) -> np.ndarray:
        """
        Exact solution for x_in (N, 2).
        """
        n = x_in.shape[0]
        p = np.zeros(n, dtype=np.float32)
        vx = np.full(n, self.v_inf, dtype=np.float32)
        vy = np.zeros(n, dtype=np.float32)
        return np.stack([p, vx, vy], axis=1)  # (N, 3)
