import numpy as np
import pandas as pd
import torch
import keras

from src.geometry.geometry import sample_points_on_boundary
from .phys_losses import PhysLoss


def mse_zero(p):
    return keras.ops.mean(keras.ops.square(p))


class CylinderViscid(PhysLoss):
    """
    Ideal fluid flow around a cylinder. Stationary viscid 2D case, rho = const, mu = const.
    The model should return (p, vx, vy)

    Equations:
    continuity: dvx/dx + dvy/dy = 0
    x-momentum: vx*dvx/dx + vy*dvx/dy + (1/rho) * dp/dx = 0
    y-momentum: vx*dvy/dx + vy*dvy/dy + (1/rho) * dp/dy = 0

    Стационарный вязкий (Navier-Stokes) 2D случай, rho = const, mu = const.
    Модель должна возвращать (p, vx, vy) в таком порядке.
    Уравнения (в форме residual == 0):
      continuity:  dvx/dx + dvy/dy = 0
      x-momentum:  vx*dvx/dx + vy*dvx/dy + (1/rho)*dp/dx - nu*(d2vx/dx2 + d2vx/dy2) = 0
      y-momentum:  vx*dvy/dx + vy*dvy/dy + (1/rho)*dp/dy - nu*(d2vy/dx2 + d2vy/dy2) = 0
    x_in: tensor shape (n, 2) columns [x, y]
    """

    def __init__(
            self,
            description: str = "",
            rho: float = 1.225,
            mu: float = 1.7894e-5,
            scale=0.001,
            cylinder=None,
            v_inf: float = 0.075,  # характерная скорость (входная)
            L: float = 0.1,
            device=None,
            path_to_data: str = "./",
            **kwargs
    ):
        description = "Steady 2D Navier-Stokes: continuity + momentum, rho = const, mu = const"
        super().__init__(description)
        self.equation_class = "Navier-Stokes (viscous) 2D"
        self.time_input = False
        self.cylinder = cylinder

        self.R = 100 * scale  # m
        self.rho = torch.tensor(rho, dtype=torch.float32)  # kg/m^3
        self.mu = torch.tensor(mu, dtype=torch.float32)  # kg/(m*s)
        self.nu = mu / rho  # m^2/s — кинематическая вязкость
        self.v_inf = v_inf  # m/s
        self.L = L  # m

        self.continuity = v_inf / L  # [1/s] ~ 0.75
        self.momentum = v_inf ** 2 / L  # [m/s^2] ~ 0.05625
        # Вязкий масштаб: nu * v_inf / L^2  [m/s^2] — совпадает по размерности с momentum,
        # поэтому оба члена нормируются на self.momentum.
        # Для справки: viscous / momentum = nu / (v_inf * L) = 1/Re_L
        self.p_scale = rho * self.v_inf ** 2  # [Pa]

        self.x_min = 0.0  # m
        self.x_max = 2000 * scale  # m
        self.y_min = 0.0  # m
        self.y_max = 1200 * scale  # m
        self.scale = scale
        self._load_dataset(path_to_data + "cylinder_viscid.csv")

        self.cylinder_points = torch.tensor(
            sample_points_on_boundary(self.cylinder, [], 2000),
            device=device,
            dtype=torch.float32
        )

        self.full_losses = [self.phys_loss]
        self.sub_losses = [
            (self.boundary_loss_1, [(self.x_min, self.y_min), (self.x_min, self.y_max)]),
            (self.boundary_loss_2, [(self.x_max, self.y_min), (self.x_max, self.y_max)]),
            (self.boundary_loss_3, [(self.x_min, self.y_max), (self.x_max, self.y_max)]),
            (self.boundary_loss_4, [(self.x_min, self.y_min), (self.x_max, self.y_min)]),
            (self.boundary_loss_cylinder, self.cylinder),
        ]
        self.loss_func = mse_zero

    def phys_loss(self, model: keras.Model, x_in, active_models, **kwargs):
        preds = model(x_in, active_models=active_models)
        p = keras.ops.squeeze(preds[:, 0:1], axis=-1)  # (n,)
        vx = keras.ops.squeeze(preds[:, 1:2], axis=-1)
        vy = keras.ops.squeeze(preds[:, 2:3], axis=-1)

        dp = torch.autograd.grad(p.sum(), x_in, create_graph=True)[0]
        dvx = torch.autograd.grad(vx.sum(), x_in, create_graph=True)[0]
        dvy = torch.autograd.grad(vy.sum(), x_in, create_graph=True)[0]
        dp_dx, dp_dy = dp[:, 0], dp[:, 1]  # Pa/m
        dvx_dx, dvx_dy = dvx[:, 0], dvx[:, 1]  # 1/s
        dvy_dx, dvy_dy = dvy[:, 0], dvy[:, 1]  # 1/s

        d2vx_dx2 = torch.autograd.grad(dvx_dx.sum(), x_in, create_graph=True)[0][:, 0]  # 1/(m*s)
        d2vx_dy2 = torch.autograd.grad(dvx_dy.sum(), x_in, create_graph=True)[0][:, 1]  # 1/(m*s)
        d2vy_dx2 = torch.autograd.grad(dvy_dx.sum(), x_in, create_graph=True)[0][:, 0]  # 1/(m*s)
        d2vy_dy2 = torch.autograd.grad(dvy_dy.sum(), x_in, create_graph=True)[0][:, 1]  # 1/(m*s)

        # Лапласианы скоростей: nu * lap(v) [m/s^2]
        lap_vx = self.nu * (d2vx_dx2 + d2vx_dy2)
        lap_vy = self.nu * (d2vy_dx2 + d2vy_dy2)

        # Невязки уравнений
        continuity = dvx_dx + dvy_dy  # 1/s
        mom_x = vx * dvx_dx + vy * dvx_dy + (1.0 / self.rho) * dp_dx - lap_vx  # m/s^2
        mom_y = vx * dvy_dx + vy * dvy_dy + (1.0 / self.rho) * dp_dy - lap_vy  # m/s^2

        loss_continuity = self.loss_func(continuity / self.continuity)
        loss_mx = self.loss_func(mom_x / self.momentum)
        loss_my = self.loss_func(mom_y / self.momentum) * 100

        phys_loss = loss_continuity + loss_mx + loss_my
        return phys_loss

    def boundary_loss_1(self, model: keras.Model, x_in, active_models, **kwargs):
        """v_x(0.0, y) = 0.0075, v_y(0.0, y) = 0, p'(0.0, y) = 0"""
        y_wout_x = x_in[:, 1]
        x = keras.ops.zeros_like(y_wout_x)
        x = keras.ops.stack([x, y_wout_x], axis=1)

        preds = model(x, active_models=active_models)
        p = preds[:, 0:1]  # (n,1)
        vx = preds[:, 1:2]
        vy = preds[:, 2:3]

        truth_vx = keras.ops.ones_like(vx) * self.v_inf
        truth_vy = keras.ops.zeros_like(vy)
        b_loss = (
                self.loss_func((truth_vx - vx) / self.v_inf) +
                self.loss_func(vy / self.v_inf) +
                self.loss_func(p / self.p_scale)
        )
        return b_loss

    def boundary_loss_2(self, model: keras.Model, x_in, active_models, **kwargs):
        """p'(2.0, y) = 0"""
        y_wout_x = x_in[:, 1]
        x = keras.ops.ones_like(y_wout_x) * (2000 * self.scale)
        x = keras.ops.stack([x, y_wout_x], axis=1)

        preds = model(x, active_models=active_models)
        p = keras.ops.squeeze(preds[:, 0:1], axis=-1)  # (n,)
        b_loss = (
            self.loss_func(p / self.p_scale)
        )
        return b_loss

    def boundary_loss_3(self, model: keras.Model, x_in, active_models, **kwargs):
        """(y=1.2): v_y = 0, ∂v_x/∂y = 0, ∂p'/∂y = 0"""
        x_wout_y = x_in[:, 0]
        y = keras.ops.ones_like(x_wout_y) * self.y_max
        x = torch.stack([x_wout_y, y], dim=1)
        x.requires_grad_(True)

        preds = model(x, active_models=active_models)
        vx = preds[:, 1:2]
        vy = preds[:, 2:3]
        p = preds[:, 0:1]
        dvx_dy = torch.autograd.grad(vx, x, torch.ones_like(vx), create_graph=True)[0][:, 1:2]
        dp_dy = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0][:, 1:2]

        b_loss = (
                self.loss_func(vy / self.v_inf) +
                self.loss_func(dvx_dy / self.continuity) +  # ∂v_x/∂y = 0
                self.loss_func(dp_dy / (self.p_scale / self.L))  # ∂p'/∂y = 0 -- Pa/m
        )
        return b_loss

    def boundary_loss_4(self, model: keras.Model, x_in, active_models, **kwargs):
        """(y=0): v_y = 0, ∂v_x/∂y = 0, ∂p'/∂y = 0"""
        x_wout_y = x_in[:, 0]
        y = keras.ops.zeros_like(x_wout_y)
        x = torch.stack([x_wout_y, y], dim=1)
        x.requires_grad_(True)

        preds = model(x, active_models=active_models)
        vx = preds[:, 1:2]
        vy = preds[:, 2:3]
        p = preds[:, 0:1]
        dvx_dy = torch.autograd.grad(vx, x, torch.ones_like(vx), create_graph=True)[0][:, 1:2]
        dp_dy = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0][:, 1:2]

        b_loss = (
                self.loss_func(vy / self.v_inf) +
                self.loss_func(dvx_dy / self.continuity) +  # ∂v_x/∂y = 0
                self.loss_func(dp_dy / (self.p_scale / self.L))  # ∂p'/∂y = 0
        )
        return b_loss

    def boundary_loss_cylinder(self, model: keras.Model, x_in, active_models, **kwargs):
        """v_x = 0, v_y = 0 (условие прилипания — no-slip)"""
        x_in = self.cylinder_points

        preds = model(x_in, active_models=active_models)
        vx = preds[:, 1]  # (n,)
        vy = preds[:, 2]  # (n,)

        b_loss = (
                self.loss_func(vx / self.v_inf) +
                self.loss_func(vy / self.v_inf)
        )
        return b_loss

    def _load_dataset(self, path_to_data: str):
        df = pd.read_csv(path_to_data, header=None, sep=",", skipinitialspace=True, skiprows=1,
                         names=["cell", "x", "y", "p", "vx", "vy"], dtype=float)
        x = df["x"].values.astype(dtype=np.float32)
        y = df["y"].values.astype(dtype=np.float32)
        p = df["p"].values.astype(dtype=np.float32)
        vx = df["vx"].values.astype(dtype=np.float32)
        vy = df["vy"].values.astype(dtype=np.float32)
        x_offset = 0.2
        y_offset = 0.6

        self.fx = dict()
        val_input = []
        for i in range(len(x)):
            _x = x[i] + x_offset
            _y = y[i] + y_offset
            _p = p[i]
            _vx = vx[i]
            _vy = vy[i]
            self.fx[(_x, _y)] = (_p, _vx, _vy)
            val_input.append([_x, _y])
        self.val_input = np.array(val_input, dtype=np.float32)
        assert min(self.val_input[:, 0]) >= 0.0
        assert max(self.val_input[:, 0]) <= 2000 * self.scale  # x coord
        assert min(self.val_input[:, 1]) >= 0.0
        assert min(self.val_input[:, 1]) <= 1200 * self.scale

    def solution(self, inp):
        res = []
        for x, y in inp:
            res.append(self.fx[x, y])
        res = keras.ops.convert_to_tensor(res, dtype=torch.float32)
        res.requires_grad_(False)
        return res
