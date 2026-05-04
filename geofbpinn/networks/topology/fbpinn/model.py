from typing import List, Optional, Callable, Union

import torch

from geofbpinn.geometry import Block
from geofbpinn.geometry.base_decomposition import BaseDecomposition
from geofbpinn.geometry.geometry import polygon_intersects_rectangle
from geofbpinn.networks import losses, metrics, optimizers, get_activation
from geofbpinn.networks.topology.base.dense import DenseNet


class FBPINN(torch.nn.Module):
    """
    Implementation of FBPINN model on PyTorch.

    For detailed description see
    1. https://github.com/benmoseley/FBPINNs
    2. Finite Basis Physics-Informed Neural Networks (FBPINNs):
    a scalable domain decomposition approach for solving differential equations,
    B. Moseley, T. Nissen-Meyer and A. Markham, Jul 2023 Advances in Computational Mathematics
    https://doi.org/10.1007/s10444-023-10065-9
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        decomposition: BaseDecomposition,
        physic_loss: Callable,
        boundary_loss: list[tuple[Callable, list[tuple[float, ...]]]],
        activation_func: str | list[str] = "linear",
        weight: Callable = torch.nn.init.normal,
        biases: Callable = torch.nn.init.zeros_,
        models_size: list[int] = [10],
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        input_size: int
            Size of input vector
        output_size: int
            Size of output vector
        decomposition: BaseDecomposition
            Class representing domain decomposition
        physic_loss: Callable
            Main physic loss
        boundary_loss: list[tuple[Callable, list[tuple[float, ...]]]]
            List of sublosses in form of `loss_func: points of area polygon`
        activation_func: str | list[str]
            List of activation func per layer
        weight: Callable
            Function for initialize weights of submodels
        biases: Callable
            Function for initialize weights of biases
        models_size: list[int]
            Number of neurons per layer for submodel
        device: Optional[str]
            Device `cpu`/`cuda`
        """
        super(FBPINN, self).__init__(**kwargs)
        self.networks: list
        self.device = device

        self.physic_loss: Callable = physic_loss

        self.decomposition = decomposition

        networks = []
        for i, block in enumerate(self.decomposition.blocks):
            nn = DenseNet(
                input_size=input_size,
                block_size=models_size,
                output_size=output_size,
                activation_func=activation_func,
                weight=weight,
                biases=biases,
                domain_block=block,
                device=device,
            )
            networks.append(nn)
        self.networks = torch.nn.ModuleList(networks)
        self.blocks: list[tuple[DenseNet, Block]] = list(
            zip(networks, self.decomposition.blocks)
        )
        self.input_size = input_size
        self.output_size = output_size

        # [(Loss, [index Model1, index Model2, ...]), (Other loss, [index Model1, index Model5, ...])]
        self.model_per_loss: list[tuple[Callable, list[int]]] = []
        for loss, polygon in boundary_loss + [(physic_loss, None)]:
            affected = [
                i
                for i, (_, block) in enumerate(self.blocks)
                if polygon_intersects_rectangle(
                    polygon, block.left_down_corner + block.right_up_corner
                )
            ]
            self.model_per_loss.append((loss, affected))

        self._build_batched_forward()

    def _build_batched_forward(self) -> None:
        """
        Stack submodel parameters into batched tensors for efficient CUDA utilization.

        Since each submodel is small, combining their weights into a single batched
        tensor allows processing all submodels simultaneously via `torch.bmm`,
        significantly improving GPU throughput.

        After calling this method the following attributes are populated:

        - `all_vmins`, `all_vmaxs` — normalization bounds, shape `(N, d)`.
        - `all_scales`, `all_shifts` — output denormalization params, shape `(N, d_out)`.
        - `stacked_w` — list of weight tensors `(N, in_i, out_i)` per layer.
        - `stacked_b` — list of bias tensors `(N, out_i)` per layer.
        - `layer_activations`` — list of activation function names per layer.
        """
        vmins, vmaxs, scales, shifts = [], [], [], []

        for nn, block in self.blocks:
            vmins.append(block.vmin)
            vmaxs.append(block.vmax)
            scales.append(block.out_denorm_scale)
            shifts.append(block.out_denorm_shift)

        self.all_vmins = torch.stack(vmins)
        self.all_vmaxs = torch.stack(vmaxs)
        self.all_scales = torch.stack(scales)
        self.all_shifts = torch.stack(shifts)

        # TODO: at this moment we consider all submodels is identical in architecture
        ref_nn = self.networks[0]
        n_hidden = len(ref_nn.blocks)
        self.layer_activations = ref_nn.activation_funcs

        stacked_w = []
        stacked_b = []
        for layer_idx in range(n_hidden):
            ws = [nn.blocks[layer_idx].w.detach() for nn, _ in self.blocks]
            bss = [nn.blocks[layer_idx].b.detach() for nn, _ in self.blocks]
            stacked_w.append(torch.stack(ws).requires_grad_(True))
            stacked_b.append(torch.stack(bss).requires_grad_(True))

        ws = [nn.out_layer.w.detach() for nn, _ in self.blocks]
        bss = [nn.out_layer.b.detach() for nn, _ in self.blocks]
        stacked_w.append(torch.stack(ws).requires_grad_(True))
        stacked_b.append(torch.stack(bss).requires_grad_(True))

        self.stacked_w = stacked_w
        self.stacked_b = stacked_b

    def _sync_stacked_params(self) -> None:
        """
        Gather weights and biases from submodels into self.stacked_w and self.stacked_b.
        """
        n_hidden = len(self.networks[0].blocks)

        for layer_idx in range(n_hidden):
            for i, (nn, _) in enumerate(self.blocks):
                self.stacked_w[layer_idx].data[i] = nn.blocks[layer_idx].w
                self.stacked_b[layer_idx].data[i] = nn.blocks[layer_idx].b

        for i, (nn, _) in enumerate(self.blocks):
            self.stacked_w[-1].data[i] = nn.out_layer.w
            self.stacked_b[-1].data[i] = nn.out_layer.b

    def _manual_forward(
        self,
        stacked_w: list[torch.Tensor],
        stacked_b: list[torch.Tensor],
        x_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched forward pass over all (or a subset of) submodels.

        Parameters
        ----------
        stacked_w: list[torch.Tensor]
            Weight tensors per layer, each of shape `(N, in_i, out_i)`,
            where `N` is the number of active submodels.
        stacked_b: list[torch.Tensor]
            Bias tensors per layer, each of shape `(N, out_i)`.
        x_norm: torch.Tensor
            Normalized input of shape `(N, N_pts, in_features)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(N, N_pts, output_size)`.
        """
        out = x_norm  # (N, N_pts, in)

        for i, (W, b, act) in enumerate(
            zip(stacked_w, stacked_b, self.layer_activations)
        ):
            # W:   (N, in, out)
            # out: (N, N_pts, in)
            # bmm: (N, N_pts, in) × (N, in, out) → (N, N_pts, out)
            out = torch.bmm(out, W) + b.unsqueeze(1)

            out = get_activation(act)(out)

        return out  # (N, N_pts, output_size)

    def _scatter_gradients(self) -> None:
        """
        In order for optimizer to correctly update the model weights,
        it is necessary to pass the gradients from stacked_w, stacked_b back
        to the model weights and biases,
        since stacked_w and stacked_b were used in the forward pass.
        """
        ref_nn = self.networks[0]
        n_hidden = len(ref_nn.blocks)

        for layer_idx in range(n_hidden):
            sw = self.stacked_w[layer_idx]
            sb = self.stacked_b[layer_idx]
            if sw.grad is None:
                continue
            for i, (nn, _) in enumerate(self.blocks):
                layer = nn.blocks[layer_idx]
                layer.w.grad = sw.grad[i].clone()
                layer.b.grad = sb.grad[i].clone()

        sw = self.stacked_w[-1]
        sb = self.stacked_b[-1]
        if sw.grad is not None:
            for i, (nn, _) in enumerate(self.blocks):
                nn.out_layer.w.grad = sw.grad[i]
                nn.out_layer.b.grad = sb.grad[i]

    def _batched_call(
        self,
        x: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluate all submodels on the given input and aggregate their outputs.

        Submodels flagged as inactive via `active_mask` are evaluated inside a
        `torch.no_grad()` context.

        Parameters
        ----------
        x: torch.Tensor
            Input points of shape `(N_pts, d)`.
        active_mask: torch.Tensor, optional
            Boolean tensor of shape `(N_blocks,)`.
            `True` marks submodels that require gradient computation.
            If `None` or all `True`, all submodels are evaluated with gradients.

        Returns
        -------
        torch.Tensor
            Aggregated output of shape `(N_pts, output_size)`, computed as the
            weighted sum of windowed submodel outputs.
        """
        N_blocks = len(self.blocks)

        x_exp = x.unsqueeze(0)  # (1, N_pts, d)
        vmins = self.all_vmins.unsqueeze(1)  # (N, 1, d)
        vmaxs = self.all_vmaxs.unsqueeze(1)  # (N, 1, d)
        x_norm = 2.0 * (x_exp - vmins) / (vmaxs - vmins) - 1.0  # (N, N_pts, d)

        if active_mask is None:
            outputs = self._manual_forward(self.stacked_w, self.stacked_b, x_norm)
        else:
            active_idx = active_mask.nonzero(as_tuple=True)[0]
            inactive_idx = (~active_mask).nonzero(as_tuple=True)[0]

            active_w = [sw[active_idx] for sw in self.stacked_w]
            active_b = [sb[active_idx] for sb in self.stacked_b]
            active_out = self._manual_forward(active_w, active_b, x_norm[active_idx])

            with torch.no_grad():
                inact_w = [sw[inactive_idx] for sw in self.stacked_w]
                inact_b = [sb[inactive_idx] for sb in self.stacked_b]
                inactive_out = self._manual_forward(
                    inact_w, inact_b, x_norm[inactive_idx]
                )

            outputs_list = [None] * N_blocks
            N_pts = x.shape[0]
            outputs = torch.empty(
                N_blocks, N_pts, self.output_size, device=x.device, dtype=x.dtype
            )
            outputs[active_idx] = active_out
            outputs[inactive_idx] = inactive_out  # (N, N_pts, d)

        scales = self.all_scales.unsqueeze(1)  # (N, 1, d)
        shifts = self.all_shifts.unsqueeze(1)
        outputs_phys = outputs * scales + shifts

        # with torch.no_grad():
        windows = self.decomposition.batched_window(x).squeeze(-1)  # (N, N_pts)
        return torch.einsum("bn,bnd->nd", windows, outputs_phys)  # (N_pts, out_d)

    def custom_compile(
        self,
        rate: float = 1e-2,
        optimizer: str = "SGD",
        loss_func: str = "MeanSquaredError",
        metric_funcs: Optional[list[str]] = None,
        run_eagerly: bool = False,
    ) -> None:
        """
        Configures the model for training

        Parameters
        ----------
        rate: float
            learning rate for optimizer
        optimizer: str
            name of optimizer
        inner_loss_func: str
            name of loss function for each network in blocks
        loss_func: str
            name of loss function
        metric_funcs: list[str]
            list with metric function names
        run_eagerly: bool
            Reserved for future use (Keras compatibility flag)
        """
        loss = losses.get_loss(loss_func)
        m = (
            [metrics.get_metric(metric) for metric in metric_funcs]
            if metric_funcs is not None
            else None
        )
        self.rate_ = rate
        self.loss_func_ = loss_func
        self.metric_funcs_ = metric_funcs
        # all_params = [p for nn, _ in self.blocks for p in nn.parameters()]
        all_params = self.stacked_w + self.stacked_b
        self.optimizer = optimizers.get_optimizer(optimizer)(
            all_params, lr=rate, fused=True
        )

    def forward(
        self,
        x: torch.Tensor,
        active_indices: Optional[list[int]] = None,
        **kwargs,
    ):
        """
        Obtaining a neural network response on the input data vector
        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        active_indices: Optional[list[int]]
            Indices subset of `self.blocks` from submodels that require gradient calculation.

        Returns
        -------
        torch.Tensor
            Predicted output
        """
        if active_indices is None or len(active_indices) == len(self.blocks):
            return self._batched_call(x, active_mask=None)

        active_mask = torch.zeros(len(self.blocks), dtype=torch.bool, device=x.device)
        active_mask[active_indices] = True
        return self._batched_call(x, active_mask=active_mask)

    def call(self, x, active_models: list = None, **kwargs):
        """
        Alias for `forward`
        """
        return self.forward(x, active_models, **kwargs)

    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the mean squared error between predictions and targets.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        y : torch.Tensor
            Ground-truth tensor

        Returns
        -------
        torch.Tensor
            Scalar MSE loss
        """
        y_pred = self(x)
        square = torch.square(y - y_pred)
        return torch.mean(square)

    def save_weights(self, path: str) -> None:
        """
        Save model state dict to disk

        Parameters
        ----------
        path: str
            File path
        """
        n_hidden = len(self.networks[0].blocks)
        for layer_idx in range(n_hidden):
            for i, (nn, _) in enumerate(self.blocks):
                nn.blocks[layer_idx].w.data = self.stacked_w[layer_idx][i].data
                nn.blocks[layer_idx].b.data = self.stacked_b[layer_idx][i].data
        for i, (nn, _) in enumerate(self.blocks):
            nn.out_layer.w.data = self.stacked_w[-1][i].data
            nn.out_layer.b.data = self.stacked_b[-1][i].data
        self._scatter_gradients()
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str) -> None:
        """
        Load model state dict from disk and synchronize batched parameter tensors

        Parameters
        ----------
        path: str
            Path to a file
        """
        self.load_state_dict(torch.load(path))
        self._sync_stacked_params()
