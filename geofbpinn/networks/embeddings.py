import torch.nn


class BaseEmbedding(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x, **kwargs):
        return x


class FourierEmbedding(BaseEmbedding):
    def __init__(
        self, input_dim: int, output_dim: int, sigma: float = 2.0, n_freqs: int = 8
    ):
        output_dim = 2 * n_freqs
        super().__init__(input_dim, output_dim)
        self._n_freqs = n_freqs

        B = torch.randn(input_dim, n_freqs) * sigma  # (d_orig, m)
        self.register_buffer("fourier_B", B)

    def forward(self, x, **kwargs):
        # (N, N_pts, d_orig) @ (d_orig, m) → (N, N_pts, m)
        proj = torch.einsum("npi,im->npm", x, self.fourier_B)
        x_embed = torch.cat(
            [torch.sin(proj), torch.cos(proj)], dim=-1
        )  # (N, N_pts, 2m)
        return x_embed
