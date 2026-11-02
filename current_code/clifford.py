import torch
import qcd_ml
import qcd_ml


class Clifford(torch.nn.Module):


    def __init__(self, n_feature_in: int, n_feature_out: int) -> None:

        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.randn(n_feature_in, n_feature_out, 16, dtype=torch.cdouble)
        )

        self.n_feature_in = n_feature_in
        self.n_feature_out = n_feature_out


        gamma = qcd_ml.qcd.static.gamma

        I_4 = torch.eye(
            4,
            dtype=torch.cdouble,
            )

        gamma_5 = gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]


        self.register_buffer("gamma_cliff", torch.stack([

            I_4,
            gamma[0],
            gamma[1],
            gamma[2],
            gamma[3],
            0.5j * (gamma[0] @ gamma[1] - gamma[1] @ gamma[0]),
            0.5j * (gamma[0] @ gamma[2] - gamma[2] @ gamma[0]),
            0.5j * (gamma[1] @ gamma[2] - gamma[2] @ gamma[1]),
            0.5j * (gamma[0] @ gamma[3] - gamma[3] @ gamma[0]),
            0.5j * (gamma[1] @ gamma[3] - gamma[3] @ gamma[1]),
            0.5j * (gamma[2] @ gamma[3] - gamma[3] @ gamma[2]),
            1j * (gamma[0] @ gamma_5),
            1j * (gamma[1] @ gamma_5),
            1j * (gamma[2] @ gamma_5),
            1j * (gamma[3] @ gamma_5),
            gamma_5
        ]))

    def forward(self, features_in: torch.Tensor) -> torch.Tensor:

        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(
                f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_in}"
            )

        return torch.einsum("iok,kjs,ixyztsc->oxyztjc", self.weights, self.gamma_cliff, features_in)

