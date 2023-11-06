import numpy as np
import torch


class BesselBasis(torch.nn.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis=8, trainable=False):
        super().__init__()

        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(
                np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )


# From https://github.com/Open-Catalyst-Project/ocp/blob/a8ac36d75fc60d5b6d54b4019502941eb868fed0/ocpmodels/models/gemnet_oc/layers/radial_basis.py#L65
class GaussianBasis(torch.nn.Module):
    def __init__(self, r_max: float, num_gaussians=50, trainable=False):
        super().__init__()

        offset = torch.linspace(0, r_max, num_gaussians)
        if trainable:
            self.offset = torch.nn.Parameter(offset, requires_grad=True)
        else:
            self.register_buffer("offset", offset)

        self.register_buffer(
            "coeff",
            torch.tensor(
                -0.5 / (r_max / (num_gaussians - 1)) ** 2,
                dtype=torch.get_default_dtype(),
            ),
        )

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist - self.offset[None, :]
        return torch.exp(self.coeff * torch.pow(dist, 2))


class PolynomialCutoff(torch.nn.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (8)
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p=6):
        super().__init__()
        self.register_buffer(
            "p", torch.tensor(p, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # yapf: disable
        envelope = (
                1.0
                - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x / self.r_max, self.p)
                + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1)
                - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2)
        )
        # yapf: enable

        # noinspection PyUnresolvedReferences
        return envelope * (x < self.r_max).type(torch.get_default_dtype())

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})"
