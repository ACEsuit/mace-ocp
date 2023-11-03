from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

from .mace_core.blocks import InteractionBlock
from .mace_core.irreps_tools import (
    lifted_skip,
    linear_out_irreps,
    reshape_irreps,
    tp_out_irreps_with_instructions,
)
from .mace_core.scatter import scatter_sum
from .mace_core.symmetric_contraction import SymmetricContraction
from .radial import BesselBasis, GaussianBasis, PolynomialCutoff


class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,  # [n_nodes, irreps]
    ):
        return self.linear(node_attrs)


class LinearReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(
            irreps_in=irreps_in, irreps_out=o3.Irreps("0e")
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


class NonLinearReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.linear_1 = o3.Linear(
            irreps_in=irreps_in, irreps_out=self.hidden_irreps
        )
        self.non_linearity = nn.Activation(
            irreps_in=self.hidden_irreps, acts=[gate]
        )
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=o3.Irreps("0e")
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]


class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        assert len(atomic_energies.shape) == 1

        self.register_buffer(
            "atomic_energies",
            torch.tensor(atomic_energies, dtype=torch.get_default_dtype()),
        )  # [n_elements, ]

    def forward(
        self, x: torch.Tensor  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, self.atomic_energies)

    def __repr__(self):
        formatted_energies = ", ".join(
            [f"{x:.4f}" for x in self.atomic_energies]
        )
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"


class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        rbf: str = "bessel",
    ):
        super().__init__()
        self.rbf = rbf
        if rbf == "bessel":
            self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        elif rbf == "gaussian":
            self.rbf_fn = GaussianBasis(r_max=r_max, num_gaussians=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
    ):
        if self.rbf == "bessel":
            bessel = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
            cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
            return bessel * cutoff  # [n_edges, n_basis]
        elif self.rbf == "gaussian":
            gaussian = self.rbf_fn(edge_lengths)  # [n_edges, n_basis]
            cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
            return gaussian * cutoff  # [n_edges, n_basis]


nonlinearities = {1: torch.nn.SiLU(), -1: torch.nn.Tanh()}


class TensorProductWeightsBlock(torch.nn.Module):
    def __init__(
        self, num_elements: int, num_edge_feats: int, num_feats_out: int
    ):
        super().__init__()

        weights = torch.empty(
            (num_elements, num_edge_feats, num_feats_out),
            dtype=torch.get_default_dtype(),
        )
        torch.nn.init.xavier_uniform_(weights)
        self.weights = torch.nn.Parameter(weights)

    def forward(
        self,
        sender_or_receiver_node_attrs: torch.Tensor,  # assumes that the node attributes are one-hot encoded
        edge_feats: torch.Tensor,
    ):
        return torch.einsum(
            "be, ba, aek -> bk",
            edge_feats,
            sender_or_receiver_node_attrs,
            self.weights,
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(shape=({", ".join(str(s) for s in self.weights.shape)}), '
            f"weights={np.prod(self.weights.shape)})"
        )


class ResidualElementDependentInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.conv_tp_weights = TensorProductWeightsBlock(
            num_elements=self.node_attrs_irreps.num_irreps,
            num_edge_feats=self.edge_feats_irreps.num_irreps,
            num_feats_out=self.conv_tp.weight_numel,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = o3.Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.irreps_out
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender, receiver = edge_index
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(node_attrs[sender], edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return message + sc  # [n_nodes, irreps]


class AgnosticNonlinearInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.SiLU(),
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = o3.Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.irreps_out, self.node_attrs_irreps, self.irreps_out
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender, receiver = edge_index
        num_nodes = node_feats.shape[0]
        tp_weights = self.conv_tp_weights(edge_feats)
        node_feats = self.linear_up(node_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp(message, node_attrs)
        return message  # [n_nodes, irreps]


class AgnosticResidualNonlinearInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.SiLU(),
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = o3.Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.irreps_out
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender, receiver = edge_index
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        message = message + sc
        return message  # [n_nodes, irreps]


class ScaleShiftBlock(torch.nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.register_buffer(
            "scale", torch.tensor(scale, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "shift", torch.tensor(shift, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.shift

    def __repr__(self):
        return f"{self.__class__.__name__}(scale={self.scale:.6f}, shift={self.shift:.6f})"


class ForceBlock(torch.nn.Module):
    def __init__(self, hidden_irreps):
        super().__init__()

        # TODO(@abhshkdz): is this assertion needed?
        self.hidden_irreps = hidden_irreps
        assert hidden_irreps[0].dim * 3 == hidden_irreps[1].dim

        l0_h = hidden_irreps[0].mul
        l1_h = hidden_irreps[1].mul

        self.output_network = torch.nn.ModuleList(
            [
                GatedEquivariantBlock(l0_h, l1_h, l1_h // 2),
                GatedEquivariantBlock(l1_h // 2, l1_h // 2, 1),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, node_feats):
        # split node_feats into scalar and vector components.
        x = node_feats[:, : self.hidden_irreps[0].dim]
        vec = node_feats[
            :,
            self.hidden_irreps[0].dim : self.hidden_irreps[0].dim
            + self.hidden_irreps[1].dim,
        ]

        vec = vec.reshape(vec.shape[0], self.hidden_irreps[1].mul, 3)
        vec = vec.transpose(1, 2)

        # pass it through the gated equivariant blocks.
        for layer in self.output_network:
            x, vec = layer(x, vec)

        # return the vector components.
        return vec.squeeze()


# Implementation based on TorchMD-Net
# https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/painn/painn.py#L607
class GatedEquivariantBlock(torch.nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        l0_channels,
        l1_channels,
        out_channels,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.l0_channels = l0_channels
        self.l1_channels = l1_channels
        self.out_channels = out_channels

        self.vec1_proj = torch.nn.Linear(l1_channels, l1_channels, bias=False)
        self.vec2_proj = torch.nn.Linear(l1_channels, out_channels, bias=False)

        self.update_net = torch.nn.Sequential(
            torch.nn.Linear(
                l0_channels + l1_channels,
                l1_channels,
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(l1_channels, out_channels * 2),
        )

        self.act = torch.nn.SiLU()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.vec1_proj.weight)
        torch.nn.init.xavier_uniform_(self.vec2_proj.weight)
        torch.nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        # x is [num_nodes x l0_hidden_channels]
        # v is [num_nodes x 3 x l1_hidden_channels]
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        x = self.act(x)
        return x, v


@compile_mode("script")
class SpeciesAgnosticResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        self.linear_down = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_down_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        input_dim = (
            self.edge_feats_irreps.num_irreps
            + 2 * self.node_feats_down_irreps.num_irreps
        )
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim]
            + 3 * [self.rbf_hidden_channels]
            + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        self.reshape = reshape_irreps(self.irreps_out)

        # Skip connection.
        self.skip_linear = o3.Linear(
            self.node_feats_irreps, self.hidden_irreps
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_linear(node_feats)
        node_feats_up = self.linear_up(node_feats)
        node_feats_down = self.linear_down(node_feats)
        augmented_edge_feats = torch.cat(
            [
                edge_feats,
                node_feats_down[sender],
                node_feats_down[receiver],
            ],
            dim=-1,
        )
        tp_weights = self.conv_tp_weights(augmented_edge_feats)
        mji = self.conv_tp(
            node_feats_up[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticResidualInteractionBlockV2(InteractionBlock):
    def _setup(self) -> None:

        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        self.linear_down = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_down_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        input_dim = (
            self.edge_feats_irreps.num_irreps
            + 2 * self.node_feats_down_irreps.num_irreps
        )
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim]
            + 3 * [self.rbf_hidden_channels]
            + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
        )
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats_up = self.linear_up(node_feats)
        node_feats_down = self.linear_down(node_feats)
        augmented_edge_feats = torch.cat(
            [
                edge_feats,
                node_feats_down[sender],
                node_feats_down[receiver],
            ],
            dim=-1,
        )
        tp_weights = self.conv_tp_weights(augmented_edge_feats)
        mji = self.conv_tp(
            node_feats_up[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class IdentityResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim]
            + 3 * [self.rbf_hidden_channels]
            + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        self.skip = lifted_skip(self.node_feats_irreps, self.hidden_irreps)
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip(node_feats)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class EdgeGatedInteractionBlock(InteractionBlock):
    def _setup(self) -> None:

        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )

        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        irreps_mid = irreps_mid.simplify()
        self.message_dim = irreps_mid.dim
        self.irreps_out = self.target_irreps

        # Convolution weights
        self.num_convs = self.num_gates if self.multi_conv else 1

        if self.use_source_target_edge_feats:
            self.linear_down = o3.Linear(
                self.node_feats_irreps,
                self.node_feats_down_irreps,
                internal_weights=True,
                shared_weights=True,
            )
            input_dim = (
                self.edge_feats_irreps.num_irreps
                + 2 * self.node_feats_down_irreps.num_irreps
            )
            self.conv_tp_weights = nn.FullyConnectedNet(
                [input_dim]
                + 3 * [self.rbf_hidden_channels]
                + [self.conv_tp.weight_numel * self.num_convs],
                torch.nn.functional.silu,
            )
        else:
            input_dim = self.edge_feats_irreps.num_irreps
            self.conv_tp_weights = nn.FullyConnectedNet(
                [input_dim]
                + 3 * [self.rbf_hidden_channels]
                + [self.conv_tp.weight_numel * self.num_convs],
                torch.nn.functional.silu,
            )

        # Edge gating operations
        if self.exponential:
            # self.scale = nn.FullyConnectedNet(
            #     [self.edge_gates_irreps.dim] + 3 * [self.rbf_hidden_channels] + [1],
            #     torch.nn.functional.silu,
            # )
            self.scale = None
        self.alpha = o3.Linear(self.node_feats_irreps, self.edge_gates_irreps)
        self.gamma = o3.Linear(
            irreps_mid * self.num_convs, self.edge_gates_irreps
        )
        self.mix = o3.FullyConnectedTensorProduct(
            self.edge_gates_irreps,
            self.edge_gates_irreps,
            f"{self.num_gates}x0e",
        )
        self.project = o3.Linear(irreps_mid * self.num_gates, self.irreps_out)

        self.skip = lifted_skip(self.node_feats_irreps, self.hidden_irreps)
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        num_edges = edge_feats.shape[0]
        sc = self.skip(node_feats)
        node_feats_up = self.linear_up(node_feats)
        if self.use_source_target_edge_feats:
            node_feats_down = self.linear_down(node_feats)
            augmented_edge_feats = torch.cat(
                [
                    edge_feats,
                    node_feats_down[sender],
                    node_feats_down[receiver],
                ],
                dim=-1,
            )
            tp_weights = self.conv_tp_weights(augmented_edge_feats).reshape(
                num_edges, self.num_convs, self.conv_tp.weight_numel
            )
        else:
            tp_weights = self.conv_tp_weights(edge_feats).reshape(
                num_edges, self.num_convs, self.conv_tp.weight_numel
            )

        mji = self.conv_tp(
            node_feats_up[sender][:, None, :],
            edge_attrs[:, None, :],
            tp_weights,
        )  # [n_edges, n_convs, irreps]

        # Gate edges
        alphas, gammas = (
            self.alpha(node_feats_up[receiver]),
            self.gamma(
                mji.reshape(num_edges, self.message_dim * self.num_convs)
            ),
        )
        gate = self.mix(alphas, gammas)  # [n_edges, n_gates]
        if self.exponential:
            # local_scale = torch.nn.functional.softplus(
            #     self.scale(alphas)
            # )
            # gate = (gate / local_scale).exp()
            #
            # Comment(@abhshkdz): The scatter_sum errors out because all_gate
            # has infs. The temperature scaling function is likely causing this.
            # Let's train with a fixed temperature for now.
            gate = gate.exp()

        all_gate = scatter_sum(
            src=gate, index=receiver, dim=0, dim_size=num_nodes
        )
        gate = torch.nan_to_num(gate / all_gate[receiver]).unsqueeze(
            -1
        )  # [n_edges, n_gates, 1]

        message = scatter_sum(
            src=(gate * mji), index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, n_gates, irreps]
        message = message.reshape(
            num_nodes, self.message_dim * self.num_gates
        )  # [n_nodes, irreps * n_gates]
        message = self.project(message)  # [n_nodes, irreps_out]
        return (
            self.reshape(message),
            sc,
        )
