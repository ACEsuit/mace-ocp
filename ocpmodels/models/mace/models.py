from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import torch
from e3nn import o3

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel

from .blocks import (
    AtomicEnergiesBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from .mace_core.blocks import (
    AgnosticInteractionBlock,
    EquivariantProductBasisBlock,
    RealAgnosticResidualInteractionBlock,
)
from .mace_core.scatter import scatter_sum
from .utils import compute_forces


@registry.register_model("mace")
class MACE(BaseModel):
    def __init__(
        self,
        # Unused legacy OCP params.
        num_atoms: Optional[int],
        bond_feat_dim: int,
        num_targets: int,
        #
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        num_interactions: int,
        num_elements: int,
        hidden_irreps: str,
        MLP_irreps: str,
        avg_num_neighbors: float,
        correlation: int,
        # Defaults from OCP / https://github.com/ACEsuit/mace/blob/main/scripts/run_train.py
        gate=torch.nn.functional.silu,
        # per-element energy references currently initialized to 0s.
        atomic_energies=str,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticResidualInteractionBlock,
        max_neighbors: int = 500,
        otf_graph: bool = True,
        use_pbc: bool = True,
        regress_forces: bool = True,
    ):
        super().__init__()
        self.cutoff = self.r_max = r_max
        self.avg_num_neighbors = avg_num_neighbors
        self.max_neighbors = max_neighbors
        self.otf_graph = otf_graph
        self.use_pbc = use_pbc
        self.regress_forces = regress_forces

        # YAML loads them as strings, initialize them as o3.Irreps.
        hidden_irreps = o3.Irreps(hidden_irreps)
        MLP_irreps = o3.Irreps(MLP_irreps)

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps(
            [(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))]
        )
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Node / e0 block
        # fmt: off
        if atomic_energies == "oc20tiny":
            atomic_energies = np.array([-0.5789446234902406, 0.0, 0.0, 0.0, 0.0, -0.5789446234902277, 0.0, -0.28947231174511384, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.631556987921935, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.631556987921935, 0.0, 0.0, 0.0])
        else:
            atomic_energies = np.array([0. for i in range(83)])
        # fmt: on

        self.nz_z = np.nonzero(atomic_energies)[0]
        assert len(self.nz_z) == num_elements

        self.register_buffer(
            "z2idx",
            torch.zeros(120, dtype=torch.int64).fill_(-1),
        )
        for i, z in enumerate(self.nz_z + 1):
            self.z2idx[z] = i

        self.atomic_energies_fn = AtomicEnergiesBlock(
            atomic_energies[self.nz_z]
        )

        # Interactions and readout
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            # MACE considers a fixed `avg_num_neighbors`. We can either compute
            # this ~fixed statistic for OC20, or compute this value on-the-fly.
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps))

    def atomic_numbers_to_compressed_one_hot(self, atomic_numbers):
        """
        Convert atomic numbers to compressed one-hot vectors.

        Returns (num_atoms, num_elements),
        where num_elements refers to the no. of elements present in the training
        dataset, not an exhaustive list of all elements. For example, if the
        only atomic numbers in the dataset are 1, 6, 8, 40, 80, then
        num_elements would be 5 with the first column in the returned tensor
        referring to hydrogen, 2nd to carbon and so on.
        """
        idx = self.z2idx[atomic_numbers]
        assert torch.all(idx >= 0)

        atomic_numbers_1hot = torch.zeros(
            atomic_numbers.shape[0],
            len(self.atomic_energies_fn.atomic_energies),
            device=atomic_numbers.device,
        ).scatter(1, idx.unsqueeze(1), 1.0)

        return atomic_numbers_1hot

    @conditional_grad(torch.enable_grad())
    def forward(self, data):

        # TODO(@abhshkdz): Fit linear references per element from training data.
        # These are currently initialized to 0.0.

        # OCP prepro boilerplate.
        pos = data.pos
        atomic_numbers = data.atomic_numbers.long()
        num_graphs = data.batch.max() + 1

        # MACE computes forces via gradients.
        pos.requires_grad_(True)

        (
            edge_index,
            D_st,
            distance_vec,
            _,
            _,
            _,
        ) = self.generate_graph(data)
        ### OCP prepro ends.

        # Atomic energies
        #
        # Comment(@abhshkdz): `data.node_attrs` is a 1-hot vector for each
        # atomic number. `self.atomic_energies_fn` just matmuls the 1-hot
        # vectors with the list of energies per atomic number, returning the
        # energy per element.
        atomic_numbers_1hot = self.atomic_numbers_to_compressed_one_hot(
            atomic_numbers
        )

        node_e0 = self.atomic_energies_fn(atomic_numbers_1hot)
        e0 = scatter_sum(
            src=node_e0, index=data.batch, dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(atomic_numbers_1hot)

        # Comment(@abhshkdz): `lengths` here is same as `D_st`, and `vectors` is
        # the same as `distance_vec` but pointing in the opposite direction.
        # vectors, lengths = get_edge_vectors_and_lengths(
        #     positions=data.positions, edge_index=data.edge_index, shifts=data.shifts
        # )
        lengths = D_st.view(-1, 1)
        vectors = -distance_vec

        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            # print((neighbors / data.natoms).mean(), self.avg_num_neighbors)
            node_feats, sc = interaction(
                node_attrs=atomic_numbers_1hot,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=atomic_numbers_1hot
            )
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies,
                index=data.batch,
                dim=-1,
                dim_size=num_graphs,
            )  # [n_graphs,]
            energies.append(energy)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]

        # Compute forces via autograd.
        forces = compute_forces(
            energy=total_energy, positions=pos, training=self.training
        )

        return total_energy, forces


@registry.register_model("scaleshift_mace")
class ScaleShiftMACE(MACE):
    def __init__(
        self,
        # Unused legacy OCP params.
        num_atoms: Optional[int],
        bond_feat_dim: int,
        num_targets: int,
        #
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        num_interactions: int,
        num_elements: int,
        hidden_irreps: str,
        MLP_irreps: str,
        avg_num_neighbors: float,
        correlation: int,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        # Defaults from OCP / https://github.com/ACEsuit/mace/blob/main/scripts/run_train.py
        gate=torch.nn.functional.silu,
        atomic_energies=str,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticResidualInteractionBlock,
        max_neighbors: int = 500,
        otf_graph: bool = True,
        use_pbc: bool = True,
        regress_forces: bool = True,
    ):
        super().__init__(
            num_atoms,
            bond_feat_dim,
            num_targets,
            #
            r_max,
            num_bessel,
            num_polynomial_cutoff,
            max_ell,
            num_interactions,
            num_elements,
            hidden_irreps,
            MLP_irreps,
            avg_num_neighbors,
            correlation,
            gate,
            atomic_energies,
            interaction_cls,
            interaction_cls_first,
            max_neighbors,
            otf_graph,
            use_pbc,
            regress_forces,
        )
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        # TODO(@abhshkdz): Fit linear references per element from training data.
        # These are currently initialized to 0.0.

        # OCP prepro boilerplate.
        pos = data.pos
        atomic_numbers = data.atomic_numbers.long()
        num_graphs = data.batch.max() + 1

        # MACE computes forces via gradients.
        pos.requires_grad_(True)

        (
            edge_index,
            D_st,
            distance_vec,
            _,
            _,
            _,
        ) = self.generate_graph(data)
        ### OCP prepro ends.

        # Atomic energies
        #
        # Comment(@abhshkdz): `data.node_attrs` is a 1-hot vector for each
        # atomic number. `self.atomic_energies_fn` just matmuls the 1-hot
        # vectors with the list of energies per atomic number, returning the
        # energy per element.
        atomic_numbers_1hot = self.atomic_numbers_to_compressed_one_hot(
            atomic_numbers
        )

        node_e0 = self.atomic_energies_fn(atomic_numbers_1hot)
        e0 = scatter_sum(
            src=node_e0, index=data.batch, dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(atomic_numbers_1hot)
        lengths = D_st.view(-1, 1)
        vectors = -distance_vec
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        node_es_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=atomic_numbers_1hot,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=atomic_numbers_1hot
            )
            node_es_list.append(
                readout(node_feats).squeeze(-1)
            )  # {[n_nodes, ], }

        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es,
            index=data.batch,
            dim=-1,
            dim_size=data.num_graphs,
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_e = e0 + inter_e
        forces = compute_forces(
            energy=inter_e, positions=pos, training=self.training
        )

        return total_e, forces


@registry.register_model("ase_nbrlist_mace")
class ASENeighborListMACE(ScaleShiftMACE):
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        # OCP prepro boilerplate.
        pos = data.pos
        atomic_numbers = data.atomic_numbers.long()
        num_graphs = data.batch.max() + 1

        # MACE computes forces via gradients.
        pos.requires_grad_(True)

        sender, receiver = data.edge_index[0], data.edge_index[1]
        vectors = pos[receiver] - pos[sender] + data.shifts  # [n_edges, 3]
        lengths = torch.linalg.norm(
            vectors, dim=-1, keepdim=True
        )  # [n_edges, 1]

        # Atomic energies
        #
        # Comment(@abhshkdz): `data.node_attrs` is a 1-hot vector for each
        # atomic number. `self.atomic_energies_fn` just matmuls the 1-hot
        # vectors with the list of energies per atomic number, returning the
        # energy per element.
        atomic_numbers_1hot = self.atomic_numbers_to_compressed_one_hot(
            atomic_numbers
        )

        node_e0 = self.atomic_energies_fn(atomic_numbers_1hot)
        e0 = scatter_sum(
            src=node_e0, index=data.batch, dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(atomic_numbers_1hot)
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        node_es_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=atomic_numbers_1hot,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=atomic_numbers_1hot
            )
            node_es_list.append(
                readout(node_feats).squeeze(-1)
            )  # {[n_nodes, ], }

        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es,
            index=data.batch,
            dim=-1,
            dim_size=data.num_graphs,
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_e = e0 + inter_e
        forces = compute_forces(
            energy=inter_e, positions=pos, training=self.training
        )

        return total_e, forces
