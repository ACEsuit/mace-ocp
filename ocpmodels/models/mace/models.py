from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import torch
from e3nn import o3

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel

from .blocks import (
    AtomicEnergiesBlock,
    ForceBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
    SpeciesAgnosticResidualInteractionBlock,
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
        # support for gaussian basis.
        rbf: str = "bessel",
        num_rbf: int = 8,
        rbf_hidden_channels: int = 64,
        direct_forces: bool = False,
        # support for species-agonstic contraction.
        contraction_type: str = "v1",
        # source and target feature size when concatenating for edge conv.
        node_feats_down_irreps: str = "64x0e",
    ):
        super().__init__()
        self.cutoff = self.r_max = r_max
        self.avg_num_neighbors = avg_num_neighbors
        self.max_neighbors = max_neighbors
        self.otf_graph = otf_graph
        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces

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
        if rbf == "bessel":
            self.radial_embedding = RadialEmbeddingBlock(
                r_max=r_max,
                num_bessel=num_bessel,
                num_polynomial_cutoff=num_polynomial_cutoff,
                rbf="bessel",
            )
        elif rbf == "gaussian":
            self.radial_embedding = RadialEmbeddingBlock(
                r_max=r_max,
                num_bessel=num_rbf,
                num_polynomial_cutoff=num_polynomial_cutoff,
                rbf="gaussian",
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
        elif atomic_energies == "oc20xxs":
            atomic_energies = np.array([-3.8315212119598576, 0.0, 0.0, 0.0, 0.0, -7.902387218067884, -7.21000990963105, -6.879033386904713, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6946341972298161, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif atomic_energies == "oc20":
            atomic_energies = np.array([-3.39973399802097, 0.0, 0.0, 0.0, -6.081884367921049, -8.293556029600811, -8.41745373621903, -4.92136266857127, 0.0, 0.0, -1.5412059197319175, 0.0, -3.6178325363353707, -5.3092430681268095, -5.291202563999228, -4.612401891658523, -3.081651431825209, 0.0, -1.5317668260051613, -2.4421854893794492, -6.728220402149601, -7.753628720142588, -8.380304712792668, -8.438050121320343, -7.841092458071865, -7.184260107389338, -6.23486296929759, -4.888130178461133, -2.962084764498329, -0.6657573646576105, -2.573145196892305, -4.067466705751315, -4.493942029750251, -3.811815951016779, 0.0, 0.0, -1.3488687481117176, -2.347882569119374, -7.026159902128673, -8.536582042496924, -9.587336073043936, -9.83366289354133, -9.268529057192797, -8.380115555600332, -6.794135382710533, -4.787344383569342, -1.8919937285040638, -0.20987260293014098, -2.0646188038840636, -3.3847372997774205, -3.648226709090113, -3.0333477411720318, 0.0, 0.0, -1.4964734007514156, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9.921176307016488, -11.142905408621559, -11.666772123514626, -11.047390248000076, -10.13349975986731, -8.286331766183878, -5.775058392981709, -2.6918083582766843, 0.28829374995119617, -1.808141562387085, -3.0229565921247694, -3.329866054812059])
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
            rbf_hidden_channels=rbf_hidden_channels,
            node_feats_down_irreps=o3.Irreps(node_feats_down_irreps),
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
            contraction_type=contraction_type,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                if self.direct_forces:
                    hidden_irreps_out = str(hidden_irreps[:2])
                else:
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
                rbf_hidden_channels=rbf_hidden_channels,
                node_feats_down_irreps=o3.Irreps(node_feats_down_irreps),
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
                contraction_type=contraction_type,
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
        # support for gaussian basis.
        rbf: str = "bessel",
        num_rbf: int = 8,
        rbf_hidden_channels: int = 64,
        # support for direct forces.
        direct_forces: bool = False,
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
            rbf=rbf,
            num_rbf=num_rbf,
            rbf_hidden_channels=rbf_hidden_channels,
            direct_forces=direct_forces,
        )
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

        if self.direct_forces:
            self.force_readout = ForceBlock(o3.Irreps(hidden_irreps))

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

        vectors = -distance_vec
        lengths = D_st.view(-1, 1)
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

        if self.direct_forces:
            forces = self.force_readout(node_feats)
        else:
            forces = compute_forces(
                energy=inter_e, positions=pos, training=self.training
            )

        return total_e, forces


@registry.register_model("interaction_energy_mace")
class InteractionEnergyMACE(MACE):
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
        atomic_energies=str,
        interaction_cls: str = "RealAgnosticResidualInteractionBlock",
        interaction_cls_first: str = "RealAgnosticResidualInteractionBlock",
        max_neighbors: int = 500,
        otf_graph: bool = True,
        use_pbc: bool = True,
        regress_forces: bool = True,
        # support for gaussian basis.
        rbf: str = "bessel",
        num_rbf: int = 8,
        rbf_hidden_channels: int = 64,
        # support for direct forces.
        direct_forces: bool = False,
        # support for species-agnostic contraction.
        contraction_type: str = "v1",
        # source and target feature size when concatenating for edge conv.
        node_feats_down_irreps: str = "64x0e",
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
            eval(interaction_cls),
            eval(interaction_cls_first),
            max_neighbors,
            otf_graph,
            use_pbc,
            regress_forces,
            rbf=rbf,
            num_rbf=num_rbf,
            rbf_hidden_channels=rbf_hidden_channels,
            direct_forces=direct_forces,
            contraction_type=contraction_type,
            node_feats_down_irreps=node_feats_down_irreps,
        )
        if self.direct_forces:
            self.force_readout = ForceBlock(o3.Irreps(hidden_irreps))

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        # TODO(@abhshkdz): Fit linear references per element from training data.
        # These are currently initialized to 0.0.

        # OCP prepro boilerplate.
        pos = data.pos
        atomic_numbers = data.atomic_numbers.long()

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

        vectors = -distance_vec
        lengths = D_st.view(-1, 1)
        ### OCP prepro ends.

        # Comment(@abhshkdz): `data.node_attrs` is a 1-hot vector for each
        # atomic number. `self.atomic_energies_fn` just matmuls the 1-hot
        # vectors with the list of energies per atomic number, returning the
        # energy per element.
        atomic_numbers_1hot = self.atomic_numbers_to_compressed_one_hot(
            atomic_numbers
        )

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

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es,
            index=data.batch,
            dim=-1,
            dim_size=data.num_graphs,
        )  # [n_graphs,]

        energy = inter_e

        if self.direct_forces:
            forces = self.force_readout(node_feats)
            return energy, forces
        else:
            raise NotImplementedError
