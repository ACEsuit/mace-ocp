import logging

import torch
import torch.optim as optim

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.modules.loss import (
    DDPLoss,
    L2MAELoss,
    MSELossForces,
    PerAtomMSELossEnergy,
)
from ocpmodels.trainers import ForcesTrainer


@registry.register_trainer("mace_trainer")
class MACETrainer(ForcesTrainer):
    # Following optimizer / weight decay defaults from
    # https://github.com/ACEsuit/mace/blob/5470b632d839358faed4e9c97f67fece1b558962/scripts/run_train.py#L210
    def load_optimizer(self):
        optimizer = self.config["optim"].get("optimizer", "AdamW")
        optimizer = getattr(optim, optimizer)

        if self.config["optim"].get("weight_decay", 0) > 0:

            int_params_decay = []
            int_params_no_decay = []
            for (
                name,
                param,
            ) in self._unwrapped_model.interactions.named_parameters():
                if "linear.weight" in name or "skip_tp_full.weight" in name:
                    int_params_decay += [param]
                else:
                    int_params_no_decay += [param]

            self.optimizer = optimizer(
                params=[
                    {
                        "name": "embedding",
                        "params": self._unwrapped_model.node_embedding.parameters(),
                        "weight_decay": 0.0,
                    },
                    {
                        "name": "interactions_decay",
                        "params": int_params_decay,
                        "weight_decay": self.config["optim"]["weight_decay"],
                    },
                    {
                        "name": "interactions_no_decay",
                        "params": int_params_no_decay,
                        "weight_decay": 0.0,
                    },
                    {
                        "name": "products",
                        "params": self._unwrapped_model.products.parameters(),
                        "weight_decay": self.config["optim"]["weight_decay"],
                    },
                    {
                        "name": "readouts",
                        "params": self._unwrapped_model.readouts.parameters(),
                        "weight_decay": 0.0,
                    },
                ],
                lr=self.config["optim"]["lr_initial"],
                **self.config["optim"].get("optimizer_params", {}),
            )
        else:
            self.optimizer = optimizer(
                params=self.model.parameters(),
                lr=self.config["optim"]["lr_initial"],
                **self.config["optim"].get("optimizer_params", {}),
            )

    def load_loss(self):
        self.loss_fn = {}
        self.loss_fn["energy"] = DDPLoss(PerAtomMSELossEnergy())
        self.loss_fn["force"] = DDPLoss(MSELossForces())

    def _compute_loss(self, out, batch_list):
        assert self.config["model_attributes"].get("regress_forces", False)

        loss = []

        natoms = torch.cat(
            [batch.natoms.to(self.device) for batch in batch_list]
        )

        # Energy loss.
        energy_target = torch.cat(
            [batch.y.to(self.device) for batch in batch_list], dim=0
        )
        if self.normalizer.get("normalize_labels", False):
            energy_target = self.normalizers["target"].norm(energy_target)
        energy_mult = self.config["optim"].get("energy_coefficient", 1)
        loss.append(
            energy_mult
            * self.loss_fn["energy"](out["energy"], energy_target, natoms)
        )

        # Force loss.
        force_target = torch.cat(
            [batch.force.to(self.device) for batch in batch_list], dim=0
        )
        if self.normalizer.get("normalize_labels", False):
            force_target = self.normalizers["grad_target"].norm(force_target)

        force_mult = self.config["optim"].get("force_coefficient", 100)

        if self.config["task"].get("train_on_free_atoms", False):
            fixed = torch.cat(
                [batch.fixed.to(self.device) for batch in batch_list]
            )
            mask = fixed == 0
            loss.append(
                force_mult
                * self.loss_fn["force"](
                    out["forces"][mask], force_target[mask]
                )
            )
        else:
            loss.append(
                force_mult * self.loss_fn["force"](out["forces"], force_target)
            )

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss


@registry.register_trainer("interaction_mace_trainer")
class InteractionMACETrainer(MACETrainer):
    def load_loss(self):
        self.loss_fn = {}
        self.loss_fn["energy"] = DDPLoss(torch.nn.L1Loss())
        self.loss_fn["force"] = DDPLoss(L2MAELoss())

    def _compute_loss(self, out, batch_list):
        loss = []

        # Energy loss.
        energy_target = torch.cat(
            [batch.y.to(self.device) for batch in batch_list], dim=0
        )
        if self.normalizer.get("normalize_labels", False):
            energy_target = self.normalizers["target"].norm(energy_target)
        energy_mult = self.config["optim"].get("energy_coefficient", 1)
        loss.append(
            energy_mult * self.loss_fn["energy"](out["energy"], energy_target)
        )

        # Force loss.
        if self.config["model_attributes"].get("regress_forces", True):
            force_target = torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            )
            if self.normalizer.get("normalize_labels", False):
                force_target = self.normalizers["grad_target"].norm(
                    force_target
                )

            tag_specific_weights = self.config["task"].get(
                "tag_specific_weights", []
            )
            if tag_specific_weights != []:
                # handle tag specific weights as introduced in forcenet
                assert len(tag_specific_weights) == 3

                batch_tags = torch.cat(
                    [
                        batch.tags.float().to(self.device)
                        for batch in batch_list
                    ],
                    dim=0,
                )
                weight = torch.zeros_like(batch_tags)
                weight[batch_tags == 0] = tag_specific_weights[0]
                weight[batch_tags == 1] = tag_specific_weights[1]
                weight[batch_tags == 2] = tag_specific_weights[2]

                if self.config["optim"].get("loss_force", "l2mae") == "l2mae":
                    # zero out nans, if any
                    found_nans_or_infs = not torch.all(
                        out["forces"].isfinite()
                    )
                    if found_nans_or_infs is True:
                        logging.warning("Found nans while computing loss")
                        out["forces"] = torch.nan_to_num(
                            out["forces"], nan=0.0
                        )

                    dists = torch.norm(
                        out["forces"] - force_target, p=2, dim=-1
                    )
                    weighted_dists_sum = (dists * weight).sum()

                    num_samples = out["forces"].shape[0]
                    num_samples = distutils.all_reduce(
                        num_samples, device=self.device
                    )
                    weighted_dists_sum = (
                        weighted_dists_sum
                        * distutils.get_world_size()
                        / num_samples
                    )

                    force_mult = self.config["optim"].get(
                        "force_coefficient", 30
                    )
                    loss.append(force_mult * weighted_dists_sum)
                else:
                    raise NotImplementedError
            else:
                # Force coefficient = 30 has been working well for us.
                force_mult = self.config["optim"].get("force_coefficient", 30)
                if self.config["task"].get("train_on_free_atoms", False):
                    fixed = torch.cat(
                        [batch.fixed.to(self.device) for batch in batch_list]
                    )
                    mask = fixed == 0
                    if (
                        self.config["optim"]
                        .get("loss_force", "mae")
                        .startswith("atomwise")
                    ):
                        force_mult = self.config["optim"].get(
                            "force_coefficient", 1
                        )
                        natoms = torch.cat(
                            [
                                batch.natoms.to(self.device)
                                for batch in batch_list
                            ]
                        )
                        natoms = torch.repeat_interleave(natoms, natoms)
                        force_loss = force_mult * self.loss_fn["force"](
                            out["forces"][mask],
                            force_target[mask],
                            natoms=natoms[mask],
                            batch_size=batch_list[0].natoms.shape[0],
                        )
                        loss.append(force_loss)
                    else:
                        loss.append(
                            force_mult
                            * self.loss_fn["force"](
                                out["forces"][mask], force_target[mask]
                            )
                        )
                else:
                    loss.append(
                        force_mult
                        * self.loss_fn["force"](out["forces"], force_target)
                    )

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss
