import torch
import torch.optim as optim

from ocpmodels.common.registry import registry
from ocpmodels.modules.loss import DDPLoss, MSELossForces, PerAtomMSELossEnergy
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
