"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch.optim as optim

from ocpmodels.common.registry import registry
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
            for name, param in self._unwrapped_model.interactions.named_parameters():
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