import os
import errno
import logging

import torch
import torch.optim as optim

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import load_state_dict
from ocpmodels.trainers.mace_trainer import MACETrainer


@registry.register_trainer("eval_mace_chkpts_trainer")
class EvalMACECheckpointsTrainer(MACETrainer):
    def load_checkpoint(self, checkpoint_path):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                errno.ENOENT, "Checkpoint file not found", checkpoint_path
            )

        logging.info(f"Loading checkpoint from: {checkpoint_path}")
        map_location = torch.device("cpu") if self.cpu else self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.epoch = checkpoint.get("epoch", 0)
        self.step = checkpoint.get("step", 0)
        self.best_val_metric = checkpoint.get("best_val_metric", None)
        self.primary_metric = checkpoint.get("primary_metric", None)

        # Match the "module." count in the keys of model and checkpoint state_dict
        # DataParallel model has 1 "module.",  DistributedDataParallel has 2 "module."
        # Not using either of the above two would have no "module."

        ckpt_key_count = next(iter(checkpoint["model"])).count("module")
        mod_key_count = next(iter(self.model.state_dict())).count("module")
        key_count_diff = mod_key_count - ckpt_key_count

        if key_count_diff > 0:
            new_dict = {
                key_count_diff * "module." + k: v
                for k, v in checkpoint["model"].items()
            }
        elif key_count_diff < 0:
            new_dict = {
                k[len("module.") * abs(key_count_diff) :]: v
                for k, v in checkpoint["model"].items()
            }
        else:
            new_dict = checkpoint["model"]

        strict = self.config["task"].get("strict_load", True)
        load_state_dict(self.model, new_dict, strict=strict)

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint and checkpoint["lr_scheduler"] is not None:
            self.scheduler.scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if "ema" in checkpoint and checkpoint["ema"] is not None:
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self.ema = None
