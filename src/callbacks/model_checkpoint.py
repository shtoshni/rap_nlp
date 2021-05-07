"""
Model Checkpointing
===================

Automatically save model checkpoints during training.

"""
import gc
import os
import re

import numpy as np
from typing import Dict, Any

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, **kwargs):
        super(MyModelCheckpoint, self).__init__(**kwargs)

    def _do_save(self, trainer, filepath: str, weights_only=None):
        # in debugging, track when we save checkpoints
        trainer.dev_debugger.track_checkpointing_history(filepath)

        # make paths
        if trainer.is_global_zero:
            self._fs.makedirs(os.path.dirname(filepath), exist_ok=True)

        # delegate the saving to the trainer
        if self.save_function is not None:
            if weights_only is None:
                self.save_function(filepath, self.save_weights_only)
            else:
                self.save_function(filepath, weights_only)
        else:
            raise ValueError(".save_function() not set")

    def _save_last_checkpoint(self, trainer, pl_module, ckpt_name_metrics):
        super(MyModelCheckpoint, self)._save_last_checkpoint(trainer, pl_module, ckpt_name_metrics)
        torch.cuda.synchronize()
        gc.collect()

    def _update_best_and_save(
        self, current: torch.Tensor, epoch: int, step: int, trainer, pl_module, metrics
    ):
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, torch.Tensor) and torch.isnan(current):
            current = torch.tensor(float('inf' if self.mode == "min" else '-inf'))

        filepath = self._get_metric_interpolated_filepath_name(metrics, epoch, step, trainer, del_filepath)

        # save the current score
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor} reached {current:0.5f}"
                f' (best {self.best_model_score:0.5f}), saving model to "{filepath}" as top {k}'
            )
        self._do_save(trainer, filepath, weights_only=True)

        if del_filepath is not None and filepath != del_filepath:
            self._del_model(del_filepath)

