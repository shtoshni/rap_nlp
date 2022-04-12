import os
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
            self, save_step_frequency,
            accumulate_grad_batches=1, prefix="N-Step-Checkpoint",
            use_model_checkpoint_filename=True,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_model_checkpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_model_checkpoint_filename = use_model_checkpoint_filename
        self.accumulate_grad_batches = accumulate_grad_batches
        self._counter = 0

    def on_train_batch_end(
        self, trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """ Check if we should save a checkpoint after every train batch """
        global_step = trainer.global_step
        # print("Global step: %d, Counter: %d" % (global_step, self._counter))
        if global_step > 0 and global_step % self.save_step_frequency == 0 and \
                (self._counter % self.accumulate_grad_batches == 0):
            trainer.run_evaluation()

        self._counter += 1
