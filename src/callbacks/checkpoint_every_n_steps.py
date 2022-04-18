import sys
import time
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
            self, save_step_frequency,
            accumulate_grad_batches=1, prefix="N-Step-Checkpoint",
            use_model_checkpoint_filename=True,
            slurm_id=None, slurm_job_mins=236,
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

        # Slurm counter
        self.slurm_id = slurm_id
        self.eval_counter = 0
        self.start_time = time.time()
        self.slurm_job_mins = slurm_job_mins

    def on_train_batch_end(
        self, trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """ Check if we should save a checkpoint after every train batch """
        global_step = trainer.global_step
        # print("Global step: %d, Counter: %d" % (global_step, self._counter))
        if global_step > 0 and global_step % self.save_step_frequency == 0 and \
                (self._counter % self.accumulate_grad_batches == 0):
            # trainer.
            trainer.training = False
            stage = trainer.state.stage
            trainer.state.stage = RunningStage.VALIDATING
            trainer._run_evaluate()
            trainer.state.stage = stage
            trainer.training = True
            trainer._logger_connector._epoch_end_reached = False

            self.eval_counter += 1
            if self.slurm_id:
                total_time = (time.time() - self.start_time) / 60
                avg_eval_time = (total_time / self.eval_counter)
                rem_time = self.slurm_job_mins - total_time
                print("Remaining time: %.2f, Avg eval time: %.2f"
                      % (rem_time, avg_eval_time))
                if rem_time < avg_eval_time:
                    sys.exit()

        self._counter += 1
