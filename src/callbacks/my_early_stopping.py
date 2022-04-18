from pytorch_lightning.callbacks import EarlyStopping


class MyEarlyStopping(EarlyStopping):
    def __init__(self, **kwargs):
        super(MyEarlyStopping, self).__init__(**kwargs)

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state
    ) -> None:
        # print(callback_state)
        if callback_state['wait_count'] >= callback_state['patience']:
            # stop every ddp process if any world process decides to stop
            should_stop = trainer.strategy.reduce_boolean_decision(True)
            trainer.should_stop = trainer.should_stop or should_stop
            if should_stop:
                self.stopped_epoch = trainer.current_epoch
            if self.verbose:
                self._log_info(trainer, "EarlyStopping triggered")