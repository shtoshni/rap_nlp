import sys
import math
import os
import json
import torch

from os import path
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from cloze_task.model import ClozeModel
from callbacks.model_checkpoint import MyModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def experiment(args):
    # WandB logger
    if args.use_wandb:
        logger = WandbLogger(name=args.model_name, id=args.model_name, project='lambada_nlp', save_dir=args.save_dir)
    else:
        logger = TensorBoardLogger(args.save_dir, name=args.model_name)

    # Callbacks
    lr_logger = LearningRateMonitor()
    args.model_dirpath = path.join(args.save_dir, args.model_name)

    checkpoint_callback = MyModelCheckpoint(
        verbose=True,
        dirpath=path.join(args.model_dirpath, "checkpoints"),
        monitor='val_perp',
        mode='min',
        save_top_k=1,
        save_last=True,
        period=1,
        prefix='cloze')

    early_stop_callback = EarlyStopping(
        monitor='val_perp',
        mode='min',
        patience=args.patience,
        verbose=True,
    )

    sys.stdout.flush()

    # Create datamodule
    from cloze_task.data_utils.cloze_datamodule import ClozeTaskDataModule
    datamodule = ClozeTaskDataModule(**vars(args))
    datamodule.setup()

    num_train_examples, one_epoch_batches = datamodule.estimate_train_batches()
    effective_batch_size = int(math.ceil(num_train_examples / one_epoch_batches))
    # args.num_training_steps = one_epoch_batches * args.max_epochs
    # print(f"\n\nOne epoch batches: {one_epoch_batches}, Amortized batch size: {effective_batch_size}")
    # print(f"Number of training steps: {args.num_training_steps}\n\n")

    args.accumulate_grad_batches = int(math.ceil(args.real_batch_size / effective_batch_size))
    args.num_training_steps = one_epoch_batches * args.max_epochs / args.accumulate_grad_batches

    print(f"One epoch batches: {one_epoch_batches}, Amortized batch size: {effective_batch_size}")
    print("Acc. grad steps: %d, Number of training steps: %d" %
          (args.accumulate_grad_batches, args.num_training_steps))

    to_train = True
    last_checkpoint = path.join(checkpoint_callback.dirpath, "cloze-last.ckpt")
    if path.isfile(last_checkpoint):
        print("Resuming training from: ", last_checkpoint)
        # Currently, I don't see a way to early stopping when resuming training
        # Below is a hacky way of checking for early stopping criteria from saved checkpoint
        checkpoint = torch.load(last_checkpoint, map_location='cpu')['callbacks']
        early_stop_callback_state = checkpoint[EarlyStopping]
        if early_stop_callback_state['wait_count'] >= early_stop_callback_state['patience']:
            print("Early Stopping Triggered on Resumption!")
            to_train = False
            from argparse import Namespace
            checkpoint_callback = Namespace(**checkpoint[MyModelCheckpoint])

    if to_train:
        trainer = Trainer.from_argparse_args(
            args,
            amp_level='O2',
            gpus=-1,
            precision=args.precision,
            weights_save_path=args.save_dir,
            resume_from_checkpoint=last_checkpoint,
            logger=logger,
            callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
            reload_dataloaders_every_epoch=True,
            gradient_clip_val=1.0, terminate_on_nan=True)
        if path.exists(last_checkpoint):
            lm_model = ClozeModel.load_from_checkpoint(last_checkpoint, tokenizer=datamodule.tokenizer)
        else:
            lm_model = ClozeModel(args, tokenizer=datamodule.tokenizer)
        trainer.fit(lm_model, datamodule=datamodule)
    else:
        trainer = Trainer.from_argparse_args(
            args,
            gpus=-1,
            precision=args.precision,
            weights_save_path=args.save_dir,
            logger=logger,
        )

    lm_model = ClozeModel.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path, tokenizer=datamodule.tokenizer)

    print("Best validation model path: ", checkpoint_callback.best_model_path)
    print("Best validation performance:", checkpoint_callback.best_model_score)

    test_perf = trainer.test(lm_model, datamodule=datamodule)
    print(test_perf)
    test_perf = test_perf[0]

    # Get best model path
    test_perf['best_model_path'] = checkpoint_callback.best_model_path
    test_perf['best_val_score'] = checkpoint_callback.best_model_score

    for key in test_perf:
        if isinstance(test_perf[key], torch.Tensor):
            test_perf[key] = round(test_perf[key].item(), 4)

    perf_file = path.join(args.model_dirpath, "perf.json")
    with open(perf_file, 'w') as f:
        f.write(json.dumps(test_perf))

    if args.slurm_id is not None:
        slurm_perf_dir = path.join(args.save_dir, 'perf')
        slurm_perf_file = path.join(slurm_perf_dir, f'{args.slurm_id}.json')
        with open(slurm_perf_file, 'w') as f:
            f.write(json.dumps(test_perf))



