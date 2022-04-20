import os
import sys
import math
import json
import torch

from os import path
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from cloze_task.model import ClozeModel
from callbacks.my_early_stopping import MyEarlyStopping
from callbacks.checkpoint_every_n_steps import CheckpointEveryNSteps
from cloze_task.data_utils.cloze_datamodule import ClozeTaskDataModule
from pytorch_lightning.callbacks import ModelCheckpoint


def experiment(args):
    # WandB logger
    if args.use_wandb:
        logger = WandbLogger(name=args.model_name, id=args.model_name, project='lambada_nlp', save_dir=args.save_dir)
    else:
        logger = TensorBoardLogger(args.save_dir, name=args.model_name)

    # Create datamodule
    datamodule = ClozeTaskDataModule(**vars(args))
    datamodule.setup()

    num_train_examples, one_epoch_batches = datamodule.estimate_train_batches()
    effective_batch_size = int(math.ceil(num_train_examples / one_epoch_batches))

    # args.accumulate_grad_batches = int(math.ceil(args.real_batch_size / effective_batch_size))
    # TODO: Remove this
    if args.accumulate_grad_batches is None:
        args.accumulate_grad_batches = 1

    if args.max_steps < 0:
        if args.train_size is not None:
            args.save_step_frequency = args.train_size // effective_batch_size
            args.max_steps = args.num_save_checkpoint * args.save_step_frequency
        else:
            # 100K steps chosen as the arbitrary number
            args.max_steps = 25_000
            args.save_step_frequency = args.max_steps // args.num_save_checkpoint
    else:
        args.save_step_frequency = args.max_steps // args.num_save_checkpoint

    print(f"One epoch batches: {one_epoch_batches}, Amortized batch size: {effective_batch_size}")
    # print("Acc. grad steps: %d, Number of training steps: %d" %
    #       (args.accumulate_grad_batches, args.max_steps))

    sys.stdout.flush()

    # Callbacks
    lr_logger = LearningRateMonitor()
    args.model_dirpath = path.join(args.save_dir, args.model_name)
    if not path.exists(args.model_dirpath):
        os.makedirs(args.model_dirpath)

    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        dirpath=path.join(args.model_dirpath, "checkpoints"),
        monitor='val_perp',
        mode='min',
        save_top_k=1,
        save_last=True,
    )

    early_stop_callback = MyEarlyStopping(
        monitor='val_perp',
        mode='min',
        patience=args.patience,
        verbose=True,
    )

    # Callback for saving checkpoint every N steps
    checkpoint_n_steps_callback = CheckpointEveryNSteps(
        save_step_frequency=args.save_step_frequency, accumulate_grad_batches=args.accumulate_grad_batches,
        slurm_id=args.slurm_id, slurm_job_mins=args.slurm_job_mins,
    )

    # Check whether to train the model or evaluate
    to_train = True
    last_checkpoint = path.join(checkpoint_callback.dirpath, "last.ckpt")

    if path.isfile(last_checkpoint):
        print("Resuming training from: ", last_checkpoint)
        # Currently, I don't see a way to early stopping when resuming training
        # Below is a hacky way of checking for early stopping criteria from saved checkpoint
        callbacks_state_dict = torch.load(last_checkpoint, map_location='cpu')['callbacks']
        early_stop_callback_state = None
        checkpoint_callback_state = None
        for callback_obj, callback_state in callbacks_state_dict.items():
            if 'EarlyStopping' in callback_obj:
                early_stop_callback_state = callback_state
            if 'ModelCheckpoint' in callback_obj:
                checkpoint_callback_state = callback_state
                # print("Hello checkpoint")

        if early_stop_callback_state['wait_count'] >= early_stop_callback_state['patience']:
            print("Early Stopping Triggered on Resumption!")
            to_train = False
            from argparse import Namespace
            checkpoint_callback = Namespace(**checkpoint_callback_state)
        print()
        print(early_stop_callback_state)
        print(checkpoint_callback)
        print()

    if to_train:
        trainer = Trainer.from_argparse_args(
            args,
            gpus=-1,
            precision=args.precision,
            logger=logger,
            callbacks=[lr_logger, early_stop_callback, checkpoint_callback, checkpoint_n_steps_callback],
            reload_dataloaders_every_n_epochs=1,
            gradient_clip_val=1.0,
            max_steps=args.max_steps,
            check_val_every_n_epoch=1000_000,
        )

        if path.exists(last_checkpoint):
            lm_model = ClozeModel.load_from_checkpoint(last_checkpoint, tokenizer=datamodule.tokenizer)
            trainer.fit(lm_model, datamodule=datamodule, ckpt_path=last_checkpoint)
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

    if args.max_steps:
        lm_model = ClozeModel.load_from_checkpoint(
            checkpoint_path=checkpoint_callback.best_model_path, tokenizer=datamodule.tokenizer)
    else:
        lm_model = ClozeModel(args, tokenizer=datamodule.tokenizer)

    doc_encoder_dir = path.join(args.model_dirpath, "huggingface")
    if not path.exists(doc_encoder_dir):
        os.makedirs(doc_encoder_dir)
    lm_model.model.save_pretrained(save_directory=doc_encoder_dir, save_config=True)
    lm_model.tokenizer.save_pretrained(save_directory=doc_encoder_dir)

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



