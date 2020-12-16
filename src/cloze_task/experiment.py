import os
import torch
import numpy as np
import sys
import logging
import time

from os import path
from collections import OrderedDict
from transformers import get_linear_schedule_with_warmup
from cloze_task.utils import load_lambada_data
from cloze_task.model import ClozeModel
from pytorch_utils.utils import print_model_info

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()


class Experiment():
    def __init__(self, args, data_dir=None, **kwargs):
        self.args = args
        self.model_args = vars(args)

        self.seed = args.seed
        self.update_frequency = 1e4

        self.train_examples, self.dev_examples, self.test_examples = load_lambada_data(
            data_dir, num_train_docs=args.num_train_docs, filt_train=args.filt_train)

        # Get model paths
        self.model_dir = args.model_dir
        self.data_dir = data_dir
        self.model_path = path.join(args.model_dir, 'model.pth')
        self.best_model_path = path.join(args.best_model_dir, 'model.pth')

        # Initialize model and training metadata
        self.finetune = args.finetune

        self.model = ClozeModel(finetune=self.finetune, **kwargs)

        self.max_epochs = args.max_epochs
        self.max_stuck_epochs = args.max_stuck_epochs
        self.train_info, self.optimizer, self.optim_scheduler, self.optimizer_params = {}, {}, {}, {}

        self.initialize_setup(init_lr=args.init_lr)
        print_model_info(self.model)
        sys.stdout.flush()

        if not eval:
            self.train(max_epochs=args.max_epochs, max_gradient_norm=5.0)

        # Finally evaluate model
        self.final_eval()

    def initialize_setup(self, init_lr, ft_lr=5e-5):
        """Initialize model and training info."""
        param_list = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_list.append(param)

        self.optimizer = torch.optim.AdamW(
            param_list, lr=init_lr, eps=1e-6)

        total_training_steps = len(self.train_examples) * self.max_epochs
        self.optim_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0.1 * total_training_steps,
            num_training_steps=total_training_steps)

        if not path.exists(self.model_path):
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            # Try to initialize the mention model part
        else:
            logger.info('Loading previous model: %s' % self.model_path)
            # Load model
            self.load_model(self.model_path)

    def train(self, max_epochs, max_gradient_norm):
        """Train model"""
        model = self.model
        epochs_done = self.train_info['epoch']
        optimizer = self.optimizer
        scheduler = self.optim_scheduler

        if self.train_info['num_stuck_epochs'] >= self.max_stuck_epochs:
            return

        for epoch in range(epochs_done, max_epochs):
            logger.info("\n\nStart Epoch %d" % (epoch + 1))
            start_time = time.time()
            # Setup training
            model.train()
            np.random.shuffle(self.train_examples)

            for example_idx, cur_example in enumerate(self.train_examples):
                def handle_example(example):
                    optimizer.zero_grad()

                    # Send the copy of the example, as the document could be truncated during training
                    loss = model(example)
                    total_loss = loss['total']
                    if total_loss is None:
                        return None

                    if torch.isnan(total_loss):
                        print("Loss is NaN")
                        sys.exit()

                    total_loss.backward()
                    # Perform gradient clipping and update parameters
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_gradient_norm)

                    optimizer.step()
                    scheduler.step()

                    self.train_info['global_steps'] += 1
                    return total_loss.item()

                example_loss = handle_example(cur_example)

                if self.train_info['global_steps'] % self.update_frequency == 0:
                    logger.info('{} {:.3f} Max mem {:.3f} GB'.format(
                        example_idx, example_loss,
                        (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0)
                    )

                    if torch.cuda.is_available():
                        try:
                            torch.cuda.reset_peak_memory_stats()
                        except AttributeError:
                            # In case of an earlier torch version
                            torch.cuda.reset_max_memory_allocated()

            sys.stdout.flush()
            # Update epochs done
            self.train_info['epoch'] = epoch + 1

            # Dev performance
            eval_perf = self.eval_model()

            # Assume that the model didn't improve
            self.train_info['num_stuck_epochs'] += 1

            # Update model if dev performance improves
            if eval_perf > self.train_info['val_perf']:
                self.train_info['num_stuck_epochs'] = 0
                self.train_info['val_perf'] = eval_perf
                logger.info('Saving best model')
                self.save_model(self.best_model_path, model_type='best')

            # Save model
            self.save_model(self.model_path)

            # Get elapsed time
            elapsed_time = time.time() - start_time
            logger.info("Epoch: %d, Acc: %.1f, Max Acc: %.1f, Time: %.2f"
                        % (epoch + 1, eval_perf, self.train_info['val_perf'], elapsed_time))

            sys.stdout.flush()
            logger.handlers[0].flush()

            if self.train_info['num_stuck_epochs'] >= self.max_stuck_epochs:
                return

    def eval_model(self, split='dev', final_eval=False):
        pass

    def load_model(self, location, model_type='last'):
        if torch.cuda.is_available():
            checkpoint = torch.load(location)
        else:
            checkpoint = torch.load(location, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.train_info = checkpoint['train_info']

        if model_type != 'best':
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.optim_scheduler.load_state_dict(checkpoint['scheduler'])
            torch.set_rng_state(checkpoint['rng_state'])
            np.random.set_state(checkpoint['np_rng_state'])

    def save_model(self, location, model_type='last'):
        """Save model"""
        save_dict = {}

        model_state_dict = OrderedDict(self.model.state_dict())
        for key in self.model.state_dict():
            if 'bert.' in key:
                del model_state_dict[key]

        save_dict['model'] = model_state_dict
        save_dict['model_args'] = self.model_args
        save_dict['train_info'] = self.train_info

        # Don't need optimizers for inference, hence not saving these for the best models.
        if model_type != 'best':
            # Regular model saved during training.
            save_dict.update({
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.optim_scheduler.state_dict(),
                'rng_state': torch.get_rng_state(),
                'np_rng_state': np.random.get_state()
            })

        torch.save(save_dict, location)
        logging.info(f"Model saved at: {location}")






