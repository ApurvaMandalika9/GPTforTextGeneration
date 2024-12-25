"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import logging

import math
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e2  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e7  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    run_name = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, n_layer, n_head, n_embd):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.tokens = 0
        self.results = []
        
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, epoch, model, best_loss, optimizer, tokens, scaler, save_path):
        raw_model = model.module if hasattr(model, "module") else model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),  # Include scaler state
            'tokens': tokens,
            'best_loss': best_loss,
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, load_path, optimizer, scaler):
        checkpoint = torch.load(load_path, map_location='cuda')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.tokens = checkpoint['tokens']
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        return checkpoint['epoch'], checkpoint['best_loss']

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        scaler = GradScaler()
        start_epoch = -1
        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader), desc=f"epoch {epoch + 1} iter {0}: train loss {0:.5f}. lr {0:.4e}") if is_train else enumerate(loader)
            for it, (input_ids, targets, condition_split_id) in pbar:

                # place data on the correct device
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                condition_split_id = condition_split_id.to(self.device)

                # forward the model
                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(is_train):
                        logits, loss, _ = model(input_ids, targets=targets, condition_split_id=condition_split_id)
                        loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (
                                    targets >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    if (it + epoch * len(loader)) % 500 == 0 and it > 0:
                        print(f"\rstep_train_loss: {loss} train_step: {it + epoch * len(loader)}, learning_rate: {lr}")
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:.4e}")

            if is_train:
                return float(np.mean(losses))

            if not is_train:
                test_loss = float(np.mean(losses))
                print("test loss: %f", test_loss)
                return test_loss

        for epoch in range(start_epoch + 1, config.max_epochs):
            start_time = time.time()
            train_loss = run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            epoch_time = time.time() - start_time

            # Record the results for this epoch
            self.results.append({
                "n_layer": self.n_layer,
                "n_head": self.n_head,
                "n_embd": self.n_embd,
                "validation_loss": test_loss,
                "time_per_epoch": epoch_time
            })

            print(f"epoch_valid_loss: {test_loss}, epoch_train_loss: {train_loss}, epoch: {epoch + 1}")

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                print(f'Saving at epoch {epoch + 1}: {self.config.ckpt_path}')
                self.save_checkpoint(epoch, model, best_loss, optimizer, self.tokens, scaler, self.config.ckpt_path)
        # printing the table with validation_loss and time_per_epoch, for each epoch
        print("\nResults Table:")
        print("{:<10} {:<10} {:<10} {:<20} {:<20}".format(
            "n_layer", "n_head", "n_embd", "validation_loss", "time_per_epoch"))
        
        for result in self.results:
            print("{:<10} {:<10} {:<10} {:<20} {:<20}".format(
                result["n_layer"], result["n_head"], result["n_embd"],
                result["validation_loss"], result["time_per_epoch"]))



