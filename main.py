import argparse
import os
import random

import numpy as np
import torch

from generate import generate
from train import train


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str,
                        help='name of this task: train/generate', required=True)
    parser.add_argument('--run_name', type=str,
                        help="name for an experiment run", required=False)
    parser.add_argument('--data_split', type=str, default='simple',
                        help="data split of SCAN dataset", required=False)
    parser.add_argument('--n_layer', type=int, default=2,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=2,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=16,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=60,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=32,
                        help="batch size", required=False)
    parser.add_argument('--num_workers', type=int, default=8,
                        help="number of workers for data loaders", required=False)
    parser.add_argument('--learning_rate', type=float,
                        default=4e-4, help="learning rate", required=False)
    parser.add_argument('--max_len', type=int, default=128,
                        help="max_len", required=False)
    parser.add_argument('--seed', type=int, default=44,
                        help="seed", required=False)
    parser.add_argument('--grad_norm_clip', type=float, default=1.0,
                        help="gradient norm clipping. smaller values mean stronger normalization.", required=False)
    parser.add_argument('--output_tokenizer_dir',
                        default='./tokenizer',
                        help="Path to the saved tokenizer directory", required=False)

    ### YOUR CODE HERE ###
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ### YOUR CODE HERE ###

    args = parser.parse_args()
    args.ckpt_path = f'./cond_gpt/weights/{args.run_name}_{args.data_split}split_{args.n_layer}layer_{args.n_head}head_{args.n_embd}embd_{args.batch_size}bs.pt'

    set_seed(args.seed)

    if args.task == 'train':
        train(args)
    elif args.task == 'generate':
        generate(args)
    else:
        raise ValueError("Invalid task")
