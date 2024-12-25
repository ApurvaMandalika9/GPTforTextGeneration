import os

from datasets import load_dataset

from dataset import TextDataset
from tokenizer import build_tokenizer
from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig


def train(args):

    # --- load dataset ---
    data_SCAN = load_dataset("scan", args.data_split)

    max_len = args.max_len
    tokenizer, vocab_size = build_tokenizer(args, data_SCAN, max_len, args.output_tokenizer_dir)

    train_val_data = data_SCAN['train'].train_test_split(test_size=0.1, seed=args.seed)
    train_data = train_val_data['train']
    val_data = train_val_data['test']
    # Don't be confused by the name: this is a validation set since it's from the original training set

    train_dataset = TextDataset(train_data['actions'], tokenizer, max_len, conditions=train_data['commands'])
    valid_dataset = TextDataset(val_data['actions'], tokenizer, max_len, conditions=val_data['commands'])
    print(f"train dataset size: {len(train_dataset)}")
    print(f"val dataset size: {len(valid_dataset)}")

    print("loading model")
    mconf = GPTConfig(vocab_size, max_len, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      isconditional=True)
    model = GPT(mconf)

    print('total params:', sum(p.numel() for p in model.parameters()))
    os.makedirs(f'./cond_gpt/weights/', exist_ok=True)
    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                          lr_decay=True, warmup_tokens=0.1 * len(train_data) * max_len,
                          final_tokens=args.max_epochs * len(train_data) * max_len,
                          num_workers=args.num_workers, ckpt_path=args.ckpt_path,
                          run_name=args.run_name, block_size=max_len, generate=False,
                          save_start_epoch=120,
                          grad_norm_clip=args.grad_norm_clip, load_checkpoint_path=None,
                          save_interval_epoch=10)
    trainer = Trainer(model, train_dataset, valid_dataset, tconf, args.n_layer, args.n_head, args.n_embd)
    trainer.train()


