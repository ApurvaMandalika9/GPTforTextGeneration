import json
import os

import numpy as np
import torch
from tqdm import tqdm


def load_tokenizer(tokenizer_path, max_length):
    tokenizer = SimpleTokenizer(max_length)
    tokenizer.load_vocab(tokenizer_path)
    return tokenizer


def build_tokenizer(args, data_SCAN, max_len, tokenizer_root):
    os.makedirs(tokenizer_root, exist_ok=True)
    tokenizer_path = tokenizer_root + f"/{args.data_split}_vocab.json"
    if os.path.exists(tokenizer_path):
        print(f"The file '{tokenizer_path}' exists. Loading tokenizer.")
        tokenizer = load_tokenizer(tokenizer_path, max_len)
    elif args.task == 'train':
        print(f"Building tokenizer at {tokenizer_path}.")
        tokenizer = SimpleTokenizer(max_length=max_len)

        # --- actions come first ---
        for data in tqdm(data_SCAN['train'], desc="Building tokenizer for actions"):
            tokenizer.fit_on_text(data['actions'])

        # --- commands come second ---
        for data in tqdm(data_SCAN['train'], desc="Building tokenizer for commands"):
            tokenizer.fit_on_text(data['commands'])
        tokenizer.save_vocab(tokenizer_path)
        print("tokenizer saved")
    else:
        raise ValueError("Tokenizer file does not exist. Please train the model first.")

    print(tokenizer.get_vocab())  # Print vocabulary
    return tokenizer, tokenizer.get_vocab_size()


class SimpleTokenizer:
    def __init__(self, max_length):
        self.vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
        self.count = 4
        self.max_length = max_length
        self.token_decoder_func = np.vectorize(self.token_decode)

    def fit_on_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.fit_on_text(line.strip())

    def fit_on_text(self, text):
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = self.count
                self.count += 1

    def encode(self, text):
        sequence = [self.vocab.get(word, self.vocab["<unk>"]) for word in text.split()]
        sequence = [self.vocab["<s>"]] + sequence + [self.vocab["</s>"]]
        padding_length = self.max_length - len(sequence)

        if padding_length > 0:
            sequence.extend([self.vocab["<pad>"]] * padding_length)

        return sequence[:self.max_length]

    def decode(self, token_ids):
        # --- Remove any characters after the <pad> and </s> ---
        end_ids = torch.nonzero((token_ids == self.vocab["<pad>"]) | (token_ids == self.vocab["</s>"]))
        end = end_ids.min() if len(end_ids) > 0 else len(token_ids)
        token_ids = token_ids[:end]
        # --- Remove the <s> token ---
        token_ids = token_ids[token_ids != self.vocab["<s>"]]
        assert (token_ids == self.vocab["<pad>"]).sum() + (token_ids == self.vocab["<s>"]).sum() + (
                token_ids == self.vocab[
            "</s>"]).sum() == 0, "There are still <s>, <pad>, or </s> tokens in the decoded sequence"

        decoded_tokens = self.token_decoder_func(token_ids.cpu())

        return ' '.join(decoded_tokens)

    def generation_encode(self, text):
        sequence = [self.vocab.get(word, self.vocab["<unk>"]) for word in text.split()]
        sequence = [self.vocab["<s>"]] + sequence  # Do not add the ending token for generation
        return sequence

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    def get_vocab(self):
        return self.vocab

    def get_vocab_size(self):
        return len(self.vocab)

    def save_vocab(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(self.vocab, file)

    def token_decode(self, token_id):
        return self.reverse_vocab.get(token_id, "<unk>")

    def load_vocab(self, file_path):
        with open(file_path, 'r') as file:
            self.vocab = json.load(file)
            self.count = len(self.vocab)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def batch_encode_plus(self, texts):
        encodings = self.encode_batch(texts)
        attention_masks = [[float(token != self.vocab["<pad>"]) for token in encoding] for encoding in encodings]

        return {
            "input_ids": encodings,
            "attention_mask": attention_masks
        }
