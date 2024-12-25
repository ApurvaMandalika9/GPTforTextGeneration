import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, conditions=None, conditions_split_id=None):
        self.texts = texts
        self.conditions = conditions  # New addition
        self.conditions_split_id = conditions_split_id  # New addition
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].strip()
        if self.conditions is not None:
            # Concatenate condition and text
            condition = self.conditions[idx].strip()
            full_text = condition + " " + text
        else:
            full_text = text
        if self.conditions_split_id is not None:
            condition_split_id = int(self.conditions_split_id[idx].strip())
        elif self.conditions is not None:
            condition_split_id = len(condition.split())
        else:
            condition_split_id = 0
        encoded_text = self.tokenizer.batch_encode_plus([full_text])
        raw_input_ids = torch.tensor(encoded_text["input_ids"], dtype=torch.long).squeeze()
        input_ids = raw_input_ids[:-1]
        targets = raw_input_ids[1:]
        return input_ids, targets, condition_split_id


