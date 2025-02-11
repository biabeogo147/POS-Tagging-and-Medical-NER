from torch.utils.data import Dataset
from typing import List
import torch


class PosTagging_Dataset(Dataset):
    def __init__(self,
                 sentences: List[List[str]],
                 tags: List[List[str]],
                 tokenizer,
                 label2id,
                 max_len=128):
        super().__init__()
        self.sentences = sentences
        self.tags = tags
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        input_token = self.sentences[idx]
        label_token = self.tags[idx]

        input_token = self.tokenizer.convert_tokens_to_ids(input_token)
        attention_mask = [1] * len(input_token)
        labels = [self.label2id[token] for token in label_token]

        return {
            "input_ids": self.pad_and_truncate(input_token, pad_id=self.tokenizer.pad_token_id),
            "labels": self.pad_and_truncate(labels, pad_id=self.label2id["0"]),
            "attention_mask": self.pad_and_truncate(attention_mask, pad_id=0)
        }

    def pad_and_truncate(self, inputs: List[int], pad_id: int):
        if len(inputs) < self.max_len:
            padded_inputs = inputs + [pad_id] * (self.max_len - len(inputs))
        else:
            padded_inputs = inputs[:self.max_len]
        return torch.as_tensor(padded_inputs)