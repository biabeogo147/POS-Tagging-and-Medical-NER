import torch
from typing import List
from torch.utils.data import Dataset
from medical_ner.preprocessing_maccrobat import preprocessing

MAX_LEN = 512


class NER_Dataset(Dataset):
    def __init__(self, input_texts, input_labels, tokenizer, label2id, max_len=MAX_LEN):
        super().__init__()
        self.tokens = input_texts
        self.labels = input_labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        input_token = self.tokens[idx]
        label_token = [self.label2id[label] for label in self.labels[idx]]

        input_token = self.tokenizer.convert_tokens_to_ids(input_token)
        attention_mask = [1] * len(input_token)

        input_ids = self.pad_and_truncate(input_token, pad_id=self.tokenizer.pad_token_id)
        labels = self.pad_and_truncate(label_token, pad_id=0)
        attention_mask = self.pad_and_truncate(attention_mask, pad_id=0)

        return {
            "input_ids": torch.as_tensor(input_ids),
            "labels": torch.as_tensor(labels),
            "attention_mask": torch.as_tensor(attention_mask)
        }

    def pad_and_truncate(self, inputs: List[int], pad_id: int):
        if len(inputs) < self.max_len:
            padded_inputs = inputs + [pad_id] * (self.max_len - len(inputs))
        else:
            padded_inputs = inputs[:self.max_len]
        return padded_inputs


def get_dataset():
    inputs_train, labels_train, inputs_val, labels_val, label2id, id2label, tokenizer = preprocessing()
    train_set = NER_Dataset(inputs_train, labels_train, tokenizer, label2id)
    val_set = NER_Dataset(inputs_val, labels_val, tokenizer, label2id)
    return train_set, val_set, label2id, id2label, tokenizer