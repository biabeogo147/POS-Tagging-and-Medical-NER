import os
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


class Preprocessing_Maccrobat:
    def __init__(self, dataset_folder, tokenizer):
        self.file_ids = [f.split(".")[0] for f in os.listdir(dataset_folder) if f.endswith('.txt')]
        self.text_files = [f + ".txt" for f in self.file_ids]
        self.anno_files = [f + ".ann" for f in self.file_ids]
        self.num_samples = len(self.file_ids)
        self.texts: List[str] = []
        for i in range(self.num_samples):
            file_path = os.path.join(dataset_folder, self.text_files[i])
            with open(file_path, 'r') as f:
                self.texts.append(f.read())

        self.tags: List[Dict[str, str]] = []
        for i in range(self.num_samples):
            file_path = os.path.join(dataset_folder, self.anno_files[i])
            with open(file_path, 'r') as f:
                text_bound_ann = [t.split("\t") for t in f.read().split("\n") if t.startswith("T")]
                text_bound_lst = []
                for text_b in text_bound_ann:
                    label = text_b[1].split(" ")
                    try:
                        _ = int(label[1])
                        _ = int(label[2])
                        tag = {
                            "text": text_b[-1],
                            "label": label[0],
                            "start": label[1],
                            "end": label[2]
                        }
                        text_bound_lst.append(tag)
                    except ValueError:
                        pass
                self.tags.append(text_bound_lst)
        self.tokenizer = tokenizer

    def process(self) -> Tuple[List[List[str]], List[List[str]]]:
        input_texts = []
        input_labels = []
        for idx in range(self.num_samples):
            full_text = self.texts[idx]
            tags = self.tags[idx]
            label_offset = []
            continuous_label_offset = []
            for tag in tags:
                offset = list(range(int(tag["start"]), int(tag["end"]) + 1))
                label_offset.append(offset)
                continuous_label_offset.extend(offset)

            all_offset = list(range(len(full_text)))
            zero_offset = [offset for offset in all_offset if offset not in continuous_label_offset]
            zero_offset = Preprocessing_Maccrobat.find_continuous_ranges(zero_offset)

            self.tokens = []
            self.labels = []
            self._merge_offset(full_text, tags, zero_offset, label_offset)
            assert len(self.tokens) == len(self.labels), "Length of tokens and labels are not equal"

            input_texts.append(self.tokens)
            input_labels.append(self.labels)

        return input_texts, input_labels

    def _merge_offset(self, full_text, tags, zero_offset, label_offset):
        i = j = 0
        while i < len(zero_offset) and j < len(label_offset):
            if zero_offset[i][0] < label_offset[j][0]:
                self._add_zero(full_text, zero_offset, i)
                i += 1
            else:
                self._add_label(full_text, label_offset, j, tags)
                j += 1

        while i < len(zero_offset):
            self._add_zero(full_text, zero_offset, i)
            i += 1

        while j < len(label_offset):
            self._add_label(full_text, label_offset, j, tags)
            j += 1

    def _add_zero(self, full_text, offset, index):
        start, *_, end = offset[index] if len(offset[index]) > 1 else (offset[index][0], offset[index][0] + 1)
        text = full_text[start:end]
        text_tokens = self.tokenizer.tokenize(text)
        self.tokens.extend(text_tokens)
        self.labels.extend(["O"] * len(text_tokens))

    def _add_label(self, full_text, offset, index, tags):
        start, *_, end = offset[index] if len(offset[index]) > 1 else (offset[index][0], offset[index][0] + 1)
        text = full_text[start:end]
        text_tokens = self.tokenizer.tokenize(text)
        self.tokens.extend(text_tokens)
        self.labels.extend(
            [f"B-{tags[index]['label']}"] + [f"I-{tags[index]['label']}"] * (len(text_tokens) - 1)
        )

    @staticmethod
    def build_label2id(tokens: List[List[str]]):
        label2id = {}
        id_counter = 0
        for token in [token for sublist in tokens for token in sublist]:
            if token not in label2id:
                label2id[token] = id_counter
                id_counter += 1
        return label2id

    @staticmethod
    def find_continuous_ranges(data: List[int]):
        if not data:
            return []
        ranges = []
        start = data[0]
        prev = data[0]
        for number in data[1:]:
            if number != prev + 1:
                ranges.append(list(range(start, prev + 1)))
                start = number
            prev = number
        ranges.append(list(range(start, prev + 1)))
        return ranges


def preprocessing():
    tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
    dataset_folder = "./MACCROBAT2018"

    Maccrobat_builder = Preprocessing_Maccrobat(dataset_folder, tokenizer)
    input_texts, input_labels = Maccrobat_builder.process()

    label2id = Preprocessing_Maccrobat.build_label2id(input_labels)
    id2label = {v: k for k, v in label2id.items()}

    inputs_train, inputs_val, labels_train, labels_val = train_test_split(
        input_texts,
        input_labels,
        test_size=0.2,
        random_state=42
    )

    return  inputs_train, labels_train, inputs_val, labels_val, label2id, id2label, tokenizer