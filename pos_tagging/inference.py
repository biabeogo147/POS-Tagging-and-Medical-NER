import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from pos_tagging.pos_tagging_dataset.pos_tagging_preprocess import build_tag, get_dataset

if __name__ == "__main__":
    sentences, sentence_tags = get_dataset()
    _, _, id2label = build_tag(sentence_tags)

    model_name = "QCRI/bert-base-multilingual-cased-pos-english"
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )

    test_sentence = "We are exploring the topic of deep learning"
    input_ids = torch.as_tensor([tokenizer.convert_tokens_to_ids(test_sentence.split())])
    input_ids = input_ids.to("cuda")

    outputs = model(input_ids)
    _, preds = torch.max(outputs.logits, -1)
    preds = preds[0].cpu().numpy()

    pred_tags = " ".join([id2label[pred] for pred in preds])
    print(pred_tags)