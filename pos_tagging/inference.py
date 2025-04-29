import os
import torch
import argparse
from pos_tagging.pos_tagging_preprocess import build_tag, get_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POS Tagging Inference")
    parser.add_argument("--input", type=str, required=True, help="Input sentence for POS tagging")
    args = parser.parse_args()

    sentences, sentence_tags = get_dataset()
    _, _, id2label = build_tag(sentence_tags)

    model_dir = os.path.join("..\\out_dir", "pos_model")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    test_sentence = args.input
    input_ids = torch.as_tensor([tokenizer.convert_tokens_to_ids(test_sentence.split())])
    input_ids = input_ids.to("cuda")

    outputs = model(input_ids)
    _, preds = torch.max(outputs.logits, -1)
    preds = preds[0].cpu().numpy()

    pred_tags = " ".join([id2label[pred] for pred in preds])
    print(pred_tags)