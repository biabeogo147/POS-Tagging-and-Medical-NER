import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical NER Inference")
    parser.add_argument("--input", type=str, required=True, help="Input sentence for Medical NER")
    args = parser.parse_args()
    test_sentence = args.input

    # test_sentence = """A 48 year-old female presented with vaginal bleeding and abnormal
    # Pap smears. Upon diagnosis of invasive non-keratinizing SCC of the cervix, she
    # underwent a radical hysterectomy with salpingo-oophorectomy which demonstrated
    # positive spread to the pelvic lymph nodes and the parametrium. Pathological
    # examination revealed that the tumour also extensively involved the lower uterine
    # segment."""

    model_dir = os.path.join("..\\out_dir", "ner_model")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    inputs = tokenizer(test_sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)

    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    model = model.to("cuda")

    outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().cpu().numpy())
    for token, pred in zip(tokens, preds):
        if token not in ["[CLS]", "[SEP]", "[PAD]"]:
            print(f"{token}\t{id2label[pred]}")