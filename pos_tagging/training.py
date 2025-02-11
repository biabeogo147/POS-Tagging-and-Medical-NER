import evaluate
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForTokenClassification
from pos_tagging_dataset.pos_tagging_dataset import PosTagging_Dataset
from pos_tagging_dataset.pos_tagging_preprocess import get_dataset, build_tag
from util.metric import compute_metrics

if __name__ == "__main__":
    sentences, sentence_tags = get_dataset()
    _, label2id, _ = build_tag(sentence_tags)

    train_sentences, test_sentences, train_tags, test_tags = train_test_split(
        sentences, sentence_tags, test_size=0.3, random_state=42
    )
    valid_sentences, test_sentences, valid_tags, test_tags = train_test_split(
        test_sentences, test_tags, test_size=0.5, random_state=42
    )

    model_name = "QCRI/bert-base-multilingual-cased-pos-english"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )

    train_dataset = PosTagging_Dataset(train_sentences, train_tags, tokenizer, label2id)
    val_dataset = PosTagging_Dataset(valid_sentences, valid_tags, tokenizer, label2id)
    test_dataset = PosTagging_Dataset(test_sentences, test_tags, tokenizer, label2id)

    model = AutoModelForTokenClassification.from_pretrained(model_name)
    accuracy = evaluate.load("accuracy")
    ignore_label = len(label2id)
    training_args = TrainingArguments(
        output_dir="out_dir",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, ignore_label, accuracy)
    )

    trainer.train()