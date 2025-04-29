from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from medical_ner.medical_ner_dataset import get_dataset
from util.metric import compute_metrics
import os


if __name__ == "__main__":
    train_set, val_set, label2id, id2label, tokenizer = get_dataset()

    model = AutoModelForTokenClassification.from_pretrained(
        "d4data/biomedical-ner-all",
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir="out_dir",
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, label2id["O"])
    )

    trainer.train()

    model.save_pretrained(os.path.join("..\\out_dir", "ner_model"))
    tokenizer.save_pretrained(os.path.join("..\\out_dir", "ner_model"))