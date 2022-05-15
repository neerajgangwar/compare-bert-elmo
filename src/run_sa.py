import numpy as np

from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoConfig,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    HfArgumentParser
)
from utils.args import ModelArguments, SADataTrainingArguments


tokenizer = None
input_field = None
metric_accuracy = load_metric("accuracy")
metric_f1 = load_metric("f1")
f1_avg_vals = None


def computeMetrics(eval_pred):
    assert not f1_avg_vals is None, "f1_avg_vals is not defined"

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    metrics = {"accuracy": accuracy}
    for f1_avg_val in f1_avg_vals.split(","):
        f1 = metric_f1.compute(predictions=predictions, references=labels, average=f1_avg_val)["f1"]
        metrics[f1_avg_val] = f1

    return metrics


def preprocess(examples):
    assert not tokenizer is None, "tokenizer is not defined"
    assert not input_field is None, "input_field is not defined"
    return tokenizer(examples[input_field], truncation=True)


def trainSST(model, tokenizer, outdir, train_dataset, eval_dataset):
    # Training arguments
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=outdir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        save_strategy="steps",
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=computeMetrics,
    )
    trainer.train()
    trainer.save_model(outdir)

    metrics = trainer.evaluate()
    print(f"eval metrics: {metrics}")


def main():
    global tokenizer, input_field, f1_avg_vals

    parser = HfArgumentParser((ModelArguments, TrainingArguments, SADataTrainingArguments))
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=data_args.num_labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=True,
        max_len=data_args.max_seq_length,
    )

    # Load dataset
    dataset = load_dataset(data_args.dataset_name)
    input_field = data_args.input_field
    tokenized_dataset = dataset.map(preprocess, batched=True)
    train_dataset = tokenized_dataset[data_args.train_split_name]
    eval_dataset = tokenized_dataset[data_args.val_split_name]
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
    f1_avg_vals = data_args.f1_average_vals

    # Train
    trainSST(model, tokenizer, training_args.output_dir, train_dataset, eval_dataset)  


if __name__ == "__main__":
    main()
