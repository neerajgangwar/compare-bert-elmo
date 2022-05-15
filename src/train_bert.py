import os
from datetime import datetime
from optparse import OptionParser

import numpy as np
from transformers import BertConfig
from transformers import BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
from datasets import load_metric


def checkAndCreateDirectories(options):
    # Data files
    datafiles = options.data
    assert datafiles is not None, f"data argument is mandatory"
    datafiles = datafiles.split(",")
    assert len(datafiles) == 2, f"Training and validation files are mandatory"
    trainfile = datafiles[0]
    valfile = datafiles[1]

    # Tokenizer
    assert os.path.exists(options.tokenizer), f"{options.tokenizer} does not exist"
    tokenizer = options.tokenizer

    # Output
    if not os.path.exists(options.out):
        os.mkdir(options.out)

    return trainfile, valfile, tokenizer, options.out


def computeMetrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(f"This line is getting printed")
    return {metric.compute(predictions=predictions, references=labels)}


def train(trainfile, valfile, tokenizerfile, outdir, init_model=None):
    # Create output directory
    outdir = f"{outdir}/{datetime.now().strftime('%Y%m%d-%H%M%S%f')}"
    if os.path.exists(outdir):
        raise Exception(f"{outdir} already exists.")

    os.mkdir(outdir)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizerfile, max_len=512)

    # Initialize model
    config = BertConfig(vocab_size=30_000)
    if init_model is None:
        print(f"Training the model from scratch.")
        model = BertForMaskedLM(config=config)
    else:
        print(f"Loading pretrained model from: {init_model}")
        model = BertForMaskedLM.from_pretrained(init_model)

    # Dataset
    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=trainfile,
        block_size=128
    )
    val_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=valfile,
        block_size=128
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # Trainer config
    training_args = TrainingArguments(
        output_dir=outdir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_gpu_train_batch_size=32,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        evaluation_strategy="epoch",
        do_train=True,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=computeMetrics
    )


    # Train
    trainer.train()

    # Save the model
    trainer.save_model(outdir, "bert_wikitext")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-I", "--init", dest="init", type="str", help="Pretrained model directory", default=None)
    parser.add_option("-T", "--tokenizer", dest="tokenizer", type="str", help="Tokenizer model path")
    parser.add_option("-D", "--data", dest="data", type="str", help="Comma separated paths for training and validation dataset")
    parser.add_option("-O", "--out", dest="out", type="str", help="Model output directory")
    (options, args) = parser.parse_args()

    trainfile, valfile, tokenizer, outdir = checkAndCreateDirectories(options)
    train(trainfile, valfile, tokenizer, outdir, options.init)
