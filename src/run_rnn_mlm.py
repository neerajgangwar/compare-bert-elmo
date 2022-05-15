import os
import time
import math
import logging, logging.handlers
from datetime import datetime
from optparse import OptionParser
from pydoc import describe
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import DataCollatorForLanguageModeling
from transformers import BertTokenizer
from transformers import LineByLineTextDataset

from lm.mlm import LstmMlm


log = logging.getLogger(__name__)


def checkAndCreateDirectories(options):
    # Data files
    assert not options.do_train or options.train_data is not None, f"No training dataset provided"
    assert not options.do_eval or options.eval_data is not None, f"No eval dataset provided"

    # Tokenizer
    assert os.path.exists(options.tokenizer), f"{options.tokenizer} does not exist"

    # Output
    if not os.path.exists(options.out):
        os.mkdir(options.out)

    return options.train_data, options.eval_data, options.tokenizer, options.out


def train(trainfile, valfile, tokenizerfile, outdir, n_epochs, batch_size, lr=0.0001, do_train=True, do_eval=True, model_state_path=None):
    # Create output directory
    outdir = f"{outdir}/{datetime.now().strftime('%Y%m%d-%H%M%S%f')}"

    if os.path.exists(outdir):
        raise Exception(f"{outdir} already exists.")

    os.mkdir(outdir)

    # Configure logging
    logdir = f"{outdir}/logs/"
    if not os.path.isdir(logdir):
        os.mkdir(logdir)
    logging_handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", f"{logdir}/training.log"))
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        handlers=[logging_handler]
    )

    # Initialize tokenizer
    log.info(f"Initializing tokenizer from {tokenizerfile}")
    tokenizer = BertTokenizer.from_pretrained(tokenizerfile, max_len=512, add_special_tokens=False)

    # Initialize model
    log.info(f"Initializing model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LstmMlm(num_tokens=30_000)
    if model_state_path:
        print(f"Loading model from {model_state_path}")
        saved_model = torch.load(model_state_path, map_location=device)
        model.load_state_dict(saved_model["model"])
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Dataset
    log.info(f"Reading dataset")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    if do_train:
        train_dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=trainfile,
            block_size=128,
        )
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=True,
        )
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        log.info("Starting training")
        losses = []
        for epoch in range(n_epochs):
            epochstarttime = time.time()
            for idx, batch in enumerate(train_data_loader):
                iterstarttime = time.time()
                optimizer.zero_grad()
                input = batch["input_ids"].to(device)
                mask = batch["attention_mask"]
                labels = batch["labels"].to(device)
                input_len = torch.stack([sum(elem) for elem in mask]).to(device)
                output, _ = model(input, input_len)
                masked_idx = torch.where(labels > 0)
                target = labels[masked_idx]
                actual = output[masked_idx]
                loss = F.cross_entropy(actual, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                log.info(f"Epoch {epoch + 1}: completed {idx}/{len(train_data_loader)} iterations, loss {loss.item()}, time {time.time() - iterstarttime} sec")
            
            torch.save({
                "model": model.state_dict(),
                "losses": losses,
            }, f"{outdir}/checkpoint-{epoch}.pth")
            log.info(f"Epoch {epoch + 1}/{n_epochs} completed. Time: {time.time() - epochstarttime} sec")

    if do_eval:
        with torch.no_grad():
            val_dataset = LineByLineTextDataset(
                tokenizer=tokenizer,
                file_path=valfile,
                block_size=128,
            )
            eval_data_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                collate_fn=data_collator,
            )
            losses = []
            for idx, batch in enumerate(eval_data_loader):
                input = batch["input_ids"].to(device)
                mask = batch["attention_mask"]
                labels = batch["labels"].to(device)
                input_len = torch.stack([sum(elem) for elem in mask]).to(device)
                output, _ = model(input, input_len)
                masked_idx = torch.where(labels > 0)
                target = labels[masked_idx]
                actual = output[masked_idx]
                loss = F.cross_entropy(actual, target)
                if not loss.isnan():
                    losses.append(loss.item())

            mean_loss = sum(losses) / len(losses)
            log.info(f"validation losses sum: {sum(losses)} / {len(losses)}\nmean: {mean_loss}")
            perplexity = math.exp(mean_loss)
            log.info(f"perplexity of language model: {perplexity}")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--init", dest="init", type="str", help="Pretrained model directory", default=None)
    parser.add_option("--tokenizer", dest="tokenizer", type="str", help="Tokenizer model path")
    parser.add_option("--train-data", dest="train_data", type="str", help="Path for training dataset", default=None)
    parser.add_option("--eval-data", dest="eval_data", type="str", help="Path for eval dataset", default=None)
    parser.add_option("--out", dest="out", type="str", help="Model output directory")
    parser.add_option("--batch-size", dest="batch_size", type="int", help="Batch size", default=8)
    parser.add_option("--epochs", dest="n_epochs", type="int", help="Number of epochs", default=1)
    parser.add_option("--lr", dest="lr", type="float", help="Learning rate", default=0.0001)
    parser.add_option("--do-train", dest="do_train", action="store_true", help="Run training", default=False)
    parser.add_option("--do-eval", dest="do_eval", action="store_true", help="Run evaluation", default=False)
    parser.add_option("--saved-model", dest="saved_model", type="str", help="Path to a saved model", default=None)

    (options, args) = parser.parse_args()

    trainfile, valfile, tokenizer, outdir = checkAndCreateDirectories(options)
    train(trainfile, valfile, tokenizer, outdir, options.n_epochs, options.batch_size, options.lr, options.do_train, options.do_eval, options.saved_model)
