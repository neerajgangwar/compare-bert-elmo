import os
import time
import logging, logging.handlers
from datetime import datetime
from optparse import OptionParser
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import BertTokenizer

from lm.mlm import LstmMlm
from models.rnn_mlm_sa import *


log = logging.getLogger(__name__)


def train(dataset, num_classes, tokenizerfile, outdir, n_epochs, batch_size, bilm_path, lr=0.0001, do_train=True, do_eval=True, model_state_path=None):
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
    print(f"Loading BiLM from {bilm_path}")
    bilm = LstmMlm(num_tokens=30_000)
    bilm_model = torch.load(bilm_path, map_location=device)
    bilm.load_state_dict(bilm_model["model"])
    config = {
        "emb_size": bilm.model_dim,
        "fc_hidden_size": 300,
        "bilstm_encoder_size": 300,
        "bilstm_integrator_size": 600,
        "fc_hidden_size1": 300,
        "mem_size": 300,
        "dropout": 0.1,
        "device": device,
        "hid_sizes_cls": "300",
    }
    model = RepModel(config, num_classes, bilm).to(device)

    # Dataset
    log.info(f"Reading dataset")
    data_collator = DataCollatorWithPadding(tokenizer, padding=True)
    dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, add_special_tokens=False, padding="max_length"), batched=True)

    if do_train:
        train_dataset = dataset["train"]
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            # collate_fn=data_collator,
            shuffle=True,
        )
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        log.info("Starting training")
        model.train()
        losses = []
        for epoch in range(n_epochs):
            epochstarttime = time.time()
            for idx, batch in enumerate(train_data_loader):
                iterstarttime = time.time()
                mask = torch.stack(batch["attention_mask"]).permute(1, 0).to(device)
                input_len = torch.stack([sum(elem) for elem in mask]).to(device)
                max_len = input_len.max()
                optimizer.zero_grad()
                input = torch.stack(batch["input_ids"]).permute(1, 0).to(device)
                input = input[:, :max_len]            
                labels = batch["label"].to(device)
                output, _ = model(input, input_len)
                if num_classes == 1:
                    output = output.squeeze(dim=-1)
                    loss = F.binary_cross_entropy_with_logits(output, labels.float())
                else:
                    loss = F.cross_entropy(output, labels)
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
        model.eval()
        with torch.no_grad():
            val_dataset = dataset["validation"]
            eval_data_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                # collate_fn=data_collator,
            )
            accuracy = 0
            for idx, batch in enumerate(eval_data_loader):
                mask = torch.stack(batch["attention_mask"]).permute(1, 0)
                input_len = torch.stack([sum(elem) for elem in mask]).to(device)
                max_len = input_len.max()
                optimizer.zero_grad()
                input = torch.stack(batch["input_ids"]).permute(1, 0)
                input = input[:, :max_len].to(device)
                labels = batch["label"].to(device)
                output, _ = model(input, input_len)
                if num_classes == 1:
                    output = output.squeeze(dim=-1)
                    output[output <= 0.5] = 0
                    output[output > 0.5] = 1
                    accuracy += (output == labels).sum()
                else:
                    predicted = output.argmax(dim=1)
                    accuracy += (predicted == labels).sum()

            accuracy = accuracy / len(val_dataset)
            log.info(f"accuracy: {accuracy}")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--dataset", dest="dataset", type="str", help="Dataset to use for training/evaluation", default="SetFit/sst2")
    parser.add_option("--tokenizer", dest="tokenizer", type="str", help="Tokenizer model path")
    parser.add_option("--out", dest="out", type="str", help="Model output directory")
    parser.add_option("--batch-size", dest="batch_size", type="int", help="Batch size", default=8)
    parser.add_option("--epochs", dest="n_epochs", type="int", help="Number of epochs", default=1)
    parser.add_option("--lr", dest="lr", type="float", help="Learning rate", default=0.0001)
    parser.add_option("--do-train", dest="do_train", action="store_true", help="Run training", default=False)
    parser.add_option("--do-eval", dest="do_eval", action="store_true", help="Run evaluation", default=False)
    parser.add_option("--saved-model", dest="saved_model", type="str", help="Path to a saved model", default=None)
    parser.add_option("--bilm", dest="bilm", type="str", help="Path to a BiLM", default=None)
    parser.add_option("--num-classes", dest="num_classes", type="int", help="Number of classes in the dataset", default=1)

    (options, args) = parser.parse_args()

    dataset = load_dataset(options.dataset)

    train(dataset, options.num_classes, options.tokenizer, options.out, options.n_epochs, options.batch_size, options.bilm, options.lr, options.do_train, options.do_eval)
