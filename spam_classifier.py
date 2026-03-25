import argparse
import os
import requests
import time
import zipfile
from pathlib import Path

import pandas as pd
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt


MODEL_NAME = "gpt2-small (124M)"
MODEL_PATH = "review_classifier.pth"

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

MODEL_CONFIGS = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

SPAM_DATA_URL     = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SPAM_DATA_BACKUP  = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
ZIP_PATH          = "sms_spam_collection.zip"
EXTRACTED_PATH    = "sms_spam_collection"
DATA_FILE_PATH    = Path(EXTRACTED_PATH) / "SMSSpamCollection.tsv"


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = max(len(e) for e in self.encoded_texts)
        else:
            self.max_length = max_length
            self.encoded_texts = [e[:max_length] for e in self.encoded_texts]

        # Pad all sequences to max_length
        self.encoded_texts = [
            e + [pad_token_id] * (self.max_length - len(e))
            for e in self.encoded_texts
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            torch.tensor(self.encoded_texts[index], dtype=torch.long),
            torch.tensor(self.data.iloc[index]["Label"], dtype=torch.long),
        )


def download_spam_data():
    if DATA_FILE_PATH.exists():
        return

    def _download(url):
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(EXTRACTED_PATH)
        os.rename(Path(EXTRACTED_PATH) / "SMSSpamCollection", DATA_FILE_PATH)

    try:
        _download(SPAM_DATA_URL)
    except (requests.exceptions.RequestException, TimeoutError):
        print("Primary URL failed, trying backup...")
        _download(SPAM_DATA_BACKUP)

    print(f"Dataset saved to {DATA_FILE_PATH}")


def prepare_datasets(tokenizer):
    download_spam_data()

    df = pd.read_csv(DATA_FILE_PATH, sep="\t", header=None, names=["Label", "Text"])

    # Balance classes
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    df["Label"] = df["Label"].map({"ham": 0, "spam": 1})

    # Split
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * 0.7)
    val_end   = train_end + int(len(df) * 0.1)
    df[:train_end].to_csv("train.csv", index=False)
    df[train_end:val_end].to_csv("validation.csv", index=False)
    df[val_end:].to_csv("test.csv", index=False)

    train_dataset = SpamDataset("train.csv", tokenizer)
    val_dataset   = SpamDataset("validation.csv", tokenizer, max_length=train_dataset.max_length)
    test_dataset  = SpamDataset("test.csv",       tokenizer, max_length=train_dataset.max_length)

    return train_dataset, val_dataset, test_dataset


def make_loaders(train_dataset, val_dataset, test_dataset, batch_size=8):
    torch.manual_seed(123)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, drop_last=False)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            return torch.device("mps")
    return torch.device("cpu")


def build_model(config):
    model_size = MODEL_NAME.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(config)
    load_weights_into_gpt(model, params)

    # Freeze all layers, then unfreeze only the classifier head and final norm
    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)
    model.out_head = nn.Linear(in_features=config["emb_dim"], out_features=2)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return torch.nn.functional.cross_entropy(logits, target_batch)


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct, total = 0, 0
    limit = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    for i, (inputs, targets) in enumerate(data_loader):
        if i >= limit:
            break
        with torch.no_grad():
            logits = model(inputs.to(device))[:, -1, :]
        correct += (torch.argmax(logits, dim=-1) == targets.to(device)).sum().item()
        total   += targets.shape[0]

    return correct / total


def train(model, train_loader, val_loader, device, num_epochs=5, eval_freq=50, eval_iter=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(inputs, targets, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    t_loss = sum(
                        calc_loss_batch(x.to(device), y.to(device), model, device).item()
                        for i, (x, y) in enumerate(train_loader) if i < eval_iter
                    ) / eval_iter
                    v_loss = sum(
                        calc_loss_batch(x.to(device), y.to(device), model, device).item()
                        for i, (x, y) in enumerate(val_loader) if i < eval_iter
                    ) / eval_iter
                model.train()
                print(f"Ep {epoch+1} (step {global_step:05d}): train loss {t_loss:.3f}, val loss {v_loss:.3f}")

        train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_acc   = calc_accuracy_loader(val_loader,   model, device, num_batches=eval_iter)
        print(f"  -> Epoch {epoch+1} accuracy — train: {train_acc*100:.2f}%  val: {val_acc*100:.2f}%")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def classify(text, model, tokenizer, device, max_length, pad_token_id=50256):
    model.eval()
    supported = model.pos_emb.weight.shape[0]
    max_len   = min(max_length, supported)

    input_ids  = tokenizer.encode(text)[:max_len]
    input_ids += [pad_token_id] * (max_len - len(input_ids))
    tensor     = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)[:, -1, :]
    return "spam" if torch.argmax(logits, dim=-1).item() == 1 else "not spam"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPT-2 spam classifier")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train",   action="store_true", help="Download data, train, and save model")
    group.add_argument("--predict", metavar="TEXT",      help="Classify an email text as spam or not spam")
    args = parser.parse_args()

    config = {**BASE_CONFIG, **MODEL_CONFIGS[MODEL_NAME]}
    tokenizer = tiktoken.get_encoding("gpt2")
    device    = get_device()
    print(f"Using device: {device}")

    if args.train:
        print("Preparing datasets...")
        train_dataset, val_dataset, test_dataset = prepare_datasets(tokenizer)

        assert train_dataset.max_length <= config["context_length"], (
            f"Max sequence length ({train_dataset.max_length}) exceeds model context "
            f"({config['context_length']}). Use max_length={config['context_length']}."
        )

        train_loader, val_loader, test_loader = make_loaders(train_dataset, val_dataset, test_dataset)

        print("Loading pre-trained GPT-2 weights...")
        model = build_model(config).to(device)

        print("Training classifier...")
        t0 = time.time()
        train(model, train_loader, val_loader, device)
        print(f"Training completed in {(time.time() - t0) / 60:.2f} minutes.")

        test_acc = calc_accuracy_loader(test_loader, model, device)
        print(f"Final test accuracy: {test_acc*100:.2f}%")

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    elif args.predict:
        if not os.path.exists(MODEL_PATH):
            print(f"No trained model found at '{MODEL_PATH}'. Run with --train first.")
            return

        # Infer the max_length used during training from the saved CSV if available
        if os.path.exists("train.csv"):
            train_dataset = SpamDataset("train.csv", tokenizer)
            max_length    = train_dataset.max_length
        else:
            max_length = config["context_length"]

        model = build_model(config).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

        result = classify(args.predict, model, tokenizer, device, max_length=max_length)
        print(f"Result: {result}")


if __name__ == "__main__":
    main()