# Spam Detector with a Fine-Tuned GPT-2

A binary SMS spam classifier built by fine-tuning GPT-2, based on
[Build a Large Language Model From Scratch](https://github.com/rasbt/LLMs-from-scratch)
by Sebastian Raschka.

The encoder is a pre-trained GPT-2 model whose weights are frozen except for the
last transformer block, the final layer norm and a new 2-class classification
head. Only those parts are trained on the SMS Spam Collection dataset.

## Requirements

- Python 3.10+
- PyTorch (CUDA or Apple MPS supported; CPU works too)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Train the classifier (downloads the dataset and GPT-2 weights on first run):

```bash
python spam_classifier.py --train
```

Classify a message:

```bash
python spam_classifier.py --predict "Congratulations! You won a free iPhone, claim it now"
# Result: spam
```

`--train` and `--predict` are mutually exclusive.

### What training does

1. Downloads the SMS Spam Collection dataset (with a fallback mirror) into
   `sms_spam_collection/`.
2. Balances the classes, then splits the data 70/10/20 into `train.csv`,
   `validation.csv` and `test.csv`.
3. Loads the pre-trained GPT-2 small (124M) weights and fine-tunes the last
   block plus a new classification head for 5 epochs.
4. Prints validation accuracy during training and the final test accuracy.
5. Saves the model to `review_classifier.pth`.

`--predict` needs a previously trained `review_classifier.pth`, and uses
`train.csv` (if present) to recover the padding length used during training.

## Files

| File | Description |
| --- | --- |
| `spam_classifier.py` | Main script: dataset prep, fine-tuning, evaluation and inference. |
| `previous_chapters.py` | GPT model implementation and weight loading helpers (from the book). |
| `gpt_download.py` | Downloads and loads the original GPT-2 checkpoints. |
| `gpt_train.py` | Chapter code for pre-training GPT-2 from scratch. |
| `gpt_generate.py` | Chapter code for loading GPT-2 and generating text. |
| `self-pretrained-model.py` | Walkthrough of the book's chapters (attention, training, sampling). |
| `tests.py` | Internal tests from the book's repository. |

The PDFs (`loss-plot.pdf`, `temperature-plot.pdf`) and `the-verdict.txt` are
artifacts/inputs of the book's pre-training chapters, not part of the spam
classifier itself.

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
from the UCI Machine Learning Repository (5,574 labeled SMS messages).

## Credits

Based on the code and book *Build a Large Language Model (From Scratch)* by
Sebastian Raschka, licensed under Apache 2.0.
