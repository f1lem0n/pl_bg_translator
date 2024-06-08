import pickle
import random
import regex as re
from os import walk
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sacrebleu import sentence_bleu
from torchtext.data import Field, Dataset, Example


def _normalize_text(text: str) -> str:
    """Lowercase, trim, remove xml tags and unusual characters"""
    text = text.replace("\t", " ")  # replace tabs with spaces
    text = re.sub(r"<[^>]+>", r"", text)  # remove xml tags
    text = re.sub(r"^[0-9XVI]+[\)\.] ", r"", text)  # remove 1) 2) I) II) etc.
    text = re.sub(r"^.{1}[\)\.] ", r"", text)  # remove a) b) c) etc.
    text = re.sub(r"[^\s0-9\p{L}.,']+", r"", text)  # remove unusual characters
    text = text.lower().strip()
    return text


def _read_tmx(filename: str) -> tuple[list[str], list[str]]:
    """Reads a TMX file and returns lists of Polish and Bulgarian lines."""
    lines_pl = []
    lines_bg = []
    with open(filename, encoding="utf-16") as f:
        for line in f:
            if line.startswith('<tuv xml:lang="pl">'):
                lines_pl.append(_normalize_text(line))
            elif line.startswith('<tuv xml:lang="bg">'):
                lines_bg.append(_normalize_text(line))
            else:
                continue
    return lines_pl, lines_bg


def _read_corpus(corpus_path: str) -> tuple[list[str], list[str]]:
    """Reads a directory of TMX files and returns
    lists of Polish and Bulgarian lines."""
    lines_pl, lines_bg = [], []
    for root, _, files in walk(corpus_path):
        root = Path(root)
        for f in files:
            if not f.endswith(".tmx"):
                continue
            print("Reading file:", root / f)
            filename = root / f
            pl, bg = _read_tmx(filename)
            lines_pl += pl
            lines_bg += bg
    assert len(lines_pl) == len(lines_bg)
    return lines_pl, lines_bg


def _load_dataset_splits(
    train_dataset_path: str, test_dataset_path: str, fields: dict[str, Field]
) -> tuple[Dataset, Dataset]:
    """Load train and test datasets from pickle files."""
    print("Loading dataset from:", train_dataset_path)
    with open(train_dataset_path, "rb") as f:
        train_dataset = pickle.load(f)
    train_dataset = Dataset(train_dataset, fields)
    print("Loading dataset from:", test_dataset_path)
    with open(test_dataset_path, "rb") as f:
        test_dataset = pickle.load(f)
    test_dataset = Dataset(test_dataset, fields)
    return train_dataset, test_dataset


def filter_examples(src, tgt, max_len):
    """Filter examples with too long sentences."""
    return len(src) <= max_len and len(tgt) <= max_len


def _create_dataset_splits(
    corpus_path: str,
    fields: dict[str, Field],
    max_len: int,
) -> tuple[Dataset, Dataset]:
    """Create train and test datasets from a corpus."""
    print("Reading corpus from:", corpus_path)
    data = zip(*_read_corpus(corpus_path))
    print("Splitting data...")
    train_data, test_data = train_test_split(list(data), test_size=0.1)
    print("Creating training set...")
    train_examples = [
        Example.fromlist([src, tgt], fields)
        for src, tgt in train_data
        if filter_examples(src, tgt, max_len)
    ]
    train_dataset = Dataset(train_examples, fields)
    print("Creating test set...")
    test_examples = [
        Example.fromlist([src, tgt], fields)
        for src, tgt in test_data
        if filter_examples(src, tgt, max_len)
    ]
    test_dataset = Dataset(test_examples, fields)
    return train_dataset, test_dataset


def _save_dataset_splits(
    train_dataset: Dataset,
    test_dataset: Dataset,
    train_dataset_path: str,
    test_dataset_path: str,
) -> None:
    """Save train and test datasets to pickle files."""
    print("Saving trainging set to:", train_dataset_path)
    with open(train_dataset_path, "wb") as f:
        pickle.dump(list(train_dataset), f)
    print("Saving test set to:", test_dataset_path)
    with open(test_dataset_path, "wb") as f:
        pickle.dump(list(test_dataset), f)
    return None


def print_dataset_info(dataset: Dataset, name: str) -> None:
    total_src_tokens = sum(
        [len(dataset[i].src) for i, _ in enumerate(dataset)]
    )
    total_tgt_tokens = sum(
        [len(dataset[i].tgt) for i, _ in enumerate(dataset)]
    )
    print(
        f"\n<|{name.upper()} SET|>\n"
        f"number of examples: {len(dataset)}\n"
        f"number of source tokens: {total_src_tokens}\n"
        f"number of target tokens: {total_tgt_tokens}\n"
    )


def get_dataset_splits(
    corpus_path: str,
    fields: dict[str, Field],
    train_dataset_path: str,
    test_dataset_path: str,
    max_len: int,
) -> tuple[Dataset, Dataset]:
    """Load train and test datasets if they exist,
    otherwise create them from TMX files in a corpus path."""
    if Path(train_dataset_path).exists() and Path(test_dataset_path).exists():
        train_dataset, test_dataset = _load_dataset_splits(
            train_dataset_path=train_dataset_path,
            test_dataset_path=test_dataset_path,
            fields=fields,
        )
    else:
        train_dataset, test_dataset = _create_dataset_splits(
            corpus_path=corpus_path,
            fields=fields,
            max_len=max_len,
        )
        _save_dataset_splits(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_dataset_path=train_dataset_path,
            test_dataset_path=test_dataset_path,
        )
    print("Done.")
    print_dataset_info(train_dataset, "train")
    print_dataset_info(test_dataset, "test")
    return train_dataset, test_dataset


def get_subset(dataset: Dataset, ratio: float) -> Dataset:
    """Get a random subset of a dataset."""
    fields = dataset.fields
    dataset = list(dataset)
    random.shuffle(dataset)
    dataset = dataset[: int(len(dataset) * ratio)]
    return Dataset(dataset, fields)


def _decode(tensor: torch.Tensor, lang: Field) -> list[str]:
    decoded_tokens = [lang.vocab.itos[i] for i in tensor]
    while "<bos>" in decoded_tokens:
        decoded_tokens.remove("<bos>")
    while "<eos>" in decoded_tokens:
        decoded_tokens.remove("<eos>")
    while "<pad>" in decoded_tokens:
        decoded_tokens.remove("<pad>")
    return decoded_tokens


def translate(model, tokens, src_lang, tgt_lang, device, token_limit):
    tokens = [src_lang.init_token] + tokens + [src_lang.eos_token]
    if len(tokens) > token_limit:
        print("Token limit exceeded.")
        return
    src = [src_lang.vocab.stoi[token] for token in tokens]
    src = torch.LongTensor(src).unsqueeze(1).to(device)
    outs = [tgt_lang.vocab.stoi["<bos>"]]
    for _ in range(token_limit):
        tgt = torch.LongTensor(outs).unsqueeze(1).to(device)
        with torch.no_grad():
            out = model(src, tgt)
        prd_token = out.argmax(2)[-1, :].item()
        outs.append(prd_token)
        if prd_token == tgt_lang.vocab.stoi["<eos>"]:
            break
    return _decode(outs, tgt_lang)


def random_eval(
    model: nn.Module,
    n_examples: int,
    dataset: Dataset,
    src_lang: Field,
    tgt_lang: Field,
    device: torch.device,
    token_limit: int,
) -> None:
    bleu = 0
    examples = []
    for _ in range(n_examples):
        ri = random.randint(0, len(dataset) - 1)
        src_tokens = list(dataset.src)[ri]
        tgt_tokens = list(dataset.tgt)[ri]
        prd_tokens = translate(
            token_limit=token_limit,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            device=device,
            model=model,
            tokens=src_tokens,
        )
        bleu += sentence_bleu(
            " ".join(prd_tokens), [" ".join(tgt_tokens)]
        ).score
        examples.append(f"> {src_tokens}\n= {tgt_tokens}\n< {prd_tokens}\n")
    bleu /= n_examples
    return {"score": bleu, "examples": examples}


def get_batch_bleu(
    model: nn.Module, src: torch.tensor, tgt: torch.tensor, tgt_lang: Field
) -> float:
    with torch.no_grad():
        model.eval()
        eval_out = model(src, tgt)
        eval_out = eval_out.argmax(2).transpose(0, 1)
        eval_tgt = tgt.transpose(0, 1)
        assert len(eval_out) == len(eval_tgt)
        batch_bleu = 0
        for t, p in zip(eval_tgt, eval_out):
            batch_bleu += sentence_bleu(
                " ".join(_decode(p, tgt_lang)),
                [" ".join(_decode(t, tgt_lang))],
            ).score
    return batch_bleu / len(eval_tgt)
