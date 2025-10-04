from datasets import DatasetDict
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt

from config import model,system_message
pd.set_option("display.max_colwidth", None)  # show full text in DataFrame


def peek(dataset: DatasetDict, split: str = "train", n: int = 5, random: bool = True):
    ds_split = dataset[split]
    if random:
        df = ds_split.to_pandas().sample(n)
    else:
        df = ds_split.select(range(n)).to_pandas()

    print(df.head(n))
    return df

import re

import re




import re


def split_record(record: str):
    """
    Split a raw record into clean text (without price mentions)
    and numeric price.
    """
    # Extract price from "Price is $227.00"
    match = re.search(r'Price\s*is\s*\$?(\d+(?:\.\d{1,2})?)', record, flags=re.IGNORECASE)
    price = float(match.group(1)) if match else None

    # Remove all price mentions from text
    text = re.sub(r'Price\s*is\s*\$?\d+(?:\.\d{1,2})?', '', record, flags=re.IGNORECASE)
    text = re.sub(r'\$\d+(?:\.\d{1,2})?', '', text)  # standalone prices
    text = re.sub(r'\s+', ' ', text).strip()

    return text, price

import json
import re



def clean_text(text: str) -> str:
    """Remove price mentions from product text."""
    text = re.sub(r'Price\s*is\s*\$?\d+(?:\.\d{1,2})?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\$\d+(?:\.\d{1,2})?', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def dataset_to_jsonl(dataset, split="train", output_path="train.jsonl"):
    with open(output_path, "w", encoding="utf-8") as f:
        for row in dataset[split]:
            user_text = clean_text(row["text"])
            price_val = round(float(row["price"]))  # nearest whole dollar
            assistant_out = f"PRICE: {price_val}"  # no decimals

            record = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_out}
                ]
            }
            f.write(json.dumps(record) + "\n")
    print(f"✅ Saved {split} split to {output_path}")




def sample_to_jsonl(dataset, split="train", n=5, output_path="sample.jsonl"):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(dataset[split]):
            if i >= n:
                break
            user_text = clean_text(row["text"])
            price_val = round(float(row["price"]))  # nearest whole dollar
            assistant_out = f"PRICE: {price_val}"  # no decimals

            record = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_out}
                ]
            }
            f.write(json.dumps(record) + "\n")
    print(f"✅ Saved {n} samples from {split} split to {output_path}")


import json
import tiktoken

# pick the right encoding for your model
enc = tiktoken.encoding_for_model("gpt-4o-mini")



import json
import tiktoken



def estimate_finetune_cost(train_file, test_file, epochs=3, model="gpt-4o-mini"):
    # pricing per 1M tokens
    prices = {
        "gpt-4o-mini": 3,   # $3 per million tokens
        "gpt-4o": 25,        # $25 per million tokens
        "gpt-3.5-turbo-1106": 0.80,  # $0.80 per million tokens

    }
    if model not in prices:
        raise ValueError(f"Model {model} not supported. Choose from {list(prices.keys())}")

    # count tokens
    train_tokens = count_tokens_in_jsonl(train_file, model)
    test_tokens = count_tokens_in_jsonl(test_file, model)

    # total tokens processed during training
    total_train_tokens = train_tokens * epochs
    cost = (total_train_tokens / 1_000_000) * prices[model]

    print(f"Train tokens : {train_tokens:,}")
    print(f"Test tokens  : {test_tokens:,}")
    print(f"Epochs       : {epochs}")
    print(f"Model        : {model}")
    print(f"Total tokens processed: {total_train_tokens:,}")
    print(f"💰 Estimated cost: ${cost:,.2f}")

    return cost






if __name__ == "__main__":
    dataset = load_dataset("ed-donner/pricer-data")

    # Peek at 5 random rows from train
    peek(dataset, "train", n=5, random=True)
    # Peek at first 5 rows from test
    peek(dataset, "test", n=5, random=False)

    from datasets import DatasetDict
    import pandas as pd

    pd.set_option("display.max_colwidth", None)

    sample = """How much does this cost to the nearest dollar?

    Delphi FG0166 Fuel Pump Module
    Delphi brings 80 years of OE Heritage into each Delphi pump...
    Price is $227.00"""

    text, price = split_record(sample)

    print("TEXT:\n", text)
    print("\nPRICE:", price)
    sample_to_jsonl(dataset, split="train", n=5, output_path="sample.jsonl")

    train_tokens = count_tokens_in_jsonl("train.jsonl")
    test_tokens = count_tokens_in_jsonl("test.jsonl")

    print(f"Train tokens: {train_tokens:,}")
    print(f"Test tokens : {test_tokens:,}")

    estimate_finetune_cost("train.jsonl", "test.jsonl", epochs=3, model="gpt-4o-mini")

