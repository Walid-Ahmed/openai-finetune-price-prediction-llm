import tiktoken
import json
import config
extra_tokens=4 #base tokens per message (<|im_start|>role ... <|im_end|>)
def count_tokens_in_jsonl(path, model="gpt-4o-mini"):
    enc = tiktoken.encoding_for_model(model)
    total_tokens = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for msg in obj["messages"]:
                total_tokens += extra_tokens  # base tokens per message (<|im_start|>role ... <|im_end|>)
                total_tokens += len(enc.encode(msg["content"]))
    return total_tokens




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


    train_tokens = count_tokens_in_jsonl("train.jsonl")
    test_tokens = count_tokens_in_jsonl("test.jsonl")

    print(f"Train tokens: {train_tokens:,}")
    print(f"Test tokens : {test_tokens:,}")

    estimate_finetune_cost("train.jsonl", "test.jsonl", epochs=config.epochs, model="gpt-4o-mini")

