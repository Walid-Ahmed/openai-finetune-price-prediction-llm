from datasets import load_dataset
from dataUtil import peek,clean_text,dataset_to_jsonl,estimate_finetune_cost
from config import model
from config import openai_client,system_message
dataset = load_dataset("ed-donner/pricer-data")
print(dataset)
#peek(dataset, "train", n=5, random=True)
import config
from cost_estimation import estimate_finetune_cost
from config import epochs

# Downsample the train if needed
num_train_records=config.num_train_records
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(num_train_records))


train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset["train"] = train_val["train"]
dataset["validation"] = train_val["test"]



'''
train.jsonl → fine-tuning (training_file).
validation.jsonl → fine-tuning validation (validation_file).
test.jsonl → local evaluation (never uploaded).
'''
dataset_to_jsonl(dataset, split="train", output_path="train.jsonl")
dataset_to_jsonl(dataset, split="validation", output_path="validation.jsonl")
dataset_to_jsonl(dataset, split="test", output_path="test.jsonl")  # for your own eval
estimate_finetune_cost("train.jsonl", "test.jsonl", epochs=epochs, model=model)





