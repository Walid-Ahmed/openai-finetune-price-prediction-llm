from openai import OpenAI
from config   import openai_client,model,epochs
client = OpenAI()


# You own the access and control of the fine-tuned model in your account, but OpenAI hosts it and you can’t download/transfer it elsewhere.
# Upload training file
with open("train.jsonl", "rb") as f:
    train_file = client.files.create(file=f, purpose="fine-tune")

# Upload validation file
with open("validation.jsonl", "rb") as f:
    val_file = client.files.create(file=f, purpose="fine-tune")

# Start fine-tune job
job = openai_client.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=val_file.id,
    model=model,
    hyperparameters={
        "n_epochs": epochs  # 👈 set epochs here
        #"batch_size": 16,
        #"learning_rate_multiplier": 1.5
    }
)

print("✅ Fine-tune job started:", job.id)

# Save job ID so monitor_finetune.py can pick it up automatically
with open("last_job_id.txt", "w") as f:
    f.write(job.id)
