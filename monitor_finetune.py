from openai import OpenAI
import matplotlib.pyplot as plt
import sys
from config import openai_client

client = OpenAI()

# --- Get job ID ---
if len(sys.argv) > 1:
    job_id = sys.argv[1]   # pass as argument if you want
else:
    with open("last_job_id.txt") as f:
        job_id = f.read().strip()

print(f"📡 Monitoring fine-tune job: {job_id}")

# --- Fetch events ---
events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=1000)

steps, train_loss, val_loss = [], [], []

for e in reversed(events.data):  # chronological
    if hasattr(e, "metrics") and e.metrics:
        step = getattr(e, "step", None)
        if step is None:
            continue
        steps.append(step)
        train_loss.append(e.metrics.get("train.loss"))
        val_loss.append(e.metrics.get("valid.loss"))
        print(f"Step {step}: train_loss={train_loss[-1]}, val_loss={val_loss[-1]}")

# --- Plot ---
if steps:
    plt.plot(steps, train_loss, label="Train Loss")
    if any(val_loss):
        plt.plot(steps, val_loss, label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Fine-tuning Loss Curves")
    plt.legend()
    plt.show()
else:
    print("⚠️ No training metrics found (maybe job already finished or no validation file).")




