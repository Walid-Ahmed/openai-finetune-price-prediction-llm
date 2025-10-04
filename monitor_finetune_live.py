import time
import sys
import matplotlib.pyplot as plt
from openai import OpenAI
from config   import openai_client,model

client = OpenAI()

# --- Load job ID ---
if len(sys.argv) > 1:
    job_id = sys.argv[1]
else:
    with open("last_job_id.txt") as f:
        job_id = f.read().strip()

print(f"📡 Live monitoring fine-tune job: {job_id}")

steps, train_loss, val_loss = [], [], []
seen_events = set()

plt.ion()  # interactive mode for live updates
fig, ax = plt.subplots()

while True:
    # Fetch the latest events
    events = client.fine_tuning.jobs.list_events(job_id, limit=1000)
    new = False

    for e in reversed(events.data):  # chronological order
        if e["id"] not in seen_events and "metrics" in e and e["metrics"]:
            seen_events.add(e["id"])
            step = e["step"]
            steps.append(step)
            train_loss.append(e["metrics"].get("train_loss"))
            val_loss.append(e["metrics"].get("valid_loss"))
            print(f"Step {step}: train_loss={train_loss[-1]}, val_loss={val_loss[-1]}")
            new = True

    if new:
        ax.clear()
        ax.plot(steps, train_loss, label="Train Loss")
        if any(val_loss):
            ax.plot(steps, val_loss, label="Validation Loss")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Fine-tuning Loss Curves (Live)")
        ax.legend()
        plt.pause(0.1)

    # Check job status
    job = client.fine_tuning.jobs.retrieve(job_id)
    if job.status in ["succeeded", "failed", "cancelled"]:
        print(f"✅ Job finished with status: {job.status}")
        break

    time.sleep(30)  # poll every 30 seconds
