import json
import random
import numpy as np
from openai import OpenAI
from config   import openai_client,model,epochs,system_message
import matplotlib.pyplot as plt

num_of_samples=100
# === Setup ===
client = OpenAI()
model_name = "ft:gpt-3.5-turbo-1106:walid::CMlot2Tx"

# Load test dataset
with open("test.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Pick 100 random samples
sampled = random.sample(lines, min(num_of_samples, len(lines)))

y_true, y_pred = [], []

print("🔎 Running evaluation on 100 random test samples...\n")

for i, line in enumerate(sampled, 1):
    record = json.loads(line)
    user_msg = next(m["content"] for m in record["messages"] if m["role"] == "user")
    label_msg = next(m["content"] for m in record["messages"] if m["role"] == "assistant")

    # Ground truth price (strip "PRICE: " and cast to int)
    true_price = int(label_msg.replace("PRICE:", "").strip())

    # Query model
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_msg}
        ],
        temperature=0
    )

    pred_text = response.choices[0].message.content.strip()
    try:
        pred_price = int(pred_text.replace("PRICE:", "").strip())
    except:
        print(f"⚠️ Could not parse prediction: {pred_text}")
        continue

    y_true.append(true_price)
    y_pred.append(pred_price)

    # Print some samples
    if i <= 5:
        print(f"Sample {i}:")
        print("  User:", user_msg[:120].replace("\n", " ") + "...")
        print("  True:", true_price)
        print("  Pred:", pred_price)
        print()

# --- Metrics ---
if y_true and y_pred:
    errors = np.array(y_pred) - np.array(y_true)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))

    print("✅ Evaluation complete")
    print(f"MAE  (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Square Error): {rmse:.2f}")
    print(f"Tested on {len(y_true)} samples")


    # --- Plot actual vs predicted ---
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.6, label="Predictions")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Perfect prediction")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices (100 random samples)")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("⚠️ No predictions collected (check parsing or model output).")
