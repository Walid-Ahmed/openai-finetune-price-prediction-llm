from openai import OpenAI
from config   import openai_client,model,epochs,system_message

client = OpenAI()

# Keep job_id as variable
job_id = "ftjob-9uHN6LW9GoSMBu5lAUQHIg3r"

# Retrieve job info to get model name
job = client.fine_tuning.jobs.retrieve(job_id)
model_name = job.fine_tuned_model

# Example query
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "How much does this cost to the nearest dollar? Samsung 32-inch LED Monitor, Full HD, HDMI input, lightweight design."}
]

response = openai_client.chat.completions.create(
    model=model_name,
    messages=messages
)

print(f"Job ID: {job_id}")
print(f"Model: {model_name}")
print("Prediction:", response.choices[0].message.content)
