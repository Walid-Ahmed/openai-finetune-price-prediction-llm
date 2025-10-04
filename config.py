import os
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import login
from transformers import  BitsAndBytesConfig

# --- Load environment variables ---
load_dotenv(override=True)


# ----------------------------
# Training configuration
# ----------------------------

num_train_records=1500
epochs=1


import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# --- API keys ---
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_token=os.getenv("HF_TOKEN")
weather_api_key = os.getenv("OPENWEATHER_API_KEY")


# --- Check keys ---
if openai_api_key:
    print(f"✅ OpenAI API Key found, begins with: {openai_api_key[:8]}")
else:
    print("❌ OpenAI API Key not set in .env")

if weather_api_key:
    print("✅ OpenWeather API Key found")
else:
    print("❌ OpenWeather API Key missing")

if hf_token:
    print("✅ HF Token Key found")
else:
    print("❌ HF Token Key missing")

# --- OpenAI client ---
openai_client = OpenAI(api_key=openai_api_key)

epochs=1
# Sign in to HuggingFace Hub

login(hf_token, add_to_git_credential=True)


# --- Models ---
AUDIO_MODEL = "whisper-1"
#https://huggingface.co/settings/gated-repos
#meeting_summarizer_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
meeting_summarizer_model = "meta-llama/Llama-3.2-3B-Instruct"

model="gpt-3.5-turbo-1106"


# --- System message ---
system_message = (
    "You estimate prices of items. "
    "Reply ONLY with the format: PRICE: <number>. "
    "Do not explain, do not output ranges, just one number."
)

meeting_title="Denver council meeting"
audio_filename="denver_extract.mp3"

if torch.cuda.is_available():
    quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4")
elif torch.backends.mps.is_available():
    quant_config = None
    dtype = torch.float16  # efficient on Apple MPS
    #dtype = torch.float32  #Some models behave better on MPS with float32 instead of float16

else:
    quant_config = None
    dtype = torch.float32   # safest on CPU

