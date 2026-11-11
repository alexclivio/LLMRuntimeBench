import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


# Which engine to test: "trtllm" or "vllm" (from the command line)
engine = sys.argv[1]

ENGINES = {
    "trtllm": {
        "model_id": os.environ["TRTLLM_MODEL_ID"],
        "model_name": "Qwen 2.5 3B TRTLLM",
    },
    "vllm": {
        "model_id": os.environ["VLLM_MODEL_ID"],
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
    },
}

config = ENGINES[engine]

client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url=f"https://model-{config['model_id']}.api.baseten.co/environments/production/sync/v1",
)

print("Calling:", client.base_url)

try:
    response = client.chat.completions.create(
        model=config["model_name"],
        messages=[{"role": "user", "content": "Explain what a GPU is in one sentence."}],
        max_tokens=100,
    )
except Exception as e:
    print("Error:", e)
    print("Details:", e.response.text)

print(f"[{engine}]", response.choices[0].message.content)