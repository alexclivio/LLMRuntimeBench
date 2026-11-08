import os
from openai import OpenAI

model_id = os.environ["TRTLLM_MODEL_ID"]

client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url=f"https://model-{model_id}.api.baseten.co/environments/production/sync/v1",
)

try:
    response = client.chat.completions.create(
        model="Qwen 2.5 3B TRTLLM",
        messages=[{"role": "user", "content": "Explain what a GPU is in one sentence."}],
        max_tokens=100,
    )
    print(response.choices[0].message.content)
except Exception as e:
    print("Error:", e)
    print("Details:", e.response.text)