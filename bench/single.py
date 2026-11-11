import os
import sys
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

ENGINES = {
    "trtllm": {"model_id": os.environ["TRTLLM_MODEL_ID"], "model_name": "Qwen 2.5 3B TRTLLM"},
    "vllm":   {"model_id": os.environ["VLLM_MODEL_ID"],   "model_name": "Qwen/Qwen2.5-3B-Instruct"},
}

engine = sys.argv[1]
config = ENGINES[engine]

client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url=f"https://model-{config['model_id']}.api.baseten.co/environments/production/sync/v1",
)

start = time.perf_counter()
first_token_time = None
chunk_count = 0
usage = None

stream = client.chat.completions.create(
    model=config["model_name"],
    messages=[{"role": "user", "content": "Explain how a car engine works in one paragraph."}],
    max_tokens=200,
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in stream:
    if chunk.usage:
        usage = chunk.usage
    if chunk.choices and chunk.choices[0].delta.content:
        if first_token_time is None:
            first_token_time = time.perf_counter()
        chunk_count += 1

end = time.perf_counter()

output_tokens = usage.completion_tokens if usage else chunk_count
ttft = first_token_time - start
generation_time = end - first_token_time

print(f"Engine:             {engine}")
print(f"Output tokens:      {output_tokens}")
print(f"Time to 1st token:  {ttft * 1000:.0f} ms")
print(f"Total time:         {(end - start) * 1000:.0f} ms")
print(f"Tokens per second:  {output_tokens / generation_time:.1f}")