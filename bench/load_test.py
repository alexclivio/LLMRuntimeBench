import asyncio
import os
import sys
import time
import statistics
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(override=True)

ENGINES = {
    "trtllm": {"model_id": os.environ["TRTLLM_MODEL_ID"], "model_name": "Qwen 2.5 3B TRTLLM"},
    "vllm":   {"model_id": os.environ["VLLM_MODEL_ID"],   "model_name": "Qwen/Qwen2.5-3B-Instruct"},
}

engine = sys.argv[1]
users = int(sys.argv[2])
config = ENGINES[engine]

client = AsyncOpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url=f"https://model-{config['model_id']}.api.baseten.co/environments/production/sync/v1",
)


async def one_request():
    start = time.perf_counter()
    first_token_time = None
    chunk_count = 0
    usage = None

    stream = await client.chat.completions.create(
        model=config["model_name"],
        messages=[{"role": "user", "content": "Explain how a car engine works in one paragraph."}],
        max_tokens=200,
        stream=True,
        stream_options={"include_usage": True},
    )

    async for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            chunk_count += 1

    end = time.perf_counter()
    tokens = usage.completion_tokens if usage else chunk_count
    return {
        "ttft": first_token_time - start,
        "tokens": tokens,
        "tps": tokens / (end - first_token_time),
    }


async def main():
    start = time.perf_counter()
    results = await asyncio.gather(*[one_request() for _ in range(users)])
    wall_time = time.perf_counter() - start

    ttfts = [r["ttft"] * 1000 for r in results]
    per_user_tps = [r["tps"] for r in results]
    total_tokens = sum(r["tokens"] for r in results)

    print(f"Engine:                 {engine}")
    print(f"Users at once:          {users}")
    print(f"TTFT median:            {statistics.median(ttfts):.0f} ms")
    print(f"TTFT worst:             {max(ttfts):.0f} ms")
    print(f"Tokens/sec per user:    {statistics.median(per_user_tps):.1f}")
    print(f"Total tokens/sec (GPU): {total_tokens / wall_time:.1f}")


asyncio.run(main())