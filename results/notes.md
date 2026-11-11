# Benchmark notes

## Setup

- Model: Qwen 2.5 3B Instruct (fp8, max context 8192)
- GPU: 1x NVIDIA L4 (24 GB) on Baseten, max replicas = 1
- Engines: TensorRT-LLM, vLLM
- Prompt: "Explain how a car engine works in one paragraph." (max 200 output tokens)
- Client: Python script run from Pleasanton, CA (TTFT includes network time)
- Cost estimates assume L4 at ~$0.85/hr and the GPU staying this busy

---

## 1. Single request (3 runs each)

### vLLM

| Run | Output tokens | TTFT   | Total time | Tokens/sec |
| --- | ------------- | ------ | ---------- | ---------- |
| 1   | 153           | 253 ms | 2,830 ms   | 59.4       |
| 2   | 119           | 233 ms | 2,232 ms   | 59.5       |
| 3   | 145           | 231 ms | 2,674 ms   | 59.3       |

### TensorRT-LLM

| Run | Output tokens | TTFT   | Total time | Tokens/sec |
| --- | ------------- | ------ | ---------- | ---------- |
| 1   | 153           | 266 ms | 2,851 ms   | 59.2       |
| 2   | 163           | 234 ms | 2,987 ms   | 59.2       |
| 3   | 191           | 254 ms | 3,485 ms   | 59.1       |

**Finding:** With one user, both engines start replying in ~0.25 s and generate
~59 tokens/sec. Output length varies between runs (the model's answer changes),
so tokens/sec is a better speed measure than total time.

---

## 2. vLLM: default autoscaling (concurrency target = 1) — BEFORE

| Users | TTFT median | TTFT worst | Tokens/sec per user | Total tokens/sec (GPU) |
| ----- | ----------- | ---------- | ------------------- | ---------------------- |
| 1     | 303 ms      | 303 ms     | 59.5                | 53.3                   |
| 8     | 10,104 ms   | 19,158 ms  | 59.1                | 57.5                   |
| 32    | 41,070 ms   | 82,698 ms  | 59.3                | 58.3                   |

**Finding:** Requests were processed one at a time. Per-user speed stayed at ~59
tok/s (each request had the GPU to itself), but TTFT grew with the queue (83 s
worst case at 32 users) and total GPU throughput stayed flat at ~58 tok/s.
Cause: Baseten's default concurrency target of 1 sends only one request at a
time to each replica, so the GPU never batches requests together.

---

## 3. vLLM: concurrency target = 32 — AFTER

### Attempt 1 (inconclusive)

| Users | TTFT median | TTFT worst | Tokens/sec per user | Total tokens/sec (GPU) |
| ----- | ----------- | ---------- | ------------------- | ---------------------- |
| 1     | 1,109 ms    | 1,109 ms   | 59.2                | 43.1                   |
| 8     | 9,483 ms    | 18,777 ms  | 59.2                | 57.6                   |
| 32    | 14,706 ms   | 20,970 ms  | 57.6                | 208.9                  |

Run immediately after changing the setting. The 8-user run still looked fully
queued, while the 32-user run showed partial batching. The setting likely had
not fully applied. Rerun below.

### Attempt 2

| Users | TTFT median | TTFT worst | Tokens/sec per user | Total tokens/sec (GPU) |
| ----- | ----------- | ---------- | ------------------- | ---------------------- |
| 1     | 276 ms      | 276 ms     | 59.3                | 53.5                   |
| 8     | 324 ms      | 345 ms     | 58.4                | 359.2                  |
| 32    | 460 ms      | 486 ms     | 52.6                | 1,183.3                |

**Finding:** With batching enabled, total throughput rose 22x (53.5 → 1,183
tok/s) from 1 to 32 users, while per-user speed dropped only ~11% (59.3 → 52.6
tok/s) and median TTFT stayed under 0.5 s (vs. 41 s median / 83 s worst before).

| Users | Tokens/hour | Cost per 1M output tokens |
| ----- | ----------- | ------------------------- |
| 1     | ~193K       | ~$4.41                    |
| 8     | ~1.29M      | ~$0.66                    |
| 32    | ~4.26M      | ~$0.20                    |

Same GPU, same hourly price: ~22x cheaper per token at 32 concurrent users.

---

## 4. TensorRT-LLM: concurrency target = 32

| Users | TTFT median | TTFT worst | Tokens/sec per user | Total tokens/sec (GPU) |
| ----- | ----------- | ---------- | ------------------- | ---------------------- |
| 1     | 273 ms      | 273 ms     | 58.9                | 51.0                   |
| 8     | 290 ms      | 327 ms     | 56.8                | 326.3                  |
| 32    | 486 ms      | 678 ms     | 54.8                | 1,117.5                |

Not tested with the default concurrency target of 1. That setting is a Baseten
platform setting, not an engine feature, and its effect was already shown with vLLM.

---

## 5. Comparison: vLLM vs TensorRT-LLM (concurrency target = 32)

| Users | Engine  | TTFT median | Per-user tok/s | Total GPU tok/s | Cost / 1M tokens |
| ----- | ------- | ----------- | -------------- | --------------- | ---------------- |
| 1     | vLLM    | 276 ms      | 59.3           | 53.5            | ~$4.41           |
| 1     | TRT-LLM | 273 ms      | 58.9           | 51.0            | ~$4.63           |
| 8     | vLLM    | 324 ms      | 58.4           | 359.2           | ~$0.66           |
| 8     | TRT-LLM | 290 ms      | 56.8           | 326.3           | ~$0.72           |
| 32    | vLLM    | 460 ms      | 52.6           | 1,183.3         | ~$0.20           |
| 32    | TRT-LLM | 486 ms      | 54.8           | 1,117.5         | ~$0.21           |

**Findings:**

1. The two engines performed within ~5–10% of each other at every load level.
   These are single runs with variable output lengths, so the differences are
   likely within noise. No clear winner at this model size and load.
2. Both engines
