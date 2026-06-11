# Custom Model Deployment — Two NIM Endpoints (TP=2 each)

Two OpenAI-compatible endpoints on this 8× H100 NVL host. NVLink pairs: (0,3), (1,2), (4,5), (6,7).

| # | Container         | Image                                          | GPUs | Host port | Served model name        |
|---|-------------------|------------------------------------------------|------|-----------|--------------------------|
| A | `custom-task-nim` | `nvcr.io/nim/nvidia/model-free-nim:2.0.5`      | 0,3  | 8088      | `aml-custom-task-nim`    |
| B | `nemotron-3-nano` | `nvcr.io/nim/nvidia/nemotron-3-nano:latest`    | 1,2  | 8089      | `nvidia/nemotron-3-nano` |

## 1. Prerequisites (run once)

```bash
export NGC_API_KEY=<PASTE_API_KEY_HERE>
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

export LOCAL_NIM_CACHE=/data/swami/gsi-training/9.custom_model_deployment/nim-cache
mkdir -p "$LOCAL_NIM_CACHE" && chmod 777 "$LOCAL_NIM_CACHE"
```

---

## 2. Model A — Custom AML - 1 (Model-Free NIM, TP=2 on GPUs 0,3)

### 2.1 Pull

```bash
export MODEL_DIR=/data/swami/gsi-training/7.run_sft/4.run_sft/checkpoints/LOWEST_VAL/model/consolidated
export NIM_LLM_IMAGE=nvcr.io/nim/nvidia/model-free-nim:2.0.5
docker pull "$NIM_LLM_IMAGE"
```

### 2.2 Start (detached)

```bash
docker run -d \
  --name custom-task-nim-1 \
  --restart unless-stopped \
  --runtime=nvidia \
  --gpus '"device=0,3"' \
  --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -v "$MODEL_DIR:$MODEL_DIR:ro" \
  -e NIM_MODEL_PATH="$MODEL_DIR" \
  -e NIM_SERVED_MODEL_NAME="aml-custom-task-nim-1" \
  -e NIM_TRUST_CUSTOM_CODE=1 \
  -p 8088:8000 \
  "$NIM_LLM_IMAGE" \
  --tensor-parallel-size 2

docker logs -f custom-task-nim-1   # wait for "Uvicorn running on http://0.0.0.0:8000"
```

### 2.3 Test

```bash
curl -s http://localhost:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "aml-custom-task-nim-1",
    "messages": [{"role":"user","content":"In one sentence, what is structuring?"}],
    "max_tokens": 64
  }' | jq -r '.choices[0].message.content'
```

---

## 3. Model B — Nemotron-3-Nano (TP=2 on GPUs 1,2)

### 3.1 Pull

```bash
export NEMOTRON_NANO_IMAGE=nvcr.io/nim/nvidia/nemotron-3-nano:latest
docker pull "$NEMOTRON_NANO_IMAGE"
```

### 3.2 Start (detached)

```bash
docker run -d \
  --name nemotron-3-nano \
  --restart unless-stopped \
  --runtime=nvidia \
  --gpus '"device=1,2"' \
  --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NGC_API_KEY \
  -e NIM_TENSOR_PARALLEL_SIZE=2 \
  -e LLM_ENABLE_THINKING=false \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -p 8089:8000 \
  "$NEMOTRON_NANO_IMAGE"

docker logs -f nemotron-3-nano   # wait for "Uvicorn running on http://0.0.0.0:8000"
```

### 3.3 Test (thinking disabled via `/no_think`)

The `/no_think` system directive disables the reasoning phase for Nemotron-Nano
models (per the [RAG blueprint reasoning docs](https://docs.nvidia.com/rag/latest/enable-nemotron-thinking.html));
include it in every request you don't want reasoning tokens on.

```bash
curl -s http://localhost:8089/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/nemotron-3-nano",
    "messages": [
      {"role":"system","content":"/no_think"},
      {"role":"user","content":"Which number is larger, 9.11 or 9.8?"}
    ],
    "max_tokens": 64
  }' | jq -r '.choices[0].message.content'
```

---

## 4. Model C — gemma-4 (Model-Free NIM, TP=2 on GPUs 4,5)

### 4.1 Pull

```bash
export NEMOTRON_NANO_IMAGE=nvcr.io/nim/google/gemma-4-31b-it:latest
docker pull "$NEMOTRON_NANO_IMAGE"
```

### 4.2 Start (detached)

```bash
docker run -d \
  --name gemma-4 \
  --restart unless-stopped \
  --runtime=nvidia \
  --gpus '"device=4,5"' \
  --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NGC_API_KEY \
  -e NIM_TENSOR_PARALLEL_SIZE=2 \
  -e LLM_ENABLE_THINKING=false \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -p 8090:8000 \
  "$NEMOTRON_NANO_IMAGE"

docker logs -f gemma-4
```

### 3.3 Test

The `/no_think` system directive disables the reasoning phase for Nemotron-Nano
models (per the [RAG blueprint reasoning docs](https://docs.nvidia.com/rag/latest/enable-nemotron-thinking.html));
include it in every request you don't want reasoning tokens on.

```bash
curl -s http://localhost:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/gemma-4-31b-it",
    "messages": [
      {"role":"system","content":"/no_think"},
      {"role":"user","content":"Which number is larger, 9.11 or 9.8?"}
    ],
    "max_tokens": 64
  }' | jq -r '.choices[0].message.content'
```

---

## 4. Tear down

```bash
docker stop custom-task-nim nemotron-3-nano
docker rm   custom-task-nim nemotron-3-nano
```
