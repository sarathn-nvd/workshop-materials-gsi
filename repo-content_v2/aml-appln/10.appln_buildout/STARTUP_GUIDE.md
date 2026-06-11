# AML Investigator — Startup Guide

Three services to bring up, in this order: **NIM → Backend → Frontend**.

---

## 0. Prerequisites

| | Version | Check |
|---|---|---|
| Python | 3.12 | `python3.12 --version` |
| Node | 18+ | `node --version` |
| Docker | with NVIDIA runtime | `docker info` |
| NGC API key | for the NIM image | `echo $NGC_API_KEY` |

Repo layout (relative to this file):

```
10.appln_buildout/
├── backend/           ← FastAPI + NeMo Agent Toolkit
├── frontend/          ← Next.js 14 (App Router)
└── STARTUP_GUIDE.md   ← this file
```

---

## 1. Start the Custom Task NIM  (port **8088**)

The agent calls a local NIM that serves the fine-tuned model. Pull and run it once per machine.

```bash
# One-time login
docker login nvcr.io -u '$oauthtoken' -p $NGC_API_KEY

# Launch (TP=2 on GPUs 0 + 3; adjust if you only have 1 GPU)
docker run -d --name aml-nim \
  --gpus '"device=0,3"' \
  --shm-size=16g \
  -e NGC_API_KEY \
  -e NIM_MODEL_NAME=aml-custom-task-nim-1 \
  -p 8088:8000 \
  nvcr.io/nim/nvidia/model-free-nim:2.0.5

# Verify
curl -s http://localhost:8088/v1/models | jq '.data[0].id'
# → "aml-custom-task-nim-1"
```

If you don't have a local GPU, point the backend at any OpenAI-compatible endpoint that serves the same model — see `NAT_AML_NIM_BASE_URL` in step 2.

---

## 2. Start the Backend  (port **8010**)

### 2a. One-time install

```bash
cd backend
python3.12 -m venv env
./env/bin/pip install --upgrade pip
./env/bin/pip install -e ./src
```

This pulls `nvidia-nat[langchain,phoenix,opentelemetry,profiler]` and installs the local `aml-app` package (which registers all NAT functions via the `nat.components` entry point).

### 2b. Launch

```bash
cd backend
NAT_PORT=8010 NAT_HOST=0.0.0.0 \
NAT_AML_NIM_BASE_URL=http://localhost:8088/v1 \
NAT_AML_NIM_MODEL=aml-custom-task-nim-1 \
NAT_AML_NIM_API_KEY=EMPTY \
NAT_AML_DATA_DIR="$PWD/data" \
./env/bin/nat serve --config_file ./src/configs/workflow.yaml \
                    --host 0.0.0.0 --port 8010
```

You should see `Application startup complete` after ~5 seconds. Verify:

```bash
curl -s http://localhost:8010/api/health
# → {"value":{"ok":true,"n_transactions":71714,"n_entities":2088,...}}
```

### 2c. Optional: full API smoke test

```bash
bash backend/scripts/smoke_test_api.sh
# → 44/44 PASS
```

---

## 3. Start the Frontend  (port **3000**)

### 3a. One-time install

```bash
cd frontend
npm install
```

### 3b. Launch

```bash
cd frontend
npm run dev
```

Open <http://localhost:3000/dashboard>. The top-right pill should read "online" with live entity / transaction counts pulled from `:8010`.

---

## 4. Verify end-to-end (60 seconds)

1. **Dashboard** loads, donut shows ≥8 typologies, no console errors.
2. **Alert Queue** lists ≥194 alerts; click the green **Run** on any open alert. After ~10–30 s the row flips to "In progress".
3. **Investigation Cockpit** for that alert renders the 7-phase rail, the four auxiliary findings, the SAR narrative, and the analyst disposition form.
4. **Leaderboard** loads the pre-compiled 4-way scorecard from `data/benchmarks/latest.json`.
5. **Skill Playgrounds** — pick Behavioral, press Run, get a typed finding from the NIM.

If all five pass, you're ready to demo.

---

## 5. Common issues

| Symptom | Fix |
|---|---|
| `Cannot connect to host localhost:8088` in skill / investigation calls | NIM container is down — `docker start aml-nim` |
| `aml-app` not registered (Pydantic discriminator error on backend startup) | Forgot step 2a — `./env/bin/pip install -e ./src` |
| Frontend pill says "backend offline" | Backend isn't on `:8010`. Either re-launch with `NAT_PORT=8010`, or update `frontend/next.config.mjs` to proxy to whatever port you're using |
| `bad interpreter: /bin/bash^M` when running `launch_nat_e2e.sh` | Script has CRLF line endings. Either run `bash launch_nat_e2e.sh` directly, or run `dos2unix launch_nat_e2e.sh` once |
| Dashboard donut shows only "None" | Only a few traces exist. Either run more investigations from the Alert Queue, or run `python -m scripts.run_batch` to populate `data/traces/` in bulk |

---

## 6. Ports and where they're configured

| Port | Service | Configured in |
|---|---|---|
| 8088 | Custom Task NIM | docker `-p 8088:8000` |
| 8010 | NAT backend | `NAT_PORT` env var or `--port` flag |
| 3000 | Next.js dev server | `frontend/package.json` (`next dev -p 3000`) |

The frontend reaches the backend through a Next.js rewrite in `frontend/next.config.mjs` — change that file if you move `:8010` elsewhere.
