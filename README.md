# RxTriage

**A fine-tuned small language model that reads messy pharma supply-chain messages and flags what's wrong — running on your own hardware, at a fraction of frontier-model cost.**

![Python](https://img.shields.io/badge/python-3.12-blue) ![Model](https://img.shields.io/badge/model-Qwen2.5--3B%20%2B%20QLoRA-orange) ![Status](https://img.shields.io/badge/status-Phases%200--3.2%20complete-green) ![Demo](https://img.shields.io/badge/context-portfolio%20demo-lightgrey)

> **Status:** Phases 0–3.2 complete — synthetic-data pipeline + a fine-tuned 3B model with a working eval harness. Serving, confidence routing, and the full evaluation report are on the [roadmap](#roadmap--status). This is an portfolio demo, not production software.

---

## What & Why

In serialized pharmaceutical supply chains (US DSCSA, EU FMD), every trading partner — manufacturers, contract manufacturers, wholesalers, pharmacies — sends shipment data in a **different format**: EPCIS XML events, X12 EDI advance-ship-notices, and plain CSVs with partner-specific column layouts. When these messages disagree (shipped quantity ≠ the ASN, a missing DSCSA transaction statement, an expired or recalled lot, a broken pallet→case→unit hierarchy), the result is an **"exception"** that a human has to chase down. That work is slow, repetitive, and expensive.

**RxTriage** shows that a small, fine-tuned model can automate the two core steps of that workflow:

1. **Normalize** — turn any of the three input formats into one canonical JSON schema.
2. **Triage** — classify the exception, explain it with a cited rule, and recommend a resolution.

**The thesis:** a fine-tuned 1–3B model, running **inside the customer's own network** (no data leaves the building), can match a frontier API on this narrow task at a fraction of the cost — which matters because pharma transaction data is sensitive and often can't be sent to a third-party API at all.

**What's built so far:**
- 🏭 A synthetic data generator that builds realistic, internally-consistent shipment bundles from **real** openFDA drug identifiers and GS1 EPCIS templates.
- 🐞 Seven defect injectors + a deterministic **rule engine** that produces the ground-truth labels (never an LLM).
- 📊 A dependency-light evaluation harness (schema-validity, per-class F1, hallucination rate).
- 🧠 A **QLoRA fine-tune** of Qwen2.5-3B — via both a plain Hugging Face stack and Unsloth — that beats the base model by a wide margin.

---

## Headline result

Fine-tuning a 3B model on 6,744 synthetic examples takes it from nearly unusable to near-target on this task:

| Model | Schema-valid | Exception macro-F1 |
|---|---|---|
| Qwen2.5-3B — **base, zero-shot** | 13.1% | 0.126 |
| gpt-4o-mini — frontier, zero-shot | 76.9% | 0.317 |
| Qwen2.5-3B — **fine-tuned (Unsloth QLoRA)** | **93.0%** | **0.771** |

> ⚠️ **Read honestly:** the fine-tuned numbers are from a 100-sample **validation** acceptance check; the baselines are from a 199-sample `test_iid` subset. They are directionally comparable (same metric core), but the full apples-to-apples report across all test splits is [Phase 5](#roadmap--status) and not yet done. See [Results](#results) for the caveats — including two exception types the model still can't do (they need pharma reference data at inference time).

---

## Quick Start

**Prerequisites**
- **Python 3.12** and [`uv`](https://docs.astral.sh/uv/) (dependency manager)
- *(optional)* an **OpenAI API key** — only to run the frontier baseline
- *(optional)* [Ollama](https://ollama.com/) — only to run the local base-model baseline
- *(optional)* an **NVIDIA GPU + CUDA** — only to run fine-tuning (the data pipeline and baselines run without one)

```bash
git clone <this-repo-url> rxtriage-slm-demo
cd rxtriage-slm-demo
cp .env.example .env      # then add your OPENAI_API_KEY (see Configuration)
make setup                # uv sync — installs the core dependencies
make test                 # pytest — generators + rule engine (no GPU needed)
make data                 # assemble the corpus + chat-format it (no GPU needed)
```

That's the whole non-GPU path — corpus and baselines run on any laptop. Fine-tuning needs a one-time GPU setup, kept separate on purpose:

<details>
<summary><strong>GPU / training setup</strong> (only if you want to run <code>make train</code>)</summary>

`make setup` intentionally does **not** install the training stack. The CUDA build of PyTorch must come from PyTorch's own index, and `uv sync --group train` would silently replace it with a CPU-only wheel (disabling the GPU). Install it manually:

```bash
# CUDA PyTorch (adjust cu130 to match your CUDA version)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# training libraries (transformers / peft / trl / bitsandbytes / accelerate)
uv pip install transformers peft trl bitsandbytes datasets accelerate pyyaml

# optional: the faster Unsloth path (Windows works via triton-windows)
uv pip install unsloth --torch-backend=auto
```

> ⚠️ **Never run `uv sync --group train` or `uv run --group train`** — the resolver will overwrite your CUDA torch wheel with a CPU-only build. The `make train*` targets deliberately call the venv Python directly for this reason.

Validated on an RTX 4090 Laptop, torch 2.10.0+cu130.
</details>

---

## Configuration

Copy `.env.example` to `.env` and fill in what you need. Secrets are never committed.

| Variable | Required? | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | For the frontier baseline only | — | Runs the gpt-4o-mini baseline in `make eval` (~$0.10 for a full run). |
| `OPENFDA_API_KEY` | No | — | Optional. Raises openFDA rate limits when re-pulling reference data (`datagen/fetch_ndc.py`, `datagen/fetch_recalls.py`). The pipeline works without it. |

---

## Usage

Every workflow is a `make` target. Each writes its outputs under `data/` (gitignored) or `eval/results/`.

**Generate the dataset** — assembles 10k labeled samples into frozen splits, then converts them to chat-format training records:
```bash
make data
# → data/corpus/*.jsonl  (train/val/test splits)
# → data/corpus/*_chat.jsonl, stats.md, token histogram
```

**Run the baselines** — zero/few-shot base model + frontier, scored on a frozen `test_iid` subset:
```bash
ollama pull qwen2.5:3b-instruct   # one-time, only for the local base-model baseline
uv run python -m eval.subset      # build/verify the frozen 199-sample subset
make eval
# → eval/results/baselines.json + baselines.md
```

**Fine-tune** (needs the [GPU setup](#quick-start) above):
```bash
make train           # plain Hugging Face stack (portable reference implementation)
make train-unsloth   # same config via Unsloth — ~1.8× faster (recommended)
# → training/runs/qwen3b_qlora/{hf,unsloth}/adapter + metrics.csv + run_meta.json
```

**Acceptance eval** — score a trained adapter against the base-model baseline on `val`:
```bash
make train-eval
# → training/runs/qwen3b_qlora/<backend>/val_eval.json
```

---

## Architecture

### Data generation pipeline (built)

```
  openFDA NDC directory ─┐
  openFDA recall lists  ─┤   (real product IDs + real recalled lots)
  GS1 EPCIS templates   ─┘
            │
            ▼
  datagen/generate.py     clean, internally-consistent shipment bundles
            │              (SGTINs, GLNs, lots, expiry, unit→case→pallet trees)
            ▼
  datagen/render.py       render each bundle into 3 formats:
            │              EPCIS XML · X12-856 ASN · CSV (5+ partner dialects)
            ▼
  datagen/inject.py       inject 1 of 7 defect types + update ground truth
            │
            ▼
  rules/engine.py         deterministic oracle → THE ground-truth labels
            │              (never an LLM — 100% agreement with injectors)
            ▼
  datagen/assemble.py     frozen splits (train / val / test_iid /
            │              test_heldout_defect / test_heldout_format)
            ▼
  datagen/format_chat.py  chat records: system(schema) + user(raw) + assistant(JSON)
            │
            ▼
  training/train*.py  ──▶  eval/  (metrics.py scores schema-validity, F1, hallucination)
```

### Target serving architecture (🚧 planned — Phases 4–5)

```
  raw message ──▶ [ fine-tuned SLM ] ──▶ canonical JSON + exceptions
   (XML/X12/CSV)         │
                         ▼
                 [ schema validate ] ──(invalid)──▶ retry once
                         │
                         ▼
                 [ confidence router ] ──(low confidence / fails twice)──▶ frontier API
                         │
                         ▼
                    triaged result
```

### Directory layout

```
rxtriage-slm-demo/
  schema.py / schema.json  # canonical JSON schema + pydantic models
  datagen/                 # generators, format renderers, defect injectors
  rules/                   # deterministic exception rule engine (ground truth)
  training/                # QLoRA scripts (HF + Unsloth), YAML config, val eval
  eval/                    # scoring core + baseline runner + frozen subset
  serving/                 # vLLM + FastAPI + router        (🚧 Phase 4)
  agent/                   # optional supervisor agent      (🚧 Phase 6)
  tests/                   # pytest: schema, generators, injectors↔rules
```

### Tech stack

- **Language / tooling:** Python 3.12, `uv`, Makefile
- **Data gen:** lxml, faker, pandas, requests (openFDA)
- **Fine-tuning:** Qwen2.5-3B-Instruct, 4-bit QLoRA (r=16) via Unsloth **or** transformers + peft + trl + bitsandbytes
- **Eval:** pydantic + jsonschema validation, OpenAI SDK (one client for both Ollama and gpt-4o-mini), hand-rolled F1 (no scikit-learn — Simplicity Rule)
- **Testing:** pytest
- **Planned:** vLLM (AWQ/Q4 serving), FastAPI triage service, confidence router

---

## Dataset & Methodology

10,000 synthetic samples, seed=42, ~30% clean (`NONE`) with the rest spread across defect types and an ~even format mix. Splits are **frozen with a SHA-256 manifest** — nothing is ever trained on a held-out split.

| Split | Total | Purpose |
|---|---|---|
| `train` | 6,744 | training |
| `val` | 843 | checkpoint selection + acceptance eval |
| `test_iid` | 843 | in-distribution test |
| `test_heldout_defect` | 1,000 | **RECALLED_LOT excluded from train** — tests defect generalization |
| `test_heldout_format` | 570 | **partner_e CSV dialect excluded from train** — tests format generalization |

The two held-out splits exist to measure honestly how the model handles a defect type and a data format it has **never seen** during training — the real test of whether it learned the task or memorized the corpus.

**Ground truth comes only from the deterministic rule engine** (`rules/engine.py`), never from an LLM. Each defect injector mutates a clean bundle *and* updates its label, severity, cited rationale, and recommended resolution; the rule engine independently re-derives the same label with 100% agreement.

---

## Results

### Baselines — 199-sample `test_iid` subset

| Model | Schema-valid | Exc. macro-F1 | Header exact-match | Hallucination |
|---|---|---|---|---|
| base zero-shot (Qwen2.5-3B, Ollama Q4) | 13.1% | 0.126 | 44.7% | 1.6% |
| base few-shot (2-shot) | 92.5% | 0.189 | 82.9% | 0.3% |
| frontier (gpt-4o-mini, 0-shot) | 76.9% | 0.317 | 82.4% | 0.0% |

This is the intended **weak-baseline signal**: even gpt-4o-mini reaches only 0.317 macro-F1 — far below the project target (≥0.90) — leaving clear headroom for fine-tuning to prove its worth.

### Fine-tuned — 100 val samples, greedy decode

| Metric | base zero-shot | HF adapter | Unsloth adapter |
|---|---|---|---|
| Schema-valid | 13.1% | 92.0% | **93.0%** |
| Macro-F1 | 0.126 | 0.758 | **0.771** |
| Header exact-match | 44.7% | 91.0% | **93.0%** |
| Hallucination | 1.6% | 0.0% | **0.0%** |

### Training time — HF vs. Unsloth (identical config, same GPU)

| Metric | HF (`train.py`) | Unsloth (`train_unsloth.py`) | Advantage |
|---|---|---|---|
| Wall time | ~3.05 h | ~1.70 h | **1.79× faster** (−44%) |
| Throughput (samples/s) | 0.614 | 1.101 | 1.79× |

Both backends ran 1 epoch over 6,744 train / 843 val samples (422 steps, effective batch 16). Unsloth is the recommended path: same-or-better eval quality, ~78 minutes less wall time.

<details>
<summary><strong>Honest findings & known limitations</strong></summary>

- **Two exception types score 0.0 F1 — by design.** `INVALID_NDC` and `RECALLED_LOT` require pharmaceutical reference data (the NDC directory, active recall lists) that isn't in the prompt — they aren't learnable from chat data alone. The plan is to inject this context at serving time (Phase 4), not to expect the model to memorize it.
- **`BROKEN_AGGREGATION` is weak (F1 ~0.63).** It needs structural reasoning over the aggregation hierarchy; a larger model (the 3.3 ablation) or targeted examples may help.
- **The base-model baseline is a serving-quant floor, not a training-precision floor.** Ollama serves a Q4_K_M quant, while the trainer starts from bf16 weights. When comparing "fine-tuned vs. base," this caveat holds.
- **Frontier schema-validity (76.9%) is *lower* than base few-shot (92.5%)** — a genuine format-fidelity gap: gpt-4o-mini renders numeric-looking lot codes as JSON ints instead of strings and occasionally emits invalid `severity` enums. Few-shot examples pin Qwen to the exact shape. This is exactly the kind of thing fine-tuning (and a retry-on-invalid loop) fixes.
- **Numbers are not yet fully apples-to-apples.** Fine-tuned = val (100); baselines = `test_iid` subset (199). The full harness re-run across all splits is Phase 5.
</details>

---

## Testing

```bash
make test        # uv run pytest
```

Covers: canonical-schema round-trip (pydantic → JSON → schema-valid), the clean-bundle generators, the three format renderers (deterministic parse-back), and each defect injector against the rule engine (an injected bundle must trigger exactly the intended rule).

---

## Roadmap / Status

| Phase | Status |
|---|---|
| **0** — Scaffolding + canonical schema | ✅ |
| **1** — Reference data (openFDA NDC + recalls, GS1 EPCIS templates) | ✅ |
| **2** — Synthetic corpus (generate · render · inject · rule-engine oracle · assemble · chat-format) | ✅ |
| **3.1** — Baselines (zero/few-shot Qwen + gpt-4o-mini) | ✅ |
| **3.2** — QLoRA fine-tune Qwen2.5-3B (HF + Unsloth) | ✅ |
| **3.3** — 1.5B ablation | ⬜ |
| **3.4** — Merge LoRA + quantize (AWQ / Q4 GGUF) | ⬜ |
| **4** — vLLM serving + FastAPI `/triage` + confidence router + partner cache | ⬜ |
| **5** — Full eval harness + latency/cost benchmark + error analysis + report | ⬜ |
| **6** — Optional supervisor agent | ⬜ |

`make serve` and `make demo` are placeholders until Phases 4–5.

---

## Notes & credits

A **portfolio demo**, not production software. Built entirely on open data: the [openFDA NDC Directory](https://open.fda.gov/apis/drug/ndc/) and [Drug Enforcement API](https://open.fda.gov/apis/drug/enforcement/), and [GS1 EPCIS](https://github.com/gs1/EPCIS) example events.

**Use of AI for development:** the architecture and task plan were designed by the developer, drafted with Claude, then reviewed and edited by the developer. Tasks are executed with Claude Code in plan mode — the developer reviews and edits each plan before execution.

**Limitations:** all transaction data is **synthetic**; the X12-856 rendering is a simplified segment subset, not a full EDI implementation. Training was kept to a single GPU under a <$20 budget. No license — not intended for production use.
