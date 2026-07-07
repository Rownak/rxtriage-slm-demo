# Phase 3.1 baselines

n=199 · subset sha256 `04de31da4d1b…` · few-shot=2 · Ollama default quant (Q4_K_M); not the bf16 base the trainer starts from.

| model | n | schema-valid | exc macro-F1 | exc micro-F1 | header exact-match | hallucination | sec/sample |
|---|---|---|---|---|---|---|---|
| base_zeroshot | 199 | 13.1% | 0.126 | 0.351 | 44.7% | 1.6% | 3.298 |
| base_fewshot | 199 | 92.5% | 0.189 | 0.390 | 82.9% | 0.3% | 2.675 |
| frontier | 199 | 76.9% | 0.317 | 0.435 | 82.4% | 0.0% | 4.317 |
