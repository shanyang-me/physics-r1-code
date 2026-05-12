# Physics-R1 — Audited Olympiad Corpus and Recipe for Visual Physics Reasoning

Reference implementation accompanying the **Physics-R1** paper.

- **Project page:** <https://shanyang-me.github.io/physics-r1-page/>
- **Datasets:** [`shanyangmie/*`](https://huggingface.co/shanyangmie) on Hugging Face
- **Author:** Shan Yang · <alexyangshan@gmail.com>

This repository contains the audit pipeline, dense physics-native reward, training recipe, and evaluation harness used to produce the results reported in the paper.

## Layout

```
audit/                     two-stage 5-gram Jaccard + mxbai-embed-large cosine audit
  audit_two_stage.py         pairwise audit pool against held-out splits
  audit_check.py             diagnostic / per-pair leak inspection
data/
  make_splits.py             build the held-out olympiad split + train pool from raw sources
reward/
  reward_physics.py          5-component dense physics-native reward
                             r = r_ans + r_fmt + r_dim + r_sym + r_cons
                             toggle dense -> binary via env DENSE_REWARD=0
eval/
  eval_batch_phyx.py         PhyX-mini-1k / PhyX-3k MCQ harness (vLLM batched)
  eval_physreason.py         PhysReason-full numerical-multipart harness
  eval_physunibench_oe.py    PhysUniBench-en-OE open-ended harness
  eval_phyx_closed_api.py    PhyX-3k via closed APIs (GPT-4o, Gemini 2.5 Pro)
judge/
  llm_judge_v2_alignment.py  best-match alignment Sonnet judge (PR + PUB-OE)
  llm_judge_v3_pubeo.py      v3 judge with cached gold + CoT-tail fallback (PUB-OE)
  judge_olympiad.py          PhysOlym-A strict + liberal Sonnet judge
LICENSE
README.md
```

## Datasets

| Dataset | Records | Purpose |
|---|---|---|
| [`shanyangmie/physcorp-a`](https://huggingface.co/datasets/shanyangmie/physcorp-a) | 6,432 | Audited multimodal physics corpus (Stage-3 clean against six public evals) |
| [`shanyangmie/physr1corp`](https://huggingface.co/datasets/shanyangmie/physr1corp) | 2,268 | Closed-form RL training pool (numeric / MCQ-gradable carve-out) |
| [`shanyangmie/physolym-a`](https://huggingface.co/datasets/shanyangmie/physolym-a) | 500 | Held-out olympiad eval, 99.8% novel-source, EN/ET bilingual |
| [`shanyangmie/physcorp-pre-audit`](https://huggingface.co/datasets/shanyangmie/physcorp-pre-audit) | 14,294 | Raw pre-audit pool — released so users can re-run the audit |

```python
from datasets import load_dataset
ds = load_dataset("shanyangmie/physolym-a", split="test")
```

## Quick start

```bash
git clone https://github.com/shanyang-me/physics-r1-code
cd physics-r1-code
pip install -r requirements.txt

# Run the contamination audit pipeline
python audit/audit_two_stage.py \
    --train_jsonl your_pool.jsonl \
    --eval_jsonl  data/physolym_a.jsonl \
    --jaccard_thr 0.4 \
    --cosine_thr  0.85 \
    --emit report.json
```

## Citation

```bibtex
@misc{yang2026physicsr1,
  title  = {Physics-R1: An Audited Olympiad Corpus and Recipe for Visual Physics Reasoning},
  author = {Yang, Shan},
  year   = {2026}
}
```

## License

Code: Apache-2.0. Datasets: per-source provenance preserved (mix of CC BY 4.0, CC BY-SA 4.0, CC BY-NC 4.0, public-domain). See dataset cards for details.
