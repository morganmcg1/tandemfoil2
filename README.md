# TandemFoilSet-Balanced

**[TandemFoilSet Paper](https://openreview.net/forum?id=4Z0P4Nbosn)**

This repo contains a custom TandemFoilSet dataset split as well as a base set of files that the [senpai](https://github.com/wandb/senpai) autoresearch harness can use as input to try and improve performance.

## Splits
The goal behind the split is to be able to say:
- The model is good/bad at generalizing to unseen geometries (train on low + high camber, val/test on moderate)
- The same model works across different Reynolds numbers for both race car and cruise (train on low, med, high Re, val/test on low, moderate, high) - random split across all Re numbers
- General single-foil random split as a sanity check

See **[SPLITS.MD](https://github.com/morganmcg1/tandemfoil2/blob/main/data/SPLITS.md)** for a full description of the dataset splits

## Targets

CFD surrogate research target: predict the full velocity `(Ux, Uy)` and pressure `p` field on tandem-airfoil meshes from TandemFoilSet.

## Layout

```
.
├── README.md
├── program.md                 # research contract — read this first
├── train.py                   # Transolver trainer; primary editable entrypoint
├── instructions/
│   ├── prompt-advisor.md
│   └── prompt-student.md
└── data/
    ├── __init__.py
    ├── loader.py              # SplitDataset, TestDataset, load_data, load_test_data
    ├── scoring.py             # MAE accumulation shared by val and test
    ├── prepare_splits.py      # one-shot: materialize .pt samples onto the PVC
    ├── generate_manifest.py   # regenerate split_manifest.json from raw data
    ├── split_manifest.json    # pinned split definition (source of truth)
    └── SPLITS.md              # split design, holdout reasons, per-file counts
```

## Data

Pre-processed samples live on the PVC at `/mnt/new-pvc/datasets/tandemfoil/splits_v2/`. Materialize them once from the manifest:

```
python data/prepare_splits.py
```

After that, `train.py` streams `.pt` samples directly — no re-preprocessing per run.

See `program.md` for feature layout (24 input dims), target channels, split design, and the full metric contract.

## Training

```
python train.py [--debug] [--epochs 50] [--agent <name>] [--wandb_name <name>]
```

The trainer logs per-epoch metrics to W&B and prints a per-split MAE breakdown to the console every epoch. At the end of the run it loads the best checkpoint (selected on `val_avg/mae_surf_p`) and evaluates it on the four held-out test splits, logging `test_avg/mae_surf_p` plus per-split per-channel MAEs.

Environment:

- `SENPAI_TIMEOUT_MINUTES` — wall-clock cap (default 30)
- `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_MODE` — W&B routing

## SENPAI Students vs KAgent

Results from the 12-hour KAgent competition on 2026-04-24, tracked in the [W&B project](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students), after the `pai-amf1-cfd` training fleet timed out. Rows are the best non-debug run per PR, ranked by the average test metric, `test_avg/mae_surf_p` (lower is better). Split columns are test-set surface-pressure MAE.

| Rank | Test avg | PR | Experiment | Test SID | Test RC | Test Cruise | Test Re | Best W&B run | Closed |
|---:|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | 40.927 | [#32](https://github.com/morganmcg1/tandemfoil2/pull/32) | Single-head nl3/sn16 triple compound | 46.569 | 52.859 | 24.717 | 39.561 | [ip8hn4tx](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/ip8hn4tx) | OPEN |
| 2 | 42.426 | [#39](https://github.com/morganmcg1/tandemfoil2/pull/39) | nl3 plus slice_num 8 compound | 49.835 | 54.915 | 24.852 | 40.103 | [3fyx76kw](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/3fyx76kw) | 2026-04-24 06:49 UTC |
| 3 | 47.187 | [#35](https://github.com/morganmcg1/tandemfoil2/pull/35) | 3-layer depth reduction at sn32 | 57.754 | 58.484 | 28.384 | 44.127 | [rze3vuj0](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/rze3vuj0) | 2026-04-24 05:56 UTC |
| 4 | 47.187 | [#41](https://github.com/morganmcg1/tandemfoil2/pull/41) | n_hidden shrink sweep on nl3/sn32 | 57.754 | 58.484 | 28.384 | 44.127 | [m8kjw0ih](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/m8kjw0ih) | OPEN |
| 5 | 50.583 | [#37](https://github.com/morganmcg1/tandemfoil2/pull/37) | n_head sweep at sn16 | 63.897 | 62.083 | 30.333 | 46.019 | [kbuurqi9](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/kbuurqi9) | 2026-04-24 06:19 UTC |
| 6 | 53.417 | [#38](https://github.com/morganmcg1/tandemfoil2/pull/38) | MLP ratio sweep at sn16 | 65.433 | 65.246 | 33.182 | 49.808 | [x9fxpvqp](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/x9fxpvqp) | 2026-04-24 06:19 UTC |
| 7 | 54.588 | [#36](https://github.com/morganmcg1/tandemfoil2/pull/36) | slice_num floor sweep at sn16 | 64.313 | 63.835 | 37.432 | 52.772 | [n6hdxux2](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/n6hdxux2) | 2026-04-24 06:19 UTC |
| 8 | 54.640 | [#34](https://github.com/morganmcg1/tandemfoil2/pull/34) | slice_num lower sweep from sn32 | 64.432 | 64.578 | 35.413 | 54.135 | [moydqx8l](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/moydqx8l) | 2026-04-24 05:11 UTC |
| 9 | 58.358 | [#27](https://github.com/morganmcg1/tandemfoil2/pull/27) | slice_num sweep on sigma0.7+SwiGLU | 68.001 | 68.536 | 40.465 | 56.429 | [nrba5yg8](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/nrba5yg8) | 2026-04-24 04:21 UTC |
| 10 | 62.603 | [#20](https://github.com/morganmcg1/tandemfoil2/pull/20) | Fourier sigma plus SwiGLU | 73.018 | 71.892 | 44.680 | 60.823 | [vqfohirl](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/vqfohirl) | 2026-04-24 01:10 UTC |
| 11 | 62.603 | [#24](https://github.com/morganmcg1/tandemfoil2/pull/24) | sigma0.7 plus SwiGLU multi-seed | 73.018 | 71.892 | 44.680 | 60.823 | [j12mrpeb](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/j12mrpeb) | 2026-04-24 02:21 UTC |
| 12 | 62.603 | [#28](https://github.com/morganmcg1/tandemfoil2/pull/28) | fine sigma sweep around 0.7 | 73.018 | 71.892 | 44.680 | 60.823 | [4kpkjrml](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/4kpkjrml) | 2026-04-24 03:48 UTC |
| 13 | 62.603 | [#30](https://github.com/morganmcg1/tandemfoil2/pull/30) | per-block Fourier reinjection | 73.018 | 71.892 | 44.680 | 60.823 | [8yc2zoj7](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/8yc2zoj7) | 2026-04-24 04:07 UTC |
| 14 | 62.603 | [#31](https://github.com/morganmcg1/tandemfoil2/pull/31) | post-hoc Reynolds scale correction | 73.018 | 71.892 | 44.680 | 60.823 | [7rm8aggp](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/7rm8aggp) | 2026-04-24 04:23 UTC |
| 15 | 62.603 | [#33](https://github.com/morganmcg1/tandemfoil2/pull/33) | alpha-gated per-block Fourier | 73.018 | 71.892 | 44.680 | 60.823 | [ko9iufzq](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/ko9iufzq) | 2026-04-24 05:08 UTC |
| 16 | 63.482 | [#17](https://github.com/morganmcg1/tandemfoil2/pull/17) | input feature jitter | 74.587 | 74.919 | 43.267 | 61.157 | [5uh500fl](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/5uh500fl) | 2026-04-24 05:08 UTC |
| 17 | 63.983 | [#23](https://github.com/morganmcg1/tandemfoil2/pull/23) | zero-init residual surface decoder | 73.204 | 76.819 | 44.043 | 61.867 | [m90nas5y](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/m90nas5y) | 2026-04-24 02:42 UTC |
| 18 | 63.983 | [#25](https://github.com/morganmcg1/tandemfoil2/pull/25) | SwiGLU decoder/head refinements | 73.204 | 76.819 | 44.043 | 61.867 | [ozt5rhq4](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/ozt5rhq4) | 2026-04-24 03:01 UTC |
| 19 | 63.983 | [#26](https://github.com/morganmcg1/tandemfoil2/pull/26) | sample-wise Reynolds normalization | 73.204 | 76.819 | 44.043 | 61.867 | [tci0og5u](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/tci0og5u) | 2026-04-24 03:01 UTC |
| 20 | 75.244 | [#7](https://github.com/morganmcg1/tandemfoil2/pull/7) | Fourier PE plus Reynolds FiLM | 90.582 | 83.394 | 54.368 | 72.632 | [91z1948k](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/91z1948k) | 2026-04-24 00:01 UTC |
| 21 | 75.244 | [#19](https://github.com/morganmcg1/tandemfoil2/pull/19) | Fourier m extension and learned B | 90.582 | 83.394 | 54.368 | 72.632 | [7kf0h22b](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/7kf0h22b) | 2026-04-24 01:12 UTC |
| 22 | 75.244 | [#21](https://github.com/morganmcg1/tandemfoil2/pull/21) | near-surface volume weighting | 90.582 | 83.394 | 54.368 | 72.632 | [b4zrwrc3](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/b4zrwrc3) | 2026-04-24 01:34 UTC |
| 23 | 76.284 | [#22](https://github.com/morganmcg1/tandemfoil2/pull/22) | attention temperature annealing | 93.932 | 85.331 | 53.419 | 72.452 | [3mngcmbn](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/3mngcmbn) | 2026-04-24 02:02 UTC |
| 24 | 79.386 | [#9](https://github.com/morganmcg1/tandemfoil2/pull/9) | pressure target reparameterization | 93.659 | 89.365 | 57.339 | 77.182 | [ixfjicxm](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/ixfjicxm) | 2026-04-23 23:44 UTC |
| 25 | 79.641 | [#12](https://github.com/morganmcg1/tandemfoil2/pull/12) | AMP plus grad accumulation throughput | 96.081 | 89.938 | 56.266 | 76.279 | [705k9e40](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/705k9e40) | 2026-04-23 23:02 UTC |
| 26 | 80.332 | [#14](https://github.com/morganmcg1/tandemfoil2/pull/14) | surf_weight plus AMP sweep | 91.552 | 90.495 | 58.883 | 80.398 | [u511jzaa](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/u511jzaa) | 2026-04-24 00:26 UTC |
| 27 | 81.915 | [#6](https://github.com/morganmcg1/tandemfoil2/pull/6) | LR schedule plus AMP anchor | 97.005 | 92.410 | 58.967 | 79.280 | [d3eydf7d](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/d3eydf7d) | 2026-04-24 00:47 UTC |
| 28 | 81.915 | [#18](https://github.com/morganmcg1/tandemfoil2/pull/18) | cross-attention surface decoder | 97.005 | 92.410 | 58.967 | 79.280 | [o9hcltd3](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/o9hcltd3) | 2026-04-24 01:11 UTC |
| 29 | 82.739 | [#16](https://github.com/morganmcg1/tandemfoil2/pull/16) | capacity sweep on AMP baseline | 94.282 | 95.991 | 58.412 | 82.271 | [2w8sjpna](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/2w8sjpna) | 2026-04-24 00:03 UTC |
| 30 | 89.344 | [#15](https://github.com/morganmcg1/tandemfoil2/pull/15) | horizontal flip augmentation | 107.680 | 106.061 | 59.338 | 84.296 | [wvpwgjvs](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/wvpwgjvs) | 2026-04-23 23:43 UTC |

PRs without a finite W&B `test_avg/mae_surf_p` at this snapshot: [#1](https://github.com/morganmcg1/tandemfoil2/pull/1), [#2](https://github.com/morganmcg1/tandemfoil2/pull/2), [#3](https://github.com/morganmcg1/tandemfoil2/pull/3), [#4](https://github.com/morganmcg1/tandemfoil2/pull/4), [#5](https://github.com/morganmcg1/tandemfoil2/pull/5), [#8](https://github.com/morganmcg1/tandemfoil2/pull/8), [#11](https://github.com/morganmcg1/tandemfoil2/pull/11), [#29](https://github.com/morganmcg1/tandemfoil2/pull/29), [#40](https://github.com/morganmcg1/tandemfoil2/pull/40), [#42](https://github.com/morganmcg1/tandemfoil2/pull/42), [#43](https://github.com/morganmcg1/tandemfoil2/pull/43).
