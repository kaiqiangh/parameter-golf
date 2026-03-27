# Recurrent Shared-Depth SP1024 4x3 rank4

This folder stages the record-track version of the **recurrent/shared-depth** `sp1024` submission. The model stores 4 unique transformer blocks and reuses them across 3 loops, giving 12 effective passes while staying well aligned with the challenge artifact budget.

## Core Design

- `VOCAB_SIZE=1024 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4`
- `NUM_LAYERS=4 NUM_LOOPS=3 LORA_RANK=4`
- tied embeddings enabled
- QAT enabled for shared linear weights: `QAT=1`
- standard final int8+zlib roundtrip eval by default: `EVAL_STRIDE=0`
- fp16 passthrough for `tok_emb.weight`
- fp16 passthrough for small loop adapters and control tensors
- long warmdown and gradient clipping for compression-aware training

The loop adapters target only `q`, `v`, and `proj`, keeping the recurrent/shared-depth model as the main capacity source rather than turning the submission into a nearly dense model in disguise.

## Record-Track Command

```bash
RUN_ID=recurrent_shared_depth_stage_b \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=4 \
NUM_LOOPS=3 \
LORA_RANK=4 \
QAT=1 \
EVAL_STRIDE=0 \
WARMDOWN_ITERS=20000 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
GRAD_CLIP_NORM=1.0 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=1000 \
TRAIN_LOG_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Run this three times for `SEED=42`, `SEED=1337`, and `SEED=7`, then replace the placeholder metadata and logs with the actual outputs.

## Status

This record folder is implemented as a submission-ready code snapshot, but the current workspace lacks a runnable PyTorch/CUDA environment. The included logs and `submission.json` are therefore explicit placeholders rather than claimed results.

## Included Files

- `train_gpt.py`
- `submission.json`
- `train_seed42.log`
- `train_seed1337.log`
- `train_seed7.log`

