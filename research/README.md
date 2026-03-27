# Autoresearch Harness

This workspace keeps the autoresearch loop outside `records/` while targeting the recurrent/shared-depth submission script at:

- `/Users/kai/Desktop/parameter-golf/records/track_10min_16mb/2026-03-20_RecurrentSharedDepth_SP1024_4x3_rank4/train_gpt.py`

`manifest.yaml` is stored as JSON-compatible YAML so the harness can parse it with the Python standard library.

## Commands

Initialize and build the first report:

```bash
python3 research/autoresearch.py init
```

Preview the batch-0 commands without launching anything:

```bash
python3 research/autoresearch.py batch0 --dry-run
```

Run the baseline tiers:

```bash
python3 research/autoresearch.py batch0
```

Run the first env-search batch:

```bash
python3 research/autoresearch.py phase-a --mode initial
```

Run the local-neighborhood env-search batch:

```bash
python3 research/autoresearch.py phase-a --mode local
```

Run a phase-B template family:

```bash
python3 research/autoresearch.py phase-b --family activation
```

Promote the best proxy candidates to final verification:

```bash
python3 research/autoresearch.py finalize --top-k 1
```

Build the report:

```bash
python3 research/autoresearch.py report
```

Parse an existing log file:

```bash
python3 research/autoresearch.py parse-log records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train.log
```
