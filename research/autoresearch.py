#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "research"
MANIFEST_PATH = WORKSPACE / "manifest.yaml"
RESULTS_PATH = WORKSPACE / "results.tsv"
RUNS_DIR = WORKSPACE / "runs"
REPORT_PATH = WORKSPACE / "report.md"

LEDGER_COLUMNS = [
    "exp_id",
    "parent_exp_id",
    "batch",
    "tier",
    "budget_group",
    "change_family",
    "summary",
    "status",
    "decision",
    "val_loss",
    "val_bpb",
    "artifact_bytes",
    "peak_memory_mib",
    "elapsed_s",
    "estimated_cost_usd",
    "returncode",
    "notes",
]

FINAL_EXACT_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+)"
)
SUBMISSION_SIZE_RE = re.compile(
    r"Total submission size int8\+zlib: (?P<bytes>\d+) bytes"
)
PEAK_MEMORY_RE = re.compile(
    r"peak memory allocated: (?P<allocated>\d+) MiB reserved: (?P<reserved>\d+) MiB"
)
STEP_RE = re.compile(r"step:(?P<step>\d+)/(?P<iterations>\d+)")


def fail(message: str) -> None:
    raise SystemExit(message)


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if not path.exists():
        fail(f"Manifest not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"Failed to parse {path}: {exc}")


def ensure_workspace() -> None:
    WORKSPACE.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)
    if not RESULTS_PATH.exists():
        write_results([])


def read_results() -> list[dict[str, str]]:
    if not RESULTS_PATH.exists():
        return []
    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def write_results(rows: list[dict[str, Any]]) -> None:
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: stringify(row.get(key, "")) for key in LEDGER_COLUMNS}
            )


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def next_exp_id() -> str:
    existing = []
    for child in RUNS_DIR.glob("exp_*"):
        suffix = child.name.removeprefix("exp_")
        if suffix.isdigit():
            existing.append(int(suffix))
    return f"exp_{(max(existing, default=0) + 1):03d}"


def create_exp_dir() -> tuple[str, Path]:
    while True:
        exp_id = next_exp_id()
        exp_dir = RUNS_DIR / exp_id
        try:
            exp_dir.mkdir(parents=True, exist_ok=False)
            return exp_id, exp_dir
        except FileExistsError:
            continue


def target_cwd(manifest: dict[str, Any]) -> Path:
    return ROOT / manifest["target"]["cwd"]


def target_script_path(manifest: dict[str, Any]) -> Path:
    return target_cwd(manifest) / manifest["target"]["script"]


def build_command(manifest: dict[str, Any], tier_name: str) -> list[str]:
    tier = manifest["tiers"][tier_name]
    command = list(manifest["target"]["command"])
    nproc = str(tier["nproc_per_node"])
    return [part.format(nproc_per_node=nproc) for part in command]


def filter_env(
    manifest: dict[str, Any], tier_name: str, overrides: dict[str, str], exp_id: str
) -> dict[str, str]:
    merged: dict[str, str] = {}
    merged.update({k: str(v) for k, v in manifest["target"]["base_env"].items()})
    merged.update({k: str(v) for k, v in manifest["tiers"][tier_name]["env"].items()})
    merged.update({k: str(v) for k, v in overrides.items()})
    merged["RUN_ID"] = exp_id
    return merged


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def estimate_cost(
    manifest: dict[str, Any], tier_name: str, elapsed_s: float | None = None
) -> float:
    tier = manifest["tiers"][tier_name]
    runtime = elapsed_s if elapsed_s is not None else float(tier["estimated_runtime_s"])
    hourly_rate = float(tier["hourly_rate_usd"])
    return hourly_rate * runtime / 3600.0


def spent_by_group(
    manifest: dict[str, Any], rows: list[dict[str, str]], budget_group: str
) -> float:
    return sum(
        parse_float(row.get("estimated_cost_usd")) or 0.0
        for row in rows
        if row.get("budget_group") == budget_group
    )


def total_spent(rows: list[dict[str, str]]) -> float:
    return sum(parse_float(row.get("estimated_cost_usd")) or 0.0 for row in rows)


def ensure_budget(
    manifest: dict[str, Any], tier_name: str, rows: list[dict[str, str]]
) -> None:
    tier = manifest["tiers"][tier_name]
    budget_group = tier["budget_group"]
    estimate = estimate_cost(manifest, tier_name)
    group_budget = float(manifest["budget"]["groups"][budget_group])
    group_spend = spent_by_group(manifest, rows, budget_group)
    if group_spend + estimate > group_budget + 1e-9:
        fail(
            f"Budget exceeded for group {budget_group}: "
            f"spent={group_spend:.2f} estimate={estimate:.2f} budget={group_budget:.2f}"
        )
    reserve = float(manifest["budget"]["buffer_usd"])
    total_budget = float(manifest["budget"]["total_usd"])
    spent = total_spent(rows)
    if spent + estimate + reserve > total_budget + 1e-9:
        fail(
            f"Insufficient total budget: spent={spent:.2f} estimate={estimate:.2f} "
            f"reserve={reserve:.2f} total={total_budget:.2f}"
        )


def parse_run_log(log_path: Path, returncode: int | None = None) -> dict[str, Any]:
    if not log_path.exists():
        return {"status": "crash", "notes": "run log missing"}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics: dict[str, Any] = {}
    final_match = None
    for match in FINAL_EXACT_RE.finditer(text):
        final_match = match
    if final_match is not None:
        metrics["status"] = "success"
        metrics["val_loss"] = float(final_match.group("val_loss"))
        metrics["val_bpb"] = float(final_match.group("val_bpb"))
    else:
        metrics["status"] = "crash"
    size_match = None
    for match in SUBMISSION_SIZE_RE.finditer(text):
        size_match = match
    if size_match is not None:
        metrics["artifact_bytes"] = int(size_match.group("bytes"))
    peak_match = None
    for match in PEAK_MEMORY_RE.finditer(text):
        peak_match = match
    if peak_match is not None:
        metrics["peak_memory_mib"] = int(peak_match.group("allocated"))
    step_match = None
    for match in STEP_RE.finditer(text):
        step_match = match
    if step_match is not None:
        metrics["step"] = int(step_match.group("step"))
        metrics["iterations"] = int(step_match.group("iterations"))
    if metrics["status"] == "crash":
        lowered = text.lower()
        if "out of memory" in lowered:
            metrics["notes"] = "oom"
        elif "cuda is required" in lowered:
            metrics["notes"] = "cuda_required"
        elif returncode is not None:
            metrics["notes"] = f"returncode={returncode}"
        else:
            metrics["notes"] = "missing_final_metric"
    return metrics


def snapshot_patch(manifest: dict[str, Any], output_path: Path) -> None:
    target = target_script_path(manifest).relative_to(ROOT)
    result = subprocess.run(
        ["git", "diff", "--binary", "--", str(target)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output_path.write_text(result.stdout, encoding="utf-8")


def read_run_env(exp_id: str) -> dict[str, str]:
    env_path = RUNS_DIR / exp_id / "env.json"
    if not env_path.exists():
        fail(f"env.json missing for {exp_id}")
    payload = json.loads(env_path.read_text(encoding="utf-8"))
    return {k: str(v) for k, v in payload["overrides"].items()}


def successful_rows(rows: list[dict[str, str]], tier: str) -> list[dict[str, str]]:
    good = []
    for row in rows:
        if row.get("tier") != tier or row.get("status") != "success":
            continue
        if parse_float(row.get("val_bpb")) is None:
            continue
        good.append(row)
    return sorted(good, key=lambda row: float(row["val_bpb"]))


def choose_top_short_rows(
    manifest: dict[str, Any],
    rows: list[dict[str, str]],
    batch_name: str,
    tier: str,
    top_k: int,
) -> list[dict[str, str]]:
    limit = int(manifest["target"]["artifact_limit_bytes"])
    batch_rows = [
        row
        for row in rows
        if row.get("batch") == batch_name
        and row.get("tier") == tier
        and row.get("status") == "success"
        and (parse_int(row.get("artifact_bytes")) or (limit + 1)) <= limit
    ]
    return sorted(batch_rows, key=lambda row: float(row["val_bpb"]))[:top_k]


def current_best_proxy(rows: list[dict[str, str]]) -> dict[str, str] | None:
    proxy_rows = successful_rows(rows, "proxy1xh100")
    return proxy_rows[0] if proxy_rows else None


def render_summary(overrides: dict[str, str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(overrides.items()))


def run_experiment(
    manifest: dict[str, Any],
    *,
    tier_name: str,
    batch_name: str,
    change_family: str,
    summary: str,
    overrides: dict[str, str],
    parent_exp_id: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    ensure_workspace()
    rows = read_results()
    if not dry_run:
        ensure_budget(manifest, tier_name, rows)
    exp_id, exp_dir = create_exp_dir()

    command = build_command(manifest, tier_name)
    target_dir = target_cwd(manifest)
    env_overrides = filter_env(manifest, tier_name, overrides, exp_id)
    env_payload = {
        "exp_id": exp_id,
        "parent_exp_id": parent_exp_id,
        "batch": batch_name,
        "tier": tier_name,
        "change_family": change_family,
        "summary": summary,
        "cwd": str(target_dir),
        "command": command,
        "command_shell": shell_join(command),
        "overrides": overrides,
        "env": env_overrides,
    }
    write_json(exp_dir / "env.json", env_payload)
    snapshot_patch(manifest, exp_dir / "diff.patch")

    row: dict[str, Any] = {
        "exp_id": exp_id,
        "parent_exp_id": parent_exp_id,
        "batch": batch_name,
        "tier": tier_name,
        "budget_group": manifest["tiers"][tier_name]["budget_group"],
        "change_family": change_family,
        "summary": summary,
        "status": "dry_run" if dry_run else "crash",
        "decision": "preview" if dry_run else "pending",
        "estimated_cost_usd": estimate_cost(manifest, tier_name),
        "returncode": "",
        "notes": "",
    }

    log_path = exp_dir / "run.log"
    metrics: dict[str, Any]
    if dry_run:
        log_path.write_text(
            f"[dry-run]\ncd {shlex.quote(str(target_dir))}\n{shell_join(command)}\n",
            encoding="utf-8",
        )
        metrics = {"status": "dry_run", "notes": "preview only"}
    else:
        start = time.monotonic()
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"[cwd] {target_dir}\n")
            handle.write(f"[command] {shell_join(command)}\n")
            handle.flush()
            proc = subprocess.run(
                command,
                cwd=target_dir,
                env={**os.environ, **env_overrides},
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        elapsed_s = time.monotonic() - start
        metrics = parse_run_log(log_path, returncode=proc.returncode)
        row["elapsed_s"] = elapsed_s
        row["estimated_cost_usd"] = estimate_cost(manifest, tier_name, elapsed_s)
        row["returncode"] = proc.returncode
        row["status"] = metrics["status"]
        row["decision"] = "pending" if metrics["status"] == "success" else "crash"
        row["notes"] = metrics.get("notes", "")

    for key in ("val_loss", "val_bpb", "artifact_bytes", "peak_memory_mib"):
        if key in metrics:
            row[key] = metrics[key]
    write_json(exp_dir / "metrics.json", {"row": row, "metrics": metrics})

    if not dry_run:
        rows.append({key: stringify(row.get(key, "")) for key in LEDGER_COLUMNS})
        write_results(rows)
    return row


def update_decisions(updates: dict[str, str]) -> None:
    rows = read_results()
    changed = False
    for row in rows:
        decision = updates.get(row["exp_id"])
        if decision is not None:
            row["decision"] = decision
            changed = True
    if changed:
        write_results(rows)


def generate_phase_a_initial_candidates() -> list[dict[str, Any]]:
    return [
        {
            "summary": "layers=3 loops=3 rank=4",
            "overrides": {"NUM_LAYERS": "3", "NUM_LOOPS": "3", "LORA_RANK": "4"},
        },
        {
            "summary": "layers=5 loops=3 rank=4",
            "overrides": {"NUM_LAYERS": "5", "NUM_LOOPS": "3", "LORA_RANK": "4"},
        },
        {
            "summary": "layers=4 loops=2 rank=4",
            "overrides": {"NUM_LAYERS": "4", "NUM_LOOPS": "2", "LORA_RANK": "4"},
        },
        {
            "summary": "layers=4 loops=4 rank=2",
            "overrides": {"NUM_LAYERS": "4", "NUM_LOOPS": "4", "LORA_RANK": "2"},
        },
        {
            "summary": "lower matrix/scalar/embed lr",
            "overrides": {
                "MATRIX_LR": "0.018",
                "SCALAR_LR": "0.018",
                "TIED_EMBED_LR": "0.028",
                "LORA_LR": "0.008",
            },
        },
        {
            "summary": "higher matrix/scalar/embed lr",
            "overrides": {
                "MATRIX_LR": "0.024",
                "SCALAR_LR": "0.024",
                "TIED_EMBED_LR": "0.034",
                "LORA_LR": "0.012",
            },
        },
        {
            "summary": "shorter warmdown clip 0.8",
            "overrides": {"WARMDOWN_ITERS": "12000", "GRAD_CLIP_NORM": "0.8"},
        },
        {
            "summary": "longer warmdown clip 1.2",
            "overrides": {"WARMDOWN_ITERS": "24000", "GRAD_CLIP_NORM": "1.2"},
        },
        {
            "summary": "lower qk gain and softcap",
            "overrides": {"QK_GAIN_INIT": "1.2", "LOGIT_SOFTCAP": "20.0"},
        },
        {
            "summary": "higher qk gain and sliding eval",
            "overrides": {"QK_GAIN_INIT": "1.8", "EVAL_STRIDE": "64"},
        },
    ]


def generate_phase_a_local_candidates(base_env: dict[str, str]) -> list[dict[str, Any]]:
    def get_int(name: str, default: int) -> int:
        return int(base_env.get(name, str(default)))

    def get_float(name: str, default: float) -> float:
        return float(base_env.get(name, str(default)))

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()

    def add(overrides: dict[str, str]) -> None:
        signature = tuple(sorted(overrides.items()))
        if signature in seen:
            return
        seen.add(signature)
        candidates.append(
            {"summary": render_summary(overrides), "overrides": overrides}
        )

    num_layers = get_int("NUM_LAYERS", 4)
    num_loops = get_int("NUM_LOOPS", 3)
    lora_rank = get_int("LORA_RANK", 4)
    matrix_lr = get_float("MATRIX_LR", 0.02)
    scalar_lr = get_float("SCALAR_LR", 0.02)
    tied_embed_lr = get_float("TIED_EMBED_LR", 0.03)
    warmdown_iters = get_int("WARMDOWN_ITERS", 20000)
    grad_clip = get_float("GRAD_CLIP_NORM", 1.0)
    qk_gain = get_float("QK_GAIN_INIT", 1.5)
    logit_softcap = get_float("LOGIT_SOFTCAP", 30.0)

    for candidate_layers in {
        max(3, num_layers - 1),
        num_layers,
        min(6, num_layers + 1),
    }:
        for candidate_loops in {
            max(2, num_loops - 1),
            num_loops,
            min(4, num_loops + 1),
        }:
            add(
                {
                    "NUM_LAYERS": str(candidate_layers),
                    "NUM_LOOPS": str(candidate_loops),
                    "LORA_RANK": str(lora_rank),
                }
            )
    for candidate_rank in (0, 2, 4, 8):
        add({"LORA_RANK": str(candidate_rank)})
    for scale in (0.9, 1.1):
        add(
            {
                "MATRIX_LR": f"{matrix_lr * scale:.5f}",
                "SCALAR_LR": f"{scalar_lr * scale:.5f}",
                "TIED_EMBED_LR": f"{tied_embed_lr * scale:.5f}",
            }
        )
    for clip in (max(0.1, grad_clip * 0.8), grad_clip, grad_clip * 1.2):
        add({"GRAD_CLIP_NORM": f"{clip:.3f}"})
    for warmdown in (max(4000, warmdown_iters - 4000), warmdown_iters + 4000):
        add({"WARMDOWN_ITERS": str(warmdown)})
    for gain in (max(0.8, qk_gain - 0.2), qk_gain + 0.2):
        add({"QK_GAIN_INIT": f"{gain:.3f}"})
    for softcap in (max(10.0, logit_softcap - 10.0), logit_softcap + 10.0):
        add({"LOGIT_SOFTCAP": f"{softcap:.1f}"})
    return candidates[:10]


def generate_phase_b_candidates(family: str) -> list[dict[str, Any]]:
    if family == "activation":
        return [
            {
                "summary": "leaky relu2 slope 0.10",
                "overrides": {
                    "MLP_ACTIVATION": "leaky_relu2",
                    "LEAKY_RELU_SLOPE": "0.10",
                },
            },
            {
                "summary": "leaky relu2 slope 0.25",
                "overrides": {
                    "MLP_ACTIVATION": "leaky_relu2",
                    "LEAKY_RELU_SLOPE": "0.25",
                },
            },
            {
                "summary": "leaky relu2 slope 0.50",
                "overrides": {
                    "MLP_ACTIVATION": "leaky_relu2",
                    "LEAKY_RELU_SLOPE": "0.50",
                },
            },
        ]
    if family == "rope":
        return [
            {"summary": "partial rope 16 dims", "overrides": {"ROPE_DIMS": "16"}},
            {"summary": "partial rope 32 dims", "overrides": {"ROPE_DIMS": "32"}},
            {"summary": "full rope 64 dims", "overrides": {"ROPE_DIMS": "64"}},
        ]
    if family == "scope":
        return [
            {"summary": "lora scope qproj", "overrides": {"LORA_SCOPE": "qproj"}},
            {"summary": "lora scope qvproj", "overrides": {"LORA_SCOPE": "qvproj"}},
            {"summary": "lora scope qkvproj", "overrides": {"LORA_SCOPE": "qkvproj"}},
        ]
    if family == "depth":
        return [
            {
                "summary": "depth scale layer sqrt",
                "overrides": {"DEPTH_SCALE_MODE": "layer_sqrt"},
            },
            {
                "summary": "depth scale loop sqrt",
                "overrides": {"DEPTH_SCALE_MODE": "loop_sqrt"},
            },
        ]
    if family == "ema_qat":
        return [
            {
                "summary": "ema 0.995 late qat 0.20",
                "overrides": {
                    "EMA_ENABLED": "1",
                    "EMA_DECAY": "0.995",
                    "LATE_QAT_THRESHOLD": "0.20",
                },
            },
            {
                "summary": "ema 0.997 late qat 0.15",
                "overrides": {
                    "EMA_ENABLED": "1",
                    "EMA_DECAY": "0.997",
                    "LATE_QAT_THRESHOLD": "0.15",
                },
            },
            {
                "summary": "ema 0.999 late qat 0.10",
                "overrides": {
                    "EMA_ENABLED": "1",
                    "EMA_DECAY": "0.999",
                    "LATE_QAT_THRESHOLD": "0.10",
                },
            },
        ]
    fail(f"Unsupported phase-B family: {family}")


def run_batch(
    manifest: dict[str, Any],
    *,
    batch_name: str,
    short_tier: str,
    promote_tier: str,
    change_family: str,
    candidates: list[dict[str, Any]],
    top_k: int,
    dry_run: bool = False,
) -> None:
    short_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        short_rows.append(
            run_experiment(
                manifest,
                tier_name=short_tier,
                batch_name=batch_name,
                change_family=change_family,
                summary=candidate["summary"],
                overrides=candidate["overrides"],
                dry_run=dry_run,
            )
        )
    if dry_run:
        return

    rows = read_results()
    promoted = choose_top_short_rows(manifest, rows, batch_name, short_tier, top_k)
    updates: dict[str, str] = {}
    promoted_ids = {row["exp_id"] for row in promoted}
    for row in rows:
        if row.get("batch") == batch_name and row.get("tier") == short_tier:
            if row["status"] != "success":
                updates[row["exp_id"]] = "crash"
            elif row["exp_id"] in promoted_ids:
                updates[row["exp_id"]] = "promote"
            else:
                updates[row["exp_id"]] = "discard"
    update_decisions(updates)

    promotion_threshold = float(manifest["promotion"]["proxy_keep_delta_bpb"])
    for promoted_row in promoted:
        overrides = read_run_env(promoted_row["exp_id"])
        before_proxy_rows = read_results()
        previous_best = current_best_proxy(before_proxy_rows)
        result = run_experiment(
            manifest,
            tier_name=promote_tier,
            batch_name=batch_name,
            change_family=change_family,
            summary=f"{promoted_row['summary']} -> {promote_tier}",
            overrides=overrides,
            parent_exp_id=promoted_row["exp_id"],
            dry_run=False,
        )
        proxy_updates = {}
        if result["status"] != "success":
            proxy_updates[result["exp_id"]] = "crash"
        elif previous_best is None:
            proxy_updates[result["exp_id"]] = "baseline"
        else:
            previous_best_bpb = float(previous_best["val_bpb"])
            current_bpb = float(result["val_bpb"])
            proxy_updates[result["exp_id"]] = (
                "keep"
                if current_bpb <= previous_best_bpb - promotion_threshold
                else "discard"
            )
        update_decisions(proxy_updates)


def find_best_proxy_env(manifest: dict[str, Any]) -> dict[str, str]:
    rows = read_results()
    best = current_best_proxy(rows)
    if best is None:
        return {k: str(v) for k, v in manifest["target"]["base_env"].items()}
    run_env = json.loads(
        (RUNS_DIR / best["exp_id"] / "env.json").read_text(encoding="utf-8")
    )
    return {k: str(v) for k, v in run_env["env"].items()}


def run_batch0(manifest: dict[str, Any], dry_run: bool = False) -> None:
    baseline_smoke = run_experiment(
        manifest,
        tier_name="smoke5090",
        batch_name="batch0",
        change_family="baseline",
        summary="5090 smoke baseline",
        overrides={},
        dry_run=dry_run,
    )
    baseline_proxy = run_experiment(
        manifest,
        tier_name="proxy1xh100",
        batch_name="batch0",
        change_family="baseline",
        summary="1xH100 baseline",
        overrides={},
        dry_run=dry_run,
    )
    if dry_run:
        return
    updates = {
        baseline_smoke["exp_id"]: (
            "baseline_smoke" if baseline_smoke["status"] == "success" else "crash"
        ),
        baseline_proxy["exp_id"]: (
            "baseline" if baseline_proxy["status"] == "success" else "crash"
        ),
    }
    update_decisions(updates)


def finalize_candidates(
    manifest: dict[str, Any], top_k: int, dry_run: bool = False
) -> None:
    rows = read_results()
    candidates = [
        row
        for row in successful_rows(rows, "proxy1xh100")
        if row.get("decision") == "keep"
    ]
    if not candidates:
        fail("No proxy keep candidates available for finalization.")
    for candidate in candidates[:top_k]:
        overrides = read_run_env(candidate["exp_id"])
        result = run_experiment(
            manifest,
            tier_name="final8xh100",
            batch_name="batch4",
            change_family="finalize",
            summary=f"finalize {candidate['exp_id']}: {candidate['summary']}",
            overrides=overrides,
            parent_exp_id=candidate["exp_id"],
            dry_run=dry_run,
        )
        if dry_run:
            continue
        update_decisions(
            {
                result["exp_id"]: (
                    "final_candidate" if result["status"] == "success" else "crash"
                )
            }
        )


def build_report(manifest: dict[str, Any]) -> str:
    rows = read_results()
    by_group = {
        group: spent_by_group(manifest, rows, group)
        for group in manifest["budget"]["groups"]
    }
    lines = [
        "# Autoresearch Report",
        "",
        "## Budget",
        "",
        f"- Total spent: ${total_spent(rows):.2f} / ${float(manifest['budget']['total_usd']):.2f}",
    ]
    for group, spent in by_group.items():
        group_budget = float(manifest["budget"]["groups"][group])
        lines.append(f"- {group}: ${spent:.2f} / ${group_budget:.2f}")

    lines.extend(["", "## Best Per Tier", ""])
    for tier in ("smoke5090", "short5090", "proxy1xh100", "final8xh100"):
        tier_rows = successful_rows(rows, tier)
        if not tier_rows:
            lines.append(f"- {tier}: none")
            continue
        best = tier_rows[0]
        lines.append(
            f"- {tier}: {best['exp_id']} val_bpb={best['val_bpb']} decision={best['decision']} summary={best['summary']}"
        )

    lines.extend(
        [
            "",
            "## 5090 To 1xH100 Correlation",
            "",
            "| 5090 exp | 5090 bpb | Proxy exp | Proxy bpb | Family |",
            "|---|---:|---|---:|---|",
        ]
    )
    proxy_rows = [
        row
        for row in rows
        if row.get("tier") == "proxy1xh100" and row.get("parent_exp_id")
    ]
    for proxy in proxy_rows:
        parent = next(
            (row for row in rows if row["exp_id"] == proxy["parent_exp_id"]), None
        )
        if parent is None:
            continue
        lines.append(
            f"| {parent['exp_id']} | {parent.get('val_bpb', '')} | {proxy['exp_id']} | {proxy.get('val_bpb', '')} | {proxy['change_family']} |"
        )
    report = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")
    return report


def parse_set_overrides(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            fail(f"Expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = value
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter Golf autoresearch harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create research workspace files if missing")

    parse_log_parser = subparsers.add_parser(
        "parse-log", help="Parse an existing run log"
    )
    parse_log_parser.add_argument("log_path", type=Path)
    parse_log_parser.add_argument("--returncode", type=int, default=None)

    run_parser = subparsers.add_parser("run", help="Run a single experiment")
    run_parser.add_argument(
        "--tier",
        required=True,
        choices=["smoke5090", "short5090", "proxy1xh100", "final8xh100"],
    )
    run_parser.add_argument("--batch", default="manual")
    run_parser.add_argument("--family", default="manual")
    run_parser.add_argument("--summary", default="")
    run_parser.add_argument("--parent-exp-id", default="")
    run_parser.add_argument("--set", action="append", default=[])
    run_parser.add_argument("--dry-run", action="store_true")

    batch0_parser = subparsers.add_parser("batch0", help="Run batch 0 baselines")
    batch0_parser.add_argument("--dry-run", action="store_true")

    phase_a_parser = subparsers.add_parser("phase-a", help="Run phase-A search batch")
    phase_a_parser.add_argument("--mode", choices=["initial", "local"], required=True)
    phase_a_parser.add_argument("--batch", default="")
    phase_a_parser.add_argument("--dry-run", action="store_true")

    phase_b_parser = subparsers.add_parser("phase-b", help="Run phase-B template batch")
    phase_b_parser.add_argument(
        "--family",
        choices=["activation", "rope", "scope", "depth", "ema_qat"],
        required=True,
    )
    phase_b_parser.add_argument("--batch", default="")
    phase_b_parser.add_argument("--dry-run", action="store_true")

    finalize_parser = subparsers.add_parser(
        "finalize", help="Promote top proxy candidates to final tier"
    )
    finalize_parser.add_argument("--top-k", type=int, default=1)
    finalize_parser.add_argument("--dry-run", action="store_true")

    subparsers.add_parser("report", help="Build a markdown report")

    args = parser.parse_args()
    ensure_workspace()

    if args.command == "init":
        manifest = load_manifest()
        build_report(manifest)
        print(f"Initialized workspace at {WORKSPACE}")
        return

    if args.command == "parse-log":
        metrics = parse_run_log(args.log_path, args.returncode)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    manifest = load_manifest()

    if args.command == "run":
        overrides = parse_set_overrides(args.set)
        summary = args.summary or render_summary(overrides) or f"manual {args.tier}"
        row = run_experiment(
            manifest,
            tier_name=args.tier,
            batch_name=args.batch,
            change_family=args.family,
            summary=summary,
            overrides=overrides,
            parent_exp_id=args.parent_exp_id,
            dry_run=args.dry_run,
        )
        print(json.dumps(row, indent=2, sort_keys=True))
        return

    if args.command == "batch0":
        run_batch0(manifest, dry_run=args.dry_run)
        print("Batch 0 completed.")
        return

    if args.command == "phase-a":
        batch_name = args.batch or ("batch1" if args.mode == "initial" else "batch2")
        candidates = (
            generate_phase_a_initial_candidates()
            if args.mode == "initial"
            else generate_phase_a_local_candidates(find_best_proxy_env(manifest))
        )
        run_batch(
            manifest,
            batch_name=batch_name,
            short_tier="short5090",
            promote_tier="proxy1xh100",
            change_family=f"phase_a_{args.mode}",
            candidates=candidates,
            top_k=int(manifest["promotion"]["short_top_k"]),
            dry_run=args.dry_run,
        )
        print(f"Phase-A {args.mode} batch completed.")
        return

    if args.command == "phase-b":
        batch_name = args.batch or "batch3"
        run_batch(
            manifest,
            batch_name=batch_name,
            short_tier="short5090",
            promote_tier="proxy1xh100",
            change_family=f"phase_b_{args.family}",
            candidates=generate_phase_b_candidates(args.family),
            top_k=int(manifest["promotion"]["phase_b_top_k"]),
            dry_run=args.dry_run,
        )
        print(f"Phase-B {args.family} batch completed.")
        return

    if args.command == "finalize":
        finalize_candidates(manifest, top_k=args.top_k, dry_run=args.dry_run)
        print("Finalize step completed.")
        return

    if args.command == "report":
        print(build_report(manifest))
        return

    fail(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
