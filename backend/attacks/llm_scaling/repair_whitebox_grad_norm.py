"""Repair failed ``grad_norm`` entries in existing white-box precomputations.

This utility intentionally recomputes only records whose stored gradient norm is
zero or absent. It loads the original saved Hugging Face splits from
``<output_path>/members`` and ``<output_path>/nonmembers`` so that the exact
full text used during precomputation is recovered; ``_text_preview`` is never
used as a substitute.

Run from the backend directory, for example:

    python -u -m attacks.llm_scaling.repair_whitebox_grad_norm \
        --model-name EleutherAI/pythia-6.9b \
        --output-path results/whitebox_mia/EleutherAI/pythia-6.9b/haritzpuerto/the_pile_00_arxiv/2048 \
        --cache-dir /tmp --resume

The original JSONL file is retained as ``*.before_gradnorm_repair`` before a
completed temporary output is atomically moved into place. Interrupted runs
keep their temporary file and can continue with ``--resume``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from attacks.llm_scaling.whitebox_utils import compute_gradient_norm


def log(message: str) -> None:
    """Emit a line immediately so ``tail -f`` shows live progress."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stdout, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair only failed grad_norm values in white-box JSONL outputs."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--cache-dir", default="/tmp")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("members", "nonmembers"),
        default=("members", "nonmembers"),
        help="Saved data splits to repair. Both are selected by default.",
    )
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute forward activations during backward to reduce GPU memory usage.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue appending to a previous temporary rewrite for each JSONL file.",
    )
    return parser.parse_args()


def count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def requires_repair(record: dict) -> bool:
    value = record.get("pred", {}).get("grad_norm")
    if value is None:
        return True
    try:
        return not float(value) > 0.0
    except (TypeError, ValueError):
        return True


def repair_split(
    split_name: str,
    output_path: Path,
    model,
    tokenizer,
    max_length: int,
    log_every: int,
    resume: bool,
) -> None:
    jsonl_name = f"mia_{split_name}.jsonl"
    source_path = output_path / jsonl_name
    saved_split_path = output_path / split_name
    temp_path = output_path / f"{jsonl_name}.gradnorm_repair.tmp"
    backup_path = output_path / f"{jsonl_name}.before_gradnorm_repair"
    error_path = output_path / f"{jsonl_name}.gradnorm_repair_errors.log"

    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source JSONL: {source_path}")
    if not saved_split_path.is_dir():
        raise FileNotFoundError(
            f"Missing saved {split_name} split at {saved_split_path}; cannot safely recover full texts."
        )

    split_data = load_from_disk(str(saved_split_path))
    source_count = count_nonempty_lines(source_path)
    if len(split_data) != source_count:
        raise ValueError(
            f"{split_name}: saved split has {len(split_data)} rows but JSONL has {source_count}; aborting."
        )

    completed = count_nonempty_lines(temp_path) if resume else 0
    if completed > source_count:
        raise ValueError(f"{split_name}: temporary output has more rows than the source JSONL.")
    mode = "a" if completed else "w"
    repaired = 0
    still_failed = 0
    skipped_valid = 0

    log(
        f"{split_name}: source rows={source_count}, resume rows={completed}, "
        f"temporary file={temp_path.name}"
    )
    with source_path.open("r", encoding="utf-8") as source, temp_path.open(mode, encoding="utf-8") as target, error_path.open("a", encoding="utf-8") as errors:
        output_index = 0
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            if output_index < completed:
                output_index += 1
                continue

            record = json.loads(line)
            if requires_repair(record):
                text = split_data[output_index]["text"]
                try:
                    # This is the same white-box operator and token limit used
                    # by precompute_whitebox.py, but only for the missing field.
                    value = float(compute_gradient_norm(text, model, tokenizer, max_length))
                    if value > 0.0 and torch.isfinite(torch.tensor(value)):
                        record.setdefault("pred", {})["grad_norm"] = value
                        repaired += 1
                        status = f"repaired grad_norm={value:.6g}"
                    else:
                        record.setdefault("pred", {})["grad_norm"] = 0.0
                        still_failed += 1
                        status = (
                            "returned a non-positive/non-finite grad_norm="
                            f"{value!r}; retained 0"
                        )
                        errors.write(
                            f"index={output_index} line={line_number} "
                            f"non_positive_or_non_finite_grad_norm={value!r} "
                            f"text_chars={len(text)}\n"
                        )
                        errors.flush()
                        log(f"{split_name}: {status}")
                except Exception as exc:  # Preserve the old value but expose the error this time.
                    record.setdefault("pred", {})["grad_norm"] = 0.0
                    still_failed += 1
                    status = f"failed ({type(exc).__name__}: {exc}); retained 0"
                    errors.write(
                        f"index={output_index} line={line_number} {type(exc).__name__}: {exc}\n"
                    )
                    errors.flush()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                finally:
                    # Only input_embeds needs gradients.  This also releases any
                    # partially materialized gradients when a backward pass OOMs.
                    model.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                skipped_valid += 1
                status = "kept existing grad_norm"

            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            target.flush()
            output_index += 1
            if output_index == 1 or output_index % max(log_every, 1) == 0 or output_index == source_count:
                log(
                    f"{split_name}: {output_index}/{source_count}; {status}; "
                    f"repaired={repaired}, still_failed={still_failed}, valid={skipped_valid}"
                )

    if output_index != source_count:
        raise RuntimeError(f"{split_name}: wrote {output_index} rows, expected {source_count}.")
    if not backup_path.exists():
        shutil.copy2(source_path, backup_path)
        log(f"{split_name}: backed up original JSONL to {backup_path.name}")
    os.replace(temp_path, source_path)
    log(
        f"{split_name}: complete; repaired={repaired}, still_failed={still_failed}, "
        f"valid={skipped_valid}; merged into {source_path.name}"
    )


def main() -> None:
    args = parse_args()
    output_path = args.output_path.resolve()
    log("[1/3] Grad-norm repair started.")
    log(f"Model: {args.model_name}")
    log(f"Output: {output_path}")
    log(f"Max length: {args.max_length}; splits: {', '.join(args.splits)}")

    log("[2/3] Loading tokenizer and model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        return_dict=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        output_attentions=False,
    )
    model.eval()
    # grad_norm is the gradient with respect to input embeddings, not model
    # weights. Freezing weights preserves that derivative while avoiding a
    # second 6.9B-sized allocation for parameter gradients during backward.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    log("Model weights frozen; gradients will be retained only for input embeddings.")
    if args.gradient_checkpointing:
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise RuntimeError("The loaded model does not support gradient checkpointing.")
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        log("Gradient checkpointing enabled; KV cache disabled for grad-norm repair.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    log(f"Model loaded. Primary device: {getattr(model, 'device', 'unknown')}")

    log("[3/3] Recomputing only missing or zero gradient norms ...")
    for split_name in args.splits:
        repair_split(
            split_name,
            output_path,
            model,
            tokenizer,
            args.max_length,
            args.log_every,
            args.resume,
        )
    log("Done. Existing non-gradient features were preserved unchanged.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted. The temporary JSONL is retained; rerun with --resume to continue.")
        raise
    except Exception:
        traceback.print_exc()
        raise
