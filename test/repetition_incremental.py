from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from vllm_service.detector.repetition_incremental import IncrementalTokenRepetitionDetector


from transformers import AutoTokenizer


def render_progress(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        total = 1
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({ratio * 100:5.1f}%)"


def print_progress(current: int, total: int, current_file: Path) -> None:
    msg = f"\rProcessing JSON files {render_progress(current, total)} | {current_file.name}"
    sys.stdout.write(msg)
    sys.stdout.flush()


def iter_json_files(root: Path, recursive: bool = True) -> Iterable[Path]:
    pattern = "**/*.json" if recursive else "*.json"
    file_name_pattern = re.compile(r"^step_\d+\.json$")
    for p in sorted(root.glob(pattern)):
        if file_name_pattern.match(p.name):
            yield p


def iter_responses(data: Any) -> Iterable[Tuple[int, int, Dict[str, Any]]]:
    if not isinstance(data, list):
        return

    for batch_idx, batch in enumerate(data):
        if not isinstance(batch, dict):
            continue
        responses = batch.get("responses", [])
        if not isinstance(responses, list):
            continue

        for response_idx, item in enumerate(responses):
            if isinstance(item, dict):
                yield batch_idx, response_idx, item


def predict_stop_reason(
    token_ids: Sequence[int],
    min_repeat_tokens: int,
    min_repeat_count: int,
    chunk_size_tokens: int,
    allow_gap: bool,
    max_gap_ratio: float,
    detection_mode: str,
) -> Tuple[str, str]:
    detector = IncrementalTokenRepetitionDetector(
        min_repeat_tokens=min_repeat_tokens,
        min_repeat_count=min_repeat_count,
        allow_gap=allow_gap,
        max_gap_ratio=max_gap_ratio,
    )

    if not token_ids:
        return "stop", ""

    if detection_mode == "streaming":
        for start in range(0, len(token_ids), chunk_size_tokens):
            chunk = token_ids[start : start + chunk_size_tokens]
            is_rep, reason = detector.append_and_detect(chunk)
            if is_rep:
                return "repetition_abort", reason
    elif detection_mode == "final":
        detector.append(token_ids)
        is_rep, reason = detector.detect()
        if is_rep:
            return "repetition_abort", reason
    else:
        raise ValueError(f"Unknown detection mode: {detection_mode}")

    return "stop", ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate stop_reason in JSON files against incremental repetition detector."
        )
    )
    parser.add_argument(
        "json_dir",
        type=Path,
        help="Directory containing JSON files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search JSON files under json_dir",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Tokenizer path passed to transformers.AutoTokenizer.from_pretrained",
    )
    parser.add_argument(
        "--min-repeat-tokens",
        type=int,
        default=120,
        help="min_repeat_tokens for detector",
    )
    parser.add_argument(
        "--min-repeat-count",
        type=int,
        default=30,
        help="min_repeat_count for detector",
    )
    parser.add_argument(
        "--chunk-size-tokens",
        type=int,
        default=32,
        help="Chunk size in token units to simulate incremental streaming input",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Return non-zero exit code when mismatches exist",
    )
    parser.add_argument(
        "--max-mismatch-print",
        type=int,
        default=50,
        help="Maximum mismatch rows to print",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Maximum number of JSON files to process (0 means no limit)",
    )
    parser.add_argument(
        "--only-stop-reason",
        type=str,
        default="",
        help="Only validate rows whose original stop_reason exactly matches this value",
    )
    parser.add_argument(
        "--allow-gap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gap-tolerant repetition matching (default: enabled)",
    )
    parser.add_argument(
        "--max-gap-ratio",
        type=float,
        default=1.0,
        help="Gap size formula uses int(pattern_len * ratio - 1)",
    )
    parser.add_argument(
        "--mismatch-output",
        type=Path,
        default=Path("repetition_mismatches.json"),
        help="Output JSON file path for all mismatched samples",
    )
    parser.add_argument(
        "--detection-mode",
        type=str,
        choices=["streaming", "final"],
        default="final",
        help=(
            "streaming: simulate incremental append; "
            "final: append full token sequence once and detect from tail"
        ),
    )
    args = parser.parse_args()

    if args.chunk_size_tokens <= 0:
        raise ValueError("chunk-size-tokens must be > 0")
    if args.max_files < 0:
        raise ValueError("max-files must be >= 0")
    if args.max_gap_ratio < 0:
        raise ValueError("max-gap-ratio must be >= 0")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    root = args.json_dir
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    files = list(iter_json_files(root, recursive=args.recursive))
    if not files:
        print(f"No JSON files found under: {root}")
        return

    if args.max_files > 0:
        files = files[: args.max_files]

    total = 0
    matched = 0
    mismatches: List[Dict[str, Any]] = []
    unsupported = 0
    filtered_out = 0
    detection_time_total = 0.0
    consistency_waived = 0

    for file_idx, json_file in enumerate(files, start=1):
        print_progress(file_idx, len(files), json_file)
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Failed to parse JSON: {json_file} ({exc})")
            continue

        for batch_idx, response_idx, item in iter_responses(data):
            actual = item.get("stop_reason")
            if not isinstance(actual, str):
                unsupported += 1
                continue
            if args.only_stop_reason and actual != args.only_stop_reason:
                filtered_out += 1
                continue

            resp_obj = item.get("response", {})
            if not isinstance(resp_obj, dict):
                unsupported += 1
                continue

            text = resp_obj.get("content", "")
            if not isinstance(text, str):
                unsupported += 1
                continue

            token_ids = tokenizer.encode(text, add_special_tokens=False)

            start_t = time.perf_counter()
            predicted, reason = predict_stop_reason(
                token_ids=token_ids,
                min_repeat_tokens=args.min_repeat_tokens,
                min_repeat_count=args.min_repeat_count,
                chunk_size_tokens=args.chunk_size_tokens,
                allow_gap=args.allow_gap,
                max_gap_ratio=args.max_gap_ratio,
                detection_mode=args.detection_mode,
            )
            detection_time_total += time.perf_counter() - start_t

            total += 1
            # Special rule: treat actual=length and predicted=stop as consistent.
            if actual == predicted or (actual == "length" and predicted == "stop"):
                matched += 1
                if actual == "length" and predicted == "stop":
                    consistency_waived += 1
            else:
                mismatches.append(
                    {
                        "file": str(json_file),
                        "batch_index": batch_idx,
                        "response_index": response_idx,
                        "response_content": text,
                        "actual": actual,
                        "original_stop_reason": actual,
                        "original_step_reason": actual,
                        "predicted": predicted,
                        "content_len": len(text),
                        "token_len": len(token_ids),
                        "reason": reason,
                    }
                )

    print()

    mismatch_count = len(mismatches)
    print("=== Repetition Stop Reason Consistency Report ===")
    print(f"json_files: {len(files)}")
    print(f"checked_responses: {total}")
    print(f"matched: {matched}")
    print(f"mismatched: {mismatch_count}")
    print(f"unsupported_rows: {unsupported}")
    print(f"filtered_out_rows: {filtered_out}")
    print(f"waived_length_to_stop_rows: {consistency_waived}")
    if args.only_stop_reason:
        print(f"only_stop_reason: {args.only_stop_reason}")
    print(f"tokenizer_path: {args.tokenizer_path}")
    print(f"detection_mode: {args.detection_mode}")
    print(f"allow_gap: {args.allow_gap}")
    print(f"max_gap_ratio: {args.max_gap_ratio}")
    accuracy = (matched / total) if total else 0.0
    print(f"match_rate: {accuracy:.4f}")
    avg_detection_ms = (detection_time_total / total * 1000.0) if total else 0.0
    print(f"total_detection_time_sec: {detection_time_total:.6f}")
    print(f"avg_detection_time_per_string_ms: {avg_detection_ms:.6f}")

    args.mismatch_output.parent.mkdir(parents=True, exist_ok=True)
    mismatch_payload = {
        "json_dir": str(root),
        "json_files": len(files),
        "checked_responses": total,
        "matched": matched,
        "mismatched": mismatch_count,
        "unsupported_rows": unsupported,
        "filtered_out_rows": filtered_out,
        "waived_length_to_stop_rows": consistency_waived,
        "only_stop_reason": args.only_stop_reason,
        "tokenizer_path": args.tokenizer_path,
        "detection_mode": args.detection_mode,
        "allow_gap": args.allow_gap,
        "max_gap_ratio": args.max_gap_ratio,
        "min_repeat_tokens": args.min_repeat_tokens,
        "min_repeat_count": args.min_repeat_count,
        "chunk_size_tokens": args.chunk_size_tokens,
        "mismatches": mismatches,
    }
    args.mismatch_output.write_text(
        json.dumps(mismatch_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"mismatch_output_file: {args.mismatch_output}")

    if mismatches:
        print("\n--- Mismatch Details ---")
        for idx, row in enumerate(mismatches[: args.max_mismatch_print], start=1):
            print(
                f"[{idx}] file={row['file']}, "
                f"batch={row['batch_index']}, response={row['response_index']}, "
                f"actual={row['actual']}, predicted={row['predicted']}, "
                f"content_len={row['content_len']}"
            )
            if row["reason"]:
                print(f"    detector_reason={row['reason']}")

        if mismatch_count > args.max_mismatch_print:
            remain = mismatch_count - args.max_mismatch_print
            print(f"... {remain} more mismatches omitted")

    if args.fail_on_mismatch and mismatch_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
