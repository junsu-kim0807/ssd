#!/usr/bin/env python3
"""
Generate profile visualizations from verification-level JSONL.

Input JSONL is produced by:
  profile/run_intermediate_verifier_profile.py --verification-jsonl ...

Each row is one verification round for one request, with:
  - target.acceptance_rate / acceptance_length / acceptance_per_position_*
  - intermediate.acceptance_rate / acceptance_length / acceptance_per_position_*
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate profile result visualizations.")
    p.add_argument(
        "--input-jsonl",
        type=str,
        default="profile/results/verification_metrics.jsonl",
        help="Input verification metrics jsonl.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="profile/results/plots",
        help="Directory to write PNG plots.",
    )
    p.add_argument(
        "--max-requests-plot",
        type=int,
        default=100,
        help="Limit number of per-request plots (to avoid too many files).",
    )
    p.add_argument(
        "--bins-rate",
        type=int,
        default=20,
        help="Histogram bins for acceptance rate.",
    )
    p.add_argument(
        "--bins-length",
        type=int,
        default=20,
        help="Histogram bins for acceptance length.",
    )
    return p.parse_args()


def _request_key(row: dict) -> str:
    dataset = row.get("dataset", "unknown")
    sample_id = str(row.get("sample_id", "unknown"))
    request_index = row.get("request_index", -1)
    return f"{dataset}__{sample_id}__{request_index}"


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def save_request_distribution_plot(
    req_key: str,
    rows: list[dict],
    out_dir: Path,
    bins_rate: int,
    bins_length: int,
) -> None:
    target_rates = [r["target"]["acceptance_rate"] for r in rows]
    inter_rates = [r["intermediate"]["acceptance_rate"] for r in rows]
    target_lengths = [r["target"]["acceptance_length"] for r in rows]
    inter_lengths = [r["intermediate"]["acceptance_length"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(target_rates, bins=bins_rate, alpha=0.6, label="target")
    axes[0].hist(inter_rates, bins=bins_rate, alpha=0.6, label="intermediate")
    axes[0].set_title(f"Request Distribution: Acceptance Rate\n{req_key}")
    axes[0].set_xlabel("acceptance rate")
    axes[0].set_ylabel("frequency")
    axes[0].legend()

    axes[1].hist(target_lengths, bins=bins_length, alpha=0.6, label="target")
    axes[1].hist(inter_lengths, bins=bins_length, alpha=0.6, label="intermediate")
    axes[1].set_title(f"Request Distribution: Acceptance Length\n{req_key}")
    axes[1].set_xlabel("acceptance length")
    axes[1].set_ylabel("frequency")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"request_distribution_{req_key}.png", dpi=150)
    plt.close(fig)


def save_request_trend_plot(req_key: str, rows: list[dict], out_dir: Path) -> None:
    rows_sorted = sorted(rows, key=lambda x: x["verification_round"])
    rounds = [r["verification_round"] for r in rows_sorted]
    target_rates = [r["target"]["acceptance_rate"] for r in rows_sorted]
    inter_rates = [r["intermediate"]["acceptance_rate"] for r in rows_sorted]
    target_lengths = [r["target"]["acceptance_length"] for r in rows_sorted]
    inter_lengths = [r["intermediate"]["acceptance_length"] for r in rows_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(rounds, target_rates, marker="o", label="target")
    axes[0].plot(rounds, inter_rates, marker="o", label="intermediate")
    axes[0].set_title(f"Request Trend: Acceptance Rate\n{req_key}")
    axes[0].set_xlabel("verification round")
    axes[0].set_ylabel("acceptance rate")
    axes[0].legend()

    axes[1].plot(rounds, target_lengths, marker="o", label="target")
    axes[1].plot(rounds, inter_lengths, marker="o", label="intermediate")
    axes[1].set_title(f"Request Trend: Acceptance Length\n{req_key}")
    axes[1].set_xlabel("verification round")
    axes[1].set_ylabel("acceptance length")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"request_trend_{req_key}.png", dpi=150)
    plt.close(fig)


def save_global_avg_distribution_plot(
    round_avg_rows: list[dict],
    out_dir: Path,
    bins_rate: int,
    bins_length: int,
) -> None:
    t_rate = [r["target_rate_avg"] for r in round_avg_rows]
    i_rate = [r["inter_rate_avg"] for r in round_avg_rows]
    t_len = [r["target_len_avg"] for r in round_avg_rows]
    i_len = [r["inter_len_avg"] for r in round_avg_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(t_rate, bins=bins_rate, alpha=0.6, label="target")
    axes[0].hist(i_rate, bins=bins_rate, alpha=0.6, label="intermediate")
    axes[0].set_title("Global Distribution of Per-Round Average Acceptance Rate")
    axes[0].set_xlabel("average acceptance rate (across requests at round r)")
    axes[0].set_ylabel("frequency")
    axes[0].legend()

    axes[1].hist(t_len, bins=bins_length, alpha=0.6, label="target")
    axes[1].hist(i_len, bins=bins_length, alpha=0.6, label="intermediate")
    axes[1].set_title("Global Distribution of Per-Round Average Acceptance Length")
    axes[1].set_xlabel("average acceptance length (across requests at round r)")
    axes[1].set_ylabel("frequency")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "global_distribution_per_round_avg.png", dpi=150)
    plt.close(fig)


def save_global_avg_trend_plot(round_avg_rows: list[dict], out_dir: Path) -> None:
    rounds = [r["verification_round"] for r in round_avg_rows]
    t_rate = [r["target_rate_avg"] for r in round_avg_rows]
    i_rate = [r["inter_rate_avg"] for r in round_avg_rows]
    t_len = [r["target_len_avg"] for r in round_avg_rows]
    i_len = [r["inter_len_avg"] for r in round_avg_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(rounds, t_rate, marker="o", label="target")
    axes[0].plot(rounds, i_rate, marker="o", label="intermediate")
    axes[0].set_title("Global Per-Round Average Acceptance Rate Trend")
    axes[0].set_xlabel("verification round")
    axes[0].set_ylabel("average acceptance rate")
    axes[0].legend()

    axes[1].plot(rounds, t_len, marker="o", label="target")
    axes[1].plot(rounds, i_len, marker="o", label="intermediate")
    axes[1].set_title("Global Per-Round Average Acceptance Length Trend")
    axes[1].set_xlabel("verification round")
    axes[1].set_ylabel("average acceptance length")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "global_trend_per_round_avg.png", dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    rows = load_rows(input_path)
    if not rows:
        raise RuntimeError("Input JSONL is empty.")

    # Group by request
    by_request: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_request[_request_key(row)].append(row)

    # 1) Request-wise distribution, 3) Request-wise trend
    request_keys = sorted(by_request.keys())
    limit = min(args.max_requests_plot, len(request_keys))
    for req_key in request_keys[:limit]:
        req_rows = by_request[req_key]
        save_request_distribution_plot(
            req_key=req_key,
            rows=req_rows,
            out_dir=out_dir,
            bins_rate=args.bins_rate,
            bins_length=args.bins_length,
        )
        save_request_trend_plot(req_key=req_key, rows=req_rows, out_dir=out_dir)

    # Aggregate by verification round across requests
    by_round: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_round[int(row["verification_round"])].append(row)

    round_avg_rows: list[dict] = []
    for vr in sorted(by_round.keys()):
        rr = by_round[vr]
        target_rates = [x["target"]["acceptance_rate"] for x in rr]
        inter_rates = [x["intermediate"]["acceptance_rate"] for x in rr]
        target_lens = [x["target"]["acceptance_length"] for x in rr]
        inter_lens = [x["intermediate"]["acceptance_length"] for x in rr]
        round_avg_rows.append(
            {
                "verification_round": vr,
                "num_requests": len(rr),
                "target_rate_avg": sum(target_rates) / len(target_rates),
                "inter_rate_avg": sum(inter_rates) / len(inter_rates),
                "target_len_avg": sum(target_lens) / len(target_lens),
                "inter_len_avg": sum(inter_lens) / len(inter_lens),
            }
        )

    # 2) Global distribution of per-round averages, 4) Global trend of per-round averages
    save_global_avg_distribution_plot(
        round_avg_rows=round_avg_rows,
        out_dir=out_dir,
        bins_rate=args.bins_rate,
        bins_length=args.bins_length,
    )
    save_global_avg_trend_plot(round_avg_rows=round_avg_rows, out_dir=out_dir)

    # Save computed round averages as jsonl for reuse
    with (out_dir / "global_round_averages.jsonl").open("w", encoding="utf-8") as f:
        for row in round_avg_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[done] input={input_path}")
    print(f"[done] request_plots={limit}, output_dir={out_dir}")
    print("[done] wrote:")
    print("  - request_distribution_<request>.png")
    print("  - request_trend_<request>.png")
    print("  - global_distribution_per_round_avg.png")
    print("  - global_trend_per_round_avg.png")
    print("  - global_round_averages.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
