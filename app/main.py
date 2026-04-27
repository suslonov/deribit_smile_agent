"""CLI entry point for the Deribit Smile Research Agent.

If the first user argument is not a known subcommand, ``run-quick`` is inserted
into ``sys.argv`` (after the program name) so parsing matches an explicit
``run-quick`` invocation.

Commands:
    run-quick       30-day CPU research run.
    run-train       Full walk-forward training study.
    run-test        Final test on held-out set using best accepted sandbox.
    run-live-paper  Live paper trading via Deribit WebSocket.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from app.data.loader import load_date_range
from app.llm.artifact_log import build_artifact, save_artifact
from app.llm.critic import evaluate_candidate
from app.llm.prompts import render_rewrite_prompt
from app.llm.proposer import LLMProposer
from app.opt.search import walk_forward_search
from app.sim.metrics import compute_metrics
from app.sim.pipeline import run_pipeline
from app.sim.pricing import ExecConfig, FeeConfig
from app.sim.simulator import run_simulation
from app.split.time_split import get_test_data, make_splits
from app.utils.clock import utc_now_str
from app.utils.hashing import hash_dict
from app.utils.io import load_yaml
from sandbox.runner import SandboxRunner

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def _build_pricing(config: dict) -> tuple[ExecConfig, FeeConfig]:
    exec_cfg_raw = config.get("execution", {})
    exec_cfg = ExecConfig(
        model=exec_cfg_raw.get("model", "mid"),
        slippage_bps=float(exec_cfg_raw.get("slippage_bps", 0.0)),
    )
    fee_cfg_raw = config.get("fees", {})
    fee_cfg = FeeConfig(
        pct_of_underlying=float(fee_cfg_raw.get("pct_of_underlying", 0.0003)),
        max_pct_of_option=float(fee_cfg_raw.get("max_pct_of_option", 0.125)),
    )
    return exec_cfg, fee_cfg


def _build_runner(config: dict) -> SandboxRunner:
    sb_cfg = config.get("sandbox", {})
    return SandboxRunner(
        calculator_path="sandbox/calculator.py",
        timeout_seconds=float(sb_cfg.get("timeout_seconds", 60)),
        allowed_imports=frozenset(sb_cfg.get("allowed_imports", [])),
        verify_determinism=bool(sb_cfg.get("verify_determinism", False)),
    )


def _print_metrics(metrics: dict) -> None:
    summary = metrics.get("summary", {})
    table = Table(title="Summary Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for k, v in summary.items():
        if v is None:
            val_str = "N/A"
        elif isinstance(v, float):
            val_str = f"{v:.4f}"
        else:
            val_str = str(v)
        table.add_row(k, val_str)
    console.print(table)


def run_quick(args) -> None:
    config_path = args.config
    llm = args.llm
    dry_run = args.dry_run
    """Quick 30-day research run on CPU."""
    config = load_yaml(config_path)
    config_hash = hash_dict(config)
    if not os.path.exists(config["data"]["path"]):
        config["data"]["path"] = config["data"]["path"].replace("mnt", "media")
        config["data"]["cache_dir"] = config["data"]["cache_dir"].replace("mnt", "media")
    run_id = f"quick_{utc_now_str()}"

    data_cfg = config["data"]
    quick_start = data_cfg.get("quick_start", "2025-01-01")
    quick_days = int(data_cfg.get("quick_days", 30))
    chunk_days = int(data_cfg.get("chunk_days", 1))

    start = date.fromisoformat(quick_start)
    end = start + timedelta(days=quick_days - 1)

    console.print(f"[bold]Run ID:[/bold] {run_id}")
    console.print(f"[bold]Period:[/bold] {start} → {end} ({quick_days} days, {chunk_days}-day chunks)")

    runner = _build_runner(config)
    runner.reload()
    exec_cfg, fee_cfg = _build_pricing(config)
    horizons = config.get("horizons", [1, 3])

    console.print("Running chunked pipeline (load → calc → simulate)...")
    errors: list[str] = []
    try:
        signals, results, metrics = run_pipeline(
            data_root=data_cfg["path"],
            start=start,
            end=end,
            config=config,
            sandbox_runner=runner,
            exec_cfg=exec_cfg,
            fee_cfg=fee_cfg,
            horizons=horizons,
            assets=data_cfg.get("assets"),
            cache_dir=data_cfg.get("cache_dir"),
            max_workers=int(data_cfg.get("max_workers", 1)),
            chunk_days=chunk_days,
        )
    except Exception as exc:
        console.print(f"[red]Pipeline error:[/red] {exc}")
        errors.append(str(exc))
        signals, results, metrics = pd.DataFrame(), pd.DataFrame(), {}

    console.print(f"Generated {len(signals):,} signals, {len(results):,} trade results")
    _print_metrics(metrics)

    # Load a sample for the artifact (first day only to avoid huge artifact)
    sample_df = load_date_range(
        root=data_cfg["path"],
        start=start,
        end=start,
        assets=data_cfg.get("assets"),
        cache_dir=data_cfg.get("cache_dir"),
        max_workers=1,
    )
    artifact = build_artifact(
        run_id=run_id,
        config=config,
        config_hash=config_hash,
        sandbox_path="sandbox/calculator.py",
        metrics=metrics,
        fold_metrics=[],
        results_df=results,
        options_df=sample_df,
        errors=errors,
        stage="quick",
    )
    out_path = save_artifact(artifact, config["output"]["artifacts_dir"])
    console.print(f"[green]Artifact saved:[/green] {out_path}")

    if llm:
        _run_llm_loop(artifact, config, runner, sample_df, exec_cfg, fee_cfg,
                      horizons, dry_run, run_id)


def run_train(args) -> None:
    """Full walk-forward training study."""
    config_path = args.config
    llm = args.llm
    dry_run = args.dry_run

    config = load_yaml(config_path)
    config_hash = hash_dict(config)
    if not os.path.exists(config["data"]["path"]):
        config["data"]["path"] = config["data"]["path"].replace("mnt", "media")
        config["data"]["cache_dir"] = config["data"]["cache_dir"].replace("mnt", "media")
    run_id = f"train_{utc_now_str()}"

    data_cfg = config["data"]
    start_str = data_cfg.get("start_date")
    end_str = data_cfg.get("end_date")

    start = date.fromisoformat(start_str) if start_str else date(2023, 1, 1)
    end = date.fromisoformat(end_str) if end_str else date.today()

    console.print(f"[bold]Run ID:[/bold] {run_id}")
    console.print(f"[bold]Loading data:[/bold] {start} → {end}")

    options_df = load_date_range(
        root=data_cfg["path"],
        start=start,
        end=end,
        assets=data_cfg.get("assets"),
        cache_dir=data_cfg.get("cache_dir"),
        max_workers=int(data_cfg.get("max_workers", 8)),
    )
    console.print(f"Loaded {len(options_df):,} rows")

    split_cfg = config["splits"]
    folds, test_cutoff = make_splits(
        df=options_df,
        test_days=int(split_cfg["test_days"]),
        train_window_days=int(split_cfg["train_window_days"]),
        val_window_days=int(split_cfg["val_window_days"]),
        gap_days=int(split_cfg["gap_days"]),
        step_days=int(split_cfg["step_days"]),
    )
    console.print(f"Created {len(folds)} folds, test cutoff: {test_cutoff.date()}")

    train_df, _ = get_test_data(options_df, test_cutoff)
    runner = _build_runner(config)
    runner.reload()
    exec_cfg, fee_cfg = _build_pricing(config)
    horizons = config.get("horizons", [1, 3])

    console.print("Running walk-forward search...")
    fold_results = walk_forward_search(
        folds=folds,
        options_df=train_df,
        config=config,
        sandbox_runner=runner,
        exec_cfg=exec_cfg,
        fee_cfg=fee_cfg,
        horizons=horizons,
    )

    fold_metrics = [
        {
            "fold": r.fold,
            "train": r.train_metrics.get("summary", {}),
            "val": r.val_metrics.get("summary", {}),
            "config": r.best_config.get("calculator", {}),
        }
        for r in fold_results
    ]

    # Aggregate val metrics for reporting
    agg_signals = runner.run(train_df, config.get("calculator", {}))
    agg_results = run_simulation(agg_signals, train_df, exec_cfg, fee_cfg, horizons)
    metrics = compute_metrics(agg_results)
    _print_metrics(metrics)

    errors: list[str] = []
    artifact = build_artifact(
        run_id=run_id,
        config=config,
        config_hash=config_hash,
        sandbox_path="sandbox/calculator.py",
        metrics=metrics,
        fold_metrics=fold_metrics,
        results_df=agg_results,
        options_df=train_df,
        errors=errors,
        stage="train",
    )
    out_path = save_artifact(artifact, config["output"]["artifacts_dir"])
    console.print(f"[green]Artifact saved:[/green] {out_path}")

    if llm:
        _run_llm_loop(artifact, config, runner, train_df, exec_cfg, fee_cfg,
                      horizons, dry_run, run_id)


def run_test(args) -> None:
    """Run the final test on the held-out test set (run once, never iterate)."""
    config_path = args.config
    config = load_yaml(config_path)
    config_hash = hash_dict(config)
    if not os.path.exists(config["data"]["path"]):
        config["data"]["path"] = config["data"]["path"].replace("mnt", "media")
        config["data"]["cache_dir"] = config["data"]["cache_dir"].replace("mnt", "media")
    run_id = f"test_{utc_now_str()}"

    data_cfg = config["data"]
    start_str = data_cfg.get("start_date")
    end_str = data_cfg.get("end_date")

    start = date.fromisoformat(start_str) if start_str else date(2023, 1, 1)
    end = date.fromisoformat(end_str) if end_str else date.today()

    console.print("[bold]FINAL TEST RUN[/bold] — results will NOT be used for training")

    options_df = load_date_range(
        root=data_cfg["path"],
        start=start,
        end=end,
        assets=data_cfg.get("assets"),
        cache_dir=data_cfg.get("cache_dir"),
        max_workers=int(data_cfg.get("max_workers", 8)),
    )

    split_cfg = config["splits"]
    _, test_cutoff = make_splits(
        df=options_df,
        test_days=int(split_cfg["test_days"]),
        train_window_days=int(split_cfg["train_window_days"]),
        val_window_days=int(split_cfg["val_window_days"]),
        gap_days=int(split_cfg["gap_days"]),
        step_days=int(split_cfg["step_days"]),
    )
    _, test_df = get_test_data(options_df, test_cutoff)
    console.print(f"Test set: {test_cutoff.date()} → {test_df['timestamp'].max().date()}, {len(test_df):,} rows")

    runner = _build_runner(config)
    runner.reload()
    exec_cfg, fee_cfg = _build_pricing(config)
    horizons = config.get("horizons", [1, 3])

    calc_cfg = config.get("calculator", {})
    signals = runner.run(test_df, calc_cfg)
    results = run_simulation(signals, test_df, exec_cfg, fee_cfg, horizons)
    metrics = compute_metrics(results)
    _print_metrics(metrics)

    artifact = build_artifact(
        run_id=run_id,
        config=config,
        config_hash=config_hash,
        sandbox_path="sandbox/calculator.py",
        metrics=metrics,
        fold_metrics=[],
        results_df=results,
        options_df=test_df,
        errors=[],
        stage="test",
    )
    out_path = save_artifact(artifact, config["output"]["artifacts_dir"])
    console.print(f"[green]Final test artifact saved:[/green] {out_path}")


def run_live_paper(args) -> None:
    """Start live paper trading via Deribit WebSocket."""
    config_path = args.config
    config = load_yaml(config_path)
    if not os.path.exists(config["data"]["path"]):
        config["data"]["path"] = config["data"]["path"].replace("mnt", "media")
        config["data"]["cache_dir"] = config["data"]["cache_dir"].replace("mnt", "media")

    runner = _build_runner(config)
    runner.reload()

    from app.live.deribit_ws import DeribitWSClient
    from app.live.paper_router import PaperRouter

    router = PaperRouter(config, runner)
    client = DeribitWSClient(config, on_snapshot=router.on_snapshot)

    console.print("[bold]Starting live paper trading...[/bold]")
    console.print(f"Trades log: {config.get('paper', {}).get('trades_log')}")

    asyncio.run(client.run())


# ---------------------------------------------------------------------------
# LLM improvement loop
# ---------------------------------------------------------------------------

def _run_llm_loop(
    artifact: dict,
    config: dict,
    runner: SandboxRunner,
    options_df: pd.DataFrame,
    exec_cfg: ExecConfig,
    fee_cfg: FeeConfig,
    horizons: list[int],
    dry_run: bool,
    run_id: str,
) -> None:
    """Send artifact to LLM, receive new calculator, validate, and optionally accept."""
    proposer = LLMProposer(config)
    prompt = render_rewrite_prompt(artifact)

    console.print("[bold]Calling LLM proposer...[/bold]")
    new_source, new_calc_cfg = proposer.propose(prompt, dry_run=dry_run)

    if new_source is None:
        console.print("[yellow]LLM returned no new source (dry run or error)[/yellow]")
        return

    console.print("Evaluating candidate...")
    critic_result = evaluate_candidate(
        candidate_source=new_source,
        current_metrics=artifact["summary_metrics"],
        options_slice=options_df.head(5000),
        config=config,
        tests_dir="tests",
    )

    if not critic_result.accepted:
        console.print(f"[red]Candidate rejected:[/red] {critic_result.reason}")
        return

    # Write candidate to disk for inspection
    candidate_path = Path(config["output"]["artifacts_dir"]) / f"candidate_{run_id}.py"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(new_source)
    console.print(f"[green]Candidate passed static checks:[/green] {candidate_path}")
    console.print("[yellow]Metric gate: run manually or extend LLM loop to auto-accept[/yellow]")


_KNOWN_SUBCOMMANDS: frozenset[str] = frozenset(
    ("run-quick", "run-train", "run-test", "run-live-paper")
)


def _inject_default_run_quick(default_command) -> None:
    argv = sys.argv
    if len(argv) <= 1:
        argv.insert(1, default_command)
        return
    first = argv[1]
    if first in _KNOWN_SUBCOMMANDS:
        return
    if first in ("-h", "--help"):
        return
    argv.insert(1, default_command)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="deribit-smile-agent",
        description="Deribit Volatility Smile Research Agent",
    )
    sub = parser.add_subparsers(
        dest="command",
        required=False,
        metavar="COMMAND",
        help="Action to run (omitted: run-quick).",
    )
    p_quick = sub.add_parser("run-quick", help="Quick 30-day research run on CPU.")
    p_quick.add_argument(
        "-c", "--config", default="configs/quick.yaml", help="Path to YAML config"
    )
    p_quick.add_argument(
        "--llm", action="store_true", help="Run LLM proposer after simulation"
    )
    p_quick.add_argument("--dry-run", action="store_true", help="Skip LLM call")
    p_train = sub.add_parser("run-train", help="Full walk-forward training study.")
    p_train.add_argument(
        "-c", "--config", default="configs/train_full.yaml", help="Path to YAML config"
    )
    p_train.add_argument("--llm", action="store_true")
    p_train.add_argument("--dry-run", action="store_true")
    p_test = sub.add_parser(
        "run-test", help="Final test on held-out set using accepted sandbox"
    )
    p_test.add_argument(
        "-c", "--config", default="configs/train_full.yaml", help="Path to YAML config"
    )
    p_live = sub.add_parser(
        "run-live-paper", help="Live paper trading via Deribit WebSocket"
    )
    p_live.add_argument(
        "-c", "--config", default="configs/live_paper.yaml", help="Path to YAML config"
    )
    _inject_default_run_quick("run-quick")
    args = parser.parse_args()
    if args.command == "run-quick":
        run_quick(args)
    elif args.command == "run-train":
        run_train(args)
    elif args.command == "run-test":
        run_test(args)
    elif args.command == "run-live-paper":
        run_live_paper(args)
    exit(0)


if __name__ == "__main__":
    main()
