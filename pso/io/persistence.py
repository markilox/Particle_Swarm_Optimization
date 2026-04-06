
import csv
import json
import os  # used to get the number of CPU cores
import platform
import subprocess  # used to run git commands and get the current commit hash
from datetime import datetime
from pathlib import Path  # cross-platform file path handling

from pso.core.types import OptimizationResult, PSOConfig


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _hardware_info() -> dict:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def _config_to_dict(config: PSOConfig) -> dict:
    return {
        "dimension": config.dimension,
        "swarm_size": config.swarm_size,
        "inertia_weight": config.inertia_weight,
        "cognitive_weight": config.cognitive_weight,
        "social_weight": config.social_weight,
        "seed": config.seed,
        "stop": {
            "max_iterations": config.stop.max_iterations,
            "tolerance": config.stop.tolerance,
            "stagnation_iterations": config.stop.stagnation_iterations,
            "min_delta": config.stop.min_delta,
        },
    }


def save_result(
    result: OptimizationResult,
    config: PSOConfig,
    objective_name: str,
    evaluator_name: str,
    output_dir: str | Path,
    run_id: str | None = None,
) -> Path:
    """Save result to output_dir/run_id/.

    Creates two files:
      - metadata.json: config, environment info, and final metrics
      - history.csv:   best_fitness + times per iteration

    Returns the path to the run directory.
    """
    output_dir = Path(output_dir)
    if run_id is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_id = f"{objective_name}_{evaluator_name}_d{config.dimension}_{ts}"

    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "objective": objective_name,
        "evaluator": evaluator_name,
        "git_commit": _git_commit_hash(),
        "hardware": _hardware_info(),
        "config": _config_to_dict(config),
        "summary": {
            "best_value": result.best_value,
            "best_position": result.best_position.tolist(),
            "total_time_s": result.total_time_s,
            "total_eval_time_s": result.total_eval_time_s,
            "total_update_time_s": result.total_update_time_s,
            "converged_iteration": result.converged_iteration,
            "total_iterations": len(result.history),
        },
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "best_fitness", "eval_time_s", "update_time_s"])
        for m in result.history:
            writer.writerow([m.iteration, m.best_fitness, m.eval_time_s, m.update_time_s])

    return run_dir


def load_metadata(run_dir: str | Path) -> dict:
    with open(Path(run_dir) / "metadata.json") as f:
        return json.load(f)


def load_history(run_dir: str | Path) -> list[dict]:
    rows = []
    with open(Path(run_dir) / "history.csv") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "iteration": int(row["iteration"]),
                    "best_fitness": float(row["best_fitness"]),
                    "eval_time_s": float(row["eval_time_s"]),
                    "update_time_s": float(row["update_time_s"]),
                }
            )
    return rows


def load_all_results(output_dir: str | Path) -> list[dict]:
    """Load all runs under output_dir.

    Returns a list of dicts with keys 'metadata', 'history' and 'run_dir'.
    Ignores subdirectories without metadata.json.
    """
    output_dir = Path(output_dir)
    results = []
    for run_dir in sorted(output_dir.iterdir()):
        if run_dir.is_dir() and (run_dir / "metadata.json").exists():
            results.append(
                {
                    "metadata": load_metadata(run_dir),
                    "history": load_history(run_dir),
                    "run_dir": str(run_dir),
                }
            )
    return results
