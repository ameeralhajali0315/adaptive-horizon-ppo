import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"


def read_csv(filename: str) -> list[dict]:
    path = RESULTS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required results file not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def to_number(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (ValueError, TypeError):
        return value


def numeric_rows(rows: list[dict]) -> list[dict]:
    return [
        {key: to_number(value) for key, value in row.items()}
        for row in rows
    ]


def get_research_results() -> dict:
    models = numeric_rows(read_csv("sparse7_evaluation_by_model.csv"))
    comparisons = numeric_rows(read_csv("sparse7_paired_comparison.csv"))
    statistics = numeric_rows(read_csv("sparse7_statistical_analysis.csv"))

    fixed_models = [row for row in models if row["algorithm"] == "fixed"]
    adaptive_models = [row for row in models if row["algorithm"] == "adaptive"]

    if not fixed_models or not adaptive_models:
        raise ValueError("Expected fixed and adaptive model evaluations.")

    return {
        "environment": "SparseGridWorld-v1",
        "training_seed_pairs": len(set(
            row["training_seed"] for row in comparisons
        )),
        "evaluation_episodes": sum(row["episodes"] for row in models),
        "models": models,
        "comparisons": comparisons,
        "statistics": statistics,
        "summary": {
            "fixed_success_rate": sum(
                row["success_rate"] for row in fixed_models
            ) / len(fixed_models),
            "adaptive_success_rate": sum(
                row["success_rate"] for row in adaptive_models
            ) / len(adaptive_models),
            "fixed_mean_steps": sum(
                row["mean_steps"] for row in fixed_models
            ) / len(fixed_models),
            "adaptive_mean_steps": sum(
                row["mean_steps"] for row in adaptive_models
            ) / len(adaptive_models),
        },
    }
