
import csv
from pathlib import Path

results_path = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "sparse7_evaluation_by_model.csv"
)

with results_path.open("r", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

print(
    f"{'Algorithm':<10} {'Seed':>6} {'Success':>10} "
    f"{'Mean steps':>12} {'Successful steps':>19}"
)
print("-" * 62)

for row in rows:
    algorithm = row["algorithm"]
    seed = int(row["training_seed"])
    success = float(row["success_rate"])
    mean_steps = float(row["mean_steps"])

    successful_steps = row["mean_successful_steps"]
    if successful_steps:
        successful_steps = f"{float(successful_steps):.2f}"
    else:
        successful_steps = "N/A"

    print(
        f"{algorithm:<10} {seed:>6} {success:>9.1%} "
        f"{mean_steps:>12.2f} {successful_steps:>19}"
    )
