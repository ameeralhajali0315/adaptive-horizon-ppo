
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

fixed_path = RESULTS_DIR / "ppo_fixed_sparse.pt"
adaptive_path = RESULTS_DIR / "ppo_adaptive_sparse.pt"

fixed = torch.load(
    fixed_path,
    map_location="cpu",
    weights_only=True,
)
adaptive = torch.load(
    adaptive_path,
    map_location="cpu",
    weights_only=True,
)

print("Fixed checkpoint keys:", list(fixed.keys()))
print("Adaptive checkpoint keys:", list(adaptive.keys()))

fixed_actor = fixed["actor"]
adaptive_actor = adaptive["actor"]

if fixed_actor.keys() != adaptive_actor.keys():
    raise ValueError("Actor parameter names differ between checkpoints.")

total_parameters = 0
different_parameters = 0
absolute_difference_sum = 0.0
maximum_difference = 0.0

for name in fixed_actor:
    fixed_tensor = fixed_actor[name].float()
    adaptive_tensor = adaptive_actor[name].float()

    difference = (fixed_tensor - adaptive_tensor).abs()

    total_parameters += difference.numel()
    different_parameters += int((difference > 1e-8).sum().item())
    absolute_difference_sum += difference.sum().item()
    maximum_difference = max(
        maximum_difference,
        difference.max().item(),
    )

print("\n=== Actor Weight Comparison ===")
print("Total parameters:", total_parameters)
print("Parameters differing by more than 1e-8:", different_parameters)
print(
    "Fraction differing:",
    f"{different_parameters / total_parameters:.2%}",
)
print(
    "Mean absolute parameter difference:",
    f"{absolute_difference_sum / total_parameters:.8f}",
)
print("Maximum absolute parameter difference:", f"{maximum_difference:.8f}")
