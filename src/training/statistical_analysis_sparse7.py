
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
INPUT_FILE = RESULTS_DIR / "sparse7_paired_comparison.csv"


def exact_sign_flip_test(differences):
    """Calculate the exact two-sided sign-flip permutation p-value."""
    differences = np.asarray(differences, dtype=float)
    observed = abs(np.mean(differences))
    n = len(differences)

    permuted_means = (
        np.mean(differences * np.asarray(signs))
        for signs in itertools.product([-1, 1], repeat=n)
    )

    extreme_count = sum(
        abs(value) >= observed - 1e-12
        for value in permuted_means
    )

    return extreme_count / (2 ** n)


def analyze_metric(name, differences, unit=""):
    differences = np.asarray(differences, dtype=float)
    n = len(differences)

    mean_diff = np.mean(differences)
    sd_diff = np.std(differences, ddof=1)

    standard_error = sd_diff / np.sqrt(n)
    critical_value = t.ppf(0.975, df=n - 1)

    ci_low = mean_diff - critical_value * standard_error
    ci_high = mean_diff + critical_value * standard_error

    p_value = exact_sign_flip_test(differences)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Number of paired seeds: {n}")
    print(f"Mean paired difference: {mean_diff:+.4f}{unit}")
    print(f"Sample SD of differences: {sd_diff:.4f}{unit}")
    print(
        f"95% t confidence interval: "
        f"[{ci_low:+.4f}, {ci_high:+.4f}]{unit}"
    )
    print(f"Exact two-sided sign-flip p-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Permutation-test result: statistically significant at 0.05.")
    else:
        print("Permutation-test result: not statistically significant at 0.05.")

    return {
        "metric": name,
        "n_pairs": n,
        "mean_difference": mean_diff,
        "sample_sd": sd_diff,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "sign_flip_p_value": p_value,
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Results file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print("Columns found in paired-comparison CSV:")
    print(df.columns.tolist())
    print(f"\nRows found: {len(df)}")

    print(
        "\nBefore continuing, confirm that the CSV contains "
        "exactly 15 paired training seeds."
    )

    # The column names are intentionally checked before analysis
    # so we do not silently analyze the wrong data.
    required_columns = {
    "success_rate_difference",
    "mean_steps_difference",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Expected columns were not found. "
            "Use the printed column names to adjust the script. "
            f"Required: {sorted(required_columns)}"
        )

    if len(df) != 15:
        raise ValueError(
            f"Expected 15 paired seeds, but found {len(df)}. "
            "Check the CSV before interpreting results."
        )

    success_results = analyze_metric(
        "Success-rate difference (Adaptive - Fixed)",
        df["success_rate_difference"] * 100,
        unit=" percentage points",
    )

    step_results = analyze_metric(
        "Episode-step difference (Adaptive - Fixed)",
        df["mean_steps_difference"],
        unit=" steps",
    )

    output = pd.DataFrame([success_results, step_results])
    output_path = RESULTS_DIR / "sparse7_statistical_analysis.csv"
    output.to_csv(output_path, index=False)

    print(f"\nStatistical summary saved to: {output_path}")
    print(
        "\nInterpretation note: training seeds are the independent "
        "experimental units. Treat the results cautiously and report "
        "both confidence intervals and permutation-test p-values."
    )


if __name__ == "__main__":
    main()
