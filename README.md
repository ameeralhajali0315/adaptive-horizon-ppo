# Adaptive-Horizon PPO with Dynamic Discount and GAE Scheduling

An experimental implementation comparing **fixed-parameter Proximal Policy Optimization (PPO)** with an adaptive variant that schedules the discount factor (\(\gamma\)) and Generalized Advantage Estimation (GAE) parameter (\(\lambda\)) during training.

> **Project status:** Core experiments and paired statistical analysis have been completed for a custom 7×7 SparseGridWorld environment. The reported results below summarize the current saved evaluation and analysis outputs.

## Research question

Can gradually changing PPO's discount factor and GAE parameter during training improve performance in a sparse-reward grid-world task compared with keeping those parameters fixed?

The project compares two approaches:

- **Fixed PPO:** Uses constant \(\gamma\) and \(\lambda\) values throughout training.
- **Adaptive PPO:** Linearly schedules \(\gamma\) from 0.90 to 0.99 and \(\lambda\) from 0.80 to 0.95.

The hypothesis is that changing these parameters over the training horizon may alter the balance between short-term and long-term rewards and the bias–variance trade-off in advantage estimates. Whether this helps is an empirical question; the results below apply to the tested environment and setup.

## Method

### Algorithms

Both approaches use PPO. The adaptive variant changes two training parameters over the course of training:

| Parameter | Adaptive schedule |
|---|---:|
| Discount factor, \(\gamma\) | 0.90 → 0.99 |
| GAE parameter, \(\lambda\) | 0.80 → 0.95 |
| Schedule | Linear |

The fixed baseline uses constant discount and GAE values as specified in the corresponding configuration files.

### Evaluation design

- **Environment:** Custom `SparseGridWorld-v1` 7×7 grid world
- **Episode limit:** 40 steps
- **Training seeds:** 15 paired seeds — `42`, `123`, `456`, `789`, `1024`, and `2026`–`2035`
- **Evaluation episodes:** 100 per trained model
- **Evaluation seeds:** Shared episode seeds `10000`–`10099` across fixed/adaptive models in each seed pair
- **Action selection during evaluation:** Deterministic argmax
- **Total evaluation rows:** 3,000 (30 models × 100 episodes)

Using the same evaluation seeds for each fixed/adaptive pair makes the comparison paired and helps control for variation caused by different evaluation episodes.

## Results

The completed paired analysis reports the following differences, calculated as **adaptive PPO minus fixed PPO**.

| Metric | Mean paired difference | 95% confidence interval | Exact two-sided sign-flip p-value |
|---|---:|---:|---:|
| Success rate | **+7.13 percentage points** | +2.73 to +11.53 percentage points | 0.00012207 |
| Episode steps | **−2.541 steps** | −4.082 to −1.000 steps | 0.00012207 |

For success rate, a positive difference favors adaptive PPO. For episode steps, a negative difference means the adaptive agent used fewer steps on average under the evaluation metric.

The statistical analysis used 15 paired seed-level differences. The confidence intervals are 95% paired t-intervals, and the p-values come from exact two-sided sign-flip tests.

### Interpretation and limitations

In this experiment, adaptive PPO achieved a higher success rate and required fewer episode steps on average than fixed PPO. The paired statistical results provide evidence of a difference under this specific experimental setup.

These findings do **not** establish that adaptive scheduling will improve PPO in every environment. The result is limited to the implemented grid world, tested configurations, seeds, and evaluation procedure. Further work should test additional environments, conduct ablations that schedule only \(\gamma\) or only \(\lambda\), and report computational cost and sensitivity to schedule endpoints.

## Repository structure

```text
adaptive-horizon-ppo/
├── configs/                 # Experiment and smoke-test configurations
├── src/
│   ├── algorithms/           # PPO, GAE, rollout buffer, schedule manager
│   ├── environments/         # Grid-world and test environment code
│   ├── models/               # Actor–critic model
│   ├── training/             # Training, evaluation, analysis, plotting scripts
│   ├── agents/               # Agent utilities
│   └── utils/                # Shared utilities
└── .gitignore
```

Generated experiment artifacts and the local virtual environment are excluded from version control. The `REPORTS/` directory is intentionally not tracked in this repository.

## Getting started

### Requirements

- Python 3.12 is the development version used for this project.
- PyTorch
- Gymnasium
- NumPy
- Matplotlib and SciPy may be needed for plotting and statistical analysis.
- Pytest may be needed to run the test suite.

The repository does not yet include a pinned dependency file. The package list above is a starting point; verify the imports in the relevant scripts and your working environment before attempting to reproduce an experiment.

### Set up a virtual environment

From the repository root, create and activate a virtual environment:

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the required packages in that environment after confirming the project's dependency versions. A pinned `requirements.txt` should be added to make setup and reproduction more reliable.

### Configuration files

Experiment configurations are stored in `configs/`. Filenames distinguish fixed and adaptive runs, and the `sparse7` configurations correspond to the 7×7 grid-world experiments. Review the selected JSON configuration before running any training or evaluation script.

Training and evaluation entry points are located in `src/training/`, including:

- `train_ppo.py`
- `train_adaptive_ppo.py`
- `evaluate_sparse7.py`
- `analyze_sparse7.py`
- `statistical_analysis_sparse7.py`
- `plot_sparse7_results.py`

Check each script's command-line arguments and expected output paths before running it. Exact reproduction commands and pinned dependencies are future documentation improvements.

## Testing

Test modules are currently located alongside the implementation code in `src/`. The project has passed **19 tests** in the current development environment. Because the tests are not all organized under a conventional `tests/` directory, review their imports and invocation requirements before assuming a generic test command will discover every test.

A future improvement is to standardize test discovery and add a documented, repeatable test command.

## Reproducibility notes

- The saved experiment configurations define the training parameters.
- Fixed and adaptive models were evaluated on matching evaluation-seed sets.
- Paired differences were independently recomputed and checked against the saved summary.
- Evaluation artifacts are not included in this repository because generated `results/` files are ignored. To independently verify the numerical findings, the result files and exact run commands will need to be archived or regenerated.

## Future work

- Add a pinned `requirements.txt` or equivalent environment specification.
- Document exact training, evaluation, and analysis commands.
- Publish compact result tables and plots with the code used to generate them.
- Move tests into a conventional `tests/` directory and standardize test discovery.
- Evaluate on additional environments and compare separate \(\gamma\)-only and \(\lambda\)-only schedules.
- Investigate schedule shapes and endpoint sensitivity.

## Author

**Ameer Alsabagh Alhajali**

Repository: [adaptive-horizon-ppo](https://github.com/ameeralhajali0315/adaptive-horizon-ppo)

---

This project is an experimental capstone implementation and should be interpreted in the context of its stated environment, configurations, and evaluation protocol.
