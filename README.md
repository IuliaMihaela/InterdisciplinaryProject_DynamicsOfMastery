Notebooks 1-5 focus on the original $333:10$ case study and use the separated folder `case_study_333_10players` so the exploratory phase stays distinct from the later generalized pipeline runs.


# The Dynamics of Mastery

## Non-Linear Dynamics and Statistical Physics of Human Skill Acquisition

This repository contains the code and analysis for an interdisciplinary project investigating the dynamics of human skill acquisition using competitive Rubik's Cube performance data.

The project uses historical competition results from the World Cube Association (WCA) to study how solving performance evolves over time and whether the observed learning dynamics exhibit:

1. **Power-law or exponential learning behavior**
2. **Temporal dependence in performance fluctuations**
3. **A reduction in performance variability as skill improves**

The analysis is designed to be reproducible and extensible across multiple WCA events and different numbers of players.

---

## Research Questions

### RQ1 — Learning-curve model

Does human skill acquisition in speedcubing follow a power-law or exponential learning curve?

The two models considered are:

### Power-law model

$$
y(x) = a x^{-b} + c
$$

where:

* $x$ is the sequential solve index,
* $y(x)$ is the average solve time,
* $a$ is a scale parameter,
* $b$ controls the rate of improvement,
* $c$ is the asymptotic performance floor.

### Exponential model

$$
y(x) = a e^{-bx} + c
$$

The models are fitted separately for each player and compared using:

* Residual Sum of Squares (RSS)
* $R^2$
* Akaike Information Criterion (AIC)
* Bayesian Information Criterion (BIC)

---

### RQ2 — Temporal structure of residuals
After removing the fitted learning curve, do the remaining performance fluctuations behave like independent noise, or do they retain temporal structure?

For each observation, the residual is defined as:

$$
e_t = y_t - \hat{y}_t
$$

where:

* $y_t$ is the observed performance,
* $\hat{y}_t$ is the model-predicted performance,
* $e_t$ is the residual.

Detrended Fluctuation Analysis (DFA) is used to estimate the Hurst exponent $H$.

The interpretation is approximately:

* $H < 0.45$: anti-persistent behavior
* $0.45 \leq H \leq 0.55$: approximately uncorrelated behavior
* $H > 0.55$: persistent behavior

The DFA scaling relationship is:

$$
F(n) \propto n^H
$$

where $F(n)$ is the fluctuation magnitude at scale $n$.

---

### RQ3 — Variability during skill acquisition

Does performance variability decrease as players approach mastery?

For each player, the observed trajectory is divided into an early phase and a late phase, each containing approximately one quarter of the available observations. The standard-deviation ratio is calculated as:

$$
R_{\sigma} =
\frac{\sigma_{\mathrm{late}}}
{\sigma_{\mathrm{early}}}
$$

where:

* $\sigma_{\mathrm{early}}$ is the standard deviation of early-career performance,
* $\sigma_{\mathrm{late}}$ is the standard deviation of late-career performance.

A value of:

* $R_{\sigma} < 1$ indicates reduced variability,
* $R_{\sigma} = 1$ indicates no change,
* $R_{\sigma} > 1$ indicates increased variability.

---

# Repository Structure

```text
.
├── data/
│   ├── raw/
│   │   └── WCA export files
│   │
│   ├── processed/
│   │    ├── case_study_333_10players/
│   │   ├── 333_10players/
│   │   ├── 333_25players/
│   │   ├── 333_50players/
│   │   ├── 222_25players/
│   │   ├── 444_25players/
│   │   ├── 555_25players/
│   │   ├── pyram_25players/
│   │   └── skewb_25players/
│   │
│   └── results/
│       ├── case_study_333_10players/
│       ├── 333_10players/
│       ├── 333_25players/
│       ├── 333_50players/
│       ├── 222_25players/
│       ├── 444_25players/
│       ├── 555_25players/
│       ├── pyram_25players/
│       ├── skewb_25players/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_learning_models.ipynb
│   ├── 03_dfa_analysis.ipynb
│   ├── 04_distribution_analysis.ipynb
│   ├── 05_results_summary.ipynb
│   └── 06_cross_experiment_summary.ipynb
│
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── learning_models.py
│   ├── dfa_analysis.py
│   └── visualization.py
│
├── run_experiments.py
├── requirements.txt
└── README.md
```

---

The project is organized into two analytical stages. Notebooks 1–5 focus on the original $333:10$ case study and use the separated folder `case_study_333_10players`, keeping the exploratory analysis distinct from the later generalized pipeline runs. Notebook 6 summarizes the results of the generalized multi-event experiments.

# Data

The analysis uses WCA competition data.

The raw data should be placed in:

```text
data/raw/
```

The raw data directory is excluded from git. The WCA export files are downloaded from 
https://www.worldcubeassociation.org/export/results


---

# Analysis Pipeline

The analysis is divided into several notebooks.

## Notebook 1 — Data Exploration

```text
01_data_exploration.ipynb
```

This notebook:

* loads the WCA data,
* filters the selected event,
* explores the available players,
* examines the number of results per player,
* selects the initial player cohort,
* constructs player learning trajectories.

This notebook was initially used for the focused $333:10$ case study and saves its outputs in `data/processed/case_study_333_10players/`.

---

## Notebook 2 — Learning-Curve Models

```text
02_learning_models.ipynb
```

This notebook:

1. Loads player trajectories.
2. Fits a power-law model to each player.
3. Fits an exponential model to each player.
4. Computes model-quality metrics.
5. Compares the two models using AIC and BIC.
6. Stores the fitted parameters.
7. Calculates residuals.

The main outputs are:

```text
model_comparison.csv
fit_parameters.csv
residuals.csv
```

These files are written to `data/results/case_study_333_10players/` for the case-study run, and to `data/results/333_10players/` when the generalized pipeline is rerun.

---

## Notebook 3 — Residual DFA Analysis

```text
03_dfa_analysis.ipynb
```

This notebook:

1. Loads the model residuals.
2. Selects the best-fitting model for each player.
3. Applies Detrended Fluctuation Analysis.
4. Estimates the Hurst exponent $H$.
5. Calculates the DFA scaling fit quality.
6. Classifies the residual behavior as:

   * anti-persistent,
   * uncorrelated,
   * persistent.

The main output is:

```text
dfa_results.csv
```

The notebook reads `residuals.csv` and `model_comparison.csv` from the case-study results folder and writes `dfa_results.csv` there.

---

## Notebook 4 — Distribution Analysis

```text
04_distribution_analysis.ipynb
```

This notebook investigates whether performance variability changes during learning.

For each player, the trajectory is divided into:

* an early phase,
* a late phase.

The notebook calculates:

* mean performance,
* standard deviation,
* skewness,
* kurtosis,
* late-to-early standard-deviation ratio.

The main outputs are:

```text
distribution_metrics.csv
variability_comparison.csv
distribution_summary.csv
```

Statistical tests are also used to compare early and late variability.

The main outputs are written to `data/results/case_study_333_10players/`.

---

## Notebook 5 — Results Summary

```text
05_results_summary.ipynb
```

This notebook summarizes the detailed analysis for the focused $333:10$ case study.

It combines the results of:

* model comparison,
* residual DFA analysis,
* distribution analysis.

The goal is to interpret the results in relation to the three research questions. It does not create new core CSV files; instead, it turns the case-study outputs into the final narrative and figures.

---

## Notebook 6 — Cross-Experiment Summary

```text
06_cross_experiment_summary.ipynb
```

This notebook compares multiple experiments across:

* different WCA events,
* different numbers of players.

The main input is:

```text
data/results/experiment_summary.csv
```

The notebook examines:

* power-law dominance,
* mean Hurst exponent,
* persistent-share,
* variability reduction,
* sample-size stability.

The notebook also produces cross-experiment visualizations, including:

* power-law win-rate plots,
* Hurst-exponent comparisons,
* variability-ratio comparisons,
* a cross-experiment heatmap.

Its input is the aggregated experiment table produced by the generalized pipeline, so it reflects the rerun experiments rather than the case-study folder.

---

# Reproducible Experiment Pipeline

The complete multi-experiment pipeline can be executed using:

```bash
python run_experiments.py
```

If no experiments are specified, the script uses the default experiment configuration defined in the script.

Experiments can also be specified explicitly:

```bash
python run_experiments.py --experiments \
    333:10 \
    333:25 \
    333:50 \
    222:25 \
    444:25 \
    555:25 \
    pyram:25 \
    skewb:25
```

The format is:

```text
EVENT:NUMBER_OF_PLAYERS
```

For example:

```text
333:25
```

means:

> Analyze 25 selected players for the 3x3x3 Cube event.

---

# Experiment Outputs

Each experiment receives its own directory.

For example:

```text
data/processed/333_25players/
```

contains:

```text
config.json
candidate_players.csv
selected_players.csv
player_trajectories.csv
```

The corresponding results directory is:

```text
data/results/333_25players/
```

and contains files such as:

```text
config.json
model_comparison.csv
fit_parameters.csv
residuals.csv
model_summary.csv
model_ranking.csv
dfa_results.csv
distribution_metrics.csv
variability_comparison.csv
distribution_summary.csv
```

The focused exploratory case study lives separately in:

```text
data/processed/case_study_333_10players/
data/results/case_study_333_10players/
```

The generalized pipeline recreates the 3x3x3 run in:

```text
data/processed/333_10players/
data/results/333_10players/
```

The `config.json` file records the configuration used for the experiment, including:

* event ID,
* number of players,
* selection thresholds,
* minimum trajectory requirements,
* execution timestamp.

This helps ensure that the results can be traced back to the exact experimental configuration.

---

# Aggregated Experiment Summary

After running multiple experiments, the pipeline creates:

```text
data/results/experiment_summary.csv
```

This file contains one row per experiment.

Typical columns include:

```text
experiment_name
event
players
power_wins
exp_wins
mean_power_r2
mean_exp_r2
mean_hurst
persistent_share
mean_std_ratio
power_win_pct
```

The file is intended for cross-experiment comparison.

For example:

| Event | Players | Power-Law Win Rate | Mean Hurst | Mean STD Ratio |
| ----- | ------: | -----------------: | ---------: | -------------: |
| 333   |      10 |               100% |      0.643 |          0.286 |
| 333   |      25 |                84% |      0.675 |          0.279 |
| 333   |      50 |                84% |      0.676 |          0.320 |
| 222   |      25 |                64% |      0.605 |          0.535 |
| 444   |      25 |                88% |      0.753 |          0.226 |
| 555   |      25 |                68% |      0.841 |          0.215 |
| pyram |      25 |                64% |      0.597 |          0.431 |
| skewb |      25 |                64% |      0.607 |          0.467 |

---

# Current Preliminary Findings

The current rerun indicates three broad patterns.

## 1. Power-law learning curves generally outperform exponential curves

Across the rerun experimental conditions, the power-law model is selected more frequently than the exponential model.

This suggests that improvement may be better characterized by a decelerating power-law process than by a simple exponential decay.

The effect is strongest for the larger cube events and remains present, although weaker, for other puzzle types. In the current rerun, the power-law model wins in the majority of analyzed trajectories in every experimental condition, with the strongest dominance observed for the 333, 444, and 555 events.

---

## 2. Residuals retain temporal structure

The estimated Hurst exponents are generally above $0.5$.

This indicates that the residual fluctuations are not consistent with completely uncorrelated white noise.

Instead, performance fluctuations retain temporal dependence.

The strength of this persistence varies across events, with 555 and 444 showing the strongest persistence and 222 / pyram / skewb staying closer to the threshold.

---

## 3. Performance variability tends to decrease

The late-to-early standard-deviation ratio is below $1$ across the tested conditions.

This indicates that performance distributions generally become more concentrated as players improve.

The effect appears strongest for 444 and 555 and weaker for 222, pyram, and skewb, but the general direction remains consistent.

---

# Reproducibility

To reproduce the analysis:

1. Clone or download the repository.
2. Install the required Python dependencies.
3. Place the required WCA data files in:

```text
data/raw/
```

4. Run the experiment pipeline:

```bash
python run_experiments.py
```

5. Open the notebooks in order:

```text
01_data_exploration.ipynb
02_learning_models.ipynb
03_dfa_analysis.ipynb
04_distribution_analysis.ipynb
05_results_summary.ipynb
06_cross_experiment_summary.ipynb
```

The first five notebooks should be run against `data/processed/case_study_333_10players/` and `data/results/case_study_333_10players/`; the generalized pipeline then recreates `data/processed/333_10players/` and `data/results/333_10players/`.

For the cross-experiment analysis, run the experiment pipeline first so that:

```text
data/results/experiment_summary.csv
```

contains the results of all desired experiments.

---

# Project Status

The current implementation provides a complete preliminary analysis pipeline for:

* data preparation,
* player selection,
* learning-curve fitting,
* model comparison,
* residual analysis,
* DFA-based estimation of temporal dependence,
* early-versus-late variability analysis,
* cross-event and cross-sample-size comparison.

The current results should be considered preliminary and are intended to guide further discussion and refinement of the methodology.

Potential future work includes:

* more extensive statistical testing,
* sensitivity analysis of player-selection criteria,
* alternative learning-curve models,
* robustness analysis across additional events,
* improved treatment of temporal irregularity,
* analysis of competition frequency and time-based learning trajectories,
* more detailed comparison between puzzle types.
