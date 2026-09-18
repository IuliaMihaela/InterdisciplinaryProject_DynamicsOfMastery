# The Dynamics of Mastery

## Non-Linear Dynamics and Statistical Physics of Human Skill Acquisition

This repository contains the code and analysis for an interdisciplinary project investigating the dynamics of human skill acquisition using competitive Rubik's Cube performance data.

The project uses historical competition results from the World Cube Association (WCA) to study how solving performance evolves over observed competitive experience and whether the observed learning dynamics exhibit:

1. **Power-law or exponential learning behavior**
2. **Temporal dependence in performance fluctuations**
3. **A reduction in performance variability as skill improves**

The analysis is designed to be reproducible and extensible across multiple WCA events and different numbers of players.

---

# Research Questions

### RQ1 — Learning-curve model

Does human skill acquisition in speedcubing follow a power-law or exponential learning curve?

The two models considered are:

### Power-law model

$$
y(x) = a x^{-b} + c
$$

where:

* $x$ is the sequential observation index;
* $y(x)$ is the average solve time;
* $a$ is a scale parameter;
* $b$ controls the rate of improvement;
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

The sequential observation index represents the chronological order of a player's observed competition results. It is therefore a proxy for accumulated observed competitive experience, rather than real elapsed time.

---

### RQ2 — Temporal structure of residuals

After removing the fitted learning curve, do the remaining performance fluctuations behave like independent noise or do they retain temporal structure?

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

The theoretical white-noise benchmark is $H=0.5$.

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
│   │   ├── case_study_333_10players/
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
│       └── experiment_summary.csv
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

The raw data directory is excluded from git. The WCA export files are downloaded from the World Cube Association results export.

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

These files are written to `data/results/case_study_333_10players/` for the case-study run, and to the corresponding generalized experiment directory when the pipeline is rerun.

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
   * approximately uncorrelated,
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

It also performs several additional analyses requested during the development of the project.

### Representative model-fit visualizations

Representative player trajectories are plotted together with both fitted models:

* observed solving times,
* fitted power-law curve,
* fitted exponential curve.

These visualizations provide a direct comparison between the two candidate learning curves and complement the numerical model-selection criteria.

Examples are selected from the generalized experiments to illustrate different degrees of model preference.

### Player-profile analysis

The notebook performs an exploratory comparison of players classified according to their preferred model.

The analysis considers:

* trajectory length,
* gender,
* registration/start year,
* nationality.

The purpose is to investigate whether observable player characteristics are associated with model preference.

This analysis is exploratory and descriptive. In particular, nationality groups have uneven sample sizes, so country-level differences are not interpreted as evidence of a nationality effect.

### Relative improvement analysis

Relative improvement is calculated as:

$$
\text{Relative improvement}
=
\frac{\text{early mean}-\text{late mean}}
{\text{early mean}}
$$

where the early and late means correspond to the first and last quarters of each player's trajectory.

The analysis compares relative improvement between players whose trajectories are classified as power-law or exponential.

The current results show similar average improvement between the two groups, while the exponential group displays greater variability in individual improvement.

### Equal-length trajectory robustness

Because the full player trajectories have different lengths, an additional model-comparison analysis uses only the first 200 observations for every player.

This provides a common temporal window across all 210 trajectories and tests whether the model preference observed over complete trajectories is sensitive to trajectory length.

The equal-length analysis is particularly important because the full-trajectory model comparison and the fixed-window comparison produce different model-preference patterns.

### Cross-experiment visualizations

The notebook also produces cross-experiment visualizations, including:

* power-law win-rate plots,
* Hurst-exponent comparisons,
* variability-ratio comparisons,
* a cross-experiment heatmap,
* representative model-fit plots,
* player-profile comparisons,
* equal-length model-comparison results.

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

The generalized `333:10` experiment is a separate pipeline run and does not necessarily contain the same player cohort as the original case study.

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

Across all eight experimental conditions, there are 210 analyzed trajectories.

---

# Current Preliminary Findings

The current analysis identifies several recurring patterns, while also showing that some results depend on the temporal extent of the trajectory.

## 1. Power-law learning curves are preferred over full trajectories

Across the 210 evaluated trajectories, the power-law model is preferred for 160 trajectories (76.2%), while the exponential model is preferred for 50 trajectories.

The power law is the majority-winning model in every full-trajectory experimental condition.

The strongest power-law dominance is observed in the 3x3x3 and 4x4x4 experiments, while the preference is weaker in 2x2x2, Pyraminx, Skewb, and 5x5x5.

This suggests that power-law-like improvement is a recurring pattern in the complete available trajectories, although the strength of model preference varies between events.

### Sample-size comparison for 3x3x3

When the number of analyzed players was increased from 10 to 25 and 50, the power law remained the preferred model for the majority of full trajectories.

The corresponding power-law win rates were:

* 333:10 — 100%
* 333:25 — 84%
* 333:50 — 84%

This provides an initial robustness check across sample sizes.

---

## 2. Model preference depends on temporal window length

An equal-length robustness analysis was performed using only the first 200 observations for every player.

This produces a different model-preference pattern:

| Experimental condition | Power Law | Exponential | Power-Law % |
| ---------------------- | --------: | ----------: | ----------: |
| 333:10                 |         3 |           7 |         30% |
| 333:25                 |         9 |          16 |         36% |
| 333:50                 |        22 |          28 |         44% |
| 222:25                 |         5 |          20 |         20% |
| 444:25                 |        14 |          11 |         56% |
| 555:25                 |        10 |          15 |         40% |
| Pyraminx:25            |         4 |          21 |         16% |
| Skewb:25               |         5 |          20 |         20% |

Across all 210 trajectories:

* Power law: 72 trajectories (34.3%)
* Exponential: 138 trajectories (65.7%)

This contrasts with the full-trajectory result of 160 power-law and 50 exponential preferences.

The result indicates that model preference is sensitive to the temporal extent of the trajectory. Therefore, the 76.2% power-law dominance observed in the full trajectories should not be interpreted as a uniform preference across all stages of progression.

Instead, model preference appears to depend partly on the amount of observed trajectory available.

---

## 3. Residuals retain persistent temporal structure

The estimated Hurst exponents are above the theoretical white-noise benchmark of $H=0.5$ across all tested experiments.

Mean Hurst exponents range from approximately 0.60 for Pyraminx to approximately 0.84 for 5x5x5.

Most residual trajectories are classified as persistent according to the study's threshold of $H>0.55$.

This indicates that performance fluctuations retain temporal dependence rather than behaving as purely independent white noise.

The strength of this persistence varies across events, with 555 and 444 showing the highest mean Hurst exponents and 222, Pyraminx, and Skewb remaining closer to the persistence threshold.

These results are interpreted as evidence of persistent temporal structure rather than as proof of a single universal mechanism.

---

## 4. Performance variability tends to decrease

The mean late-to-early standard-deviation ratio is below 1 across all tested experimental conditions.

This indicates that within-player performance variability generally decreases during the later part of the observed trajectory.

The reduction is strongest for 444 and 555 and weaker for 222, Pyraminx, and Skewb, but the general direction remains consistent.

The distribution analysis also shows changes in skewness and kurtosis between early and late phases, consistent with fewer extreme performance deviations in the later phase.

---

## 5. Relative improvement is similar between model groups

Both model groups show substantial improvement over their observed trajectories.

The current relative-improvement results are:

| Model       | Players | Mean Improvement | Median Improvement |     SD |
| ----------- | ------: | ---------------: | -----------------: | -----: |
| Exponential |      50 |           25.91% |             26.31% | 14.70% |
| Power Law   |     160 |           23.78% |             23.94% |  7.80% |

The mean improvement is therefore broadly similar between the two groups.

However, improvement is considerably more variable among exponential-model players. The standard deviation of individual improvement is 14.70% for the exponential group compared with 7.80% for the power-law group.

This suggests that players in the two model groups can show similar overall improvement while differing in the consistency of that improvement across individuals.

---

## 6. Exploratory player-profile analysis

The analysis was extended to investigate whether observable player characteristics distinguish power-law and exponential trajectories.

### Trajectory length

Player trajectories range from 221 to 860 observations. All selected players therefore have substantially more observations than the minimum required for inclusion.

The power-law group has longer trajectories overall:

* Power law: median 474 observations
* Exponential: median 356.5 observations

This difference is descriptive and may partly reflect differences in available career history or model discrimination over longer trajectories.

### Registration/start year

The median registration/start year is:

* Power law: 2011
* Exponential: 2013.5

This difference may reflect differences between player cohorts or competition eras. It should not be interpreted as an age effect.

### Gender

Gender shows relatively little variation between the two model groups. Both groups are predominantly male, so gender does not currently provide a strong descriptive distinction between the model classes.

### Nationality

Nationality shows some exploratory variation across countries.

However, country-level sample sizes are uneven, with several countries represented by only a small number of players. The observed differences should therefore be interpreted descriptively rather than as evidence of a nationality effect.

Overall, the current player-profile analysis does not identify a single clear demographic characteristic that distinguishes power-law from exponential trajectories.

---

# Robustness Across Sample Sizes and Events

The current cross-experiment analysis shows that the main qualitative patterns observed in the original 3x3x3 case study also appear across multiple events, although their strength varies.

Across the full available trajectories:

1. power-law learning curves are preferred for the majority of trajectories;
2. residual performance fluctuations exhibit persistent temporal structure;
3. late-stage performance variability is lower than early-stage variability.

However, the equal-length analysis demonstrates that the first conclusion is sensitive to temporal extent. When all players are restricted to their first 200 observations, exponential fits become the majority preference.

The statistical characteristics also differ between events. The 3x3x3 and 4x4x4 events currently show stronger power-law model preference over the full trajectories, while 4x4x4 and 5x5x5 show stronger residual persistence and variability contraction than 2x2x2, Pyraminx, and Skewb.

Taken together, the results suggest that the statistical characteristics of skill acquisition are broadly consistent across the tested events but are not identical between events or across different temporal windows.

---

# Reproducibility

To reproduce the analysis:

1. Clone or download the repository.
2. Ensure you have Python 3.12+ installed.
3. Install the required Python dependencies.
4. Place the required WCA data files in:

```text
data/raw/
```

5. Run the experiment pipeline:

```bash
python run_experiments.py
```

6. Open the notebooks in order:

```text
01_data_exploration.ipynb
02_learning_models.ipynb
03_dfa_analysis.ipynb
04_distribution_analysis.ipynb
05_results_summary.ipynb
06_cross_experiment_summary.ipynb
```

The first five notebooks are designed around the original 333:10 case-study outputs:

```text
data/processed/case_study_333_10players/
data/results/case_study_333_10players/
```

The generalized pipeline then recreates the corresponding experiment directories, including:

```text
data/processed/333_10players/
data/results/333_10players/
```

For the cross-experiment analysis, run the experiment pipeline first so that:

```text
data/results/experiment_summary.csv
```

contains the results of all desired experiments.
