from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data_loading import load_wca_tables, processed_data_dir, raw_data_dir, results_data_dir
from src.dfa_analysis import calculate_hurst_exponent, classify_noise
from src.learning_models import evaluate_model, exponential, fit_exponential, fit_power_law, power_law, residuals
from src.preprocessing import add_competition_dates, build_candidate_players, build_player_trajectories, shortlist_candidate_players


@dataclass(frozen=True)
class ExperimentConfig:
    event_id: str
    n_players: int
    min_results: int = 50
    selection_min_results: int = 100
    fallback_min_results: int = 50
    min_span: int = 3

    @property
    def name(self) -> str:
        return f"{self.event_id}_{self.n_players}players"


def parse_experiment_specs(specs: Iterable[str] | None) -> list[ExperimentConfig]:
    if not specs:
        specs = ["333:10", "333:25", "333:50", "222:25", "444:25", "555:25", "666:25", "777:25"]

    parsed: list[ExperimentConfig] = []
    for spec in specs:
        event_id, players = spec.split(":", maxsplit=1)
        parsed.append(ExperimentConfig(event_id=event_id.strip(), n_players=int(players)))
    return parsed


def save_config(config: ExperimentConfig, folder: Path, extra: dict | None = None) -> None:
    payload = {
        "event_id": config.event_id,
        "n_players": config.n_players,
        "min_results": config.min_results,
        "selection_min_results": config.selection_min_results,
        "fallback_min_results": config.fallback_min_results,
        "min_span": config.min_span,
        "execution_date_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)

    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def prepare_results_with_dates(results: pd.DataFrame, competitions_with_dates: pd.DataFrame, event_id: str) -> pd.DataFrame:
    event_results = results.loc[
        results["event_id"] == event_id,
        ["person_id", "person_name", "competition_id", "best", "average"],
    ].copy()

    event_results["average_sec"] = pd.to_numeric(event_results["average"], errors="coerce") / 100
    event_results["best_sec"] = pd.to_numeric(event_results["best"], errors="coerce") / 100

    event_results = event_results.loc[event_results["average_sec"].gt(0)].copy()
    event_results = event_results.merge(
        competitions_with_dates[["id", "date"]],
        left_on="competition_id",
        right_on="id",
        how="left",
    )
    return event_results


def run_model_fits(selected_trajectories: pd.DataFrame, results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis_rows = []
    parameter_rows = []
    residual_rows = []

    for person_id, group in selected_trajectories.groupby("person_id", sort=False):
        group = group.sort_values("seq_index").reset_index(drop=True)
        player_name = group["person_name"].iloc[0]
        x = group["seq_index"].to_numpy(dtype=float)
        y = group["average_seconds"].to_numpy(dtype=float)

        power_params, _ = fit_power_law(x, y)
        exp_params, _ = fit_exponential(x, y)

        power_pred = power_law(x, *power_params) if power_params is not None else None
        exp_pred = exponential(x, *exp_params) if exp_params is not None else None

        power_metrics = evaluate_model(y, power_pred, n_params=3) if power_pred is not None else None
        exp_metrics = evaluate_model(y, exp_pred, n_params=3) if exp_pred is not None else None

        if power_params is not None:
            parameter_rows.append(
                {
                    "person_id": person_id,
                    "person_name": player_name,
                    "model": "power_law",
                    "a": power_params[0],
                    "b": power_params[1],
                    "c": power_params[2],
                }
            )
        if exp_params is not None:
            parameter_rows.append(
                {
                    "person_id": person_id,
                    "person_name": player_name,
                    "model": "exponential",
                    "a": exp_params[0],
                    "b": exp_params[1],
                    "c": exp_params[2],
                }
            )

        if power_pred is not None:
            residual_rows.extend(
                pd.DataFrame(
                    {
                        "person_id": person_id,
                        "person_name": player_name,
                        "model": "power_law",
                        "seq_index": group["seq_index"].to_numpy(),
                        "competition_date": group["competition_date"].to_numpy(),
                        "observed_seconds": y,
                        "predicted_seconds": power_pred,
                        "residual_seconds": residuals(y, power_pred),
                    }
                ).to_dict(orient="records")
            )
        if exp_pred is not None:
            residual_rows.extend(
                pd.DataFrame(
                    {
                        "person_id": person_id,
                        "person_name": player_name,
                        "model": "exponential",
                        "seq_index": group["seq_index"].to_numpy(),
                        "competition_date": group["competition_date"].to_numpy(),
                        "observed_seconds": y,
                        "predicted_seconds": exp_pred,
                        "residual_seconds": residuals(y, exp_pred),
                    }
                ).to_dict(orient="records")
            )

        if power_metrics is None and exp_metrics is None:
            continue

        if power_metrics is None:
            winner = "exponential"
            winner_bic = "exponential"
        elif exp_metrics is None:
            winner = "power_law"
            winner_bic = "power_law"
        else:
            winner = "power_law" if power_metrics["aic"] < exp_metrics["aic"] else "exponential"
            if power_metrics["aic"] == exp_metrics["aic"]:
                winner = "tie"

            winner_bic = "power_law" if power_metrics["bic"] < exp_metrics["bic"] else "exponential"
            if power_metrics["bic"] == exp_metrics["bic"]:
                winner_bic = "tie"

        delta_aic = None if (power_metrics is None or exp_metrics is None) else exp_metrics["aic"] - power_metrics["aic"]
        delta_bic = None if (power_metrics is None or exp_metrics is None) else exp_metrics["bic"] - power_metrics["bic"]

        analysis_rows.append(
            {
                "person_id": person_id,
                "person_name": player_name,
                "winner": winner,
                "winner_bic": winner_bic,
                "power_rss": None if power_metrics is None else power_metrics["rss"],
                "power_r2": None if power_metrics is None else power_metrics["r2"],
                "power_aic": None if power_metrics is None else power_metrics["aic"],
                "power_bic": None if power_metrics is None else power_metrics["bic"],
                "exp_rss": None if exp_metrics is None else exp_metrics["rss"],
                "exp_r2": None if exp_metrics is None else exp_metrics["r2"],
                "exp_aic": None if exp_metrics is None else exp_metrics["aic"],
                "exp_bic": None if exp_metrics is None else exp_metrics["bic"],
                "delta_aic": delta_aic,
                "delta_aic_abs": None if delta_aic is None else abs(delta_aic),
                "delta_bic": delta_bic,
            }
        )

    model_comparison = pd.DataFrame(analysis_rows)
    fit_parameters = pd.DataFrame(parameter_rows)
    residuals_df = pd.DataFrame(residual_rows)

    if not model_comparison.empty:
        model_comparison = model_comparison.sort_values(["power_aic", "exp_aic"], na_position="last").reset_index(drop=True)
    if not fit_parameters.empty:
        fit_parameters = fit_parameters.sort_values(["person_id", "model"]).reset_index(drop=True)
    if not residuals_df.empty:
        residuals_df = residuals_df.sort_values(["person_id", "model", "seq_index"]).reset_index(drop=True)

    winner_counts = (
        model_comparison["winner"].value_counts().rename_axis("winner").reset_index(name="count")
        if not model_comparison.empty
        else pd.DataFrame(columns=["winner", "count"])
    )
    winner_bic_counts = (
        model_comparison["winner_bic"].value_counts().rename_axis("winner").reset_index(name="count")
        if not model_comparison.empty
        else pd.DataFrame(columns=["winner", "count"])
    )
    winner_counts["criterion"] = "AIC"
    winner_bic_counts["criterion"] = "BIC"
    model_summary = pd.concat([winner_counts, winner_bic_counts], ignore_index=True)[["criterion", "winner", "count"]]

    model_ranking = (
        model_comparison[
            [
                "person_id",
                "person_name",
                "delta_aic",
                "delta_bic",
                "winner",
                "winner_bic",
                "delta_aic_abs",
            ]
        ]
        .sort_values("delta_aic_abs", ascending=False)
        .reset_index(drop=True)
        if not model_comparison.empty
        else pd.DataFrame(columns=["person_id", "person_name", "delta_aic", "delta_bic", "winner", "winner_bic", "delta_aic_abs"])
    )

    model_comparison.to_csv(results_dir / "model_comparison.csv", index=False)
    fit_parameters.to_csv(results_dir / "fit_parameters.csv", index=False)
    residuals_df.to_csv(results_dir / "residuals.csv", index=False)
    model_summary.to_csv(results_dir / "model_summary.csv", index=False)
    model_ranking.to_csv(results_dir / "model_ranking.csv", index=False)

    return model_comparison, fit_parameters, residuals_df


def run_dfa(residuals_df: pd.DataFrame, model_comparison: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    if residuals_df.empty or model_comparison.empty:
        dfa_results = pd.DataFrame(
            columns=["person_id", "person_name", "model", "analysis_model", "n_obs", "hurst", "intercept", "dfa_r2", "noise_type"]
        )
        dfa_results.to_csv(results_dir / "dfa_results.csv", index=False)
        return dfa_results

    model_choice = model_comparison[["person_id", "winner", "winner_bic"]].copy()
    model_choice["analysis_model"] = np.where(
        model_choice["winner"] == "tie",
        model_choice["winner_bic"],
        model_choice["winner"],
    )

    selected_residuals = residuals_df.merge(
        model_choice[["person_id", "analysis_model"]],
        on="person_id",
        how="inner",
    )
    selected_residuals = selected_residuals.loc[
        selected_residuals["model"] == selected_residuals["analysis_model"]
    ].copy()

    rows = []
    for person_id, group in selected_residuals.groupby("person_id", sort=False):
        group = group.sort_values("seq_index").reset_index(drop=True)
        try:
            dfa_result = calculate_hurst_exponent(group["residual_seconds"].to_numpy(dtype=float))
        except ValueError:
            continue

        rows.append(
            {
                "person_id": person_id,
                "person_name": group["person_name"].iloc[0],
                "model": group["model"].iloc[0],
                "analysis_model": group["analysis_model"].iloc[0],
                "n_obs": int(len(group)),
                "hurst": dfa_result["hurst"],
                "intercept": dfa_result["intercept"],
                "dfa_r2": dfa_result["dfa_r2"],
                "noise_type": classify_noise(dfa_result["hurst"]),
            }
        )

    dfa_results = pd.DataFrame(rows)
    if not dfa_results.empty:
        dfa_results = dfa_results.sort_values(["hurst", "dfa_r2"], ascending=[False, False]).reset_index(drop=True)

    dfa_results.to_csv(results_dir / "dfa_results.csv", index=False)
    return dfa_results


def run_distribution_analysis(trajectories: pd.DataFrame, results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    comparison_rows = []

    for person_id, group in trajectories.groupby("person_id", sort=False):
        group = group.sort_values("seq_index").reset_index(drop=True)
        n_obs = len(group)
        phase_n = n_obs // 4
        if phase_n < 3:
            continue

        early = group.iloc[:phase_n]
        late = group.iloc[-phase_n:]

        def _stats(values: np.ndarray) -> dict[str, float]:
            values = np.asarray(values, dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            skewness = float(pd.Series(values).skew()) if len(values) > 2 else float("nan")
            kurt = float(pd.Series(values).kurt()) if len(values) > 3 else float("nan")
            return {"mean": mean, "std": std, "skewness": skewness, "kurtosis": kurt}

        early_stats = _stats(early["average_seconds"].to_numpy(dtype=float))
        late_stats = _stats(late["average_seconds"].to_numpy(dtype=float))

        metrics_rows.append(
            {
                "person_id": person_id,
                "person_name": group["person_name"].iloc[0],
                "phase": "early",
                "n_phase": int(len(early)),
                **early_stats,
            }
        )
        metrics_rows.append(
            {
                "person_id": person_id,
                "person_name": group["person_name"].iloc[0],
                "phase": "late",
                "n_phase": int(len(late)),
                **late_stats,
            }
        )

        std_ratio = float(late_stats["std"] / early_stats["std"]) if np.isfinite(early_stats["std"]) and early_stats["std"] > 0 else float("nan")
        comparison_rows.append(
            {
                "person_id": person_id,
                "person_name": group["person_name"].iloc[0],
                "n_obs": int(n_obs),
                "phase_n": int(phase_n),
                "mean_early": early_stats["mean"],
                "mean_late": late_stats["mean"],
                "std_early": early_stats["std"],
                "std_late": late_stats["std"],
                "std_ratio": std_ratio,
                "variability_reduction_pct": float(1 - std_ratio) if np.isfinite(std_ratio) else float("nan"),
                "skewness_early": early_stats["skewness"],
                "skewness_late": late_stats["skewness"],
                "kurtosis_early": early_stats["kurtosis"],
                "kurtosis_late": late_stats["kurtosis"],
            }
        )

    distribution_metrics = pd.DataFrame(metrics_rows)
    variability_comparison = pd.DataFrame(comparison_rows)

    if not distribution_metrics.empty:
        distribution_metrics = distribution_metrics.sort_values(["person_id", "phase"]).reset_index(drop=True)
    if not variability_comparison.empty:
        variability_comparison = variability_comparison.sort_values("std_ratio").reset_index(drop=True)

    summary_rows: list[dict[str, float]] = []
    if not variability_comparison.empty:
        summary_rows.extend(
            [
                {"metric": "players_analyzed", "value": float(len(variability_comparison))},
                {"metric": "mean_std_early", "value": float(np.nanmean(variability_comparison["std_early"]))},
                {"metric": "mean_std_late", "value": float(np.nanmean(variability_comparison["std_late"]))},
                {"metric": "mean_std_ratio", "value": float(np.nanmean(variability_comparison["std_ratio"]))},
                {
                    "metric": "mean_variability_reduction_pct",
                    "value": float(np.nanmean(variability_comparison["variability_reduction_pct"])),
                },
                {
                    "metric": "share_with_reduced_variability",
                    "value": float(np.mean(variability_comparison["std_ratio"] < 1)),
                },
            ]
        )

    distribution_summary = pd.DataFrame(summary_rows)

    distribution_metrics.to_csv(results_dir / "distribution_metrics.csv", index=False)
    variability_comparison.to_csv(results_dir / "variability_comparison.csv", index=False)
    distribution_summary.to_csv(results_dir / "distribution_summary.csv", index=False)

    return distribution_metrics, variability_comparison, distribution_summary


def run_experiment(config: ExperimentConfig, base_path: Path) -> dict[str, object]:
    tables = load_wca_tables(base_path)
    competitions_with_dates = add_competition_dates(tables["competitions"])
    prepared_results = prepare_results_with_dates(tables["results"], competitions_with_dates, config.event_id)

    processed_dir = processed_data_dir(base_path, experiment_name=config.name)
    results_dir = results_data_dir(base_path, experiment_name=config.name)
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    candidate_players = build_candidate_players(
        tables["results"],
        competitions_with_dates,
        event_id=config.event_id,
        min_results=config.min_results,
    )
    selected_players = shortlist_candidate_players(
        candidate_players,
        n_candidates=config.n_players,
        min_results=config.selection_min_results,
        fallback_min_results=config.fallback_min_results,
        min_span=config.min_span,
    )

    trajectories = build_player_trajectories(
        prepared_results,
        selected_players,
        selected_id_column="person_id",
        results_id_column="person_id",
        date_column="date",
        average_column="average_sec",
    )

    candidate_players.to_csv(processed_dir / "candidate_players.csv", index=False)
    selected_players.to_csv(processed_dir / "selected_players.csv", index=False)
    trajectories.to_csv(processed_dir / "player_trajectories.csv", index=False)

    model_comparison, _, residuals_df = run_model_fits(trajectories, results_dir)
    dfa_results = run_dfa(residuals_df, model_comparison, results_dir)
    distribution_metrics, variability_comparison, distribution_summary = run_distribution_analysis(trajectories, results_dir)

    power_wins = int((model_comparison["winner"] == "power_law").sum()) if not model_comparison.empty else 0
    exp_wins = int((model_comparison["winner"] == "exponential").sum()) if not model_comparison.empty else 0
    mean_power_r2 = float(model_comparison["power_r2"].mean()) if not model_comparison.empty else float("nan")
    mean_exp_r2 = float(model_comparison["exp_r2"].mean()) if not model_comparison.empty else float("nan")
    mean_hurst = float(dfa_results["hurst"].mean()) if not dfa_results.empty else float("nan")
    persistent_share = float((dfa_results["noise_type"] == "persistent").mean()) if not dfa_results.empty else float("nan")
    mean_std_ratio = (
        float(variability_comparison["std_ratio"].mean()) if not variability_comparison.empty else float("nan")
    )

    extra = {
        "experiment_name": config.name,
        "event": config.event_id,
        "players": config.n_players,
        "raw_data_dir": str(raw_data_dir(base_path)),
        "processed_dir": str(processed_dir),
        "results_dir": str(results_dir),
        "candidate_players_count": int(len(candidate_players)),
        "selected_players_count": int(len(selected_players)),
        "trajectory_rows": int(len(trajectories)),
        "model_rows": int(len(model_comparison)),
        "dfa_rows": int(len(dfa_results)),
        "distribution_metric_rows": int(len(distribution_metrics)),
        "variability_rows": int(len(variability_comparison)),
        "distribution_summary_rows": int(len(distribution_summary)),
    }
    save_config(config, processed_dir, extra=extra)
    save_config(config, results_dir, extra=extra)

    summary_row = {
        "event": config.event_id,
        "players": config.n_players,
        "power_wins": power_wins,
        "exp_wins": exp_wins,
        "mean_power_r2": mean_power_r2,
        "mean_exp_r2": mean_exp_r2,
        "mean_hurst": mean_hurst,
        "persistent_share": persistent_share,
        "mean_std_ratio": mean_std_ratio,
    }

    print(f"[done] {config.name}: selected={len(selected_players)}, trajectories={len(trajectories)}, dfa={len(dfa_results)}")
    return summary_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-event, multi-size mastery experiments.")
    parser.add_argument(
        "--experiments",
        nargs="*",
        help="Experiment specs formatted as EVENT:PLAYERS, e.g. 333:25 222:50",
    )
    parser.add_argument(
        "--base-path",
        default=".",
        help="Repository base path (default: current working directory).",
    )
    args = parser.parse_args()

    base_path = Path(args.base_path).resolve()
    configs = parse_experiment_specs(args.experiments)
    summary_rows = []

    for config in configs:
        summary_rows.append(run_experiment(config, base_path))

    experiment_summary = pd.DataFrame(summary_rows)
    summary_output = results_data_dir(base_path) / "experiment_summary.csv"
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    experiment_summary.to_csv(summary_output, index=False)
    print(f"Saved experiment summary to: {summary_output}")


if __name__ == "__main__":
    main()
