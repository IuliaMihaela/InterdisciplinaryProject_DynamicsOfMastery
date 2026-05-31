from __future__ import annotations

import pandas as pd


def add_competition_dates(competitions: pd.DataFrame) -> pd.DataFrame:
	competitions_with_dates = competitions.copy()
	competitions_with_dates["date"] = pd.to_datetime(
		dict(
			year=competitions_with_dates["year"],
			month=competitions_with_dates["month"],
			day=competitions_with_dates["day"],
		)
	)
	return competitions_with_dates


def build_candidate_players(
	results: pd.DataFrame,
	competitions: pd.DataFrame,
	event_id: str = "333",
	min_results: int = 50,
) -> pd.DataFrame:
	"""Summarize players for a single event and keep those with enough data."""

	event_results = results.loc[
		results["event_id"] == event_id,
		["person_id", "person_name", "competition_id", "best", "average"],
	].copy()
	event_results["best_sec"] = pd.to_numeric(event_results["best"], errors="coerce") / 100
	event_results["average_sec"] = pd.to_numeric(event_results["average"], errors="coerce") / 100

	competition_dates = competitions.loc[:, ["id", "date"]].copy()
	merged = event_results.merge(
		competition_dates,
		left_on="competition_id",
		right_on="id",
		how="left",
	)

	summary = (
		merged.groupby(["person_id", "person_name"], as_index=False)
		.agg(
			n_competitions=("competition_id", "nunique"),
			n_results=("competition_id", "size"),
			first_year=("date", lambda series: int(series.dt.year.min())),
			last_year=("date", lambda series: int(series.dt.year.max())),
			best_sec_mean=("best_sec", "mean"),
			average_sec_mean=("average_sec", "mean"),
		)
		.assign(span=lambda frame: frame["last_year"] - frame["first_year"])
		.sort_values(
			["n_results", "span", "n_competitions"],
			ascending=[False, False, False],
		)
		.reset_index(drop=True)
	)

	return summary.loc[summary["n_results"] >= min_results].reset_index(drop=True)


def shortlist_candidate_players(
	candidate_players: pd.DataFrame,
	n_candidates: int = 10,
	min_results: int = 100,
	fallback_min_results: int = 50,
	min_span: int = 3,
) -> pd.DataFrame:
	shortlist = candidate_players.loc[
		(candidate_players["n_results"] >= min_results)
		& (candidate_players["span"] >= min_span)
	].copy()
	if len(shortlist) < 5:
		shortlist = candidate_players.loc[
			(candidate_players["n_results"] >= fallback_min_results)
			& (candidate_players["span"] >= min_span)
		].copy()
	return shortlist.head(n_candidates).reset_index(drop=True)


def build_player_trajectories(
	results_with_dates: pd.DataFrame,
	selected_players: pd.DataFrame,
	selected_id_column: str | None = None,
	results_id_column: str = "person_id",
	date_column: str = "date",
	average_column: str = "average_sec",
) -> pd.DataFrame:
	"""Create cleaned player trajectories with a sequential index per player.

	The output schema is:
	- person_id
	- person_name
	- competition_date
	- average_seconds
	- seq_index
	"""

	if selected_id_column is None:
		if "WCA ID" in selected_players.columns:
			selected_id_column = "WCA ID"
		elif results_id_column in selected_players.columns:
			selected_id_column = results_id_column
		else:
			raise ValueError(
				"Could not infer selected player ID column. Expected 'WCA ID' or 'person_id'."
			)

	required_results_columns = {
		results_id_column,
		"person_name",
		date_column,
		average_column,
	}
	missing_results_columns = required_results_columns.difference(results_with_dates.columns)
	if missing_results_columns:
		raise ValueError(
			f"Missing required columns in results_with_dates: {sorted(missing_results_columns)}"
		)

	if selected_id_column not in selected_players.columns:
		raise ValueError(
			f"Missing selected player ID column '{selected_id_column}' in selected_players."
		)

	selected_ids = set(selected_players[selected_id_column].dropna().astype(str))

	trajectories = (
		results_with_dates.loc[
			results_with_dates[results_id_column].astype(str).isin(selected_ids),
			[results_id_column, "person_name", date_column, average_column],
		]
		.rename(
			columns={
				results_id_column: "person_id",
				date_column: "competition_date",
				average_column: "average_seconds",
			}
		)
		.assign(
			average_seconds=lambda frame: pd.to_numeric(frame["average_seconds"], errors="coerce"),
			competition_date=lambda frame: pd.to_datetime(frame["competition_date"], errors="coerce"),
		)
	)

	trajectories = trajectories.loc[
		trajectories["average_seconds"].gt(0) & trajectories["competition_date"].notna()
	].copy()
	trajectories = trajectories.sort_values(["person_id", "competition_date"]).reset_index(drop=True)
	trajectories["seq_index"] = trajectories.groupby("person_id").cumcount() + 1

	return trajectories[
		["person_id", "person_name", "competition_date", "average_seconds", "seq_index"]
	].reset_index(drop=True)

