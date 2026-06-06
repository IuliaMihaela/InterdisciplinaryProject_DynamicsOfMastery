from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_FILENAMES = {
	"persons": "WCA_export_persons.tsv",
	"competitions": "WCA_export_competitions.tsv",
	"results": "WCA_export_results.tsv",
	"events": "WCA_export_events.tsv",
}


def resolve_repo_root(base_path: str | Path | None = None) -> Path:
	"""Return the repository root for the workspace."""

	candidate = Path(base_path) if base_path is not None else Path.cwd()
	if (candidate / "data" / "raw").exists():
		return candidate
	if (candidate.parent / "data" / "raw").exists():
		return candidate.parent
	return candidate


def raw_data_dir(base_path: str | Path | None = None) -> Path:
	return resolve_repo_root(base_path) / "data" / "raw"


def processed_data_dir(
	base_path: str | Path | None = None,
	experiment_name: str | None = None,
) -> Path:
	"""Return the processed-data directory.

	If ``experiment_name`` is provided, return an experiment-scoped subfolder.
	"""

	root = resolve_repo_root(base_path) / "data" / "processed"
	if experiment_name:
		return root / experiment_name
	return root


def results_data_dir(
	base_path: str | Path | None = None,
	experiment_name: str | None = None,
) -> Path:
	"""Return the results directory, optionally scoped by experiment."""

	root = resolve_repo_root(base_path) / "data" / "results"
	if experiment_name:
		return root / experiment_name
	return root


def list_raw_files(base_path: str | Path | None = None) -> list[str]:
	return sorted(path.name for path in raw_data_dir(base_path).iterdir() if path.is_file())


def load_wca_tables(base_path: str | Path | None = None) -> dict[str, pd.DataFrame]:
	"""Load the WCA TSV exports needed for the exploration notebook."""

	directory = raw_data_dir(base_path)
	return {
		name: pd.read_csv(directory / filename, sep="\t", low_memory=False)
		for name, filename in RAW_FILENAMES.items()
	}


def load_selected_players(
	base_path: str | Path | None = None,
	experiment_name: str | None = None,
) -> pd.DataFrame:
	"""Load selected players from processed data."""

	return pd.read_csv(
		processed_data_dir(base_path, experiment_name=experiment_name)
		/ "selected_players.csv"
	)


def load_player_trajectories(
	base_path: str | Path | None = None,
	experiment_name: str | None = None,
) -> pd.DataFrame:
	"""Load per-player trajectories from processed data."""

	trajectories = pd.read_csv(
		processed_data_dir(base_path, experiment_name=experiment_name)
		/ "player_trajectories.csv"
	)
	if "competition_date" in trajectories.columns:
		trajectories["competition_date"] = pd.to_datetime(
			trajectories["competition_date"], errors="coerce"
		)
	elif "date" in trajectories.columns:
		trajectories["date"] = pd.to_datetime(trajectories["date"], errors="coerce")
	return trajectories

