"""Smoke tests for cellclust.data."""
import sys
import tempfile
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cellclust.data import load_data, split_by_muscle, summarize_strata


@pytest.fixture
def fake_csv():
    """A tiny CSV mimicking the VF_DATA.csv format."""
    rows = []
    for animal_id in range(1, 4):
        for diet in ["NCD", "HFD"]:
            for week in [8, 12, 16]:
                for muscle in ["SOL", "GAS"]:
                    for f in [20.0, 30.0, 40.0, 50.0]:
                        rows.append({
                            "Min Feret Diameter (um)": f"{f:.2f}".replace(".", ","),
                            "Codigo animal": animal_id,
                            "Condicion":     diet,
                            "Tiempo":        week,
                            "Musculo":       muscle,
                        })
    df = pd.DataFrame(rows)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                       delete=False) as f:
        df.to_csv(f.name, sep=";", index=False)
        path = f.name
    yield path
    Path(path).unlink()


def test_load_data_returns_dataframe(fake_csv):
    df = load_data(fake_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_data_required_columns(fake_csv):
    df = load_data(fake_csv)
    for col in ["feret", "diet", "week", "Musculo", "animal_id", "animal"]:
        assert col in df.columns, f"missing column: {col}"


def test_load_data_filter(fake_csv):
    df = load_data(fake_csv)
    assert (df["feret"] >= 2.0).all()
    assert (df["feret"] <= 120.0).all()


def test_load_data_muscle_restriction(fake_csv):
    df = load_data(fake_csv)
    assert set(df["Musculo"].unique()).issubset({"SOL", "GAS"})


def test_split_by_muscle(fake_csv):
    df = load_data(fake_csv)
    parts = split_by_muscle(df)
    assert set(parts.keys()) == {"SOL", "GAS"}
    assert len(parts["SOL"]) + len(parts["GAS"]) == len(df)


def test_summarize_strata(fake_csv):
    df = load_data(fake_csv)
    s = summarize_strata(df)
    assert {"Muscle", "Diet", "Week", "n_fibers",
             "mean", "sd", "median", "pct_below_20um"}.issubset(s.columns)
