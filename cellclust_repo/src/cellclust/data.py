"""
cellclust.data
==============
Raw CSV loading, biological-plausibility quality control, and
per-muscle splitting.

The input file (VF_DATA.csv) is a long-format export from Dragonfly
v2022.2 with one row per segmented fiber. Decimal separator is comma
(European notation), field separator is semicolon.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from .config import RAW_CSV, MUSCLES, FERET_MIN_UM, FERET_MAX_UM


def load_data(path: Path | str = RAW_CSV) -> pd.DataFrame:
    """Load and QC the raw VF_DATA.csv export.

    Parameters
    ----------
    path : Path or str
        Location of the raw CSV. Defaults to ``data/VF_DATA.csv``.

    Returns
    -------
    pd.DataFrame
        Long-format table of fibers with columns:
            feret      : minimum Feret diameter (um, float)
            diet       : NCD or HFD
            week       : 8, 12, or 16
            Musculo    : SOL or GAS
            animal_id  : within-stratum integer identifier
            animal     : unique animal id (Muscle + Diet + Week + animal_id)
    """
    df = pd.read_csv(path, sep=";", decimal=",")
    df = df.rename(columns={
        "Min Feret Diameter (um)": "feret",
        "Codigo animal":           "animal_id",
        "Condicion":               "diet",
        "Tiempo":                  "week",
    })
    df["feret"] = pd.to_numeric(df["feret"], errors="coerce")

    # Muscle restriction
    df = df[df["Musculo"].isin(MUSCLES)]

    # Biological-plausibility filter (Briguet et al. 2004)
    df = df[
        (df["feret"] >= FERET_MIN_UM) &
        (df["feret"] <= FERET_MAX_UM) &
        df["feret"].notna()
    ]

    # Build a unique animal identifier: the integer id resets within
    # each Diet x Week stratum, so we concatenate all stratifiers.
    df["animal"] = (
        df["Musculo"].astype(str)
        + "_" + df["diet"].astype(str)
        + "_W" + df["week"].astype(str)
        + "_A" + df["animal_id"].astype(str)
    )
    return df.reset_index(drop=True)


def split_by_muscle(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return ``{muscle_name: DataFrame}`` with each muscle's subset."""
    return {m: df[df["Musculo"] == m].reset_index(drop=True) for m in MUSCLES}


def summarize_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Per (Muscle x Diet x Week) descriptive statistics.

    Useful for Manuscript Table 1 and for sanity-checking the cohort
    after QC.
    """
    rows = []
    for (m, d, w), sub in df.groupby(["Musculo", "diet", "week"]):
        rows.append({
            "Muscle":         m,
            "Diet":           d,
            "Week":           int(w),
            "n_fibers":       len(sub),
            "n_animals":      sub["animal_id"].nunique(),
            "mean":           sub["feret"].mean(),
            "sd":             sub["feret"].std(),
            "sem":            sub["feret"].std() / (len(sub) ** 0.5),
            "median":         sub["feret"].median(),
            "p5":             sub["feret"].quantile(0.05),
            "p95":            sub["feret"].quantile(0.95),
            "pct_below_20um": 100 * (sub["feret"] < 20).mean(),
        })
    return pd.DataFrame(rows)
