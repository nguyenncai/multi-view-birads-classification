"""Utilities to normalize VinDr annotation CSV schemas."""

from __future__ import annotations

import pandas as pd


def _extract_numeric_label(series: pd.Series) -> pd.Series:
    """Extract the first integer from values like 'BI-RADS 4' or '4'."""
    return pd.to_numeric(
        series.astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    )


def load_vindr_annotations(csv_path: str) -> pd.DataFrame:
    """
    Load VinDr-style annotations and expose a common schema used by the repo.

    Supported inputs:
    - cleaned_label.csv style: patient_id, view, BIRADS, density
    - breast-level_annotations.csv style: study_id, view_position,
      breast_birads, breast_density
    """
    df = pd.read_csv(csv_path).copy()

    if "patient_id" not in df.columns and "study_id" in df.columns:
        df["patient_id"] = df["study_id"]
    if "view" not in df.columns and "view_position" in df.columns:
        df["view"] = df["view_position"]

    if "BIRADS" not in df.columns and "breast_birads" in df.columns:
        df["BIRADS"] = _extract_numeric_label(df["breast_birads"])
    elif "BIRADS" in df.columns:
        df["BIRADS"] = _extract_numeric_label(df["BIRADS"])

    if "density" not in df.columns and "breast_density" in df.columns:
        df["density"] = (
            df["breast_density"].astype(str).str.extract(r"([ABCD])", expand=False)
        )
    elif "density" in df.columns:
        df["density"] = df["density"].astype(str).str.extract(r"([ABCD])", expand=False)

    required_columns = ["patient_id", "image_id", "laterality", "view", "split"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Unsupported annotation schema in {csv_path}. "
            f"Missing required columns after normalization: {missing_columns}"
        )

    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["laterality"] = df["laterality"].astype(str).str.upper().str.strip()
    df["view"] = df["view"].astype(str).str.upper().str.strip()
    df["split"] = df["split"].astype(str).str.lower().str.strip()

    if "cancer" in df.columns:
        df["cancer"] = pd.to_numeric(df["cancer"], errors="coerce")

    return df
