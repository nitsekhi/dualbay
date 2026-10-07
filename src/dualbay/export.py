"""Power BI export: a star schema as CSV files plus a DAX measures file.

The dashboard layer of the digital twin is Power BI, not Grafana: it is already in the author's
toolkit, handles the regulatory-threshold logic in DAX, and needs no server.

Tables written to <out_dir>/powerbi/
  DimDate.csv        one row per day 2019-01-01 .. 2029-12-31 (Japanese fiscal year, season)
  DimStation.csv     station, bay, zone, coordinates, verified flag
  DimZone.csv
  DimSensor.csv
  DimModelRun.csv
  FactInsituDaily.csv  station × date × param × layer_group → mean, min, max, n
  FactSceneZone.csv    scene × zone pixel budget and diagnostics (the 'how much usable data' view)
  FactMatchup.csv      matchup with in-situ value
  FactPrediction.csv   predictions with interval per run
  FactMetric.csv       metrics per run and subset
  measures.dax         measures to paste into the model
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from . import db

SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
          6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}


def dim_date(start="2019-01-01", end="2029-12-31") -> pd.DataFrame:
    d = pd.DataFrame({"Date": pd.date_range(start, end, freq="D")})
    d["DateKey"] = d["Date"].dt.strftime("%Y%m%d").astype(int)
    d["Year"] = d["Date"].dt.year
    d["Month"] = d["Date"].dt.month
    d["MonthName"] = d["Date"].dt.strftime("%b")
    d["YearMonth"] = d["Date"].dt.strftime("%Y-%m")
    d["Season"] = d["Month"].map(SEASON)
    d["FiscalYearJP"] = d["Year"].where(d["Month"] >= 4, d["Year"] - 1)   # 年度 starts in April
    d["IsHypoxiaSeason"] = d["Month"].between(5, 10)
    return d[["DateKey", "Date", "Year", "Month", "MonthName", "YearMonth", "Season", "FiscalYearJP", "IsHypoxiaSeason"]]


def export(db_path: str | Path = db.DEFAULT_PATH, out_dir: str | Path = "data/processed") -> list[Path]:
    out = Path(out_dir) / "powerbi"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    with db.connect(db_path) as con:
        tables = {
            "DimDate": dim_date(),
            "DimStation": db.read(con, "SELECT station_id AS StationKey, name_en AS Station, name_ja AS StationJa, operator, bay AS Bay, zone_id AS ZoneKey, lat AS Lat, lon AS Lon, coord_verified AS CoordVerified FROM station"),
            "DimZone": db.read(con, "SELECT zone_id AS ZoneKey, bay AS Bay, description AS Description FROM zone"),
            "DimSensor": db.read(con, "SELECT sensor_id AS SensorKey, name AS Sensor, provider AS Provider, resolution_m AS ResolutionM FROM sensor"),
            "DimModelRun": db.read(con, "SELECT run_id AS RunKey, model AS Model, tier AS Tier, param AS Param, train_desc AS TrainDesc, created_utc AS CreatedUtc FROM model_run"),
            "FactInsituDaily": db.read(con, """
                SELECT station_id AS StationKey,
                       CAST(strftime('%Y%m%d', datetime(time_utc, '+9 hours')) AS INTEGER) AS DateKey,
                       param AS Param,
                       CASE WHEN layer IN ('surface','upper') OR (layer IS NULL AND IFNULL(depth_m,0) <= 2) THEN 'surface'
                            WHEN layer IN ('lower','bottom') THEN 'bottom' ELSE 'middle' END AS LayerGroup,
                       AVG(value) AS MeanValue, MIN(value) AS MinValue, MAX(value) AS MaxValue, COUNT(*) AS N
                FROM insitu_obs WHERE qc = 'ok'
                GROUP BY 1,2,3,4"""),
            "FactSceneZone": db.read(con, """
                SELECT z.scene_id AS SceneKey, s.sensor_id AS SensorKey, z.zone_id AS ZoneKey,
                       CAST(strftime('%Y%m%d', datetime(s.time_utc, '+9 hours')) AS INTEGER) AS DateKey,
                       z.n_water AS NWater, z.n_good AS NGood, z.frac_good AS FracGood,
                       z.frac_absaer AS FracAbsAer, z.frac_lowlw AS FracLowLw, z.frac_straylight AS FracStrayLight,
                       z.frac_higlint AS FracGlint, z.frac_cldice AS FracCloud, z.frac_neg_412 AS FracNeg412, z.frac_neg_443 AS FracNeg443
                FROM zone_scene_stats z JOIN scene s ON s.scene_id = z.scene_id"""),
            "FactMatchup": db.read(con, """
                SELECT matchup_id AS MatchupKey, scene_id AS SceneKey, sensor_id AS SensorKey, station_id AS StationKey,
                       CAST(strftime('%Y%m%d', datetime(scene_time, '+9 hours')) AS INTEGER) AS DateKey,
                       param AS Param, insitu_value AS InsituValue, dt_min AS DtMin, owt AS OWT, split AS Split
                FROM v_matchup_wide"""),
            "FactPrediction": db.read(con, "SELECT run_id AS RunKey, matchup_id AS MatchupKey, y_pred AS Pred, y_lo AS PredLo, y_hi AS PredHi FROM prediction"),
            "FactMetric": db.read(con, "SELECT run_id AS RunKey, subset AS Subset, n AS N, rmse AS RMSE, mae AS MAE, bias AS Bias, mape AS MAPE, mdsa AS MdSA, r2 AS R2, picp68 AS PICP68 FROM metric"),
        }
    for name, df in tables.items():
        p = out / f"{name}.csv"
        df.to_csv(p, index=False, encoding="utf-8-sig")   # BOM so Excel/Power BI read Japanese correctly
        written.append(p)
    p = out / "measures.dax"
    p.write_text(DAX, encoding="utf-8")
    written.append(p)
    return written


DAX = r"""// dualbay — Power BI measures. Paste into a measure table (e.g. _Measures).
// Relationships: FactInsituDaily[DateKey] → DimDate[DateKey]; FactInsituDaily[StationKey] → DimStation[StationKey];
// FactSceneZone[DateKey] → DimDate; FactSceneZone[ZoneKey] → DimZone; FactMatchup[DateKey] → DimDate;
// FactPrediction[MatchupKey] → FactMatchup[MatchupKey]; FactPrediction[RunKey] → DimModelRun[RunKey].

// ---------- In-situ, regulatory view ----------
Bottom DO (mg/L) =
CALCULATE ( AVERAGE ( FactInsituDaily[MeanValue] ),
    FactInsituDaily[Param] = "do", FactInsituDaily[LayerGroup] = "bottom" )

Surface Chl-a (µg/L) =
CALCULATE ( AVERAGE ( FactInsituDaily[MeanValue] ),
    FactInsituDaily[Param] = "chl", FactInsituDaily[LayerGroup] = "surface" )

// 底層DO environmental standard (MOE, 2016): class 1 ≥ 4.0, class 2 ≥ 3.0, class 3 ≥ 2.0 mg/L.
// Threshold is a what-if parameter so the user can pick the class of the water body.
Hypoxia Threshold = SELECTEDVALUE ( 'DO Threshold'[Value], 3 )

Hypoxia Days =
CALCULATE ( DISTINCTCOUNT ( FactInsituDaily[DateKey] ),
    FactInsituDaily[Param] = "do", FactInsituDaily[LayerGroup] = "bottom",
    FactInsituDaily[MinValue] < [Hypoxia Threshold] )

Hypoxia Share % =
DIVIDE ( [Hypoxia Days],
    CALCULATE ( DISTINCTCOUNT ( FactInsituDaily[DateKey] ),
        FactInsituDaily[Param] = "do", FactInsituDaily[LayerGroup] = "bottom" ) )

Bloom Days (Chl ≥ 20) =
CALCULATE ( DISTINCTCOUNT ( FactInsituDaily[DateKey] ),
    FactInsituDaily[Param] = "chl", FactInsituDaily[LayerGroup] = "surface",
    FactInsituDaily[MaxValue] >= 20 )

// ---------- Satellite data availability (gap #3 and #9 of the brief) ----------
Clear Scenes =
CALCULATE ( DISTINCTCOUNT ( FactSceneZone[SceneKey] ), FactSceneZone[FracGood] >= 0.5, FactSceneZone[NWater] >= 20 )

Clear Scenes per Month = DIVIDE ( [Clear Scenes], DISTINCTCOUNT ( DimDate[YearMonth] ) )

Usable Pixel Share % = AVERAGE ( FactSceneZone[FracGood] )

Absorbing Aerosol Flag % = AVERAGE ( FactSceneZone[FracAbsAer] )

Negative Blue Rrs % = AVERAGE ( FactSceneZone[FracNeg412] )

// ---------- Model accuracy ----------
RMSE =
VAR t = ADDCOLUMNS ( FactPrediction, "e2", ( FactPrediction[Pred] - RELATED ( FactMatchup[InsituValue] ) ) ^ 2 )
RETURN SQRT ( AVERAGEX ( t, [e2] ) )

Bias = AVERAGEX ( FactPrediction, FactPrediction[Pred] - RELATED ( FactMatchup[InsituValue] ) )

MdSA % =
VAR t = ADDCOLUMNS ( FILTER ( FactPrediction, FactPrediction[Pred] > 0 && RELATED ( FactMatchup[InsituValue] ) > 0 ),
                     "alr", ABS ( LOG10 ( DIVIDE ( FactPrediction[Pred], RELATED ( FactMatchup[InsituValue] ) ) ) ) )
RETURN 100 * ( POWER ( 10, MEDIANX ( t, [alr] ) ) - 1 )

Interval Coverage 68 % =
AVERAGEX ( FactPrediction,
    IF ( RELATED ( FactMatchup[InsituValue] ) >= FactPrediction[PredLo]
         && RELATED ( FactMatchup[InsituValue] ) <= FactPrediction[PredHi], 1, 0 ) )

// Gain of a selected run over the tier-2 reference run with the same parameter.
RMSE Gain vs Tier 2 % =
VAR thisRMSE = [RMSE]
VAR refRMSE = CALCULATE ( [RMSE], REMOVEFILTERS ( DimModelRun ), DimModelRun[Tier] = 2,
                          DimModelRun[Param] = SELECTEDVALUE ( DimModelRun[Param] ) )
RETURN 100 * DIVIDE ( refRMSE - thisRMSE, refRMSE )

Target Met (−20 %) = IF ( [RMSE Gain vs Tier 2 %] >= 20, "Met", "Not met" )
"""
