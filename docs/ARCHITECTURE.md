# dualbay — architecture and data flow

## One picture

```
 Satellite L2 (PACE OCI, SGLI, OLCI)      In-situ (monitoring posts, 千葉灯標, AERONET-OC, 公共用水域)
          │ earthaccess / STAC                         │ CSV / Excel downloads
          ▼                                            ▼
   pace.py  ── bay subset only ──►  zone_spectra.csv   insitu.py ──► insitu.csv  (long table, JST)
   flags.py    pixel budget          station_values.csv              │
          │                                   │                      │
          └──────────────► matchup.py ◄───────┴──────────────────────┘
                              │  ±3 h, surface layer, 3×3 window QC (Bailey & Werdell 2006)
                              ▼
                       db.py / schema.sql  (SQLite star schema)
                              │
        ┌───────────┬─────────┼─────────────┬───────────────┐
        ▼           ▼         ▼             ▼               ▼
    owt.py     baseline.py  models.py   transfer.py       lag.py
  optical     tier 1 / 2    tier 3 +    temporal /      chl → bottom DO,
  water types  reference   uncertainty  spatial,        hypoxia events,
                                        static vs       blue-tide index
                                        adaptive
        └───────────┴─────────┴─────────────┴───────────────┘
                              ▼
                        export.py  ──►  Power BI star schema + measures.dax  (the digital-twin dashboard)
```

`demo.py` runs this whole chain on synthetic data (`dualbay demo`), so every link is exercised
before real data arrive. Demo outputs are mock data and live under `data/demo/`.

## Why these choices

| Decision | Reason |
|---|---|
| Download each PACE granule to a temp dir, read only the bay subset, delete | A granule is hundreds of MB; the two bays are ~1 % of it. Remote streaming (`earthaccess.open`) falls back to HTTP outside AWS us-west-2 and is unreliable from Japan, so download-then-subset is the default. |
| SQLite, not Postgres / Kafka / InfluxDB | One file, no server, idempotent loaders. ~10⁶ rows is nothing for it. Infrastructure is not the research question. |
| Power BI as the dashboard | Already in the author's toolkit; regulatory-threshold logic lives in DAX; no hosting. Grafana removed. |
| NASA L3 mask without ABSAER/ATMWARN + per-zone flag fractions; bit table read from the file's `flag_meanings` | The ABSAER / LOWLW / negative-blue-Rrs fractions per zone are a direct, cheap test of the atmospheric-correction problem in Tokyo Bay (Higa et al. 2024). |
| Three baseline tiers | The −20 % RMSE claim is measured against algorithms **recalibrated on the same matchups** (tier 2). Beating untuned global coefficients proves little. |
| Random forest ensemble for uncertainty | Simplest method that gives a per-pixel interval and can be checked by coverage (PICP). GP / MDN can replace it behind the same `fit / predict` API. |
| Temporal transfer before spatial | Needs Tokyo Bay data only; a digital twin runs daily for years, so season-to-season stability matters as much as bay-to-bay. |
| Novelty flag from OWT distance | Tells the user *where* a prediction is extrapolation. This is the mechanism behind H2 (transfer fails in unseen optical types). |
| Lag analysis on in-situ first | RQ3 can start before any model access: monitoring posts measure both surface Chl-a and bottom DO hourly. |

## Store (schema.sql)

Dimensions `station`, `zone`, `sensor`, `model_run`. Facts `insitu_obs`, `scene`, `zone_scene_stats`,
`pixel_extract`, `matchup`, `prediction`, `metric`. Views `v_matchup_wide`, `v_prediction_eval`.
All loaders are `INSERT OR IGNORE / REPLACE`, so re-running never duplicates. The in-situ
de-duplication key uses `IFNULL` because NULL depth / layer would otherwise defeat UNIQUE.

Time is stored as ISO-8601 UTC. Japanese local time (JST = UTC+9) is derived in queries and in `DimDate`.

## Error handling

- `pace.process`: per-granule try/except, 3 retries with a 20 s wait on I/O errors, progress saved
  every 10 granules, automatic resume from the existing CSV, failures listed in `failed_granules.csv`.
- `insitu.parse_file`: tries UTF-8 / Shift-JIS / EUC-JP, detects the header row and the delimiter,
  turns Japanese missing-value markers into NaN, converts 24:00 to next-day 00:00, flags out-of-range values.
- `db.connect`: transaction per session, rollback and log on any exception; unknown stations are refused.
- `baseline.recalibrate`: refuses to fit with fewer matchups than coefficients + 3; non-convergence is reported, not swallowed.
- `transfer.*`: minimum sample sizes enforced; spatial transfer to a bay without data is skipped with a message.

## Known limits (honest list)

- Station coordinates are approximate until replaced with official values (`coord_verified: false`).
- `toy_rrs` in the demo is not a radiative-transfer model; it only gives algorithms something physical to recover.
- The blue-tide index coefficients are from a GOCI relation and must be recalibrated for PACE.
- `calibration_table` assumes a normal spread of the tree ensemble in log space; fine for a first check.
- The OWT scheme is generic k-means; the lab's 2020 classification of Tokyo Bay should be the starting point for the real scheme.
- No hydrodynamic model lives here. Assimilation is done inside the lab's model; this repository prepares the observations.
