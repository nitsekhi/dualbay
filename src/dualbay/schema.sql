-- dualbay data store (SQLite). One file, data/dualbay.sqlite.
-- Design: a small star around the matchup fact. Dimensions: station, zone, sensor, model_run.
-- Facts: insitu_obs (hourly/monthly measurements), pixel_extract (satellite values at stations),
-- matchup (the join of the two within a time window), prediction (model outputs with uncertainty).
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS station (
  station_id     TEXT PRIMARY KEY,
  name_ja        TEXT,
  name_en        TEXT,
  operator       TEXT,
  bay            TEXT NOT NULL CHECK (bay IN ('tokyo','sagami','ise','other')),
  lat            REAL NOT NULL,
  lon            REAL NOT NULL,
  coord_verified INTEGER NOT NULL DEFAULT 0,
  zone_id        TEXT
);

CREATE TABLE IF NOT EXISTS zone (
  zone_id     TEXT PRIMARY KEY,
  bay         TEXT NOT NULL,
  description TEXT,
  lon_min REAL, lat_min REAL, lon_max REAL, lat_max REAL
);

CREATE TABLE IF NOT EXISTS sensor (
  sensor_id    TEXT PRIMARY KEY,
  name         TEXT,
  provider     TEXT,
  resolution_m INTEGER
);

-- Long in-situ table. value_qc: ok | out_of_range | suspect
CREATE TABLE IF NOT EXISTS insitu_obs (
  obs_id      INTEGER PRIMARY KEY,
  station_id  TEXT NOT NULL REFERENCES station(station_id),
  time_utc    TEXT NOT NULL,            -- ISO 8601, UTC
  layer       TEXT,                     -- surface | upper | middle | lower | bottom | NULL
  depth_m     REAL,
  param       TEXT NOT NULL,            -- chl | turbidity | tsm | do | water_temp | salinity | ph | cdom
  value       REAL NOT NULL,
  unit        TEXT,
  qc          TEXT NOT NULL DEFAULT 'ok',
  source_file TEXT
);
-- NULLs are distinct in a plain UNIQUE constraint, so the de-duplication key uses IFNULL
CREATE UNIQUE INDEX IF NOT EXISTS ux_insitu ON insitu_obs(station_id, time_utc, IFNULL(layer,''), IFNULL(depth_m,-1), param, IFNULL(source_file,''));
CREATE INDEX IF NOT EXISTS ix_insitu_station_time ON insitu_obs(station_id, time_utc);
CREATE INDEX IF NOT EXISTS ix_insitu_param ON insitu_obs(param);

CREATE TABLE IF NOT EXISTS scene (
  scene_id   TEXT PRIMARY KEY,
  sensor_id  TEXT NOT NULL REFERENCES sensor(sensor_id),
  time_utc   TEXT NOT NULL,
  cloud_pct  REAL,
  lon_min REAL, lat_min REAL, lon_max REAL, lat_max REAL,
  processed  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_scene_time ON scene(sensor_id, time_utc);

-- Per-scene, per-zone pixel budget and diagnostics (closes the resolution/adjacency gap)
CREATE TABLE IF NOT EXISTS zone_scene_stats (
  scene_id   TEXT NOT NULL REFERENCES scene(scene_id),
  zone_id    TEXT NOT NULL REFERENCES zone(zone_id),
  n_water    INTEGER, n_good INTEGER, frac_good REAL,
  frac_absaer REAL, frac_lowlw REAL, frac_straylight REAL, frac_higlint REAL,
  frac_turbidw REAL, frac_atmwarn REAL, frac_coastz REAL, frac_cldice REAL,
  frac_neg_412 REAL, frac_neg_443 REAL,
  PRIMARY KEY (scene_id, zone_id)
);

-- Satellite values at a station: the 3x3 window median spectrum, one row per band, plus QC
CREATE TABLE IF NOT EXISTS pixel_extract (
  extract_id  INTEGER PRIMARY KEY,
  scene_id    TEXT NOT NULL REFERENCES scene(scene_id),
  station_id  TEXT NOT NULL REFERENCES station(station_id),
  wavelength  REAL,                     -- NULL for non-spectral products (chlor_a)
  rrs         REAL,
  rrs_unc     REAL,                     -- sensor-provided uncertainty when available (PACE Rrs_unc)
  product     TEXT,                     -- e.g. 'chlor_a'
  product_val REAL,
  dist_km     REAL,
  n_valid     INTEGER,
  window_cv   REAL,
  passed      INTEGER NOT NULL DEFAULT 0,
  UNIQUE (scene_id, station_id, wavelength, product)
);

-- The analysis fact: one satellite observation paired with one in-situ value
CREATE TABLE IF NOT EXISTS matchup (
  matchup_id   INTEGER PRIMARY KEY,
  scene_id     TEXT NOT NULL REFERENCES scene(scene_id),
  station_id   TEXT NOT NULL REFERENCES station(station_id),
  param        TEXT NOT NULL,
  insitu_time  TEXT NOT NULL,
  dt_min       REAL NOT NULL,
  insitu_value REAL NOT NULL,
  unit         TEXT,
  owt          INTEGER,                 -- optical water type assigned to the pixel
  split        TEXT,                    -- train | test_time | test_space | NULL
  UNIQUE (scene_id, station_id, param)
);
CREATE INDEX IF NOT EXISTS ix_matchup_param ON matchup(param, station_id);

CREATE TABLE IF NOT EXISTS model_run (
  run_id      TEXT PRIMARY KEY,
  created_utc TEXT NOT NULL,
  model       TEXT NOT NULL,            -- ocx_std | ocx_local | rf_owt | pinn | ...
  tier        INTEGER,                  -- 1 standard, 2 locally recalibrated, 3 project model
  param       TEXT NOT NULL,
  train_desc  TEXT,                     -- e.g. 'tokyo 2024-2025'
  config_json TEXT
);

CREATE TABLE IF NOT EXISTS prediction (
  run_id      TEXT NOT NULL REFERENCES model_run(run_id),
  matchup_id  INTEGER NOT NULL REFERENCES matchup(matchup_id),
  y_pred      REAL NOT NULL,
  y_lo        REAL,                     -- lower bound of the central 68 % interval
  y_hi        REAL,
  PRIMARY KEY (run_id, matchup_id)
);

CREATE TABLE IF NOT EXISTS metric (
  run_id   TEXT NOT NULL REFERENCES model_run(run_id),
  subset   TEXT NOT NULL,               -- all | bay:tokyo | split:test_time | owt:3 ...
  n        INTEGER,
  rmse REAL, mae REAL, bias REAL, mape REAL, mdsa REAL, r2 REAL, picp68 REAL,
  PRIMARY KEY (run_id, subset)
);

-- Views used by the Power BI export and the report
CREATE VIEW IF NOT EXISTS v_matchup_wide AS
SELECT m.matchup_id, m.scene_id, s.sensor_id, s.time_utc AS scene_time, m.station_id, st.bay, st.zone_id,
       m.param, m.insitu_value, m.dt_min, m.owt, m.split,
       substr(s.time_utc, 1, 7) AS year_month
FROM matchup m JOIN scene s ON s.scene_id = m.scene_id JOIN station st ON st.station_id = m.station_id;

CREATE VIEW IF NOT EXISTS v_prediction_eval AS
SELECT p.run_id, r.model, r.tier, r.param, v.*, p.y_pred, p.y_lo, p.y_hi,
       (p.y_pred - v.insitu_value) AS err,
       CASE WHEN v.insitu_value BETWEEN p.y_lo AND p.y_hi THEN 1 ELSE 0 END AS in_interval
FROM prediction p JOIN model_run r ON r.run_id = p.run_id JOIN v_matchup_wide v ON v.matchup_id = p.matchup_id;
