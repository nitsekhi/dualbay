import numpy as np
import pandas as pd

from dualbay import insitu, matchup, study_area


def _jp_csv(tmp_path):
    """A file shaped like a Japanese monitoring-post export: title rows, Shift-JIS, half-width katakana,
    per-layer columns, 欠測 (missing) markers and hour 24."""
    text = (
        "東京湾水質連続観測データ\n"
        "地点：川崎人工島\n"
        "\n"
        "観測日時,上層水温(℃),上層塩分,上層ｸﾛﾛﾌｨﾙa(μg/L),上層濁度(FTU),下層DO(mg/L)\n"
        "2025/07/14 09:00,25.1,30.2,21.5,4.2,3.1\n"
        "2025/07/14 10:00,25.3,30.1,欠測,4.4,2.9\n"
        "2025/07/14 24:00,24.9,30.4,19.8,3.9,2.7\n"
        "2025/07/15 01:00,24.8,30.5,999,3.8,2.6\n"
    )
    p = tmp_path / "kawasaki_island.csv"
    p.write_bytes(text.encode("cp932"))
    return p


def test_parse_japanese_export(tmp_path):
    df = insitu.parse_file(_jp_csv(tmp_path), station="kawasaki_island")
    assert set(df["param"]) == {"water_temp", "salinity", "chl", "turbidity", "do"}
    chl = df[df["param"] == "chl"].set_index("time")["value"]
    assert len(chl) == 3  # 欠測 dropped
    # 24:00 on the 14th is midnight of the 15th
    assert pd.Timestamp("2025-07-15 00:00", tz="Asia/Tokyo") in chl.index
    assert set(df.loc[df["param"] == "do", "layer"]) == {"lower"}
    assert set(df.loc[df["param"] == "chl", "layer"]) == {"upper"}
    bad = df[(df["param"] == "chl") & (df["value"] == 999)]
    assert (bad["qc"] == "out_of_range").all()


def test_matchups_window_and_footprint(tmp_path):
    obs = insitu.parse_file(_jp_csv(tmp_path), station="kawasaki_island")
    scenes = pd.DataFrame(
        [
            # 09:30 JST = 00:30 UTC, footprint covers Tokyo Bay → matches the 09:00 and 10:00 obs, nearest wins
            {"sensor": "s3_olci", "scene_id": "A", "time": "2025-07-14T00:30:00Z", "cloud_pct": 5,
             "lon_min": 139.0, "lat_min": 34.5, "lon_max": 140.5, "lat_max": 36.0},
            # footprint over the Pacific only → no station inside
            {"sensor": "s3_olci", "scene_id": "B", "time": "2025-07-14T00:30:00Z", "cloud_pct": 5,
             "lon_min": 141.0, "lat_min": 30.0, "lon_max": 142.0, "lat_max": 31.0},
            # a day later → outside ±3 h
            {"sensor": "pace_oci", "scene_id": "C", "time": "2025-07-16T03:00:00Z", "cloud_pct": 5,
             "lon_min": 139.0, "lat_min": 34.5, "lon_max": 140.5, "lat_max": 36.0},
        ]
    )
    m = matchup.find_matchups(obs, scenes, window_h=3, params=["chl", "turbidity", "do"])
    assert set(m["scene_id"]) == {"A"}
    # DO is lower-layer only → not a satellite matchup
    assert set(m["param"]) == {"chl", "turbidity"}
    chl = m[m["param"] == "chl"].iloc[0]
    assert chl["value"] == 21.5 and abs(chl["dt_min"]) == 30
    s = matchup.summary(m)
    assert s["n_matchups"].sum() == 2


def test_window_stats_bailey_werdell():
    arr = np.full((5, 5), 2.0)
    arr[2, 3] = 2.1
    ok = matchup.window_stats(arr, 2, 2)
    assert ok["passed"] and ok["n_valid"] == 9

    cloudy = arr.copy()
    cloudy[1:4, 1:3] = np.nan  # 6 of 9 invalid
    assert not matchup.window_stats(cloudy, 2, 2)["passed"]

    noisy = np.arange(1, 10, dtype=float).reshape(3, 3)  # smooth gradient, CV ≈ 0.5
    assert not matchup.window_stats(noisy, 1, 1)["passed"]


def test_zones():
    assert study_area.zone_of(139.95, 35.55) == "tokyo_inner"
    assert study_area.zone_of(139.35, 35.05) == "sagami_open"
    assert study_area.zone_of(139.62, 35.15) == "miura_transition"
    assert study_area.zone_of(141.5, 35.0) is None
