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
            {
                "sensor": "s3_olci",
                "scene_id": "A",
                "time": "2025-07-14T00:30:00Z",
                "cloud_pct": 5,
                "lon_min": 139.0,
                "lat_min": 34.5,
                "lon_max": 140.5,
                "lat_max": 36.0,
            },
            # footprint over the Pacific only → no station inside
            {
                "sensor": "s3_olci",
                "scene_id": "B",
                "time": "2025-07-14T00:30:00Z",
                "cloud_pct": 5,
                "lon_min": 141.0,
                "lat_min": 30.0,
                "lon_max": 142.0,
                "lat_max": 31.0,
            },
            # a day later → outside ±3 h
            {
                "sensor": "pace_oci",
                "scene_id": "C",
                "time": "2025-07-16T03:00:00Z",
                "cloud_pct": 5,
                "lon_min": 139.0,
                "lat_min": 34.5,
                "lon_max": 140.5,
                "lat_max": 36.0,
            },
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


def test_parse_mlit_monitoring_post_export(tmp_path):
    """Hourly MLIT export: 層/深度 columns, DO(%) and DO(mg/l), full-width chｌ, per-parameter flag columns, 24:00."""
    hdr = (
        "    No,YYYY/MM/DD hh:mm,    層,深度(m),水温(℃),塩分(-),DO(%),DO(mg/l),chｌ(μg/l),濁度(NTU),pH(-),ORP(mv),"
        "水温フラグ,塩分フラグ,ＤＯフラグ,クロロフィルフラグ,濁度フラグ,pHフラグ,ORPフラグ"
    )
    rows = [
        "1,2025/07/01 00:00,1,1.14,27.55,25.62,156.90,10.73,39.20,4.70,8.54,196.00,0,0,0,0,0,0,0",
        "2,2025/07/01 00:00,9,12.50,22.10,32.40,12.00,0.85,1.20,6.10,7.80,-50.00,0,0,0,0,0,0,0",
        "3,2025/07/01 24:00,1,1.12,27.50,25.63,153.20,10.49,35.70,4.80,8.54,197.00,0,0,0,9,0,0,0",
    ]
    f = tmp_path / "01kemigawa-2025.07-S.csv"
    f.write_bytes(("\n".join([hdr] + rows) + "\n").encode("cp932"))
    df = insitu.parse_many([f])
    assert set(df["station"]) == {"kemigawa_oki"}
    assert set(df["param"]) == {
        "chl",
        "do",
        "ph",
        "salinity",
        "turbidity",
        "water_temp",
    }  # DO(%) and ORP skipped
    do = df[(df["param"] == "do") & (df["depth_m"] == 1.14)]
    assert do["value"].iloc[0] == 10.73  # mg/l column, not the % column
    assert df["time"].max() == pd.Timestamp("2025-07-02 00:00", tz="Asia/Tokyo")  # 24:00 → next day
    flagged = df[(df["param"] == "chl") & (df["time"] == df["time"].max())]
    assert flagged["qc"].iloc[0] == "flagged"  # provider flag 9 on chlorophyll
