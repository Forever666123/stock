"""FOMC decision dates (statement day) and the rate action taken.

2015-2025 are scheduled meetings plus the two 2020 emergency moves, which are
tagged separately so the scheduled-only statistics can exclude them.
2026 dates come from the Fed's published calendar; the actions are *not*
verified in this repo (the Sep 2026 meeting has not happened as of data end),
so they are excluded from every statistic by default.

action: +25/+50/+75 = hike (bp), 0 = hold, negative = cut.
"""
from __future__ import annotations

import pandas as pd

_ROWS = [
    # 2015
    ("2015-01-28", 0), ("2015-03-18", 0), ("2015-04-29", 0), ("2015-06-17", 0),
    ("2015-07-29", 0), ("2015-09-17", 0), ("2015-10-28", 0), ("2015-12-16", 25),
    # 2016
    ("2016-01-27", 0), ("2016-03-16", 0), ("2016-04-27", 0), ("2016-06-15", 0),
    ("2016-07-27", 0), ("2016-09-21", 0), ("2016-11-02", 0), ("2016-12-14", 25),
    # 2017
    ("2017-02-01", 0), ("2017-03-15", 25), ("2017-05-03", 0), ("2017-06-14", 25),
    ("2017-07-26", 0), ("2017-09-20", 0), ("2017-11-01", 0), ("2017-12-13", 25),
    # 2018
    ("2018-01-31", 0), ("2018-03-21", 25), ("2018-05-02", 0), ("2018-06-13", 25),
    ("2018-08-01", 0), ("2018-09-26", 25), ("2018-11-08", 0), ("2018-12-19", 25),
    # 2019
    ("2019-01-30", 0), ("2019-03-20", 0), ("2019-05-01", 0), ("2019-06-19", 0),
    ("2019-07-31", -25), ("2019-09-18", -25), ("2019-10-30", -25), ("2019-12-11", 0),
    # 2020 (03-03 and 03-15 were unscheduled emergency cuts; 03-15 was a Sunday)
    ("2020-01-29", 0), ("2020-03-03", -50), ("2020-03-15", -100), ("2020-04-29", 0),
    ("2020-06-10", 0), ("2020-07-29", 0), ("2020-09-16", 0), ("2020-11-05", 0),
    ("2020-12-16", 0),
    # 2021
    ("2021-01-27", 0), ("2021-03-17", 0), ("2021-04-28", 0), ("2021-06-16", 0),
    ("2021-07-28", 0), ("2021-09-22", 0), ("2021-11-03", 0), ("2021-12-15", 0),
    # 2022
    ("2022-01-26", 0), ("2022-03-16", 25), ("2022-05-04", 50), ("2022-06-15", 75),
    ("2022-07-27", 75), ("2022-09-21", 75), ("2022-11-02", 75), ("2022-12-14", 50),
    # 2023
    ("2023-02-01", 25), ("2023-03-22", 25), ("2023-05-03", 25), ("2023-06-14", 0),
    ("2023-07-26", 25), ("2023-09-20", 0), ("2023-11-01", 0), ("2023-12-13", 0),
    # 2024
    ("2024-01-31", 0), ("2024-03-20", 0), ("2024-05-01", 0), ("2024-06-12", 0),
    ("2024-07-31", 0), ("2024-09-18", -50), ("2024-11-07", -25), ("2024-12-18", -25),
    # 2025
    ("2025-01-29", 0), ("2025-03-19", 0), ("2025-05-07", 0), ("2025-06-18", 0),
    ("2025-07-30", 0), ("2025-09-17", -25), ("2025-10-29", -25), ("2025-12-10", -25),
]
EMERGENCY = {"2020-03-03", "2020-03-15"}

# 2026 scheduled statement days. Three actions are recovered from the CME
# FedWatch meeting files in data/fedwatch/: a past meeting's distribution
# collapses onto the bucket the Fed chose, and 04-29, 06-17 and 07-29 all
# settle on 350-375, so those three were holds and the target range has not
# moved since at least April. The rest are still unknown -> None, excluded.
_ROWS_2026 = [
    ("2026-01-28", None), ("2026-03-18", None), ("2026-04-29", 0),
    ("2026-06-17", 0), ("2026-07-29", 0), ("2026-09-16", None),
    ("2026-10-28", None), ("2026-12-09", None),
]

# Target range in bp as last confirmed by the FedWatch files (2026-07-29).
CURRENT_TARGET_BP = (350, 375)

# Meetings where a hike was a live option (rate-hiking cycles). Used for the
# "does a pre-meeting selloff stop a hike" test. Judgement call, kept explicit.
HIKE_CYCLE_WINDOWS = [("2015-12-01", "2018-12-31"), ("2022-03-01", "2023-07-31")]


def fomc_table(include_2026: bool = False) -> pd.DataFrame:
    rows = _ROWS + (_ROWS_2026 if include_2026 else [])
    df = pd.DataFrame(rows, columns=["date", "action"])
    df["date"] = pd.to_datetime(df["date"])
    df["emergency"] = df["date"].dt.strftime("%Y-%m-%d").isin(EMERGENCY)
    df["kind"] = pd.cut(df["action"].fillna(-999), [-1000, -998, -1, 0, 1000],
                        labels=["unknown", "cut", "hold", "hike"])
    df["hike_option"] = False
    for a, b in HIKE_CYCLE_WINDOWS:
        df.loc[(df["date"] >= a) & (df["date"] <= b), "hike_option"] = True
    return df.set_index("date")


if __name__ == "__main__":
    t = fomc_table()
    print(t["kind"].value_counts())
    print("scheduled only:", t[~t.emergency]["kind"].value_counts().to_dict())
