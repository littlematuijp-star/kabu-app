"""
比較対象（実際に買えるETF）の配当込みリターン。

Yahooのデータには稀に「1日だけ株価が1/10になる」ような欠陥がある
（1306 TOPIX ETF の 2026-03-30/31 が実例。383円→37.6円→389円）。
分割と区別するため、Stock Splits 列が0なのに1日で±60%超動いた日は
データ不良とみなして除外する。
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "bench")

NAMES = {"1306": "1306 TOPIX", "1321": "1321 日経225",
         "1478": "1478 高配当MSCI", "1577": "1577 高配当70"}


PACK = os.path.join(HERE, "data", "packed")
_PACKED = None


def _packed():
    global _PACKED
    if _PACKED is None:
        f = os.path.join(PACK, "bench.parquet")
        _PACKED = pd.read_parquet(f) if os.path.exists(f) else pd.DataFrame()
    return _PACKED


def total_return(code: str) -> pd.Series:
    b = _packed()
    if len(b):
        d = b[b["code"] == code].drop(columns=["code"])
        if d.empty:
            return pd.Series(dtype=float)
        d = d.set_index("Date").sort_index()
    else:
        f = os.path.join(DIR, f"{code}.csv")
        if not os.path.exists(f):
            return pd.Series(dtype=float)
        d = pd.read_csv(f, parse_dates=["Date"]).set_index("Date").sort_index()
    sp = d["Stock Splits"].replace(0, np.nan).fillna(1.0)
    r = (d["Close"] + d["Dividends"]) / (d["Close"].shift(1) / sp) - 1
    r = r.where(r.abs() < 0.6)          # データ不良の日は「変化なし」扱い
    return (1 + r.fillna(0)).cumprod()


def all_curves(index=None) -> dict:
    out = {}
    for c, name in NAMES.items():
        s = total_return(c)
        if len(s):
            out[name] = s.reindex(index).ffill() if index is not None else s
    return out
