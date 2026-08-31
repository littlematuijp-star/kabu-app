"""
戦略の検証。ここが本体。

3つの罠を避けるために、必ずこの順で確認する:
 1) 未来情報の混入 … 指標は全て「その日までの情報」だけで作る(factors.py)
 2) 過剰最適化    … 前半5年で決めたルールを、後半5年に触らず適用して確かめる
 3) 相場の追い風  … 市場平均(全株等ウェイト)と必ず並べる。勝てていないなら意味がない
"""
import warnings; warnings.filterwarnings("ignore")
import os, itertools, json
import numpy as np
import pandas as pd

import factors
import backtest as bt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "out")
os.makedirs(OUT, exist_ok=True)

SPLIT = pd.Timestamp("2021-09-01")   # 前半=ルール決め / 後半=答え合わせ


def market(P):
    """市場代用: 全銘柄等ウェイトの配当込みリターン。"""
    return (1 + P["ret"].mean(axis=1).fillna(0)).cumprod()


def slice_stats(curve, label, lo=None, hi=None):
    c = curve.loc[lo:hi] if (lo is not None or hi is not None) else curve
    return bt.stats(c, label)


def main():
    P = factors.load_panel()
    F = factors.compute(P)
    print("パネル:", P["px"].shape, P["px"].index.min().date(), "〜", P["px"].index.max().date())
    mkt = market(P)

    # ---------- 1. 主要な設計判断を比較 ----------
    variants = {
        "配当コア25 年1回": dict(n=25, rebalance_months=12),
        "配当コア25 半年":   dict(n=25, rebalance_months=6),
        "配当コア15 年1回": dict(n=15, rebalance_months=12),
        "配当コア40 年1回": dict(n=40, rebalance_months=12),
        "利回りだけ上位25":  dict(n=25, rebalance_months=12, w=(1.0, 0, 0, 0, 0),
                             max_cuts=99, require_uptrend=False),
        "トレンド条件なし25": dict(n=25, rebalance_months=12, require_uptrend=False),
        "減配歴チェックなし25": dict(n=25, rebalance_months=12, max_cuts=99),
    }
    rows = [bt.stats(mkt, "【比較】市場平均(全株等ウェイト)")]
    curves = {"市場平均": mkt}
    for name, kw in variants.items():
        rb = kw.pop("rebalance_months", 12)
        cur, log = bt.run(P, F, rebalance_months=rb, **kw)
        curves[name] = cur
        s = bt.stats(cur, name)
        s["平均銘柄数"] = log.n.mean()
        s["年間入替"] = log.swaps.mean() * (12 / rb)
        rows.append(s)
    df = pd.DataFrame(rows).set_index("戦略")
    pct = ["CAGR", "最大DD", "12m平均", "12m最悪", "12m>=7%", "12mプラス", "年別最悪", "年別>=7%"]
    disp = df.copy()
    disp[pct] = (disp[pct] * 100).round(1)
    print("\n=== 全期間 2017/9-2026/8 ===")
    print(disp.round(1).to_string())

    # ---------- 2. 前半で決めて後半で答え合わせ ----------
    print("\n=== 前半(〜2021/8 ルール決め) / 後半(2021/9〜 答え合わせ) ===")
    rows2 = []
    for name, c in curves.items():
        a = slice_stats(c, name + " [前半]", hi=SPLIT)
        b = slice_stats(c, name + " [後半]", lo=SPLIT)
        for s in (a, b):
            if s:
                rows2.append(s)
    d2 = pd.DataFrame(rows2).set_index("戦略")
    d2[pct] = (d2[pct] * 100).round(1)
    print(d2.round(1).to_string())

    # ---------- 3. 年別 ----------
    print("\n=== 年別リターン(%) ===")
    yr = pd.DataFrame({k: v.resample("YE").last().pct_change() for k, v in curves.items()})
    yr.index = yr.index.year
    print((yr * 100).round(1).dropna(how="all").to_string())

    df.to_csv(os.path.join(OUT, "validate_all.csv"), encoding="utf-8-sig")
    yr.to_csv(os.path.join(OUT, "validate_yearly.csv"), encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(os.path.join(OUT, "curves.csv"), encoding="utf-8-sig")
    print("\n保存:", OUT)


if __name__ == "__main__":
    main()
