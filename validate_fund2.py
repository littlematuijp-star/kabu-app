"""
ファンダメンタル指標の検証（厳密版）。

先行検証で分かった問題に対処する:

 1) t値の水増し
    月次で「先12ヶ月」を見ると、隣り合う観測が11ヶ月ぶん重なる。
    独立でない観測を独立として数えるとt値が約sqrt(12)倍に膨らむ。
    → 重複を考慮した補正t値を併記する。

 2) 期間が特殊すぎる
    2023年3月の「東証PBR1倍割れ改善要請」の直撃期間しか検証できていない。
    → 期間を前半・後半に割り、両方で成立するかを見る。
       片方でしか出ないなら、それは一度きりのイベントの記録に過ぎない。

 3) 配当利回りとの重複
    高配当株はもともと低PBR・低PERであることが多い。
    「PBRが効いた」のではなく「既に使っている配当利回りの言い換え」
    かもしれない。→ 指標どうしの相関を見る。
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd

import factors, fund_panel, backtest as bt, active, bench

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "out")


def ic_series(f, fwd, liq, me, min_n=80):
    """各月の順位相関と、5分位平均リターン。"""
    ics, dec, ns, dates = [], [], [], []
    for t in me:
        m = liq.loc[t] >= 1e8
        v = pd.concat([f.loc[t], fwd.loc[t]], axis=1, keys=["x", "y"])[m].dropna()
        if len(v) < min_n:
            continue
        ics.append(v.x.corr(v.y, method="spearman"))
        ns.append(len(v))
        dates.append(t)
        q = pd.qcut(v.x, 5, labels=False, duplicates="drop")
        g = v.y.groupby(q).mean()
        dec.append(g.values if len(g) == 5 else [np.nan] * 5)
    return pd.Series(ics, index=pd.DatetimeIndex(dates)), np.array(dec), ns


def summarize(ic: pd.Series, dec, ns, H):
    """重複を補正したt値を含む要約。"""
    n = len(ic)
    if n < 3:
        return {}
    raw_t = ic.mean() / (ic.std(ddof=1) / np.sqrt(n))
    # 先H ヶ月を月次で見ると観測がH-1ヶ月ぶん重なる。
    # 実質の独立サンプル数は n/H 程度なので、t を sqrt(H) で割る。
    adj_t = raw_t / np.sqrt(H)
    d = dict(IC=ic.mean(), 見かけのt=raw_t, 補正t=adj_t, 月数=n,
             独立サンプル概算=n / H, 平均銘柄数=int(np.mean(ns)),
             ICプラスの月=float((ic > 0).mean()))
    if dec.ndim == 2 and not np.isnan(dec).all():
        d["最下位20"] = np.nanmean(dec[:, 0])
        d["最上位20"] = np.nanmean(dec[:, 4])
        d["上下差"] = np.nanmean(dec[:, 4] - dec[:, 0])
    return d


def main():
    P = factors.load_panel()
    F = factors.compute(P)
    G = fund_panel.build(P, F)
    px, tr = P["px"], P["tr"]
    idx = px.index
    me = pd.DatetimeIndex(sorted(pd.Series(idx, index=idx)
                                 .groupby([idx.year, idx.month]).last()))
    trm = tr.loc[me]
    liq = (px * P["vol"]).rolling(60).mean().loc[me]

    cand = {
        "割安さ(PBR逆数)": G["純資産倍率の逆数"].loc[me],
        "割安さ(PER逆数)": G["益回り"].loc[me],
        "ROE": G["ROE"].loc[me],
        "自己資本比率": G["自己資本比率"].loc[me],
        "営業利益率": G["営業利益率"].loc[me],
        "配当性向の低さ": -G["配当性向"].loc[me],
        "【対照】配当利回り": F["yield"].loc[me],
        "【対照】12ヶ月モメンタム": F["tr_mom12"].loc[me],
    }

    H = 12
    fwd = trm.shift(-H) / trm - 1
    store = {}
    rows = []
    for k, f in cand.items():
        ic, dec, ns = ic_series(f, fwd, liq, me)
        store[k] = ic
        s = summarize(ic, dec, ns, H)
        if s:
            rows.append(dict(指標=k, **s))
    d = pd.DataFrame(rows).sort_values("IC", ascending=False)
    for c in ["IC", "最下位20", "最上位20", "上下差", "ICプラスの月"]:
        if c in d: d[c] = (d[c] * 100).round(1)
    pd.set_option("display.width", 240)
    print("=== 先12ヶ月リターンへの予測力（重複補正つき）===")
    print(d.round(2).to_string(index=False))
    d.to_csv(os.path.join(OUT, "fund_ic_strict.csv"), encoding="utf-8-sig")

    # ---------- 期間を割って両方で成立するか ----------
    print("\n=== 期間を前半・後半に割る（片方だけなら一度きりのイベント）===")
    rows2 = []
    for k, ic in store.items():
        if len(ic) < 8:
            continue
        mid = ic.index[len(ic) // 2]
        a, b = ic.loc[:mid], ic.loc[mid:]
        rows2.append(dict(指標=k,
                          前半IC=a.mean() * 100, 前半月数=len(a),
                          後半IC=b.mean() * 100, 後半月数=len(b),
                          符号一致="○" if np.sign(a.mean()) == np.sign(b.mean()) else "×"))
    print(pd.DataFrame(rows2).round(2).to_string(index=False))

    # ---------- 配当利回りとの重複 ----------
    print("\n=== 指標どうしの順位相関（配当利回りの言い換えになっていないか）===")
    t_last = me[-13]
    m = liq.loc[t_last] >= 1e8
    X = pd.DataFrame({k: f.loc[t_last] for k, f in cand.items()})[m].dropna()
    print(f"（基準日 {t_last.date()}  {len(X)}銘柄）")
    print((X.corr(method="spearman") * 100).round(0).to_string())


if __name__ == "__main__":
    main()
