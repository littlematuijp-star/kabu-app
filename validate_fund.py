"""
PBR/PER/ROE/配当性向などファンダメンタル指標の検証。

やること:
 1) 各指標が「先1年のリターン」をどれだけ予測できたか（順位相関IC、5分位の差）
 2) その指標で銘柄を選んだ場合のポートフォリオ成績
 3) 既存の配当ルールに足したとき、成績が良くなるのか

【この検証の限界】
 決算データが2023年3月期以降しか取れないため、検証できるのは約3年分。
 しかもその3年は日本株の大相場だった。
 「効かなかった」は信用してよいが、「効いた」は割り引いて読むこと。
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd

import factors, fund_panel, backtest as bt, active, bench

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "out")


def main():
    P = factors.load_panel()
    F = factors.compute(P)
    G = fund_panel.build(P, F)
    px, tr = P["px"], P["tr"]
    cov = pd.DataFrame({k: v.notna().sum(axis=1) for k, v in G.items()})
    print("=== 指標が使えるようになった時期（その日に値がある銘柄数）===")
    print(cov.resample("YE").last().to_string())

    idx = px.index
    me = pd.DatetimeIndex(sorted(pd.Series(idx, index=idx)
                                 .groupby([idx.year, idx.month]).last()))
    trm = tr.loc[me]
    liq = (px * P["vol"]).rolling(60).mean().loc[me]

    # 検証に使う指標（高いほど良い向きに符号を揃える）
    cand = {
        "割安さ(BPS/株価=PBR逆数)": G["純資産倍率の逆数"].loc[me],
        "割安さ(益回り=PER逆数)": G["益回り"].loc[me],
        "ROE": G["ROE"].loc[me],
        "自己資本比率": G["自己資本比率"].loc[me],
        "営業利益率": G["営業利益率"].loc[me],
        "配当性向の低さ": -G["配当性向"].loc[me],
        "【対照】配当利回り": F["yield"].loc[me],
        "【対照】12ヶ月モメンタム": F["tr_mom12"].loc[me],
    }

    for H, label in ((12, "先12ヶ月"), (6, "先6ヶ月")):
        fwd = trm.shift(-H) / trm - 1
        rows = []
        for k, f in cand.items():
            ics, dec, ns = [], [], []
            for t in me:
                m = liq.loc[t] >= 1e8
                v = pd.concat([f.loc[t], fwd.loc[t]], axis=1, keys=["x", "y"])[m].dropna()
                if len(v) < 80:
                    continue
                ics.append(v.x.corr(v.y, method="spearman"))
                ns.append(len(v))
                q = pd.qcut(v.x, 5, labels=False, duplicates="drop")
                g = v.y.groupby(q).mean()
                if len(g) == 5:
                    dec.append(g.values)
            if not ics:
                continue
            dec = np.array(dec)
            rows.append(dict(指標=k, IC=np.mean(ics),
                             t値=np.mean(ics) / (np.std(ics) / np.sqrt(len(ics))) if len(ics) > 2 else np.nan,
                             月数=len(ics), 平均銘柄数=int(np.mean(ns)),
                             最下位20=dec[:, 0].mean() if dec.ndim == 2 else np.nan,
                             最上位20=dec[:, 4].mean() if dec.ndim == 2 else np.nan,
                             上下差=(dec[:, 4] - dec[:, 0]).mean() if dec.ndim == 2 else np.nan))
        d = pd.DataFrame(rows).sort_values("IC", ascending=False)
        for c in ["IC", "最下位20", "最上位20", "上下差"]:
            d[c] = (d[c] * 100).round(2)
        pd.set_option("display.width", 220)
        print(f"\n=== {label}リターンへの予測力（{me[0].date()}〜、流動性1億円以上）===")
        print(d.round(2).to_string(index=False))
        d.to_csv(os.path.join(OUT, f"fund_ic_{H}m.csv"), encoding="utf-8-sig")

    # ---------- 実際にポートフォリオを組んで比べる ----------
    print("\n=== ファンダ条件を足すと成績は良くなるか（2023/7〜2026/8）===", flush=True)
    START = pd.Timestamp("2023-07-01")
    mkt = (1 + P["ret"].mean(axis=1).fillna(0)).cumprod()
    rows = []

    def add(cur, name):
        c = cur.loc[START:]
        c = c / c.iloc[0]
        s = bt.stats(c, name)
        rows.append(s)

    add(mkt, "【比較】市場平均")
    for k, v in bench.all_curves(idx).items():
        add(v, "【比較】" + k)

    base = dict(n=20, min_yield=0.02, max_per_sector=3, max_vol=0.6,
                require_uptrend=True, stop_loss=0.15,
                use_trend_exit=True, use_divcut_exit=True, start=START)
    cur, _ = active.run(P, F, **base)
    add(cur, "現行ルール（配当のみ）")

    for name, g, mode in [
        ("＋PBR割安 上位半分", G["純資産倍率の逆数"], "half"),
        ("＋PER割安 上位半分", G["益回り"], "half"),
        ("＋ROE 上位半分", G["ROE"], "half"),
        ("＋配当性向が低い 上位半分", -G["配当性向"], "half"),
        ("＋自己資本比率 上位半分", G["自己資本比率"], "half"),
    ]:
        cur, _ = active.run(P, F, extra_filter=(g, mode), **base)
        add(cur, name)

    d = pd.DataFrame(rows).set_index("戦略")
    d = (d * 100).round(1)
    print(d.to_string())
    d.to_csv(os.path.join(OUT, "fund_portfolio.csv"), encoding="utf-8-sig")


if __name__ == "__main__":
    main()
