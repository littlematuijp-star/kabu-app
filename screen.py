"""
今日の買い候補を選び出す。

バックテストで検証した select() をそのまま今日の日付で呼ぶ。
「検証したルールと、実際に買うルールが同じ」であることが一番大事
（別の目的変数で学習していて再現しなかった、という失敗を繰り返さないため）。
"""
import os
import numpy as np
import pandas as pd

import factors
import backtest as bt

HERE = os.path.dirname(os.path.abspath(__file__))
FUND = os.path.join(HERE, "data", "fundamentals.csv")


def build(n=25, **kw):
    P = factors.load_panel()
    F = factors.compute(P)
    t = P["px"].index[-1]
    codes = bt.select(F, t, n=n, **kw)

    rows = []
    for c in codes:
        rows.append(dict(
            コード=c,
            株価=round(float(bt.at(P["px"], t)[c]), 1),
            配当利回り=float(bt.at(F["yield"], t)[c]),
            年間配当=round(float(bt.at(factors.ttm_dividend(P["div"]), t)[c]), 1),
            増配率3年=float(bt.at(F["div_growth"], t).get(c, np.nan)),
            減配月数5年=float(bt.at(F["div_cuts_5y"], t).get(c, np.nan)),
            ボラティリティ=float(bt.at(F["vol1y"], t)[c]),
            year1リターン=float(bt.at(F["tr_mom12"], t).get(c, np.nan)),
            売買代金=float(bt.at(F["turnover"], t)[c]),
        ))
    df = pd.DataFrame(rows)

    # 補助情報（検証していない。最終チェック用）
    if os.path.exists(FUND) and not df.empty:
        f = pd.read_csv(FUND, dtype={"code": str}).set_index("code")
        for col in ["銘柄名", "セクター", "PBR", "PER実績", "ROE", "配当性向", "DEレシオ", "時価総額"]:
            if col in f.columns:
                df[col] = df["コード"].map(f[col])
    return df, t


def warnings_for(row) -> list:
    """買う前に人間が見るべき注意点。機械的に切らず、警告として出す。"""
    w = []
    if pd.notna(row.get("配当性向")) and row["配当性向"] and row["配当性向"] > 0.8:
        w.append(f"配当性向が高い({row['配当性向']:.0%})→減配余地")
    if pd.notna(row.get("DEレシオ")) and row["DEレシオ"] and row["DEレシオ"] > 200:
        w.append(f"負債が重い(D/E {row['DEレシオ']:.0f}%)")
    if row.get("減配月数5年", 0) and row["減配月数5年"] > 0:
        w.append(f"過去5年に減配歴あり({int(row['減配月数5年'])}ヶ月)")
    if pd.notna(row.get("増配率3年")) and row["増配率3年"] < 0:
        w.append("3年で減配トレンド")
    if row.get("配当利回り", 0) > 0.06:
        w.append("利回りが異常に高い→株価下落/特別配当の可能性")
    return w


if __name__ == "__main__":
    df, t = build()
    pd.set_option("display.width", 250)
    print(f"基準日 {t.date()}  候補 {len(df)}銘柄")
    show = df.copy()
    for c in ["配当利回り", "増配率3年", "ボラティリティ", "year1リターン"]:
        show[c] = (show[c] * 100).round(1)
    show["売買代金"] = (show["売買代金"] / 1e8).round(1)
    print(show.to_string(index=False))
    out = os.path.join(HERE, "data", "out", "candidates.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("保存:", out)
