"""
保有銘柄の点検。検証した売却ルールを、今日の保有にそのまま当てはめる。

売却条件（バックテストと完全に同じ）:
  A 損切り    … 買値から -15%（窓を空けた急落用の保険。通常はBが先に出る）
  B トレンド割れ … 株価が200日移動平均を下回った  ← 実際に効いているのはこれ
  C 減配     … 過去12ヶ月の配当が1年前を下回った
  D 保有期限  … 12ヶ月経過（候補として選び直す）
"""
import numpy as np
import pandas as pd

import factors
import backtest as bt


def check(P, F, holdings: pd.DataFrame) -> pd.DataFrame:
    """
    holdings: コード / 買値(任意) / 買った日(任意) の列を持つ DataFrame
    """
    t = P["px"].index[-1]
    px = bt.at(P["px"], t)
    ma = bt.at(P["px"].rolling(200).mean(), t)
    ttm = F["yield"] * P["px"]
    yoy = bt.at(ttm / ttm.shift(252), t)
    y = bt.at(F["yield"], t)

    out = []
    for _, h in holdings.iterrows():
        c = str(h["コード"]).strip()
        if c not in P["px"].columns:
            out.append(dict(コード=c, 判定="データなし")); continue
        p = float(px[c]); m = float(ma.get(c, np.nan))
        sell, keep = [], []

        if pd.notna(h.get("買値")) and h.get("買値"):
            chg = p / float(h["買値"]) - 1
            if chg <= -0.15:
                sell.append(f"A 損切り: 買値から {chg:+.1%}")
            else:
                keep.append(f"買値から {chg:+.1%}")
        if pd.notna(m):
            if p < m:
                sell.append(f"B トレンド割れ: 株価 {p:,.0f}円 < 200日線 {m:,.0f}円")
            else:
                keep.append(f"200日線 {m:,.0f}円 より {p/m-1:+.1%} 上")
        if pd.notna(yoy.get(c, np.nan)):
            if yoy[c] < 0.999:
                sell.append(f"C 減配: 年間配当が1年前の {yoy[c]:.0%}")
            else:
                keep.append(f"配当は1年前の {yoy[c]:.0%}（維持〜増配）")
        if pd.notna(h.get("買った日")) and str(h.get("買った日")).strip():
            try:
                d0 = pd.Timestamp(h["買った日"])
                months = (t.year - d0.year) * 12 + (t.month - d0.month)
                if months >= 12:
                    sell.append(f"D 保有期限: {months}ヶ月経過。候補として選び直す")
                else:
                    keep.append(f"保有 {months}ヶ月")
            except Exception:
                pass

        out.append(dict(
            コード=c, 株価=round(p, 1), 配当利回り=float(y.get(c, np.nan)),
            判定="売却" if sell else "継続",
            売却理由=" / ".join(sell), 継続根拠=" / ".join(keep),
            損切りライン=round(float(h["買値"]) * 0.85, 1) if h.get("買値") else np.nan,
            トレンド線=round(m, 1) if pd.notna(m) else np.nan,
        ))
    return pd.DataFrame(out)
