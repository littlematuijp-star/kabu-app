"""
1年勝負＋途中入替エンジン。

「買って1年放置」ではなく、毎月保有を点検して、
崩れた銘柄を売り、候補上位と入れ替える運用を再現する。

売却条件（どれか1つでも該当したら売る）:
  A 損切り   … 買値から -X%
  B トレンド割れ … 200日移動平均を下回った（業績・減配の悪化は株価に先に出やすい）
  C 減配     … 過去12ヶ月の配当が1年前を下回った
  D 保有期限  … 一定月数を超えたら、候補として選び直す

売った枠は、その時点の候補上位から補充する。
※すべて「その日までの情報」だけで判定している。
"""
import numpy as np
import pandas as pd

import backtest as bt

COST = 0.003   # 売買往復コスト（手数料+スプレッド）


def month_ends(index):
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(sorted(s.groupby([index.year, index.month]).last()))


def run(P, F, n=25, stop_loss=0.15, use_trend_exit=True, use_divcut_exit=True,
        max_hold_months=12, start=None, **select_kw):
    """
    毎月末に点検して入替。等金額配分。
    戻り値: (資産曲線, 売買記録, 保有記録)
    """
    tr, px = P["tr"], P["px"]
    idx = tr.index
    me = month_ends(idx)
    me = me[me >= (start or idx[0] + pd.Timedelta(days=400))]

    ttm = F["yield"] * px          # 過去12ヶ月配当額（円）
    ttm_yoy = ttm / ttm.shift(252)  # 1年前との比

    cash = 1.0
    pos = {}       # code -> dict(shares_value_frac, entry_price, entry_date)
    curve = pd.Series(dtype=float)
    trades = []
    val = 1.0
    prev_t = me[0]

    # 保有比率で管理する（等金額 → 以後は値動きでドリフト）
    weights = {}   # code -> 現在の評価額

    ma200_all = px.rolling(200).mean()
    for i, t in enumerate(me):
        # --- 前月末→今月末の値動きを反映 ---
        if i > 0:
            seg = tr.loc[prev_t:t]
            for c in list(weights):
                s = tr[c].loc[prev_t:t].ffill()
                if s.isna().all() or s.iloc[0] == 0:
                    continue
                weights[c] *= float(s.iloc[-1] / s.iloc[0])
            # 日次曲線
            if weights or cash > 0:
                tot0 = sum(weights.values()) + cash
                sub = tr[list(weights)].loc[prev_t:t].ffill() if weights else None
                if sub is not None and len(sub) > 1:
                    w0 = pd.Series({c: weights[c] / float(sub[c].iloc[-1] / sub[c].iloc[0])
                                    for c in weights})
                    path = (sub / sub.iloc[0]).mul(w0, axis=1).sum(axis=1) + cash
                else:
                    path = pd.Series(sum(weights.values()) + cash, index=seg.index)
                curve = pd.concat([curve, path.iloc[1:]])
        else:
            curve = pd.concat([curve, pd.Series(1.0, index=[t])])

        equity = sum(weights.values()) + cash

        # --- 売却判定 ---
        p_now = bt.at(px, t)
        ma200 = bt.at(ma200_all, t)
        yoy = bt.at(ttm_yoy, t)
        for c in list(weights):
            info = pos[c]
            reasons = []
            pr = p_now.get(c, np.nan)
            if pd.notna(pr) and pd.notna(info["entry_price"]):
                chg = pr / info["entry_price"] - 1
                if chg <= -stop_loss:
                    reasons.append(f"損切り({chg:.0%})")
            if use_trend_exit and pd.notna(pr) and pd.notna(ma200.get(c, np.nan)) and pr < ma200[c]:
                reasons.append("200日線割れ")
            if use_divcut_exit and pd.notna(yoy.get(c, np.nan)) and yoy[c] < 0.999:
                reasons.append("減配")
            months = (t.year - info["entry_date"].year) * 12 + (t.month - info["entry_date"].month)
            if months >= max_hold_months:
                reasons.append("保有期限")
            if reasons:
                cash += weights[c] * (1 - COST)
                trades.append(dict(date=t, code=c, action="売", 理由="/".join(reasons),
                                   損益=float(pr / info["entry_price"] - 1) if pd.notna(pr) else np.nan))
                del weights[c]; del pos[c]

        # --- 補充 ---
        empty = n - len(weights)
        if empty > 0 and cash > 1e-9:
            cand = bt.select(F, t, n=n * 3, **select_kw)
            cand = [c for c in cand if c not in weights][:empty]
            if cand:
                per = cash / len(cand)
                for c in cand:
                    weights[c] = per * (1 - COST)
                    pos[c] = dict(entry_price=float(p_now.get(c, np.nan)), entry_date=t)
                    trades.append(dict(date=t, code=c, action="買", 理由="候補上位", 損益=np.nan))
                cash = 0.0
        prev_t = t

    curve = curve[~curve.index.duplicated()].sort_index()
    return curve, pd.DataFrame(trades)
