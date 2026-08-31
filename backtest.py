"""
長期・配当戦略のバックテスト。

・銘柄選択は「その日までの情報」のみ
・成績は配当込みトータルリターン
・売買コストは入替1銘柄あたり往復0.3%を差し引く
・年1回など低頻度リバランス（長期保有前提）
"""
import numpy as np
import pandas as pd

COST_PER_SWAP = 0.003   # 入替1銘柄あたり往復コスト(手数料+スプレッド)


# ------------------------------------------------------------------ 補助
def sector_of(code: str) -> int:
    """
    証券コードの百の位を業種の代理として使う。

    千の位で切ると 8000番台（商社・銀行・保険・不動産）が丸ごと1業種になり、
    日本の高配当投資の中心である商社と銀行を同時に持てなくなる。
    百の位なら 8000商社 / 8300銀行 / 8600証券 / 8700保険 / 8800不動産 が分かれる。
    """
    return int(code) // 100


_FFILL_CACHE = {}


def at(df: pd.DataFrame, t) -> pd.Series:
    """
    時点 t での各銘柄の値。t 以前の最後の有効値を使う（銘柄ごとに独立に前方補完）。

    前方補完はデータ全体に対して一度だけ行い、以後は使い回す。
    毎回 ffill すると 2,463日×1,057銘柄 の補完を何千回も繰り返すことになり、
    バックテストが数十分単位で遅くなる。
    """
    key = id(df)
    hit = _FFILL_CACHE.get(key)
    if hit is None or hit[0] is not df or hit[1].shape != df.shape:
        _FFILL_CACHE[key] = (df, df.ffill())
        hit = _FFILL_CACHE[key]
    f = hit[1]
    pos = f.index.searchsorted(t, side="right") - 1
    if pos < 0:
        return pd.Series(dtype=float)
    return f.iloc[pos]


def zscore(s: pd.Series) -> pd.Series:
    """順位を0〜1に直したスコア。外れ値に強い。"""
    return s.rank(pct=True)


# ------------------------------------------------------------------ 銘柄選択
def select(F, t, n=25, min_yield=0.025, max_cuts=6, max_vol=0.45,
           min_turnover=1e8, max_per_sector=4, require_uptrend=True,
           w=(1.0, 0.6, 0.8, 0.5, 0.3), extra_filter=None):
    """
    ある時点 t で保有する銘柄を選ぶ。
    戻り値: 銘柄コードのリスト（空なら現金）
    """
    y = at(F["yield"], t)
    cuts = at(F["div_cuts_5y"], t)
    nodiv = at(F["no_div_ratio"], t)
    grow = at(F["div_growth"], t)
    vol = at(F["vol1y"], t)
    turn = at(F["turnover"], t)
    up = at(F["above200"], t)
    mom = at(F["tr_mom12"], t)

    df = pd.DataFrame(dict(y=y, cuts=cuts, nodiv=nodiv, grow=grow,
                           vol=vol, turn=turn, up=up, mom=mom)).dropna(subset=["y", "vol", "turn"])
    if df.empty:
        return []

    # --- ふるい(必須条件) ---
    m = (df.turn >= min_turnover) & (df.y >= min_yield) & (df.y < 0.15)
    m &= df.cuts.fillna(99) <= max_cuts
    m &= df.nodiv.fillna(1) <= 0.2
    m &= df.vol <= max_vol
    if require_uptrend:
        m &= df.up.fillna(0) > 0
    df = df[m]

    # 追加のふるい（ファンダメンタル指標などを足して検証するための口）
    #   extra_filter = (日次パネル, "half") … その時点の候補内で上位半分だけ残す
    if extra_filter is not None and len(df):
        panel, mode = extra_filter
        v = at(panel, t).reindex(df.index)
        ok = v.notna()
        if ok.sum() >= 10:
            thr = v[ok].quantile(0.5 if mode == "half" else 0.7)
            df = df[ok & (v >= thr)]

    if len(df) < 5:
        return []

    # --- 点数付け ---
    df["score"] = (w[0] * zscore(df.y)
                   + w[1] * zscore(df.grow.fillna(0))
                   + w[2] * zscore(-df.cuts.fillna(0))
                   + w[3] * zscore(-df.vol)
                   + w[4] * zscore(df.mom.fillna(0)))

    # --- 業種分散をかけながら上位から採用 ---
    picked, count = [], {}
    for code in df.sort_values("score", ascending=False).index:
        s = sector_of(code)
        if count.get(s, 0) >= max_per_sector:
            continue
        picked.append(code)
        count[s] = count.get(s, 0) + 1
        if len(picked) >= n:
            break
    return picked


# ------------------------------------------------------------------ 実行
def run(P, F, rebalance_months=12, start=None, **kw):
    """リバランス日ごとに select() で選び、等ウェイトで保有し続ける。"""
    tr = P["tr"]
    idx = tr.index
    me = pd.DatetimeIndex(sorted(pd.Series(idx, index=idx)
                                 .groupby([idx.year, idx.month]).last()))
    me = me[me >= (start or idx[0] + pd.Timedelta(days=400))]
    dates = me[::rebalance_months]

    val, prev = 1.0, set()
    curve = pd.Series(dtype=float)
    log = []
    for i, t in enumerate(dates):
        end = dates[i + 1] if i + 1 < len(dates) else idx[-1]
        pick = select(F, t, **kw)
        seg = tr.loc[t:end]
        if pick:
            sub = seg[pick].ffill()
            p = (sub / sub.iloc[0]).mean(axis=1)
        else:
            p = pd.Series(1.0, index=seg.index)   # 条件を満たす銘柄なし → 現金
        swaps = len(set(pick) - prev)
        cost = COST_PER_SWAP * swaps / max(len(pick), 1) if pick else 0.0
        v = val * (1 - cost) * p / p.iloc[0]
        curve = pd.concat([curve, v.iloc[1:] if i > 0 else v])
        log.append(dict(date=t, n=len(pick), swaps=swaps, codes=pick))
        val = float(v.iloc[-1])
        prev = set(pick)
    return curve[~curve.index.duplicated()], pd.DataFrame(log)


def stats(curve: pd.Series, label="") -> dict:
    c = curve.dropna()
    if len(c) < 260:
        return {}
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    r12 = (c / c.shift(252) - 1).dropna()
    yearly = c.resample("YE").last().pct_change().dropna()
    return {
        "戦略": label,
        "CAGR": (c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1,
        "最大DD": float((c / c.cummax() - 1).min()),
        "12m平均": r12.mean(),
        "12m最悪": r12.min(),
        "12m>=7%": (r12 >= 0.07).mean(),
        "12mプラス": (r12 > 0).mean(),
        "年別最悪": yearly.min(),
        "年別>=7%": (yearly >= 0.07).mean(),
    }
