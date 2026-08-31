"""
長期・配当戦略のための指標計算。

「その時点で分かっていた情報だけ」で作ることを徹底する。
未来の情報が1つでも混ざるとバックテストは意味を失う
（例: 分割調整後の株価水準を使うと、将来分割する銘柄=将来上がる銘柄が
　　　安く見えて、予測力があるように錯覚する）。
"""
import os, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")


# ---------------------------------------------------------------- 読み込み
PACK = os.path.join(HERE, "data", "packed")


def _load_packed():
    """公開用に圧縮したparquetから読む（data/raw が無い環境用）。"""
    px = pd.read_parquet(os.path.join(PACK, "px.parquet")).astype("float64")
    div = pd.read_parquet(os.path.join(PACK, "div.parquet")).astype("float64")
    vol = pd.read_parquet(os.path.join(PACK, "vol.parquet")).astype("float64")
    sp = pd.read_parquet(os.path.join(PACK, "splits.parquet")).astype("float64")
    return px, div, vol, sp


def load_panel(min_days: int = 1000, _allow_packed: bool = True):
    """
    保存済みCSVから3つのパネル(日付×銘柄)を作る。

    px   : 未調整終値（その日に実際についていた株価。配当利回りの分母）
    tr   : トータルリターン指数（配当を再投資した場合の資産推移。成績計算用）
    div  : その日の1株配当（円）
    vol  : 出来高
    """
    # 圧縮済みがあればそれを使う（クラウド公開時は data/raw を持たない）
    if _allow_packed and os.path.exists(os.path.join(PACK, "px.parquet")):
        px, div, vol, sp = _load_packed()
        return _finish(px, div, vol, sp)

    px, div, vol, splits = {}, {}, {}, {}
    for f in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
        code = os.path.basename(f)[:-4]
        df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        if len(df) < min_days:
            continue
        px[code] = df["Close"]
        div[code] = df.get("Dividends", pd.Series(0.0, index=df.index)).fillna(0.0)
        vol[code] = df["Volume"]
        splits[code] = df.get("Stock Splits", pd.Series(0.0, index=df.index)).fillna(0.0)
    px = pd.DataFrame(px).sort_index()
    div = pd.DataFrame(div).reindex(px.index).fillna(0.0)
    vol = pd.DataFrame(vol).reindex(px.index)
    sp = pd.DataFrame(splits).reindex(px.index).fillna(0.0)
    return _finish(px, div, vol, sp)


def _finish(px, div, vol, sp):
    """トータルリターン指数などの派生値を作る。"""
    # 分割を考慮した日次トータルリターン
    #   分割日は Stock Splits=n (1株がn株に) なので、前日終値をnで割って比較する
    ratio = sp.replace(0.0, np.nan).fillna(1.0)
    prev = px.shift(1) / ratio
    ret = (px + div) / prev - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)
    # 1日で±60%超は異常値(データ不良)とみなして除外
    ret = ret.where(ret.abs() < 0.6)
    tr = (1 + ret.fillna(0.0)).cumprod()
    tr = tr.where(px.notna())
    return dict(px=px, tr=tr, div=div, vol=vol, ret=ret, splits=sp)


# ---------------------------------------------------------------- 指標
def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """各月の最終営業日。"""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(sorted(s.groupby([index.year, index.month]).last()))


def ttm_dividend(div: pd.DataFrame) -> pd.DataFrame:
    """過去12ヶ月に実際に支払われた1株配当の合計（実績ベース。予想は使わない）。"""
    return div.rolling("365D").sum()


def compute(P: dict) -> dict:
    """全指標を日次で計算して辞書で返す。"""
    px, tr, div, vol, ret = P["px"], P["tr"], P["div"], P["vol"], P["ret"]
    F = {}

    ttm = ttm_dividend(div)
    F["yield"] = ttm / px                       # 配当利回り(実績)

    # 増配トレンド: 直近12ヶ月配当 ÷ 3年前の12ヶ月配当（年率換算）
    prev3 = ttm.shift(756)
    F["div_growth"] = (ttm / prev3.replace(0, np.nan)) ** (1 / 3) - 1

    # 減配回数: 過去5年で、12ヶ月配当が1年前を下回った月の数（少ないほど良い）
    ttm_m = ttm.resample("ME").last()
    cut = (ttm_m < ttm_m.shift(12) * 0.999).astype(float)
    F["div_cuts_5y"] = cut.rolling(60, min_periods=24).sum().reindex(px.index).ffill()

    # 無配期間の割合: 過去5年で配当がゼロだった月の割合（低いほど良い）
    F["no_div_ratio"] = (ttm_m <= 0).astype(float).rolling(60, min_periods=24).mean() \
                                    .reindex(px.index).ffill()

    # リスク指標
    F["vol1y"] = ret.rolling(252).std() * np.sqrt(252)
    F["maxdd1y"] = tr / tr.rolling(252).max() - 1

    # 株価トレンド（配当込み）。減配・業績悪化は株価に先に出ることが多い
    F["tr_mom12"] = tr / tr.shift(252) - 1
    F["tr_mom3"] = tr / tr.shift(63) - 1
    F["above200"] = (px > px.rolling(200).mean()).astype(float)

    # 流動性: 平均売買代金(円/日, 60日)
    F["turnover"] = (px * vol).rolling(60).mean()

    return F
