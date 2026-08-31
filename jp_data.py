"""
生株価(未調整)＋配当履歴のダウンロードとキャッシュ。

長期・配当戦略では「その時点の配当利回り」が必要になる。
調整後株価だけでは過去の利回りを復元できないので、
未調整のClose と 配当の実支払い額 を保存する。
中断しても再開できる(1銘柄1ファイル)。
"""
import certfix  # noqa  証明書設定。触らないでOK
import os, time, glob, sys
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
os.makedirs(RAW, exist_ok=True)

# 既存プロジェクトのクリーンなユニバースを流用
# 銘柄リストは同じフォルダに置いたものを優先（公開時に外部フォルダへ依存しないため）
_LOCAL = os.path.join(HERE, "tickers_clean.txt")
UNIVERSE_FILE = _LOCAL if os.path.exists(_LOCAL) else os.path.join(
    os.path.dirname(HERE), "株価予測", "tickers_clean.txt")


def is_reit(code: str) -> bool:
    """J-REIT・インフラファンドのコード帯。株式とは値動きの性質が違うので長期株式戦略からは外す。"""
    try:
        n = int(code)
    except ValueError:
        return False
    return (2971 <= n <= 2999) or (3226 <= n <= 3299) or (3451 <= n <= 3499) \
        or (8951 <= n <= 8999) or (9281 <= n <= 9286)


def load_universe(exclude_reit: bool = True) -> list:
    codes = []
    for line in open(UNIVERSE_FILE, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        c = line.split()[0]
        if exclude_reit and is_reit(c):
            continue
        codes.append(c)
    return codes


def path_of(code: str) -> str:
    return os.path.join(RAW, f"{code}.csv")


def fetch_one(code: str, period: str = "10y") -> pd.DataFrame:
    """1銘柄の 未調整OHLCV + Dividends + Stock Splits を取得。"""
    h = yf.Ticker(f"{code}.T").history(period=period, auto_adjust=False)
    if h is None or len(h) == 0:
        return pd.DataFrame()
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"] if c in h.columns]
    h = h[cols].copy()
    idx = pd.to_datetime(h.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    h.index = idx
    h.index.name = "Date"
    return h


def download_all(codes=None, sleep=0.15, refresh=False):
    codes = codes or load_universe()
    todo = [c for c in codes if refresh or not os.path.exists(path_of(c))]
    print(f"対象 {len(codes)} 銘柄 / 未取得 {len(todo)} 銘柄", flush=True)
    ok = ng = 0
    for i, c in enumerate(todo, 1):
        try:
            df = fetch_one(c)
            if df.empty:
                ng += 1
            else:
                df.to_csv(path_of(c))
                ok += 1
        except Exception as e:
            ng += 1
            print(f"  失敗 {c}: {type(e).__name__}", flush=True)
        if i % 50 == 0:
            print(f"  {i}/{len(todo)} 成功{ok} 失敗{ng}", flush=True)
        time.sleep(sleep)
    print(f"完了: 成功{ok} 失敗{ng} / 保存済み合計 {len(glob.glob(os.path.join(RAW,'*.csv')))}", flush=True)


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    download_all(refresh=refresh)
