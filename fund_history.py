"""
過去の決算数値（年次）を取得する。PBR/PER/ROE/配当性向を検証するために必要。

yfinance から取れるのは年次4〜5期分だけ。10年は取れない。
それでも「今の値」しか無い状態とは決定的に違い、
2023年以降については本物の時点データで検証できる。
中断しても再開できる（1銘柄1ファイル）。
"""
import certfix  # noqa
import os, sys, time, glob
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
FIN = os.path.join(HERE, "data", "fin")
os.makedirs(FIN, exist_ok=True)

# 使う勘定科目（yfinance の英語名）
BS = ["Stockholders Equity", "Total Equity Gross Minority Interest",
      "Ordinary Shares Number", "Share Issued", "Total Assets", "Total Debt"]
IS = ["Net Income", "Net Income Common Stockholders", "Total Revenue",
      "Operating Income", "Basic EPS"]


def fetch_one(code: str) -> pd.DataFrame:
    tk = yf.Ticker(f"{code}.T")
    bs = tk.balance_sheet
    inc = tk.income_stmt
    if (bs is None or bs.empty) and (inc is None or inc.empty):
        return pd.DataFrame()
    parts = []
    if bs is not None and not bs.empty:
        parts.append(bs.reindex(BS))
    if inc is not None and not inc.empty:
        parts.append(inc.reindex(IS))
    df = pd.concat(parts)
    df = df.loc[:, sorted(df.columns)]          # 決算期を古い順に
    df.columns = [pd.Timestamp(c).date().isoformat() for c in df.columns]
    return df


def download_all(codes, sleep=0.1, refresh=False):
    todo = [c for c in codes if refresh or not os.path.exists(os.path.join(FIN, f"{c}.csv"))]
    print(f"対象 {len(codes)} 銘柄 / 未取得 {len(todo)} 銘柄", flush=True)
    ok = ng = 0
    for i, c in enumerate(todo, 1):
        try:
            df = fetch_one(c)
            if df.empty:
                ng += 1
                open(os.path.join(FIN, f"{c}.csv"), "w").close()   # 空でも印を残す
            else:
                df.to_csv(os.path.join(FIN, f"{c}.csv"), encoding="utf-8")
                ok += 1
        except Exception as e:
            ng += 1
            print(f"  失敗 {c}: {type(e).__name__}", flush=True)
        if i % 25 == 0:
            print(f"  {i}/{len(todo)} 成功{ok} 失敗{ng}", flush=True)
        time.sleep(sleep)
    print(f"完了: 成功{ok} 失敗{ng}", flush=True)


if __name__ == "__main__":
    import jp_data
    download_all(jp_data.load_universe(), refresh="--refresh" in sys.argv)
