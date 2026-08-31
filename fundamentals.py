"""
現時点のファンダメンタル指標を yfinance から取得してキャッシュする。

【重要な限界】
ここで取れるのは「今の」PER/PBR/ROE/配当性向などのスナップショットで、
過去の値は取れない。つまりこれらは過去データで検証できない。
→ バックテストでは使わず、実運用の「最終安全チェック」としてのみ使う。
   検証済みの指標（配当利回り・減配歴・増配・ボラ）が主、ここは従。
"""
import certfix  # noqa
import os, json, time, sys
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "fundamentals.csv")

KEYS = {
    "longName": "銘柄名",
    "sector": "セクター",
    "industry": "業種",
    "marketCap": "時価総額",
    "trailingPE": "PER実績",
    "forwardPE": "PER予想",
    "priceToBook": "PBR",
    "returnOnEquity": "ROE",
    "profitMargins": "純利益率",
    "operatingMargins": "営業利益率",
    "dividendYield": "配当利回り会社",
    "payoutRatio": "配当性向",
    "debtToEquity": "DEレシオ",
    "currentRatio": "流動比率",
    "revenueGrowth": "売上成長",
    "earningsGrowth": "利益成長",
    "freeCashflow": "FCF",
    "totalCash": "現金",
    "totalDebt": "有利子負債",
}


def fetch(codes, sleep=0.2, resume=True):
    have = {}
    if resume and os.path.exists(OUT):
        old = pd.read_csv(OUT, dtype={"code": str}).set_index("code")
        have = old.to_dict("index")
    todo = [c for c in codes if c not in have]
    print(f"対象 {len(codes)} / 未取得 {len(todo)}", flush=True)
    for i, c in enumerate(todo, 1):
        try:
            info = yf.Ticker(f"{c}.T").info or {}
            have[c] = {jp: info.get(k) for k, jp in KEYS.items()}
        except Exception:
            have[c] = {jp: None for jp in KEYS.values()}
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
            pd.DataFrame(have).T.rename_axis("code").to_csv(OUT, encoding="utf-8-sig")
        time.sleep(sleep)
    df = pd.DataFrame(have).T.rename_axis("code")
    df.to_csv(OUT, encoding="utf-8-sig")
    print("保存:", OUT, df.shape, flush=True)
    return df


if __name__ == "__main__":
    import jp_data
    fetch(jp_data.load_universe())
