"""
日本語の銘柄名と業種を、東証(JPX)公式の上場銘柄一覧から取得する。

yfinance の銘柄名は英語(Neturen Co., Ltd.)で、業種も
海外基準(Basic Materials)のため日本株には馴染まない。
JPXが無料公開している一覧なら、日本語名と33業種区分が正確に取れる。
認証不要・月1回更新。
"""
import certfix  # noqa
import os
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(HERE, "data", "packed")
URL = ("https://www.jpx.co.jp/markets/statistics-equities/misc/"
       "tvdivq0000001vg2-att/data_j.xls")


def fetch() -> pd.DataFrame:
    r = requests.get(URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    tmp = os.path.join(HERE, "data", "data_j.xls")
    with open(tmp, "wb") as f:
        f.write(r.content)
    d = pd.read_excel(tmp, dtype=str)
    d = d.rename(columns={"コード": "code", "銘柄名": "銘柄名",
                          "33業種区分": "業種", "規模区分": "規模",
                          "市場・商品区分": "市場"})
    d = d[["code", "銘柄名", "業種", "規模", "市場"]].copy()
    d["code"] = d["code"].str.strip()
    # ETF・ETN・REITなど業種が付かないものは除く
    d = d[d["業種"].notna() & (d["業種"] != "-")]
    os.remove(tmp)
    return d.drop_duplicates(subset="code")


def save():
    d = fetch()
    os.makedirs(PACK, exist_ok=True)
    out = os.path.join(PACK, "names.parquet")
    d.to_parquet(out, compression="zstd", index=False)
    print(f"保存: {out}  {len(d)}銘柄")
    return d


def load() -> pd.DataFrame:
    f = os.path.join(PACK, "names.parquet")
    if not os.path.exists(f):
        return pd.DataFrame()
    d = pd.read_parquet(f)
    d["code"] = d["code"].astype(str)
    return d.set_index("code")


if __name__ == "__main__":
    d = save()
    print(d.head(5).to_string(index=False))
