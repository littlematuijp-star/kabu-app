"""
公開用にデータをまとめる。

data/raw の1銘柄1CSV(146MB)と data/fin の決算CSVを、
少数のparquetファイル(合計20MB程度)に圧縮する。
GitHubに載せてStreamlit Community Cloudで動かすために必要。

生成物: data/packed/*.parquet
  px / div / vol / splits … 未調整株価・配当・出来高・分割
  fin                     … 決算数値（銘柄×決算期）
  bench_*                 … 比較用ETF
"""
import os, glob
import numpy as np
import pandas as pd

import factors
import fund_panel

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(HERE, "data", "packed")


def main():
    os.makedirs(PACK, exist_ok=True)
    total = 0

    # ---- 株価まわり ----
    P = factors.load_panel(_allow_packed=False)
    for k in ["px", "div", "vol", "splits"]:
        f = os.path.join(PACK, f"{k}.parquet")
        P[k].astype("float32").to_parquet(f, compression="zstd")
        total += os.path.getsize(f)
        print(f"  {k:7s} {P[k].shape}  {os.path.getsize(f)/1e6:.1f} MB")

    # ---- 決算 ----
    st = fund_panel.load_statements(_allow_packed=False)
    rows = []
    for code, recs in st.items():
        for fy, v in recs.items():
            rows.append(dict(code=code, fy=fy, **v))
    fin = pd.DataFrame(rows)
    f = os.path.join(PACK, "fin.parquet")
    fin.to_parquet(f, compression="zstd", index=False)
    total += os.path.getsize(f)
    print(f"  fin     {fin.shape}  {os.path.getsize(f)/1e6:.1f} MB")

    # ---- 比較用ETF ----
    bd = os.path.join(HERE, "data", "bench")
    parts = []
    for g in sorted(glob.glob(os.path.join(bd, "*.csv"))):
        code = os.path.basename(g)[:-4]
        d = pd.read_csv(g, parse_dates=["Date"])
        d["code"] = code
        parts.append(d)
    if parts:
        b = pd.concat(parts, ignore_index=True)
        f = os.path.join(PACK, "bench.parquet")
        b.to_parquet(f, compression="zstd", index=False)
        total += os.path.getsize(f)
        print(f"  bench   {b.shape}  {os.path.getsize(f)/1e6:.1f} MB")

    # ---- 現時点のファンダ（表示用）----
    src = os.path.join(HERE, "data", "fundamentals.csv")
    if os.path.exists(src):
        d = pd.read_csv(src, dtype={"code": str})
        f = os.path.join(PACK, "fundamentals.parquet")
        d.to_parquet(f, compression="zstd", index=False)
        total += os.path.getsize(f)
        print(f"  現ファンダ {d.shape}  {os.path.getsize(f)/1e6:.1f} MB")

    # ---- 日本語の銘柄名・業種（東証公式）----
    import jp_names
    try:
        jp_names.save()
        f = os.path.join(PACK, "names.parquet")
        total += os.path.getsize(f)
        print(f"  銘柄名   {os.path.getsize(f)/1e6:.1f} MB")
    except Exception as e:
        print(f"  銘柄名の取得に失敗（既存を維持）: {type(e).__name__}")

    print(f"合計 {total/1e6:.1f} MB  → {PACK}")


if __name__ == "__main__":
    main()
