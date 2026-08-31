"""
データの増分更新。GitHub Actions から毎日実行される想定。

全銘柄を最初から取り直すと30分かかるが、
既に10年分を持っているので「最後の日付以降」だけ取れば1〜2分で済む。

  python update_data.py            … 株価・配当の増分更新（毎日用）
  python update_data.py --full     … 決算・銘柄情報も更新（月1回用）
"""
import certfix  # noqa
import os, sys, time, datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

import jp_data

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(HERE, "data", "packed")

OVERLAP_DAYS = 10        # 直近この日数は取り直す（確定値の修正・配当の後追い反映に備える）
BATCH = 150              # 一度にYahooへ投げる銘柄数


def _read(name):
    # 保存はfloat32だが、更新中はfloat64で扱う（float32のまま代入すると型エラーになる）
    return pd.read_parquet(os.path.join(PACK, f"{name}.parquet")).astype("float64")


def _write(name, df):
    df.astype("float32").to_parquet(os.path.join(PACK, f"{name}.parquet"), compression="zstd")


def fetch_recent(codes, start):
    """複数銘柄をまとめて取得し、銘柄×日付の表4つに分解する。"""
    out = {"Close": {}, "Dividends": {}, "Volume": {}, "Stock Splits": {}}
    for i in range(0, len(codes), BATCH):
        chunk = codes[i:i + BATCH]
        tk = [c + ".T" for c in chunk]
        try:
            d = yf.download(tk, start=start.strftime("%Y-%m-%d"), interval="1d",
                            group_by="ticker", auto_adjust=False, actions=True,
                            threads=True, progress=False)
        except Exception as e:
            print(f"  取得失敗 {chunk[0]}〜: {type(e).__name__}", flush=True)
            continue
        if d is None or len(d) == 0:
            continue
        for c in chunk:
            key = c + ".T"
            if key not in d.columns.get_level_values(0):
                continue
            sub = d[key]
            for field in out:
                if field in sub.columns:
                    s = sub[field].dropna()
                    if len(s):
                        out[field][c] = s
        print(f"  {min(i+BATCH, len(codes))}/{len(codes)}", flush=True)
    return {k: pd.DataFrame(v) for k, v in out.items()}


def update_prices():
    px = _read("px"); div = _read("div"); vol = _read("vol"); sp = _read("splits")
    last = px.index.max()
    start = last - pd.Timedelta(days=OVERLAP_DAYS)
    print(f"保有データ: {px.index.min().date()} 〜 {last.date()}  {px.shape[1]}銘柄")
    print(f"{start.date()} 以降を取得します…", flush=True)

    codes = list(px.columns)
    t0 = time.time()
    new = fetch_recent(codes, start)
    print(f"取得完了 {time.time()-t0:.0f}秒")

    if new["Close"].empty:
        print("新しいデータがありません。終了します。")
        return False

    def merge(old, add, fill0=False):
        if add.empty:
            return old
        add = add.reindex(columns=old.columns).astype("float64")
        # 日付のタイムゾーンを外して既存データと揃える
        ai = pd.to_datetime(add.index)
        if getattr(ai, "tz", None) is not None:
            ai = ai.tz_localize(None)
        add.index = ai.normalize()
        idx = old.index.union(add.index)
        out = old.reindex(idx).astype("float64")
        # 重なる期間は新しい値で上書きする（Yahoo側の訂正を取り込む）。
        # ただし update() は「新しい側がNaNのセル」を書き換えない。
        # これが重要で、1銘柄でも取得に失敗したとき、その銘柄の
        # 既存データを空で潰してしまうのを防ぐ。
        out.update(add)
        return out.fillna(0.0) if fill0 else out

    px2 = merge(px, new["Close"])
    div2 = merge(div, new["Dividends"], fill0=True)
    vol2 = merge(vol, new["Volume"])
    sp2 = merge(sp, new["Stock Splits"], fill0=True)

    added = len(px2.index) - len(px.index)
    print(f"営業日 {len(px.index)} → {len(px2.index)}  (+{added}日)")
    print(f"最新日: {px2.index.max().date()}")
    if added == 0:
        print("新しい営業日はありませんでした（直近分の訂正のみ反映）。")

    # ---- 安全装置 ----
    # 更新で株価の欠損が増えるのは異常（Yahoo側の不調で既存データを
    # 壊しかけている）。壊れたデータをクラウドに送らないため、ここで止める。
    lost = int(px2.reindex(px.index).isna().sum().sum()) - int(px.isna().sum().sum())
    if lost > 0:
        print(f"[中止] 既存期間の株価欠損が {lost} 件増えました。保存せずに終了します。")
        return False
    keep = int((px2.iloc[-1].notna()).sum())
    if keep < px2.shape[1] * 0.8:
        print(f"[中止] 最新日に値のある銘柄が {keep}/{px2.shape[1]} しかありません。保存せずに終了します。")
        return False

    for name, d in [("px", px2), ("div", div2), ("vol", vol2), ("splits", sp2)]:
        _write(name, d)
    print("保存しました:", PACK)
    return True


def update_financials():
    """決算・銘柄情報の更新（月1回で十分）。"""
    import fund_history, fundamentals, pack_data
    codes = [c for c in jp_data.load_universe()]
    print(f"決算データを更新します（{len(codes)}銘柄・時間がかかります）", flush=True)
    fund_history.download_all(codes, refresh=True)
    fundamentals.fetch(codes, resume=False)
    # packed に反映（株価は update_prices が別途更新済みなので決算まわりのみ）
    st = fund_history  # noqa
    import fund_panel
    stmts = fund_panel.load_statements(_allow_packed=False)
    rows = []
    for code, recs in stmts.items():
        for fy, v in recs.items():
            rows.append(dict(code=code, fy=fy, **v))
    pd.DataFrame(rows).to_parquet(os.path.join(PACK, "fin.parquet"),
                                  compression="zstd", index=False)
    f = os.path.join(HERE, "data", "fundamentals.csv")
    if os.path.exists(f):
        pd.read_csv(f, dtype={"code": str}).to_parquet(
            os.path.join(PACK, "fundamentals.parquet"), compression="zstd", index=False)
    print("決算データを保存しました")


if __name__ == "__main__":
    ok = update_prices()
    if "--full" in sys.argv:
        update_financials()
    print("完了", dt.datetime.now().isoformat(timespec="seconds"))
