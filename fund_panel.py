"""
過去の決算数値から、PBR/PER/ROE/配当性向などの「時点パネル」を作る。

未来情報を入れないための2つの決めごと:

1) 決算は期末の3ヶ月後から使う
   日本企業の決算短信は期末後45日程度だが、安全側に倒して90日後から使う。
   2026年3月期の数字は 2026年6月29日以降でないと参照しない。

2) 株式分割は「株数」側で調整する
   保存してある株価は未調整（実際についていた値段）。
   決算期末時点の発行済株式数に、その後の分割を掛けて現在株数に直し、
   時価総額 = 未調整株価 × 調整後株数 で計算する。
   1株あたり指標（EPS等）を直接使うと分割でズレるため使わない。
"""
import os, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIN = os.path.join(HERE, "data", "fin")

REPORT_LAG_DAYS = 90    # 決算期末から実際に公表されるまでの余裕


def _row(df, names):
    """複数の候補名から最初に見つかった行を返す。"""
    for n in names:
        if n in df.index:
            s = df.loc[n]
            if s.notna().any():
                return s
    return pd.Series(index=df.columns, dtype=float)


PACK = os.path.join(HERE, "data", "packed")


def load_statements(_allow_packed: bool = True) -> dict:
    """銘柄コード -> 決算期(Timestamp) -> 指標 の辞書。"""
    out = {}
    pf = os.path.join(PACK, "fin.parquet")
    if _allow_packed and os.path.exists(pf):
        d = pd.read_parquet(pf)
        for code, g in d.groupby("code"):
            recs = {}
            for _, r in g.iterrows():
                recs[pd.Timestamp(r["fy"])] = dict(
                    equity=r["equity"], shares=r["shares"], net_income=r["net_income"],
                    assets=r["assets"], revenue=r["revenue"], op_income=r["op_income"])
            out[str(code)] = recs
        return out

    for f in sorted(glob.glob(os.path.join(FIN, "*.csv"))):
        code = os.path.basename(f)[:-4]
        if os.path.getsize(f) == 0:
            continue
        try:
            df = pd.read_csv(f, index_col=0)
        except Exception:
            continue
        if df.empty:
            continue
        eq = _row(df, ["Stockholders Equity", "Total Equity Gross Minority Interest"])
        sh = _row(df, ["Ordinary Shares Number", "Share Issued"])
        ni = _row(df, ["Net Income Common Stockholders", "Net Income"])
        ta = _row(df, ["Total Assets"])
        rev = _row(df, ["Total Revenue"])
        op = _row(df, ["Operating Income"])
        recs = {}
        for c in df.columns:
            try:
                d = pd.Timestamp(c)
            except Exception:
                continue
            recs[d] = dict(equity=eq.get(c, np.nan), shares=sh.get(c, np.nan),
                           net_income=ni.get(c, np.nan), assets=ta.get(c, np.nan),
                           revenue=rev.get(c, np.nan), op_income=op.get(c, np.nan))
        if recs:
            out[code] = recs
    return out


def build(P: dict, F: dict) -> dict:
    """
    日次のファンダ指標パネルを返す。
    px は未調整株価、splits は分割比率（P に入っているもの）。
    """
    px, splits, div = P["px"], P["splits"], P["div"]
    idx = px.index
    st = load_statements()

    cols = px.columns
    blank = lambda: pd.DataFrame(np.nan, index=idx, columns=cols)
    equity_ps = blank()      # 1株あたり純資産（分割調整後の株数ベース）
    ni_ps = blank()          # 1株あたり純利益
    roe = blank()            # ROE（決算ベース）
    equity_ratio = blank()   # 自己資本比率
    op_margin = blank()      # 営業利益率

    # 分割の累積係数（各日以降に起きる分割を掛けるため、日ごとの累積を用意）
    cum = splits.replace(0.0, np.nan).fillna(1.0).cumprod()

    for c in cols:
        if c not in st:
            continue
        s_cum = cum[c]
        for fy, v in sorted(st[c].items()):
            eq, sh, ni = v["equity"], v["shares"], v["net_income"]
            if not (pd.notna(eq) and pd.notna(sh) and sh > 0):
                continue
            start = fy + pd.Timedelta(days=REPORT_LAG_DAYS)
            pos = idx.searchsorted(start)
            if pos >= len(idx):
                continue
            # 次の決算が使えるようになるまでの区間
            nxt = [d for d in sorted(st[c]) if d > fy]
            end = idx.searchsorted(nxt[0] + pd.Timedelta(days=REPORT_LAG_DAYS)) if nxt else len(idx)
            seg = idx[pos:end]
            if len(seg) == 0:
                continue
            # 決算期末時点の株数を、その後の分割で現在株数に直す
            base_pos = max(idx.searchsorted(fy) - 1, 0)
            f0 = float(s_cum.iloc[base_pos]) if pd.notna(s_cum.iloc[base_pos]) else 1.0
            adj = (s_cum.loc[seg] / f0).replace(0, np.nan).fillna(1.0)
            shares_now = sh * adj
            equity_ps.loc[seg, c] = eq / shares_now
            if pd.notna(ni):
                ni_ps.loc[seg, c] = ni / shares_now
                roe.loc[seg, c] = ni / eq if eq > 0 else np.nan
            if pd.notna(v["assets"]) and v["assets"] > 0:
                equity_ratio.loc[seg, c] = eq / v["assets"]
            if pd.notna(v["revenue"]) and v["revenue"] > 0 and pd.notna(v["op_income"]):
                op_margin.loc[seg, c] = v["op_income"] / v["revenue"]

    G = {}
    G["PBR"] = px / equity_ps
    G["PER"] = px / ni_ps
    G["ROE"] = roe
    G["自己資本比率"] = equity_ratio
    G["営業利益率"] = op_margin
    ttm = div.rolling("365D").sum()
    G["配当性向"] = ttm / ni_ps
    G["益回り"] = ni_ps / px          # PERの逆数（赤字も連続的に扱えるので順位付けに使う）
    G["純資産倍率の逆数"] = equity_ps / px   # PBRの逆数（＝BPS/株価）

    # 異常値を落とす（赤字によるマイナスPERなどは順位付けから外す）
    G["PBR"] = G["PBR"].where((G["PBR"] > 0) & (G["PBR"] < 50))
    G["PER"] = G["PER"].where((G["PER"] > 0) & (G["PER"] < 300))
    G["配当性向"] = G["配当性向"].where((G["配当性向"] >= 0) & (G["配当性向"] < 10))
    G["ROE"] = G["ROE"].where(G["ROE"].abs() < 2)
    return G
