"""
推奨銘柄と「なぜ推奨するのか」の生成。

推奨理由は2段に分けて出す。

 【検証済み・10年】 配当利回り(実績)・減配歴・増配率・値動きの安定性・200日線
 【検証済み・3年のみ】 PBR・PER … 2023年3月期以降の決算しか取れないため
                      検証は2023/7〜2026/8の約3年。しかもこの期間は
                      「東証のPBR1倍割れ改善要請」(2023年3月)の直撃期で、
                      割安株が特別に上がった可能性がある。強さは割り引いて読むこと。
                      ※検証期間を延ばすにはJ-Quants Standard(3,300円/月)が必要。
                        無料プランは2年分しかなく、yfinanceの4年より短い。未実施。
 【参考・効果なし】 ROE・自己資本比率・営業利益率・配当性向
                  → 検証したところ符号が逆だった（高いほど上がらなかった）。
                    足切りに使うと成績が悪化したため、表示のみ。
"""
import os
import numpy as np
import pandas as pd

import factors
import backtest as bt

HERE = os.path.dirname(os.path.abspath(__file__))
FUND = os.path.join(HERE, "data", "fundamentals.csv")


PACK_FUND = os.path.join(HERE, "data", "packed", "fundamentals.parquet")


def load_fundamentals():
    # 公開用の圧縮版があればそちらを使う
    if os.path.exists(PACK_FUND):
        f = pd.read_parquet(PACK_FUND)
        f["code"] = f["code"].astype(str)
        f = f.set_index("code")
    elif os.path.exists(FUND):
        f = pd.read_csv(FUND, dtype={"code": str}).set_index("code")
    else:
        return pd.DataFrame()
    for c in f.columns:
        if c not in ("銘柄名", "セクター", "業種"):
            f[c] = pd.to_numeric(f[c], errors="coerce")
    return f


# ---------------------------------------------------------------- 理由生成
def verified_reasons(row) -> list:
    """過去10年のデータで検証済みの根拠。"""
    r = []
    y = row.get("配当利回り")
    if pd.notna(y):
        r.append(f"実績配当利回り {y*100:.2f}%（年間配当 {row['年間配当']:.1f}円）")
    cuts = row.get("減配月数5年")
    if pd.notna(cuts):
        r.append("過去5年 減配なし" if cuts == 0 else f"過去5年 減配期間 {int(cuts)}ヶ月")
    g = row.get("増配率3年")
    if pd.notna(g):
        r.append(f"配当は3年で年率 {g*100:+.1f}%" + ("（増配）" if g > 0.01 else "（横ばい〜減少）" if g < 0 else ""))
    v = row.get("ボラティリティ")
    if pd.notna(v):
        r.append(f"年間値動き幅 {v*100:.0f}%" + ("（安定）" if v < 0.25 else "（大きめ）" if v > 0.4 else ""))
    r.append("株価は200日移動平均より上（下降トレンドではない）")
    return r


def fundamental_reasons(row) -> list:
    """
    ファンダメンタルの根拠。

    【検証結果にもとづく注意書き】
    2023/7〜2026/8の約3年で検証したところ、
      ・割安さ(PBR/PER)は先1年のリターンを予測できた（IC +25/+20、補正t 2.9/3.3）
      ・ROE・自己資本比率・営業利益率は符号が逆だった（高いほど上がらなかった）
    そのため「ROEが高い＝買い材料」とは書かない。事実だけを示す。
    """
    r = []
    pbr, per, roe = row.get("PBR"), row.get("PER実績"), row.get("ROE")
    payout, de, cr = row.get("配当性向"), row.get("DEレシオ"), row.get("流動比率")
    rg, eg, fcf = row.get("売上成長"), row.get("利益成長"), row.get("FCF")

    if pd.notna(pbr):
        if pbr <= 1.0:
            r.append(f"◎ PBR {pbr:.2f}倍 — 解散価値割れ。検証で最も強く効いた条件"
                     "（PBR1倍以下に絞ると3年でCAGR +4.7pt）")
        elif pbr <= 1.5:
            r.append(f"○ PBR {pbr:.2f}倍 — 割安圏")
        else:
            r.append(f"△ PBR {pbr:.2f}倍 — 割安ではない。検証では高PBRほど成績が悪かった")
    if pd.notna(per):
        if 0 < per < 12:
            r.append(f"◎ PER {per:.1f}倍 — 利益に対して株価が安い")
        elif 0 < per < 20:
            r.append(f"○ PER {per:.1f}倍 — 標準的")
        elif per >= 20:
            r.append(f"△ PER {per:.1f}倍 — 利益に対して株価が高い")
        else:
            r.append("△ PERが算出不能 — 最終赤字")
    if pd.notna(roe):
        r.append(f"ROE {roe*100:.1f}%（参考）— 検証期間では ROEの高低と"
                 "1年後リターンに正の関係はなかった（IC -7.6）")
    if pd.notna(payout):
        if payout > 1.0:
            r.append(f"配当性向 {payout*100:.0f}% — 利益を超える配当。"
                     "減配リスクはあるが、減配が起きれば売却ルールCで自動的に外れる")
        else:
            r.append(f"配当性向 {payout*100:.0f}%")
    if pd.notna(de):
        r.append(f"D/Eレシオ {de:.0f}%（参考）— 金融・商社は業態上高くなる")
    if pd.notna(cr) and cr < 1.0:
        r.append(f"流動比率 {cr:.2f} — 短期の支払い能力は低い")
    if pd.notna(rg) and pd.notna(eg):
        r.append(f"売上 {rg*100:+.1f}% / 利益 {eg*100:+.1f}%")
    if pd.notna(fcf):
        r.append("フリーキャッシュフローはプラス" if fcf > 0
                 else "フリーキャッシュフローはマイナス")
    return r


def risk_flags(row) -> list:
    """
    買う前に人間が見ておくべき点。

    ※これは「除外条件」ではない。検証したところ、これらで機械的に足切りすると
      成績はむしろ悪化した。判断材料として出すだけに留める。
    """
    w = []
    if row.get("配当利回り", 0) > 0.06:
        w.append("利回りが6%超 — 株価下落や特別配当が原因の可能性。持続性を要確認")
    if pd.notna(row.get("配当性向")) and row.get("配当性向", 0) > 1.0:
        w.append(f"配当性向が100%超（{row['配当性向']*100:.0f}%）— 利益を超える配当。"
                 "ただし減配が起きれば売却ルールCで外れる")
    if row.get("減配月数5年", 0) > 0:
        w.append(f"過去5年に減配歴あり（{int(row['減配月数5年'])}ヶ月）")
    if pd.notna(row.get("増配率3年")) and row["増配率3年"] < 0:
        w.append("配当が3年で減少トレンド")
    if pd.notna(row.get("利益成長")) and row["利益成長"] < -0.2:
        w.append(f"利益が前年比{row['利益成長']*100:.0f}%の減益 — "
                 "ただし検証では減益銘柄を外すと成績が下がった")
    if row.get("ボラティリティ", 0) > 0.45:
        w.append("値動きが荒い — 1年以内に大きく下振れる可能性")
    return w


def fund_reject(row, max_pbr=None) -> str:
    """
    銘柄を除外する条件。

    【重要】以前ここに入れていた「常識的な安全弁」は、検証の結果すべて削除した。
    2023/7〜2026/8の約3年で、足切りなし(CAGR 28.6%)と比べた効果は次の通り:

        自己資本比率 40%以上だけ残す   21.2%  (-7.4pt)  ← 最悪
        ROE 8%以上だけ残す            25.8%  (-2.8pt)
        ROE 3%以上だけ残す            27.4%  (-1.2pt)
        PER 40倍以下だけ残す          27.5%  (-1.1pt)
        配当性向 100%以下だけ残す      27.7%  (-0.9pt)
        PBR 1倍以下だけ残す           33.3%  (+4.7pt)  ← 唯一プラス

    「良い会社に絞る」は全滅し、「安い会社に絞る」だけが効いた。
    そのため残すのは割安フィルタ(PBR上限)だけ。既定では無効。
    減配リスクは事前の足切りではなく、売却ルールC(減配したら売る)で対処する。

    戻り値: 除外理由（空文字なら合格）
    """
    if max_pbr is not None:
        pbr = row.get("PBR")
        if pd.isna(pbr):
            return "PBRが取得できない（割安フィルタ使用時は判定不能のため除外）"
        if pbr > max_pbr:
            return f"PBR {pbr:.2f}倍 — 割安フィルタの上限 {max_pbr:.1f}倍 を超える"
    return ""


# ---------------------------------------------------------------- 本体
def build(n=25, max_pbr=None, **kw):
    P = factors.load_panel()
    F = factors.compute(P)
    t = P["px"].index[-1]
    # ファンダ足切りで減る分を見込んで多めに拾う
    # 割安フィルタで減る分を見込んで多めに拾う
    codes = bt.select(F, t, n=n * 4 if max_pbr is not None else n, **kw)
    ttm = factors.ttm_dividend(P["div"])

    px_t = bt.at(P["px"], t); y_t = bt.at(F["yield"], t); d_t = bt.at(ttm, t)
    g_t = bt.at(F["div_growth"], t); c_t = bt.at(F["div_cuts_5y"], t)
    v_t = bt.at(F["vol1y"], t); m_t = bt.at(F["tr_mom12"], t); to_t = bt.at(F["turnover"], t)
    ma = bt.at(P["px"].rolling(200).mean(), t)

    fund = load_fundamentals()
    rows = []
    for c in codes:
        r = dict(コード=c, 株価=round(float(px_t[c]), 1),
                 配当利回り=float(y_t[c]), 年間配当=float(d_t[c]),
                 増配率3年=float(g_t.get(c, np.nan)), 減配月数5年=float(c_t.get(c, np.nan)),
                 ボラティリティ=float(v_t[c]), year1リターン=float(m_t.get(c, np.nan)),
                 売買代金億円=round(float(to_t[c]) / 1e8, 1),
                 損切り目安=round(float(px_t[c]) * 0.85, 1),
                 トレンド線=round(float(ma.get(c, np.nan)), 1) if pd.notna(ma.get(c, np.nan)) else np.nan)
        if not fund.empty and c in fund.index:
            for col in fund.columns:
                r[col] = fund.loc[c, col]
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, t, pd.DataFrame()
    df["除外理由"] = df.apply(lambda x: fund_reject(x, max_pbr), axis=1)
    rejected = df[df["除外理由"] != ""].copy()
    df = df[df["除外理由"] == ""].head(n).copy()
    for d in (df, rejected):
        if d.empty:
            continue
        d["検証済み理由"] = d.apply(lambda x: verified_reasons(x), axis=1)
        d["ファンダ理由"] = d.apply(lambda x: fundamental_reasons(x), axis=1)
        d["注意"] = d.apply(lambda x: risk_flags(x), axis=1)
    return df, t, rejected


def to_text(df, t) -> str:
    """人が読む推奨レポート。"""
    L = [f"■ 推奨銘柄リスト（基準日 {t.date()}） {len(df)}銘柄 / 等金額で分散",
         "",
         "【この推奨の作り方】",
         "  10年検証済みの選別条件: 配当利回り・減配歴・増配率・値動きの安定性・200日線",
         "  3年のみ検証: PBR・PER（割安さは効いた。ただし東証PBR改善要請の直撃期のみの検証）",
         "  効果が確認できず: ROE・自己資本比率・営業利益率・配当性向（足切りに使うと成績が悪化した）",
         ""]
    for _, r in df.iterrows():
        name = r.get("銘柄名", "")
        sec = r.get("セクター", "")
        L.append(f"── {r['コード']} {name}  [{sec}]  株価 {r['株価']:,.1f}円")
        L.append("   ◆ 検証済みの推奨理由")
        for x in r["検証済み理由"]:
            L.append(f"     ・{x}")
        if r["ファンダ理由"]:
            L.append("   ◇ ファンダメンタル面（◎○△=3年検証済み / 参考=効果なし）")
            for x in r["ファンダ理由"]:
                L.append(f"     ・{x}")
        if r["注意"]:
            L.append("   ▲ 注意")
            for x in r["注意"]:
                L.append(f"     ・{x}")
        L.append(f"   [売却ライン]: 株価 {r['損切り目安']:,.1f}円（-15%）"
                 f" / 200日線 {r['トレンド線']:,.1f}円 割れ / 減配発表")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    df, t, rej = build(n=20, min_yield=0.02, max_per_sector=3, max_vol=0.6)
    txt = to_text(df, t)
    if len(rej):
        txt += chr(10) + '■ ファンダメンタルで除外した銘柄（配当条件は満たしていた）' + chr(10)
        for _, r in rej.iterrows():
            txt += f"  x {r['コード']} {r.get('銘柄名','')} - {r['除外理由']}" + chr(10)
    try:
        print(txt)
    except UnicodeEncodeError:
        print(txt.encode('cp932', 'replace').decode('cp932'))
    out = os.path.join(HERE, 'data', 'out', 'recommend.csv')
    df.drop(columns=['検証済み理由', 'ファンダ理由', '注意']).to_csv(out, index=False, encoding='utf-8-sig')
    with open(os.path.join(HERE, 'data', 'out', 'recommend.txt'), 'w', encoding='utf-8') as f:
        f.write(txt)
    print('保存:', out)
