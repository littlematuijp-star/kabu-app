"""
日本株 1年勝負・配当ベース銘柄選択アプリ

  1. 今日の推奨銘柄を、理由つきで出す
  2. 保有中の銘柄に売却シグナルが出ていないか点検する
  3. そのルールの過去10年の成績を、実際に買えるETFと並べて見せる
"""
import os
import numpy as np
import pandas as pd
import streamlit as st

import factors
import backtest as bt
import recommend as rc
import monitor as mn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "out")

st.set_page_config(page_title="日本株 1年勝負・配当ベース", layout="wide")


@st.cache_resource(show_spinner="株価データを読み込み中…（初回20秒ほど）")
def load():
    P = factors.load_panel()
    return P, factors.compute(P)


def pc(x, d=1):
    return "-" if pd.isna(x) else f"{x*100:.{d}f}%"


# ================================================================== サイドバー
st.sidebar.header("選び方の設定")
n = st.sidebar.slider("持つ銘柄数", 5, 25, 10, 1,
                      help="点数の順位帯ごとに10年検証した結果: "
                           "1〜5位 CAGR18.0% / 6〜10位 17.6% / 11〜15位 17.0% / "
                           "16〜20位 14.6% / 26〜30位 9.3%。"
                           "順位が下がるほど確実に悪化するので、増やしすぎると"
                           "成績の悪い銘柄を無理に買うことになります。"
                           "5銘柄が数字上は最良ですが1銘柄の当たり外れで振れるため、"
                           "分散とのバランスで10を既定にしています。")
min_yield = st.sidebar.slider("最低配当利回り(%)", 0.0, 4.0, 2.0, 0.5,
                              help="検証では2%が最良。0%も3%も成績が落ちました。") / 100
max_sector = st.sidebar.slider("同じ業種の上限銘柄数", 1, 6, 3, 1)
pbr_mode = st.sidebar.selectbox(
    "割安フィルタ（PBR上限）", ["使わない", "1.5倍以下", "1.0倍以下"], index=0,
    help="2023/7〜2026/8の約3年で検証: 足切りなし CAGR 28.6% に対し "
         "PBR1.5倍以下 29.2%、PBR1.0倍以下 33.3%。"
         "ただし検証はこの3年のみで、東証のPBR1倍割れ改善要請(2023年3月)の"
         "直撃期にあたります。10年検証済みの他の条件とは信頼度が違います。")
max_pbr = {"使わない": None, "1.5倍以下": 1.5, "1.0倍以下": 1.0}[pbr_mode]
kw = dict(min_yield=min_yield, max_per_sector=max_sector, max_vol=0.6, require_uptrend=True)

st.sidebar.divider()
st.sidebar.caption("**データ更新**: `データ更新.bat`\n\n**再検証**: `検証実行.bat`")

tabs = st.tabs(["🎯 今日の推奨", "💼 保有チェック", "📊 過去10年の成績", "📖 このツールの限界"])

# ================================================================== 推奨
with tabs[0]:
    with st.spinner("銘柄を選定中…"):
        df, t, rej = rc.build(n=n, max_pbr=max_pbr, **kw)
    st.caption(f"データ基準日: {t.date()}")

    if df.empty:
        st.warning("条件を満たす銘柄がありません。左の条件をゆるめてください。")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("推奨銘柄数", f"{len(df)} 銘柄")
        c2.metric("平均配当利回り", pc(df["配当利回り"].mean(), 2))
        c3.metric("平均PBR", f"{pd.to_numeric(df.get('PBR'), errors='coerce').mean():.2f}倍"
                  if "PBR" in df else "-")
        c4.metric("割安フィルタで除外", f"{len(rej)} 銘柄")

        st.info("**等金額で全銘柄を買ってください。** "
                "上位だけ買うと、検証した成績とは別物になります。\n\n"
                "毎月1回「💼 保有チェック」で点検し、売却シグナルが出た銘柄を入れ替えます。")

        # ---- 一覧表
        show = df.copy()
        for c in ["配当利回り", "増配率3年", "ボラティリティ", "ROE", "配当性向", "利益成長"]:
            if c in show:
                show[c] = pd.to_numeric(show[c], errors="coerce").map(lambda v: pc(v))
        if "時価総額" in show:
            show["時価総額(億)"] = (pd.to_numeric(show["時価総額"], errors="coerce") / 1e8).round(0)
        cols = [c for c in ["コード", "銘柄名", "業種", "規模", "株価", "配当利回り", "年間配当",
                            "増配率3年", "減配月数5年", "PBR", "PER実績", "ROE", "配当性向",
                            "利益成長", "ボラティリティ", "時価総額(億)", "売買代金億円",
                            "損切り目安", "トレンド線"] if c in show]
        st.dataframe(show[cols], width="stretch", hide_index=True)

        # ---- 個別の推奨理由
        st.subheader("推奨理由（銘柄ごと）")
        for _, r in df.iterrows():
            with st.expander(f"{r['コード']}　{r.get('銘柄名','')}　"
                             f"（利回り {pc(r['配当利回り'],2)}／株価 {r['株価']:,.0f}円）"):
                a, b = st.columns(2)
                with a:
                    st.markdown("**◆ 検証済みの推奨理由**")
                    st.caption("過去10年のデータで効果を確認した根拠")
                    for x in r["検証済み理由"]:
                        st.markdown(f"- {x}")
                with b:
                    st.markdown("**◇ ファンダメンタル面の情報**")
                    st.caption("◎○△ = 3年検証で効果を確認（PBR/PER）。"
                               "「参考」= 効果が確認できなかった指標（ROE等）")
                    for x in r["ファンダ理由"]:
                        st.markdown(f"- {x}")
                if r["注意"]:
                    st.warning("**注意**\n\n" + "\n".join(f"- {x}" for x in r["注意"]))
                st.error(f"**売却ライン** ｜ 損切り {r['損切り目安']:,.0f}円（-15%）　/　"
                         f"200日線 {r['トレンド線']:,.0f}円 割れ　/　減配の発表")

        # ---- 除外銘柄
        if len(rej):
            with st.expander(f"⛔ 割安フィルタで除外した銘柄 {len(rej)}件（配当条件は満たしていた）"):
                st.caption("PBRの上限を超えたため外れた銘柄です。")
                for _, r in rej.iterrows():
                    st.markdown(f"- **{r['コード']} {r.get('銘柄名','')}** — {r['除外理由']}")

        st.download_button("推奨リストをCSVで保存",
                           df.drop(columns=["検証済み理由", "ファンダ理由", "注意"])
                             .to_csv(index=False).encode("utf-8-sig"),
                           f"推奨_{t.date()}.csv", "text/csv")
        st.download_button("推奨理由つきレポートを保存",
                           rc.to_text(df, t).encode("utf-8-sig"),
                           f"推奨レポート_{t.date()}.txt", "text/plain")

# ================================================================== 保有チェック
with tabs[1]:
    st.write("**毎月1回**、持っている銘柄を点検します。買値と買った日は分かる範囲で。")
    default = pd.DataFrame({"コード": ["", "", "", "", ""], "買値": [None]*5, "買った日": [""]*5})
    ed = st.data_editor(default, num_rows="dynamic", width="stretch",
                        column_config={
                            "コード": st.column_config.TextColumn("コード", help="例 7203"),
                            "買値": st.column_config.NumberColumn("買値(円)"),
                            "買った日": st.column_config.TextColumn("買った日", help="例 2026-04-15"),
                        })
    held = ed[ed["コード"].astype(str).str.strip() != ""]
    if len(held):
        P, F = load()
        res = mn.check(P, F, held)
        sell = res[res["判定"] == "売却"]
        keep = res[res["判定"] == "継続"]
        c1, c2 = st.columns(2)
        c1.metric("売却シグナル", f"{len(sell)} 銘柄")
        c2.metric("継続", f"{len(keep)} 銘柄")
        if len(sell):
            st.error("**売却シグナルが出ています**")
            for _, r in sell.iterrows():
                st.markdown(f"- **{r['コード']}**（{r['株価']:,.0f}円）— {r['売却理由']}")
            st.caption("売った資金は『今日の推奨』タブの上位から、まだ持っていない銘柄で補充します。")
        if len(keep):
            st.success("**継続でよい銘柄**")
            for _, r in keep.iterrows():
                st.markdown(f"- **{r['コード']}**（{r['株価']:,.0f}円）— {r['継続根拠']}")
        st.dataframe(res, width="stretch", hide_index=True)

# ================================================================== 成績
with tabs[2]:
    f = os.path.join(OUT, "final.csv")
    if not os.path.exists(f):
        st.info("`検証実行.bat` を実行すると、ここに過去10年の成績が出ます。")
    else:
        d = pd.read_csv(f).set_index("戦略")
        st.subheader("過去10年の成績（2017/9〜2026/8・配当込み・売買コスト0.3%差引後）")
        st.dataframe(d.round(1), width="stretch")
        st.caption("**年別>=7%** … 暦年で+7%以上を達成した年の割合。あなたの目標に直接対応する数字です。"
                   "**12m最悪** … 最も運が悪かった1年間の成績。")
        fc = os.path.join(OUT, "final_curves.csv")
        if os.path.exists(fc):
            cur = pd.read_csv(fc, index_col=0, parse_dates=True)
            st.subheader("資産推移（1.0 = 開始時）")
            st.line_chart(cur)
        fy = os.path.join(OUT, "final_yearly.csv")
        if os.path.exists(fy):
            y = pd.read_csv(fy, index_col=0)
            st.subheader("年別リターン(%)")
            st.dataframe((y * 100).round(1), width="stretch")
        fs = os.path.join(OUT, "sensitivity.csv")
        if os.path.exists(fs):
            with st.expander("設定を変えたときの感度（過剰最適化のチェック）"):
                st.dataframe(pd.read_csv(fs).set_index("戦略").round(1), width="stretch")

# ================================================================== 限界
with tabs[3]:
    st.markdown(open(os.path.join(HERE, "README.md"), encoding="utf-8").read())
