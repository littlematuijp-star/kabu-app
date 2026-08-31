"""
recommend.py に入れた「常識的な足切り」を1つずつ検証する。

常識で正しそうな条件が、データでは有害なことがある。
実際に外してみて、成績がどう変わるかを測る。
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd
import factors, fund_panel, backtest as bt, active, bench

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "out")
START = pd.Timestamp("2023-07-01")

P = factors.load_panel(); F = factors.compute(P); G = fund_panel.build(P, F)
mkt = (1 + P["ret"].mean(axis=1).fillna(0)).cumprod()
rows = []
def add(cur, name):
    c = cur.loc[START:]; c = c / c.iloc[0]
    rows.append(bt.stats(c, name))

add(mkt, "【比較】市場平均")
for k, v in bench.all_curves(P["px"].index).items(): add(v, "【比較】" + k)

base = dict(n=20, min_yield=0.02, max_per_sector=3, max_vol=0.6,
            require_uptrend=True, stop_loss=0.15,
            use_trend_exit=True, use_divcut_exit=True, start=START)
cur, _ = active.run(P, F, **base); add(cur, "足切りなし（配当ルールのみ）")

# 個別の足切りを「それ単体で適用」したときの効果
#   True を残す条件を作り、extra_filter に渡せる形（大きいほど良い）にする
cond = {
    "ROE 3%以上だけ残す":       (G["ROE"] >= 0.03).astype(float),
    "ROE 8%以上だけ残す":       (G["ROE"] >= 0.08).astype(float),
    "PER 40倍以下だけ残す":     (G["PER"] <= 40).astype(float),
    "PER 20倍以下だけ残す":     (G["PER"] <= 20).astype(float),
    "配当性向 100%以下だけ残す": (G["配当性向"] <= 1.0).astype(float),
    "配当性向 60%以下だけ残す":  (G["配当性向"] <= 0.6).astype(float),
    "自己資本比率 40%以上だけ残す": (G["自己資本比率"] >= 0.40).astype(float),
    "PBR 1倍以下だけ残す":      (G["PBR"] <= 1.0).astype(float),
    "PBR 1.5倍以下だけ残す":    (G["PBR"] <= 1.5).astype(float),
}
for name, c in cond.items():
    m = c.where(c > 0)          # 条件を満たさない銘柄は NaN → 候補から外れる
    cur, _ = active.run(P, F, extra_filter=(m, "half"), **base)
    add(cur, name)

d = pd.DataFrame(rows).set_index("戦略")
d = (d * 100).round(1)
pd.set_option("display.width", 240)
print("=== 足切り条件の個別検証（2023/7〜2026/8・約3年）===")
print(d.to_string())
d.to_csv(os.path.join(OUT, "filter_test.csv"), encoding="utf-8-sig")
