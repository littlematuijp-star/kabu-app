"""最終設定の確定と、年別成績の書き出し。"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, factors, backtest as bt, active, bench
P=factors.load_panel(); F=factors.compute(P)
mkt=(1+P["ret"].mean(axis=1).fillna(0)).cumprod()
SPLIT=pd.Timestamp("2021-09-01")
rows=[]; curves={"市場平均(全株等ウェイト)":mkt}
curves.update(bench.all_curves(P["px"].index))
def add(cur,name,tr=None):
    s=bt.stats(cur,name)
    s["前半CAGR"]=bt.stats(cur.loc[:SPLIT],"a")["CAGR"]; s["後半CAGR"]=bt.stats(cur.loc[SPLIT:],"b")["CAGR"]
    if tr is not None: s["年間売買"]=len(tr)/9.0
    rows.append(s); curves[name]=cur
add(mkt,"【比較】市場平均(全株等ウェイト)")
for k,v in bench.all_curves(P["px"].index).items(): add(v,"【比較】"+k)
base=dict(min_yield=0.02,max_per_sector=3,require_uptrend=True,max_vol=0.6,
          stop_loss=0.15,use_trend_exit=True,use_divcut_exit=True)
for n in (15,20,25):
    cur,tr=active.run(P,F,n=n,**base); add(cur,f"入替あり {n}銘柄",tr)
cur,tr=active.run(P,F,n=20,**{**base,"use_divcut_exit":False}); add(cur,"入替あり 20銘柄(減配ルールなし)",tr)
d=pd.DataFrame(rows).set_index("戦略")
pc=[c for c in d.columns if c!="年間売買"]; d[pc]=(d[pc]*100).round(1)
pd.set_option("display.width",250); print(d.round(1).to_string())
y=pd.DataFrame({k:v.resample("YE").last().pct_change() for k,v in curves.items()})
y.index=y.index.year
print("\n=== 年別リターン(%) ===")
print((y*100).round(1).dropna(how="all").to_string())
d.to_csv("data/out/final.csv",encoding="utf-8-sig")
y.to_csv("data/out/final_yearly.csv",encoding="utf-8-sig")
pd.DataFrame(curves).to_csv("data/out/final_curves.csv",encoding="utf-8-sig")
