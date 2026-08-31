"""保有銘柄数を減らすと成績はどうなるか。5〜25銘柄で比較する。"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, factors, backtest as bt, active, bench
P=factors.load_panel(); F=factors.compute(P)
mkt=(1+P["ret"].mean(axis=1).fillna(0)).cumprod()
SPLIT=pd.Timestamp("2021-09-01")
rows=[]
def add(cur,name,tr=None):
    s=bt.stats(cur,name)
    s["前半CAGR"]=bt.stats(cur.loc[:SPLIT],"a")["CAGR"]
    s["後半CAGR"]=bt.stats(cur.loc[SPLIT:],"b")["CAGR"]
    if tr is not None: s["年間売買"]=len(tr)/9.0
    rows.append(s)
add(mkt,"【比較】市場平均")
for k,v in bench.all_curves(P["px"].index).items(): add(v,"【比較】"+k)
base=dict(min_yield=0.02,max_per_sector=3,require_uptrend=True,max_vol=0.6,
          stop_loss=0.15,use_trend_exit=True,use_divcut_exit=True)
for n in (5,8,10,15,20,25):
    cur,tr=active.run(P,F,n=n,**base); add(cur,f"{n}銘柄",tr)
d=pd.DataFrame(rows).set_index("戦略")
pc=[c for c in d.columns if c!="年間売買"]; d[pc]=(d[pc]*100).round(1)
pd.set_option("display.width",250); print(d.round(1).to_string())
d.to_csv("data/out/n_compare.csv",encoding="utf-8-sig")
# 年別も
print("\n=== 年別リターン(%) ===")
cur5,_=active.run(P,F,n=5,**base); cur10,_=active.run(P,F,n=10,**base); cur20,_=active.run(P,F,n=20,**base)
y=pd.DataFrame({"市場":mkt,"5銘柄":cur5,"10銘柄":cur10,"20銘柄":cur20})
y=y.resample("YE").last().pct_change().dropna(how="all")*100
y.index=y.index.year
print(y.round(1).to_string())
