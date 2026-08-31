"""コスト感度と細部の詰め。年100回売買する前提が、コスト想定にどれだけ耐えるか。"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, factors, backtest as bt, active, bench
P=factors.load_panel(); F=factors.compute(P)
mkt=(1+P["ret"].mean(axis=1).fillna(0)).cumprod()
SPLIT=pd.Timestamp("2021-09-01")
base=dict(n=25,min_yield=0.02,max_per_sector=3,require_uptrend=True,max_vol=0.6,
          stop_loss=0.15,use_trend_exit=True,use_divcut_exit=True)
rows=[]
def add(cur,name,tr=None):
    s=bt.stats(cur,name)
    s["前半CAGR"]=bt.stats(cur.loc[:SPLIT],"a")["CAGR"]; s["後半CAGR"]=bt.stats(cur.loc[SPLIT:],"b")["CAGR"]
    if tr is not None: s["年間売買"]=len(tr)/9.0
    rows.append(s)
add(mkt,"【比較】市場平均")
for k,v in bench.all_curves(P["px"].index).items(): add(v,"【比較】"+k)

print("=== 売買コスト感度 ===",flush=True)
for c in (0.001,0.003,0.005,0.010):
    active.COST=c
    cur,tr=active.run(P,F,**base); add(cur,f"往復コスト{c*100:.1f}%",tr)
active.COST=0.003

print("=== 損切り幅 ===",flush=True)
for sl in (0.10,0.15,0.20,0.25):
    kw=dict(base); kw["stop_loss"]=sl
    cur,tr=active.run(P,F,**kw); add(cur,f"損切り-{sl*100:.0f}%",tr)

print("=== 銘柄数 ===",flush=True)
for n in (15,25,40):
    kw=dict(base); kw["n"]=n
    cur,tr=active.run(P,F,**kw); add(cur,f"{n}銘柄",tr)

print("=== 最低配当利回り ===",flush=True)
for my in (0.0,0.02,0.03):
    kw=dict(base); kw["min_yield"]=my
    cur,tr=active.run(P,F,**kw); add(cur,f"利回り{my*100:.0f}%以上",tr)

d=pd.DataFrame(rows).set_index("戦略")
pc=[c for c in d.columns if c!="年間売買"]
d[pc]=(d[pc]*100).round(1)
pd.set_option("display.width",250); print("\n"+d.round(1).to_string())
d.to_csv("data/out/sensitivity.csv",encoding="utf-8-sig")
