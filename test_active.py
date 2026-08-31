import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, factors, backtest as bt, active
P=factors.load_panel(); F=factors.compute(P)
mkt=(1+P["ret"].mean(axis=1).fillna(0)).cumprod()
SPLIT=pd.Timestamp("2021-09-01")
rows=[]
def add(cur,name):
    s=bt.stats(cur,name)
    s["前半CAGR"]=bt.stats(cur.loc[:SPLIT],"a")["CAGR"]; s["後半CAGR"]=bt.stats(cur.loc[SPLIT:],"b")["CAGR"]
    rows.append(s)
add(mkt,"【比較】市場平均")
cases=[
 ("放置(年1回入替)",       dict(stop_loss=9.9, use_trend_exit=False, use_divcut_exit=False)),
 ("損切り-15%のみ",        dict(stop_loss=0.15, use_trend_exit=False, use_divcut_exit=False)),
 ("トレンド割れのみ",       dict(stop_loss=9.9, use_trend_exit=True,  use_divcut_exit=False)),
 ("減配のみ",             dict(stop_loss=9.9, use_trend_exit=False, use_divcut_exit=True)),
 ("全部入り(損切+トレンド+減配)", dict(stop_loss=0.15, use_trend_exit=True, use_divcut_exit=True)),
 ("損切り-20%+トレンド",    dict(stop_loss=0.20, use_trend_exit=True,  use_divcut_exit=False)),
]
import time
for name,kw in cases:
    _t0=time.time()
    cur,tr=active.run(P,F,n=25,min_yield=0.02,max_per_sector=3,require_uptrend=True,max_vol=0.6,**kw)
    add(cur,name)
    rows[-1]["売買回数"]=len(tr); rows[-1]["勝率"]=(tr[tr.action=="売"].損益>0).mean()
    print(f"  {name} 完了 {time.time()-_t0:.0f}秒", flush=True)
d=pd.DataFrame(rows).set_index("戦略")
pc=[c for c in d.columns if c not in ("売買回数","勝率")]
d[pc]=(d[pc]*100).round(1); d["勝率"]=(d["勝率"]*100).round(1)
pd.set_option("display.width",250); print(d.round(1).to_string())
