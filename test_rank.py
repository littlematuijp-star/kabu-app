"""点数の順位は本当に成績に効いているか。上位5位 / 6-10位 / 11-15位 / 16-20位 を比べる。"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, factors, backtest as bt, active
P=factors.load_panel(); F=factors.compute(P)
mkt=(1+P["ret"].mean(axis=1).fillna(0)).cumprod()
SPLIT=pd.Timestamp("2021-09-01")
rows=[]
def add(cur,name):
    s=bt.stats(cur,name)
    s["前半CAGR"]=bt.stats(cur.loc[:SPLIT],"a")["CAGR"]
    s["後半CAGR"]=bt.stats(cur.loc[SPLIT:],"b")["CAGR"]
    rows.append(s)
add(mkt,"【比較】市場平均")
base=dict(n=5,min_yield=0.02,max_per_sector=3,require_uptrend=True,max_vol=0.6,
          stop_loss=0.15,use_trend_exit=True,use_divcut_exit=True)
for off,lbl in [(0,"1〜5位"),(5,"6〜10位"),(10,"11〜15位"),(15,"16〜20位"),(25,"26〜30位")]:
    cur,_=active.run(P,F,offset=off,**base); add(cur,lbl)
d=pd.DataFrame(rows).set_index("戦略")*100
pd.set_option("display.width",250); print(d.round(1).to_string())
