"""
条件の総当たり検証。

目的は「一番成績が良い条件」を探すことではない（それは過剰最適化）。
・前半と後半の両方で通用する条件はどれか
・実際に買えるETF(1306/1321/1478/1577)に勝てているか
を確かめること。勝てていないなら、素直にETFを買った方がよい。
"""
import warnings; warnings.filterwarnings("ignore")
import os, itertools
import numpy as np, pandas as pd
import factors, backtest as bt

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(HERE,"data","out"); os.makedirs(OUT,exist_ok=True)
SPLIT=pd.Timestamp("2021-09-01")
BENCH={"1306 TOPIX":"1306","1321 日経225":"1321","1478 高配当MSCI":"1478","1577 高配当70":"1577"}

def bench_curves(index):
    import bench
    return bench.all_curves(index)


def main():
    P=factors.load_panel(); F=factors.compute(P)
    print("パネル:",P["px"].shape, P["px"].index.min().date(),"〜",P["px"].index.max().date(),flush=True)
    idx=P["px"].index
    mkt=(1+P["ret"].mean(axis=1).fillna(0)).cumprod()
    curves={"市場平均(全株等ウェイト)":mkt}; curves.update(bench_curves(idx))

    grid=[]
    for n in (25,50):
        for my in (0.020,0.030):
            for mps in (2,3):
                for up in (True,False):
                    for rb in (12,6):
                        grid.append(dict(n=n,min_yield=my,max_per_sector=mps,
                                         require_uptrend=up,rebalance_months=rb))
    print("試行数",len(grid),flush=True)
    rows=[]
    for i,g in enumerate(grid,1):
        kw=dict(g); rb=kw.pop("rebalance_months")
        cur,log=bt.run(P,F,rebalance_months=rb,max_vol=0.60,**kw)
        name=f"n{g['n']} 利{g['min_yield']*100:.0f}% 業種{g['max_per_sector']} " \
             f"{'上昇のみ' if g['require_uptrend'] else '制限なし'} {rb}ヶ月"
        a=bt.stats(cur.loc[:SPLIT],"a"); b=bt.stats(cur.loc[SPLIT:],"b"); f_=bt.stats(cur,"f")
        rows.append(dict(条件=name, 全CAGR=f_["CAGR"], 前半CAGR=a["CAGR"], 後半CAGR=b["CAGR"],
                         最大DD=f_["最大DD"], 年別最悪=f_["年別最悪"],
                         **{"12m>=7%":f_["12m>=7%"], "年別>=7%":f_["年別>=7%"]},
                         平均銘柄数=log.n.mean()))
        curves[name]=cur
        if i%8==0: print(f"  {i}/{len(grid)}",flush=True)
    d=pd.DataFrame(rows).set_index("条件")

    ref=[]
    for name,c in curves.items():
        if name in BENCH or name.startswith("市場"):
            s=bt.stats(c,name); a=bt.stats(c.loc[:SPLIT],"a"); b=bt.stats(c.loc[SPLIT:],"b")
            ref.append(dict(条件="【比較】"+name, 全CAGR=s["CAGR"], 前半CAGR=a["CAGR"], 後半CAGR=b["CAGR"],
                            最大DD=s["最大DD"], 年別最悪=s["年別最悪"],
                            **{"12m>=7%":s["12m>=7%"],"年別>=7%":s["年別>=7%"]}, 平均銘柄数=np.nan))
    r=pd.DataFrame(ref).set_index("条件")
    allr=pd.concat([r,d.sort_values("全CAGR",ascending=False)])
    pcols=["全CAGR","前半CAGR","後半CAGR","最大DD","年別最悪","12m>=7%","年別>=7%"]
    disp=allr.copy(); disp[pcols]=(disp[pcols]*100).round(1)
    pd.set_option("display.width",250)
    print("\n=== 条件別（上=比較対象、下=CAGR順） ===")
    print(disp.round(1).to_string())
    allr.to_csv(os.path.join(OUT,"sweep.csv"),encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(os.path.join(OUT,"sweep_curves.csv"),encoding="utf-8-sig")

if __name__=="__main__":
    main()
