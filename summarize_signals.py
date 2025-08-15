# summarize_signals.py
# 讀取 data/ 內各公司的：
#   *_analysis_correlations.csv（特徵×目標的皮爾森 r）
#   *_regressions_sig_p05.csv（p<0.05 的顯著係數）
# 產出：
#   data/signal_shortlist.csv      → 依一致性分數排序的候選特徵（feature,target, sign）
#   data/signal_summary.txt        → 人類可讀摘要（每家公司、每組特徵的支持證據）

import pathlib, pandas as pd, numpy as np, re

ROOT = pathlib.Path(__file__).parent.resolve()
DATA = ROOT / "data"

def load_all():
    sig_rows = []
    corr_rows = []
    for p in DATA.glob("*_regressions_sig_p05.csv"):
        ticker = p.name.split("_")[0].upper()
        df = pd.read_csv(p)
        # 某些環境 p 值欄名可能不同，統一成 pval
        pcol = next((c for c in df.columns if c.lower().startswith("p>|")), "P>|t|")
        df = df.rename(columns={pcol: "pval"})
        df["ticker"] = ticker
        sig_rows.append(df[["ticker","target","term","Coef.","pval"]])

    for p in DATA.glob("*_analysis_correlations.csv"):
        ticker = p.name.split("_")[0].upper()
        df = pd.read_csv(p)
        if not {"feature","target","pearson_r"}.issubset(set(df.columns)):
            continue
        df["ticker"] = ticker
        corr_rows.append(df[["ticker","feature","target","pearson_r","n"]])

    sig = pd.concat(sig_rows, ignore_index=True) if sig_rows else pd.DataFrame(columns=["ticker","target","term","Coef.","pval"])
    cor = pd.concat(corr_rows, ignore_index=True) if corr_rows else pd.DataFrame(columns=["ticker","feature","target","pearson_r","n"])
    return sig, cor

def normalize_term(term: str) -> str:
    # 只保留我們關心的 z_ 前綴特徵；去掉 z_ 前綴供對齊
    t = str(term).strip()
    if not t.startswith("z_"):
        return None
    base = t[2:]
    return base

def shortlist(sig: pd.DataFrame, cor: pd.DataFrame) -> pd.DataFrame:
    if sig.empty:
        return pd.DataFrame()
    s = sig.copy()
    s["feature"] = s["term"].map(normalize_term)
    s = s.dropna(subset=["feature"])
    s["sign"] = np.sign(s["Coef."].astype(float)).astype(int)

    # 聚合：每個 (feature,target) 計算一致性分數（各公司相同號號的數量）
    grp = (s.groupby(["feature","target","sign"])
             .agg(n_support=("ticker","nunique"),
                  tickers=("ticker", lambda x: ",".join(sorted(set(x)))))
             .reset_index())

    # 取每個 (feature,target) 中支持最多的 sign
    top = (grp.sort_values(["feature","target","n_support"], ascending=[True,True,False])
              .groupby(["feature","target"])
              .head(1)
              .reset_index(drop=True))

    # 合併相關性支持（同號且 |r|>=0.30 視為支持）
    if not cor.empty:
        cor2 = cor.copy()
        cor2["corr_sign"] = np.sign(cor2["pearson_r"].astype(float)).astype(int)
        cor2["abs_r"] = cor2["pearson_r"].abs()
        # 對齊 feature 名稱（cor 用 'feature' 已是基名）
        sup = (top.merge(cor2, on=["feature","target"], how="left"))
        sup["corr_support"] = ((sup["corr_sign"] == sup["sign"]) & (sup["abs_r"] >= 0.30)).astype(int)
        corr_support = (sup.groupby(["feature","target"])
                           .agg(n_corr_support=("corr_support","sum"),
                                max_abs_r=("abs_r","max"))
                           .reset_index())
        top = top.merge(corr_support, on=["feature","target"], how="left")
    else:
        top["n_corr_support"] = 0
        top["max_abs_r"] = np.nan

    # 排序分數：一致性多→相關性支持多→|r|大
    top["score"] = top["n_support"]*2 + top["n_corr_support"]
    top = top.sort_values(["score","n_support","n_corr_support","max_abs_r"], ascending=False)

    # 友善欄位
    top = top[["feature","target","sign","n_support","tickers","n_corr_support","max_abs_r","score"]]
    return top

def write_summary(sig: pd.DataFrame, cor: pd.DataFrame, top: pd.DataFrame):
    lines = []
    lines.append("Signal shortlist (p<0.05 in OLS, HC1 robust) with cross-ticker consistency")
    lines.append("")
    if top.empty:
        lines.append("  (no signals)")
    else:
        for _, r in top.iterrows():
            sgn = "+" if r["sign"]>0 else "-"
            lines.append(f"- {r['feature']} → {r['target']}  [{sgn}]  support={int(r['n_support'])} tickers=({r['tickers']}) corr_support={int(r['n_corr_support'])} max|r|={r['max_abs_r']:.3f}" if pd.notna(r['max_abs_r']) else
                         f"- {r['feature']} → {r['target']}  [{sgn}]  support={int(r['n_support'])} tickers=({r['tickers']})")

    # 每家公司簡短列示（top 10 by p）
    lines.append("\nPer-ticker highlights:")
    if not sig.empty:
        for tkr, g in sig.groupby("ticker"):
            g2 = g.copy()
            g2["feature"] = g2["term"].map(normalize_term)
            g2 = g2.dropna(subset=["feature"]).sort_values("pval").head(10)
            lines.append(f"== {tkr} ==")
            for _, r in g2.iterrows():
                sgn = "+" if r["Coef."]>0 else "-"
                lines.append(f"  {r['feature']} → {r['target']}  {sgn}  p={r['pval']:.4g}")
    else:
        lines.append("  (no per-ticker results)")

    out_txt = DATA / "signal_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_txt}")

def main():
    sig, cor = load_all()
    top = shortlist(sig, cor)
    out_csv = DATA / "signal_shortlist.csv"
    top.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved {out_csv}")
    if not top.empty:
        print(top.head(12).to_string(index=False))
    write_summary(sig, cor, top)

if __name__ == "__main__":
    main()
