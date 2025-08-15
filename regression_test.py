# regression_test.py (build table from model attributes; robust p-values)
import argparse, pathlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = pathlib.Path(__file__).parent.resolve()
DATA = ROOT / "data"

FEATURES = [
    "neg_per_1000","unc_per_1000",
    "company_neg_per_1000","company_unc_per_1000",
    "external_neg_per_1000","external_unc_per_1000",
    "company_sent_ratio_mean",
    "finbert_neg_mean","finbert_neu_mean","finbert_pos_mean",
]
TARGETS = ["ar3d","ar5d","ar10d","ar20d","r5d","r10d","r20d","vol20d"]

def read_events(ticker: str) -> pd.DataFrame:
    f = DATA / f"{ticker}_item1a_events.csv"
    if not f.exists():
        raise FileNotFoundError(f"找不到 {f}，請先跑 run_mvp.py")
    df = pd.read_csv(f)
    for c in FEATURES + TARGETS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_10k"] = df.get("form", pd.Series(index=df.index, dtype=object)).astype(str).str.upper().str.contains("10-K").astype(int)
    df["year"]   = pd.to_datetime(df.get("filing_date"), errors="coerce").dt.year
    return df

def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    mu = s.mean(); sd = s.std(ddof=0); 
    if not np.isfinite(sd) or sd == 0: sd = 1.0
    return (s - mu) / sd

def design_matrix(df: pd.DataFrame, y_col: str):
    sub = df.dropna(subset=[y_col]).copy()
    if len(sub) < 8:
        return None, None
    zfeats = []
    for c in FEATURES:
        if c in sub.columns:
            sub[f"z_{c}"] = zscore_series(sub[c]); zfeats.append(f"z_{c}")
    yfe = pd.get_dummies(sub["year"], prefix="yr", drop_first=True, dummy_na=False)
    parts = []
    if zfeats: parts.append(sub[zfeats])
    parts.append(sub[["is_10k"]])
    if yfe.shape[1] > 0: parts.append(yfe)
    X = pd.concat(parts, axis=1).apply(pd.to_numeric, errors="coerce").astype(float)
    y = pd.to_numeric(sub[y_col], errors="coerce").astype(float)
    X.replace([np.inf,-np.inf], np.nan, inplace=True); y.replace([np.inf,-np.inf], np.nan, inplace=True)
    valid = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[valid], y.loc[valid]
    if len(X) < 8: return None, None
    std = X.std(axis=0, ddof=0).fillna(0.0)
    X = X[std[std>0].index.tolist()]
    X = sm.add_constant(X, has_constant="add")
    if len(X) <= X.shape[1]:
        drops = [c for c in X.columns if c.startswith("yr_")]
        while len(X) <= X.shape[1] and drops:
            X = X.drop(columns=[drops.pop()])
        if len(X) <= X.shape[1]:
            z_keep = [c for c in X.columns if c.startswith("z_")]
            while len(X) <= X.shape[1] and z_keep:
                X = X.drop(columns=[z_keep.pop()])
        if len(X) <= X.shape[1]:
            return None, None
    return y, X

def run_regression_table(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    y, X = design_matrix(df, y_col)
    if y is None or X is None:
        return pd.DataFrame()
    try:
        res = sm.OLS(y, X).fit(cov_type="HC1")
        ci = res.conf_int(alpha=0.05)
        out = pd.DataFrame({
            "term": res.params.index,
            "Coef.": res.params.values,
            "Std.Err.": res.bse.values,
            "t": res.tvalues.values,
            "P>|t|": res.pvalues.values,
            "CI_low": ci[0].values,
            "CI_high": ci[1].values,
        })
        out.insert(0, "target", y_col)
        return out
    except Exception as e:
        print(f"[warn] {y_col} regression failed: {e}")
        return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    args = ap.parse_args()
    T = args.ticker.upper()

    df = read_events(T)
    outs = [t for y in TARGETS if len((t:=run_regression_table(df,y)))]

    if not outs:
        print("No regressions produced (after cleaning / df too small)."); return

    res = pd.concat(outs, ignore_index=True)
    out_csv = DATA / f"{T}_regressions_ols_hc1.csv"
    res.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 適配 p 值欄名（雖然我們固定為 P>|t|，再保險處理一次）
    pcol = next((c for c in res.columns if c.lower().startswith("p>|")), "P>|t|")
    top = (res.sort_values(pcol).groupby("target", as_index=False, sort=False).head(10))

    out_top = DATA / f"{T}_regressions_top10.txt"
    lines = []
    for tgt, grp in top.groupby("target"):
        lines.append(f"=== {tgt} ===")
        for _, r in grp.iterrows():
            lines.append(f"{r['term']:>24s}  coef={r['Coef.']:+.4f}  t={r['t']:+.2f}  p={r[pcol]:.4f}  [{r['CI_low']:+.4f}, {r['CI_high']:+.4f}]")
        lines.append("")
    with open(out_top, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved:\n  {out_csv}\n  {out_top}")

if __name__ == "__main__":
    main()
