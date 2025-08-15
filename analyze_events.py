# analyze_events.py (use merged events CSV; handle _x/_y suffixes)
# 讀入 <TICKER>_item1a_events.csv（已包含 file-level 特徵與事件窗目標）
# 自動把 *_x/_y 對齊到標準名稱；輸出：相關表、分位數利差、摘要

import argparse, pathlib, re
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).parent.resolve()
DATA = ROOT / "data"

FEATURES = [
    "neg_per_1000","unc_per_1000",
    "company_neg_per_1000","company_unc_per_1000",
    "external_neg_per_1000","external_unc_per_1000",
    "company_sent_ratio_mean",
    "finbert_neg_mean","finbert_neu_mean","finbert_pos_mean",
]
TARGETS = ["r1d","r3d","r5d","r10d","r20d","ar1d","ar3d","ar5d","ar10d","ar20d","vol5d","vol20d"]
META    = ["accession","filing_date","form","ticker","company_title","cik","n_paras"]

def read_events(ticker: str) -> pd.DataFrame:
    f = DATA / f"{ticker}_item1a_events.csv"
    if not f.exists():
        raise FileNotFoundError(f"找不到 {f}（請先跑 run_mvp.py 產生 events 檔）")
    df = pd.read_csv(f)
    # 正規化欄位（去空白）
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_col(df: pd.DataFrame, name: str) -> str | None:
    """大小寫不敏感；允許 *_x/_y；偏好：精確==>無尾碼==> *_x ==>*_y"""
    lname = name.lower()
    # 精確命中
    for c in df.columns:
        if c.lower() == lname:
            return c
    # 去尾碼比對
    candidates = []
    for c in df.columns:
        base = re.sub(r'_(x|y)$', '', c.lower())
        if base == lname:
            candidates.append(c)
    if not candidates:
        return None
    # 偏好順序
    prefs = [name, f"{name}_x", f"{name}_y"]
    for p in prefs:
        for c in candidates:
            if c.lower() == p.lower():
                return c
    return candidates[0]

def select_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    # accession 一定要
    acc = find_col(df, "accession")
    if acc is None:
        raise KeyError("找不到 accession 欄位")
    out["accession"] = df[acc]

    # meta（若存在就帶上）
    for m in META:
        c = find_col(df, m)
        if c is not None and m != "accession":
            out[m] = df[c]

    # features / targets：找到就複製成標準名
    for c in FEATURES + TARGETS:
        col = find_col(df, c)
        if col is not None:
            out[c] = pd.to_numeric(df[col], errors="coerce")
    return out

def corr_table(df: pd.DataFrame) -> pd.DataFrame:
    fcols = [c for c in FEATURES if c in df.columns]
    tcols = [c for c in TARGETS if c in df.columns]
    if not fcols or not tcols:
        return pd.DataFrame(columns=["feature","target","pearson_r","n"])
    rows = []
    for f in fcols:
        for t in tcols:
            s = df[[f,t]].dropna()
            rho = s[f].corr(s[t]) if len(s) >= 3 else np.nan
            rows.append({"feature": f, "target": t, "pearson_r": rho, "n": int(len(s))})
    return pd.DataFrame(rows)

def quantile_spread(df: pd.DataFrame, feature: str, target: str, q=0.2):
    if feature not in df.columns or target not in df.columns:
        return np.nan, 0, 0
    sub = df[[feature, target]].dropna()
    if len(sub) < 10:
        return np.nan, 0, 0
    lo_cut = sub[feature].quantile(q)
    hi_cut = sub[feature].quantile(1-q)
    lo = sub[sub[feature] <= lo_cut][target].mean()
    hi = sub[sub[feature] >= hi_cut][target].mean()
    return float(hi - lo), int((sub[feature] <= lo_cut).sum()), int((sub[feature] >= hi_cut).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, help="e.g., AAPL")
    args = ap.parse_args()
    T = args.ticker.upper()

    raw = read_events(T)
    df  = select_and_standardize(raw)

    present_features = [c for c in FEATURES if c in df.columns]
    present_targets  = [c for c in TARGETS if c in df.columns]

    # 1) 相關係數
    corr = corr_table(df)
    corr_path = DATA / f"{T}_analysis_correlations.csv"
    (corr.sort_values(["target","feature"]) if len(corr) else pd.DataFrame(columns=["feature","target","pearson_r","n"]))\
        .to_csv(corr_path, index=False, encoding="utf-8-sig")

    # 2) 分組利差
    notes = []
    for feat in ["company_neg_per_1000", "company_unc_per_1000", "neg_per_1000", "external_neg_per_1000"]:
        for tgt in ["ar3d","ar5d","vol20d"]:
            spread, n_lo, n_hi = quantile_spread(df, feat, tgt, q=0.2)
            notes.append({"feature": feat, "target": tgt, "top_minus_bottom_mean": spread, "n_low": n_lo, "n_high": n_hi})
    notes_df = pd.DataFrame(notes)
    notes_path = DATA / f"{T}_analysis_quintile_spreads.csv"
    notes_df.to_csv(notes_path, index=False, encoding="utf-8-sig")

    # 3) 摘要
    summary_lines = []
    summary_lines.append(f"Ticker: {T}")
    summary_lines.append(f"Rows (merged filings): {len(df)}")
    summary_lines.append("")
    summary_lines.append("Available features: " + (", ".join(present_features) if present_features else "(none)"))
    summary_lines.append("Available targets: " + (", ".join(present_targets) if present_targets else "(none)"))
    summary_lines.append("")
    summary_lines.append("Top correlations (|r| >= 0.2):")
    top = corr.dropna().loc[corr["pearson_r"].abs() >= 0.2] if "pearson_r" in corr.columns else pd.DataFrame()
    if len(top) == 0:
        summary_lines.append("  (none meeting threshold)")
    else:
        for _, r in top.sort_values("pearson_r", ascending=False).iterrows():
            summary_lines.append(f"  r={r['pearson_r']:.3f}  {r['feature']}  vs  {r['target']}  (n={int(r['n'])})")
    summary_lines.append("")
    summary_lines.append("Quintile spreads (top 20% - bottom 20%):")
    for _, r in notes_df.iterrows():
        val = "nan" if pd.isna(r["top_minus_bottom_mean"]) else f"{r['top_minus_bottom_mean']:.4f}"
        summary_lines.append(f"  {r['feature']} → {r['target']}: {val}  (n_low={int(r['n_low'])}, n_high={int(r['n_high'])})")

    summary_path = DATA / f"{T}_analysis_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("Saved:\n  {}\n  {}\n  {}".format(corr_path, notes_path, summary_path))

if __name__ == "__main__":
    main()
