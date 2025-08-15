# panel_regression.py
# 以 data/all_events_merged.csv 進行跨公司 pooled OLS：
#  - 目標：預設 ar20d
#  - 訊號：預設 external_neg_per_1000
#  - 控制：公司固定效果 C(ticker)、年度固定效果 C(year)、是否 10-K（is_10k）
#  - 特徵：依「每家公司」做標準化(z-score)；並對原始特徵 winsorize（預設 1%）
#  - 標準誤：依 ticker 叢集的 cluster-robust（HC1）

import argparse, pathlib, numpy as np, pandas as pd
import statsmodels.formula.api as smf

ROOT = pathlib.Path(__file__).parent.resolve()
DATA = ROOT / "data"
OUTDIR = ROOT / "data"

def winsorize(s: pd.Series, p: float) -> pd.Series:
    p = float(p)
    p = 0.0 if p < 0 else (10.0 if p > 10 else p)  # 安全界限
    if p == 0 or s.isna().all():
        return s
    lo = np.nanpercentile(s.astype(float), p)
    hi = np.nanpercentile(s.astype(float), 100 - p)
    return s.clip(lo, hi)

def zscore_by_ticker(df: pd.DataFrame, col: str) -> pd.Series:
    def _z(g):
        x = g[col].astype(float)
        mu = x.mean()
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return x*0.0
        return (x - mu) / sd
    return df.groupby("ticker", group_keys=False).apply(_z)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", default="external_neg_per_1000", help="自變數欄名")
    ap.add_argument("--target", default="ar20d", help="目標欄名")
    ap.add_argument("--winsor", type=float, default=1.0, help="對 feature 進行 winsorize 百分位（每端 p%%，預設 1）")
    args = ap.parse_args()

    merged = DATA / "all_events_merged.csv"
    if not merged.exists():
        raise FileNotFoundError("找不到 data/all_events_merged.csv，請先執行 combine_events.py")

    df = pd.read_csv(merged)

    need = ["ticker", "form", "filing_date", args.feature, args.target]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"缺少欄位：{missing}")

    # 數值化
    df[args.feature] = pd.to_numeric(df[args.feature], errors="coerce")
    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")

    # 年度（以 filing_date，若缺用 event_date）
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        df["year"] = df["filing_date"].dt.year.fillna(df["event_date"].dt.year)
    else:
        df["year"] = df["filing_date"].dt.year

    # is_10k
    df["form"] = df["form"].astype(str).str.upper()
    df["is_10k"] = (df["form"] == "10-K").astype(int)

    # 清理 NA
    df = df.dropna(subset=[args.feature, args.target, "ticker", "year"])

    # winsorize（先 across 全體，再 by ticker 做 z-score）
    df["_feat_w"] = winsorize(df[args.feature].astype(float), args.winsor)
    df["_z_feat"] = zscore_by_ticker(df.assign(**{args.feature: df["_feat_w"]}), "_feat_w")

    # 模型：target ~ z_feat + is_10k + C(ticker) + C(year)
    formula = f"{args.target} ~ _z_feat + is_10k + C(ticker) + C(year)"

    # 叢集標準誤（按 ticker）
    model = smf.ols(formula, data=df)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df["ticker"]})

    # 輸出精簡摘要
    term_rows = res.summary2().tables[1].reset_index().rename(columns={"index":"term"})
    # 只挑與解釋變數直接相關的幾行
    key = term_rows[term_rows["term"].isin(["Intercept", "_z_feat", "is_10k"])].copy()

    out_txt = OUTDIR / f"panel_{args.feature}__{args.target}.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Panel OLS with FE (ticker, year), clustered by ticker\n")
        f.write(f"feature={args.feature} (winsor={args.winsor}%), target={args.target}\n")
        f.write(f"N={int(res.nobs)}, tickers={', '.join(sorted(df['ticker'].unique()))}\n\n")
        f.write(key.to_string(index=False))
        f.write("\n\nFull terms:\n")
        f.write(term_rows.head(20).to_string(index=False))
        f.write("\n")
    print(f"[ok] saved {out_txt}")

    # 也存成 CSV（完整係數表）
    out_csv = OUTDIR / f"panel_{args.feature}__{args.target}.csv"
    term_rows.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[ok] saved {out_csv}")

if __name__ == "__main__":
    main()
