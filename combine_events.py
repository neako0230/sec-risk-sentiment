# combine_events.py — 將 data/*_item1a_events.csv 合併成一張總表
import pathlib, pandas as pd, numpy as np

ROOT = pathlib.Path(__file__).parent.resolve()
DATA = ROOT / "data"
OUT_CSV = DATA / "all_events_merged.csv"

FEATURES = [
    "neg_per_1000","unc_per_1000",
    "company_neg_per_1000","company_unc_per_1000",
    "external_neg_per_1000","external_unc_per_1000",
    "company_sent_ratio_mean",
    "finbert_neg_mean","finbert_neu_mean","finbert_pos_mean",
]
TARGETS = ["r1d","r3d","r5d","r10d","r20d","ar1d","ar3d","ar5d","ar10d","ar20d","vol5d","vol20d"]

KEEP_BASE = ["ticker","company_title","cik","form","filing_date","event_date","accession"]

def load_one(path: pathlib.Path) -> pd.DataFrame:
    tkr = path.name.split("_")[0].upper()
    df = pd.read_csv(path)
    df["ticker"] = tkr
    # 轉數值
    for c in FEATURES + TARGETS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # 日期欄
    for c in ["filing_date","event_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    return df

def main():
    files = sorted(DATA.glob("*_item1a_events.csv"))
    if not files:
        print("No *_item1a_events.csv found in data/. Run run_mvp.py first.")
        return

    frames = []
    for p in files:
        try:
            frames.append(load_one(p))
        except Exception as e:
            print(f"[warn] skip {p.name}: {e}")

    if not frames:
        print("Nothing loaded."); return

    df = pd.concat(frames, ignore_index=True)

    # 精簡欄位（保留存在的）
    keep = [c for c in KEEP_BASE if c in df.columns] \
           + [c for c in FEATURES if c in df.columns] \
           + [c for c in TARGETS if c in df.columns]
    df = df[keep]

    # 去重：以 (ticker, accession) 優先；若無 accession 用 (ticker, filing_date, form)
    if "accession" in df.columns:
        df = df.sort_values(["ticker","filing_date"]).drop_duplicates(["ticker","accession"], keep="last")
    else:
        df = df.sort_values(["ticker","filing_date","form"]).drop_duplicates(["ticker","filing_date","form"], keep="last")

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[ok] saved {OUT_CSV} rows={len(df)} tickers={', '.join(sorted(df['ticker'].unique()))}")

    # 簡報表：各 ticker 筆數與 target 非空數量
    summary = (
        df.assign(_row=1)
          .groupby("ticker")
          .agg(
              rows=("_row","sum"),
              r5d_nonnull=("r5d", lambda s: int(s.notna().sum())),
              r10d_nonnull=("r10d", lambda s: int(s.notna().sum())),
              r20d_nonnull=("r20d", lambda s: int(s.notna().sum())),
              ar5d_nonnull=("ar5d", lambda s: int(s.notna().sum())),
              ar10d_nonnull=("ar10d", lambda s: int(s.notna().sum())),
              ar20d_nonnull=("ar20d", lambda s: int(s.notna().sum())),
          )
          .reset_index()
          .sort_values("ticker")
    )
    out_sum = DATA / "all_events_summary.csv"
    summary.to_csv(out_sum, index=False, encoding="utf-8-sig")
    print(f"[ok] saved {out_sum}")

if __name__ == "__main__":
    main()
