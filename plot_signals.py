# plot_signals.py
# 讀取 data/signal_shortlist.csv 的前幾個候選訊號，
# 針對每個 <feature,target> × 每個涵蓋的 ticker 產生散佈圖（含線性趨勢線與 r）

import pathlib, pandas as pd, numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent.resolve()
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

SHORTLIST = DATA / "signal_shortlist.csv"

def load_events(ticker: str) -> pd.DataFrame:
    f = DATA / f"{ticker}_item1a_events.csv"
    if not f.exists():
        raise FileNotFoundError(f"缺少 {f}，請先跑 run_mvp.py --ticker {ticker}")
    df = pd.read_csv(f)
    # 轉成數值
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def plot_one(df: pd.DataFrame, ticker: str, feature: str, target: str):
    if feature not in df.columns or target not in df.columns:
        print(f"[skip] {ticker} 缺 {feature} 或 {target}")
        return
    sub = df[[feature, target, "filing_date", "form"]].copy()
    sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
    sub[target]  = pd.to_numeric(sub[target], errors="coerce")
    sub = sub.dropna(subset=[feature, target])
    if len(sub) < 3:
        print(f"[skip] {ticker} {feature}->{target} 資料點過少 ({len(sub)})")
        return

    x = sub[feature].values.astype(float)
    y = sub[target].values.astype(float)
    r = float(np.corrcoef(x, y)[0,1]) if len(sub) >= 2 else np.nan

    # 線性趨勢線
    try:
        b1, b0 = np.polyfit(x, y, 1)  # y ≈ b1*x + b0
        xs = np.linspace(x.min(), x.max(), 100)
        ys = b1*xs + b0
        fit_ok = True
    except Exception:
        fit_ok = False

    plt.figure(figsize=(8,6), dpi=150)
    plt.scatter(x, y, alpha=0.75, edgecolor="none")
    if fit_ok:
        plt.plot(xs, ys, linewidth=2)
    ttl = f"{ticker}: {feature} → {target}   (n={len(sub)}, r={r:.3f})"
    plt.title(ttl)
    plt.xlabel(feature)
    plt.ylabel(target)
    # 小字註記：年份與表單類型
    try:
        forms = sub["form"].astype(str).str.upper().value_counts().to_dict()
        years = pd.to_datetime(sub["filing_date"], errors="coerce").dt.year.value_counts().sort_index().to_dict()
        txt = "forms: " + ", ".join(f"{k}:{v}" for k,v in forms.items())
        txt += "\nyears: " + ", ".join(f"{k}:{v}" for k,v in years.items())
        plt.gcf().text(0.01, 0.01, txt, fontsize=8, va="bottom")
    except Exception:
        pass

    out = PLOTS / f"{ticker}__{feature}__{target}.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"[saved] {out}")

def main():
    if not SHORTLIST.exists():
        raise FileNotFoundError("找不到 data/signal_shortlist.csv，請先跑 summarize_signals.py")

    sl = pd.read_csv(SHORTLIST)
    if sl.empty:
        print("signal_shortlist.csv 是空的"); return

    # 取前 8 個訊號（可自行調整）
    top = sl.head(8).copy()

    for _, row in top.iterrows():
        feature = str(row["feature"])
        target  = str(row["target"])
        tickers = [t.strip().upper() for t in str(row["tickers"]).split(",") if t.strip()]
        for tk in tickers:
            df = load_events(tk)
            plot_one(df, tk, feature, target)

    print(f"\n全部完成。圖片輸出在: {PLOTS}")

if __name__ == "__main__":
    main()
