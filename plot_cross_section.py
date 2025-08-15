# plot_cross_section.py — 跨公司散佈圖（預設 external_neg_per_1000 → ar20d）
import pathlib, argparse
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent.resolve()
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", default="external_neg_per_1000")
    ap.add_argument("--target", default="ar20d")
    ap.add_argument("--out", default=None, help="輸出檔名（可省略，自動命名）")
    args = ap.parse_args()

    merged = DATA / "all_events_merged.csv"
    if not merged.exists():
        raise FileNotFoundError("找不到 data/all_events_merged.csv，請先執行 combine_events.py")

    df = pd.read_csv(merged)
    if args.feature not in df.columns or args.target not in df.columns:
        raise SystemExit(f"欄位不存在：{args.feature} 或 {args.target}")

    # 清理
    sub = df[["ticker", args.feature, args.target]].copy()
    sub[args.feature] = pd.to_numeric(sub[args.feature], errors="coerce")
    sub[args.target]  = pd.to_numeric(sub[args.target], errors="coerce")
    sub = sub.dropna(subset=[args.feature, args.target])
    if len(sub) < 5:
        raise SystemExit(f"資料點太少（{len(sub)}）")

    # 計算整體皮爾森 r
    x = sub[args.feature].values.astype(float)
    y = sub[args.target].values.astype(float)
    R = float(np.corrcoef(x, y)[0,1]) if len(sub) >= 2 else np.nan

    # 散佈：不同 ticker 上不同標記
    tickers = sorted(sub["ticker"].unique())
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"]
    plt.figure(figsize=(9,7), dpi=150)

    for i, tk in enumerate(tickers):
        d = sub[sub["ticker"]==tk]
        plt.scatter(d[args.feature], d[args.target], label=f"{tk} (n={len(d)})",
                    alpha=0.8, marker=markers[i % len(markers)], edgecolor="none")

    # 整體線性回歸線
    try:
        b1, b0 = np.polyfit(x, y, 1)  # y ≈ b1*x + b0
        xs = np.linspace(x.min(), x.max(), 100)
        ys = b1*xs + b0
        plt.plot(xs, ys, linewidth=2, label=f"fit: y={b1:.3f}x+{b0:.3f}")
    except Exception:
        pass

    plt.title(f"Cross-ticker: {args.feature} → {args.target}   (N={len(sub)}, r={R:.3f})")
    plt.xlabel(args.feature)
    plt.ylabel(args.target)
    plt.legend(frameon=False)
    plt.tight_layout()

    out = args.out or (PLOTS / f"ALL__{args.feature}__{args.target}.png")
    if isinstance(out, pathlib.Path):
        out_path = out
    else:
        out_path = pathlib.Path(out)
    plt.savefig(out_path)
    plt.close()
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()
