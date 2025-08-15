# panel_collect.py — 收斂 data/panel_*.csv 的固定效果回歸結果為一張總表
import pathlib, re
import pandas as pd
from math import isnan

ROOT = pathlib.Path(__file__).parent.resolve()
DATA = ROOT / "data"
OUT = DATA / "panel_summary.csv"

def pick_pcol(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).lower().startswith("p>|"):
            return c
    # 退而求其次
    for c in df.columns:
        if str(c).strip().lower() in ("p", "pvalue", "p>|z|", "p>|t|"):
            return c
    raise KeyError("找不到 p 值欄位")

def load_one(path: pathlib.Path) -> pd.DataFrame:
    # 解析 feature/target
    m = re.match(r"panel_(.+)__(.+)\.csv$", path.name)
    feature, target = (m.group(1), m.group(2)) if m else ("?", "?")
    df = pd.read_csv(path)
    pcol = pick_pcol(df)
    # 只保留重點兩行
    keep = df[df["term"].isin(["_z_feat", "is_10k"])].copy()
    if keep.empty:
        return pd.DataFrame()
    keep["feature"] = feature
    keep["target"] = target
    keep.rename(columns={pcol: "pval"}, inplace=True)
    # 符號與顯著星號
    def stars(p):
        try:
            if p < 0.001: return "***"
            if p < 0.01:  return "**"
            if p < 0.05:  return "*"
        except Exception:
            pass
        return ""
    keep["sign"] = keep["Coef."].apply(lambda x: "+" if float(x) > 0 else "-")
    keep["sig"] = keep["pval"].apply(stars)
    # 排序
    keep = keep[["feature","target","term","Coef.","Std.Err.","pval","sign","sig"]]
    return keep

def main():
    rows = []
    for p in sorted(DATA.glob("panel_*.csv")):
        try:
            part = load_one(p)
            if not part.empty:
                rows.append(part)
        except Exception as e:
            print(f"[warn] skip {p.name}: {e}")
    if not rows:
        print("沒有找到 panel_*.csv，請先執行 panel_regression.py")
        return
    out = pd.concat(rows, ignore_index=True)
    out.sort_values(["feature","target","term"], inplace=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"[ok] saved {OUT}")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
