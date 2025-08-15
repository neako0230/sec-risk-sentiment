# make_report.py — 產生 reports/report.html
import pathlib, pandas as pd, datetime, html

ROOT = pathlib.Path(__file__).parent.resolve()
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
OUTDIR = ROOT / "reports"
OUTDIR.mkdir(exist_ok=True)
OUT = OUTDIR / "report.html"

def esc(x): return html.escape(str(x))

# 讀 shortlist
sl = None
sl_path = DATA / "signal_shortlist.csv"
if sl_path.exists():
    sl = pd.read_csv(sl_path)

# 收集每家公司顯著結果
rows = []
for p in DATA.glob("*_regressions_sig_p05.csv"):
    ticker = p.name.split("_")[0].upper()
    df = pd.read_csv(p)
    pcol = next((c for c in df.columns if c.lower().startswith("p>|")), "P>|t|")
    # 只留我們關心的 z_ 特徵，做個人類可讀欄位 feature
    df = df[df["term"].astype(str).str.startswith("z_")].copy()
    df["feature"] = df["term"].str[2:]
    df["ticker"] = ticker
    df = df.sort_values([pcol, "target"]).reset_index(drop=True)
    rows.append(df)

if rows:
    sig_all = pd.concat(rows, ignore_index=True)
else:
    sig_all = pd.DataFrame(columns=["ticker","target","feature","Coef.","P>|t|"])

# HTML 樣式
css = """
<style>
:root { --bg:#0b0d10; --card:#12161b; --text:#e8eef4; --muted:#9fb3c8; --accent:#7cc3ff; --chip:#1b2430; --br:14px; }
*{box-sizing:border-box} body{margin:0;padding:24px;background:var(--bg);color:var(--text);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans","Helvetica Neue",Arial}
h1{margin:0 0 8px 0;font-size:28px;font-weight:700}
.sub{margin:0 0 20px 0;color:var(--muted)}
.card{background:var(--card);border:1px solid #1f2833;border-radius:var(--br);padding:16px;margin:12px 0}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{border-bottom:1px solid #1f2833;padding:8px;text-align:left}
th{color:#cfe7ff;font-weight:600}
.badge{display:inline-block;background:var(--chip);border:1px solid #263140;padding:3px 8px;border-radius:999px;font-size:12px;color:#cde3ff}
a{color:#8ad1ff;text-decoration:none;border-bottom:1px dotted #456}
.small{color:var(--muted);font-size:12px}
.kv{display:flex;gap:8px;flex-wrap:wrap}
.kv .badge{background:#0e1420}
</style>
"""

# Shortlist 區塊
sl_html = "<p class='small'>(尚未產生 shortlist)</p>"
if isinstance(sl, pd.DataFrame) and not sl.empty:
    top = sl.head(20).copy()
    def sgn(v): 
        try: return "+" if int(v)>0 else "-"
        except: return "-"
    rows_html = []
    for _, r in top.iterrows():
        rows_html.append(
            f"<tr><td>{esc(r['feature'])}</td><td>{esc(r['target'])}</td>"
            f"<td>{sgn(r.get('sign',0))}</td>"
            f"<td>{int(r.get('n_support',0))}</td>"
            f"<td>{esc(r.get('tickers',''))}</td>"
            f"<td>{int(r.get('n_corr_support',0))}</td>"
            f"<td>{esc(f'{float(r.get('max_abs_r', float('nan'))):.3f}' if pd.notna(r.get('max_abs_r')) else '')}</td>"
            f"<td>{int(r.get('score',0))}</td></tr>"
        )
    sl_html = f"""
    <div class="card">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <h2 style="margin:0">候選訊號（Top 20）</h2>
        <div class="kv">
          <a href="../data/signal_shortlist.csv" class="badge">下載 signal_shortlist.csv</a>
          <a href="../plots/index.html" class="badge">開啟圖庫</a>
        </div>
      </div>
      <table>
        <thead><tr><th>feature</th><th>target</th><th>符號</th><th>支持公司數</th><th>tickers</th><th>相關性支持</th><th>max|r|</th><th>score</th></tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
      </table>
    </div>
    """

# 每家公司 Top-10 顯著
per_ticker_html = []
if not sig_all.empty:
    pcol = next((c for c in sig_all.columns if c.lower().startswith("p>|")), "P>|t|")
    for tkr, g in sig_all.groupby("ticker"):
        g2 = g.sort_values(pcol).head(10)
        rows_html = []
        for _, r in g2.iterrows():
            rows_html.append(
                f"<tr><td>{esc(r['feature'])}</td><td>{esc(r['target'])}</td>"
                f"<td>{float(r['Coef.']):+.4f}</td>"
                f"<td>{float(r[pcol]):.4g}</td></tr>"
            )
        per_ticker_html.append(f"""
        <div class="card">
          <h3 style="margin-top:0">{esc(tkr)} — 顯著結果 Top 10（依 p 值）</h3>
          <table>
            <thead><tr><th>feature</th><th>target</th><th>coef</th><th>p</th></tr></thead>
            <tbody>{''.join(rows_html)}</tbody>
          </table>
        </div>
        """)
else:
    per_ticker_html.append("<p class='small'>(尚無顯著結果。請先跑 regression_test.py 並產生 *_regressions_sig_p05.csv)</p>")

# 組頁
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
html_doc = f"""<!doctype html>
<html lang="zh-Hant">
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SEC 風險段落 × 事件窗 — 報告</title>
{css}
<body>
<h1>SEC 風險段落 × 事件窗 — 報告</h1>
<p class="sub">產生時間：{esc(now)} | 輸出位置：<code>{esc(str(OUT))}</code></p>

{sl_html}

<div class="card">
  <h2 style="margin-top:0">圖庫</h2>
  <p class="small">快速連結：<a href="../plots/index.html">plots/index.html</a></p>
</div>

{''.join(per_ticker_html)}

</body>
</html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(html_doc)

print(f"Saved {OUT}")
