# make_gallery.py — 產生 plots/index.html，把 plots/*.png 做成圖庫總覽
import pathlib, datetime, pandas as pd, html

ROOT = pathlib.Path(__file__).parent.resolve()
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
OUT = PLOTS / "index.html"

PLOTS.mkdir(exist_ok=True)

# 嘗試載入候選訊號清單（若有）
shortlist = None
sl_path = DATA / "signal_shortlist.csv"
if sl_path.exists():
    try:
        shortlist = pd.read_csv(sl_path)
    except Exception:
        shortlist = None

# 收集圖片
imgs = sorted(PLOTS.glob("*.png"))
cards = []
for p in imgs:
    name = p.name  # e.g., AAPL__external_neg_per_1000__r10d.png
    parts = name[:-4].split("__")
    if len(parts) == 3:
        ticker, feature, target = parts
    else:
        ticker, feature, target = "", name[:-4], ""
    cards.append({
        "path": p.name,
        "ticker": ticker,
        "feature": feature,
        "target": target
    })

# 依 ticker, feature, target 排序
cards.sort(key=lambda r: (r["ticker"], r["feature"], r["target"]))

# HTML 頁面
def esc(s): return html.escape(str(s))

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

head = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>SEC 風險段落 × 事件窗 圖庫</title>
<style>
  :root {{
    --bg:#0b0d10; --card:#12161b; --text:#e8eef4; --muted:#9fb3c8; --accent:#7cc3ff; --chip:#1b2430;
    --br:14px;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; padding:24px; background:var(--bg); color:var(--text);
    font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans","Helvetica Neue",Arial,"Apple Color Emoji","Segoe UI Emoji";
  }}
  h1 {{ margin:0 0 8px 0; font-size:28px; font-weight:700; }}
  .sub {{ margin:0 0 20px 0; color:var(--muted); }}
  .wrap {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap:16px; }}
  .card {{
    background:var(--card); border:1px solid #1f2833; border-radius:var(--br);
    padding:12px; box-shadow: 0 2px 10px rgba(0,0,0,.25);
  }}
  .imgbox {{
    border-radius:10px; overflow:hidden; background:#0a0c0f; border:1px solid #202a36;
    aspect-ratio: 4/3; display:flex; align-items:center; justify-content:center;
  }}
  img {{ width:100%; height:auto; display:block; }}
  .meta {{ margin-top:10px; }}
  .ticker {{ font-weight:700; letter-spacing:.5px; color:var(--accent); }}
  .feat, .tgt {{
    display:inline-block; background:var(--chip); border:1px solid #263140;
    padding:3px 8px; border-radius:999px; font-size:12px; margin-right:6px; color:#cde3ff;
  }}
  a.dl {{
    color:#cde3ff; text-decoration:none; border-bottom:1px dotted #456; padding-bottom:1px;
  }}
  .shortlist {{
    background:var(--card); border:1px solid #1f2833; border-radius:var(--br);
    padding:12px; margin: 12px 0 22px 0;
  }}
  .shortlist table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  .shortlist th, .shortlist td {{ border-bottom:1px solid #1f2833; padding:6px; text-align:left; }}
  .shortlist th {{ color:#cfe7ff; font-weight:600; }}
</style>
</head>
<body>
<h1>SEC 風險段落 × 事件窗 圖庫</h1>
<p class="sub">產生時間：{esc(now)} | 目錄：<code>{esc(str(PLOTS))}</code></p>
"""

sl_html = ""
if isinstance(shortlist, pd.DataFrame) and not shortlist.empty:
    top = shortlist.head(12).copy()
    sl_html = ["<div class='shortlist'><div style='display:flex;align-items:center;justify-content:space-between;'><h3 style='margin:0'>候選訊號（Top 12）</h3><a class='dl' href='../data/signal_shortlist.csv'>下載 CSV</a></div>"]
    sl_html.append("<table><thead><tr><th>feature</th><th>target</th><th>符號</th><th>支持公司數</th><th>tickers</th><th>相關性支持</th><th>max|r|</th><th>score</th></tr></thead><tbody>")
    for _, r in top.iterrows():
        sign = "+" if int(r.get("sign", 0)) > 0 else "-"
        sl_html.append(
            f"<tr><td>{esc(r['feature'])}</td><td>{esc(r['target'])}</td>"
            f"<td>{sign}</td><td>{int(r.get('n_support',0))}</td>"
            f"<td>{esc(r.get('tickers',''))}</td>"
            f"<td>{int(r.get('n_corr_support',0))}</td>"
            f"<td>{esc(f'{float(r.get('max_abs_r', float('nan'))):.3f}' if pd.notna(r.get('max_abs_r')) else '')}</td>"
            f"<td>{int(r.get('score',0))}</td></tr>"
        )
    sl_html.append("</tbody></table></div>")
    sl_html = "\n".join(sl_html)

grid = ["<div class='wrap'>"]
for c in cards:
    grid.append(f"""
    <div class="card">
      <div class="imgbox"><a href="{esc(c['path'])}" target="_blank"><img src="{esc(c['path'])}" alt="plot"/></a></div>
      <div class="meta">
        <div class="ticker">{esc(c['ticker'])}</div>
        <div style="margin-top:6px;">
          <span class="feat">{esc(c['feature'])}</span>
          <span class="tgt">{esc(c['target'])}</span>
        </div>
      </div>
    </div>
    """)
grid.append("</div>")

tail = """
</body>
</html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(head + sl_html + "\n".join(grid) + tail)

print(f"Saved {OUT}")
