import os, re, time, pathlib, argparse, io
from datetime import datetime, timedelta
from urllib.parse import urljoin
from typing import Optional, List, Dict, Any

import requests, pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

# =========================
# 基本設定
# =========================
ROOT = pathlib.Path(__file__).parent.resolve()
DATA = ROOT / "data"
RES  = ROOT / "resources" / "lm"
DATA.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

load_dotenv()
UA = os.getenv("SEC_USER_AGENT", "SecRiskResearch/0.1 (email: please_set)")
HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}  # 不要放 Host

# 快取（可選）
try:
    import requests_cache
    requests_cache.install_cache(str(ROOT / "cache_sec"), expire_after=60 * 60 * 24 * 7)
except Exception:
    pass

DATA_BASE = "https://data.sec.gov/"
WWW_BASE  = "https://www.sec.gov/"
SUBMISSIONS = urljoin(DATA_BASE, "submissions/")
SEC_FILES_JSON = "https://www.sec.gov/files/company_tickers.json"

# LM 鏡像來源（官方連結常變）
LM_SOURCES = [
    "https://raw.githubusercontent.com/Microkiller/Dic/master/LoughranMcDonald_MasterDictionary_2018.csv",
    "https://raw.githubusercontent.com/ark4innovation/datascience/master/ai-for-trading/5-nlp-on-financial-statements/loughran_mcdonald_master_dic_2016.csv",
]

def sleep_polite():
    time.sleep(0.2)  # SEC 建議 ≤10 req/s

# ---- 通用取檔（自動在 data.sec.gov / www.sec.gov 間切換） ----
def fetch_text_multi(path: str) -> Optional[str]:
    for base in (DATA_BASE, WWW_BASE):
        url = urljoin(base, path)
        sleep_polite()
        r = requests.get(url, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            return r.text
    return None

def fetch_json_multi(path: str) -> Optional[Dict[str, Any]]:
    for base in (DATA_BASE, WWW_BASE):
        url = urljoin(base, path)
        sleep_polite()
        r = requests.get(url, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                return None
    return None

# =========================
# 1) 取 CIK & 公司名稱（ticker -> CIK, title）
# =========================
def get_cik_and_title_from_ticker(ticker: str):
    sleep_polite()
    r = requests.get(SEC_FILES_JSON, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    for _, row in data.items():
        if row["ticker"].upper() == ticker.upper():
            cik = str(row["cik_str"]).zfill(10)
            title = row.get("title", "")
            return cik, title
    raise ValueError("Ticker {} not found in SEC mapping.".format(ticker))

# =========================
# 2) 列出 filings（submissions/CIK##########.json）
# =========================
def list_filings(cik: str, forms=("10-K", "10-Q"), years_back=5):
    url = urljoin(SUBMISSIONS, "CIK{}.json".format(cik))
    sleep_polite()
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    j = r.json()
    recent = j.get("filings", {}).get("recent", {})
    forms_list = recent.get("form", [])
    files = []
    cutoff = datetime.today().date() - timedelta(days=365 * years_back + 30)
    for i, form in enumerate(forms_list):
        if form not in forms:
            continue
        filing_date = recent["filingDate"][i]
        acc_dash = recent["accessionNumber"][i]              # 例 0000320193-25-000073
        acc_no   = acc_dash.replace("-", "")                 # 例 000032019325000073
        primary  = recent["primaryDocument"][i]              # 可能是 aapl-yyyymmdd.htm 或 accession
        try:
            fdate = datetime.strptime(filing_date, "%Y-%m-%d").date()
        except Exception:
            continue
        if fdate >= cutoff:
            files.append(
                {
                    "form": form,
                    "filing_date": str(fdate),
                    "accession_nodash": acc_no,
                    "accession_dash":  acc_dash,
                    "primary_doc": primary,
                    "cik": cik,
                }
            )
    return files

# =========================
# 3) 下載 primary HTML（多層備援）
# =========================
def split_acc_dash(nodash: str) -> str:
    a = re.sub(r"\D", "", nodash)
    if len(a) >= 18:
        return "{}-{}-{}".format(a[:10], a[10:12], a[12:])
    return nodash  # 保底

def pick_best(html_names: List[str]) -> str:
    def rank(fn: str) -> int:
        s = 0
        low = fn.lower()
        if "10-k" in low or "10k" in low: s += 6
        if "10-q" in low or "10q" in low: s += 6
        if "form10" in low: s += 4
        if "primary" in low: s += 3
        if "document" in low: s += 2
        if "ixbrl" in low: s += 1
        if "index" in low: s -= 10
        return s
    html_names = list(dict.fromkeys(html_names))
    html_names.sort(key=rank, reverse=True)
    return html_names[0]

def pick_best_html_from_submission_txt(txt: str) -> Optional[str]:
    cand = re.findall(r'(?im)^\s*<FILENAME>\s*([^\s<>]+\.(?:htm|html))\s*$', txt)
    if not cand:
        cand = re.findall(r'(?i)([A-Za-z0-9_\-./]+?\.(?:htm|html))', txt)
    return pick_best(cand) if cand else None

def download_primary_html(meta: dict) -> str:
    """嘗試順序：
    1) primary_doc 原樣
    2) primary_doc 加 .htm / .html 猜測
    3) 目錄 index.json（列檔）
    4) 目錄 index.htm / index.html（HTML 列表）
    5) {accession_dash}-index.htm（申報詳情頁）
    6) submission {acc}.txt（從 <FILENAME> 找 HTML；最後保底用其文字）
    """
    cik_int = str(int(meta["cik"]))
    acc_no   = meta["accession_nodash"]
    acc_dash = meta.get("accession_dash") or split_acc_dash(acc_no)
    pdoc = (meta.get("primary_doc") or "").strip()

    # 1) primary_doc 原樣
    if pdoc:
        path = "Archives/edgar/data/{}/{}/{}".format(cik_int, acc_no, pdoc)
        txt = fetch_text_multi(path)
        if txt:
            print("[info] fetched primary_doc: {}".format(pdoc))
            return txt
        # 2) 若沒有副檔名，猜測 .htm / .html
        base, ext = os.path.splitext(pdoc)
        if ext.lower() not in (".htm", ".html"):
            for guess_ext in (".htm", ".html"):
                gpath = "Archives/edgar/data/{}/{}/{}{}".format(cik_int, acc_no, pdoc, guess_ext)
                txt = fetch_text_multi(gpath)
                if txt:
                    print("[info] fetched guessed primary: {}{}".format(pdoc, guess_ext))
                    return txt

    # 3) 目錄 index.json
    j = fetch_json_multi("Archives/edgar/data/{}/{}/index.json".format(cik_int, acc_no))
    if j:
        items = j.get("directory", {}).get("item", [])
        htmls = [it["name"] for it in items if it.get("name","").lower().endswith((".htm", ".html"))]
        if htmls:
            best = pick_best(htmls)
            txt = fetch_text_multi("Archives/edgar/data/{}/{}/{}".format(cik_int, acc_no, best))
            if txt:
                print("[info] fetched via index.json: {}".format(best))
                return txt

    # 4) 目錄 index.htm / index.html
    for idx_name in ("index.htm", "index.html"):
        html_listing = fetch_text_multi("Archives/edgar/data/{}/{}/{}".format(cik_int, acc_no, idx_name))
        if html_listing:
            links = re.findall(r'href="([^"]+)"', html_listing, flags=re.I)
            htmls = [ln for ln in links if ln.lower().endswith((".htm", ".html"))]
            if htmls:
                best = pick_best(htmls)
                if not str(best).lower().startswith("http"):
                    best_path = "Archives/edgar/data/{}/{}/{}".format(cik_int, acc_no, best)
                else:
                    best_path = best.replace("https://www.sec.gov/","").replace("https://data.sec.gov/","")
                txt = fetch_text_multi(best_path)
                if txt:
                    print("[info] fetched via {}: {}".format(idx_name, best))
                    return txt

    # 5) 申報詳情頁 {acc_dash}-index.htm
    detail = fetch_text_multi("Archives/edgar/data/{}/{}/{}-index.htm".format(cik_int, acc_no, acc_dash))
    if detail:
        links = re.findall(r'href="([^"]+)"', detail, flags=re.I)
        htmls = [ln for ln in links if ln.lower().endswith((".htm", ".html"))]
        if htmls:
            best = pick_best(htmls)
            if not str(best).lower().startswith("http"):
                best_path = "Archives/edgar/data/{}/{}/{}".format(cik_int, acc_no, best)
            else:
                best_path = best.replace("https://www.sec.gov/","").replace("https://data.sec.gov/","")
            txt = fetch_text_multi(best_path)
            if txt:
                print("[info] fetched via filing detail page: {}".format(best))
                return txt

    # 6) submission .txt
    sub_txt = fetch_text_multi("Archives/edgar/data/{}/{}/{}.txt".format(cik_int, acc_no, acc_no))
    if sub_txt:
        best = pick_best_html_from_submission_txt(sub_txt)
        if best:
            txt = fetch_text_multi("Archives/edgar/data/{}/{}/{}".format(cik_int, acc_no, best))
            if txt:
                print("[info] fetched via submission txt: {}".format(best))
                return txt
        print("[warn] using submission .txt text as fallback HTML source")
        return sub_txt

    raise FileNotFoundError(
        "Cannot locate filing HTML. Tried primary_doc, guessed extensions, index.json, index.htm/html, filing-detail -index.htm, and submission .txt."
    )

# =========================
# 4) 從 HTML 擷取 Item 1A 文字
# =========================
ITEM_RE = re.compile(r"item\s*1a\.?\s*risk\s*factors", re.I)
NEXT_ITEM_RE = re.compile(r"(^|\n)\s*item\s*1b\.|(^|\n)\s*item\s*2\.", re.I)

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.select("table, style, script, footer"):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def extract_item_1a_text(html: str) -> str:
    text = html_to_text(html)
    low = text.lower()
    m = ITEM_RE.search(low)
    if not m:
        return ""
    start = m.start()
    m2 = NEXT_ITEM_RE.search(low, pos=start + 10)
    end = m2.start() if m2 else len(text)
    return text[start:end].strip()

def split_paragraphs(text: str):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [p for p in paras if len(p.split()) >= 5]

# =========================
# 5) LM 詞典與計數
# =========================
def ensure_lm_dicts():
    neg_path = RES / "lm_negative_words.txt"
    unc_path = RES / "lm_uncertainty_words.txt"
    if neg_path.exists() and unc_path.exists():
        return neg_path, unc_path

    last_err = None
    for url in LM_SOURCES:
        try:
            print("Downloading LM dictionary from: {}".format(url))
            sleep_polite()
            r = requests.get(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}, timeout=60)
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content), encoding="latin-1")
            cols = [c.lower() for c in df.columns]
            df.columns = cols

            # 找詞欄
            if "word" in df.columns:
                word_col = "word"
            else:
                cand = [c for c in df.columns if "word" in c]
                if not cand:
                    raise RuntimeError("word column not found")
                word_col = cand[0]

            # 將 negative / uncertainty 轉為布林
            def to_bool_mask(series: pd.Series):
                num = pd.to_numeric(series, errors="coerce")
                mask = num.fillna(0) > 0
                if mask.any():
                    return mask
                s = series.astype(str).str.strip().str.upper()
                return s.isin(["1", "TRUE", "Y", "YES"])

            if "negative" in df.columns:
                neg_mask = to_bool_mask(df["negative"])
                neg = df.loc[neg_mask, word_col].astype(str).str.lower().str.strip().unique()
            else:
                neg = []

            if "uncertainty" in df.columns:
                unc_mask = to_bool_mask(df["uncertainty"])
                unc = df.loc[unc_mask, word_col].astype(str).str.lower().str.strip().unique()
            else:
                unc = []

            pd.Series(neg).to_csv(neg_path, index=False, header=False, encoding="utf-8")
            pd.Series(unc).to_csv(unc_path, index=False, header=False, encoding="utf-8")
            return neg_path, unc_path
        except Exception as e:
            last_err = e
            print("  -> failed, try next source. ({})".format(e))
            continue
    raise RuntimeError("Failed to download LM dictionary from all sources. Last error: {}".format(last_err))

def load_wordset(path: pathlib.Path):
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set([line.strip().lower() for line in f if line.strip()])

def lm_counts(text: str, neg_set: set, unc_set: set):
    tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    N = len(tokens) if len(tokens) > 0 else 1
    neg = sum(1 for t in tokens if t in neg_set)
    unc = sum(1 for t in tokens if t in unc_set)
    return {
        "neg_cnt": neg,
        "unc_cnt": unc,
        "n_tokens": N,
        "neg_per_1000": 1000.0 * neg / N,
        "unc_per_1000": 1000.0 * unc / N,
    }

# ---- 句子級 LM 計數（用於主體對齊）----
def lm_counts_basic(text: str, neg_set: set, unc_set: set):
    tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    return (len(tokens), sum(1 for t in tokens if t in neg_set), sum(1 for t in tokens if t in unc_set))

def split_sentences(text: str) -> List[str]:
    pieces = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    sents = []
    for p in pieces:
        for s in p.split("\n"):
            s = s.strip()
            if len(s.split()) >= 3:
                sents.append(s)
    return sents

COMPANY_SUFFIXES = {"inc", "inc.", "incorporated", "corp", "co", "company", "corporation", "ltd", "limited", "plc", "sa", "ag", "nv", "oyj", "ab", "as", "kk", "gmbh", "spa", "s.p.a", "srl", "oy", "bv", "se"}

def make_company_aliases(company_title: str, ticker: str) -> List[str]:
    aliases = {"we", "our", "us", "the company"}
    name = (company_title or "").lower()
    name = re.sub(r"[\,\.]", " ", name)
    words = [w for w in re.findall(r"[a-z]+", name) if w and w not in COMPANY_SUFFIXES]
    if words:
        aliases.add(words[0])
        aliases.add(" ".join(words[:2]) if len(words)>=2 else words[0])
    if ticker:
        aliases.add(ticker.lower())
    return list(aliases)

EXTERNAL_CUES = re.compile(
    r"\b(supplier|suppliers|customer|customers|client|clients|vendor|vendors|partner|partners|"
    r"competitor|competitors|market|industry|econom(y|ies)|macroeconomic|inflation|recession|"
    r"interest\s+rates?|currency|fx|exchange\s+rate|government|regulation|regulatory|law|laws|"
    r"policy|tariff|tariffs|geopolitical|war|conflict|pandemic|covid|outbreak|china|europe|asia|"
    r"third[-\s]?party|cloud\s+provider|data\s+center|carrier|network|lessor|lessee|tenant|landlord|"
    r"bank|lender|credit\s+market|labor\s+market|weather|climate|hurricane|earthquake|wildfire)\b",
    re.I
)

def is_company_sentence(s: str, aliases: List[str]) -> bool:
    low = s.lower()
    for a in aliases:
        if re.search(r"\b{}\b".format(re.escape(a)), low):
            return True
    if re.search(r"\bwe\s+(expect|face|may|could|will|intend|plan|believe|estimate|anticipate)\b", low):
        return True
    return False

def is_external_sentence(s: str) -> bool:
    return EXTERNAL_CUES.search(s) is not None

def company_external_lm_features(paragraph: str, neg_set: set, unc_set: set, aliases: List[str]) -> Dict[str, float]:
    sents = split_sentences(paragraph)
    comp_tokens = comp_neg = comp_unc = 0
    ext_tokens = ext_neg = ext_unc = 0
    comp_sents = ext_sents = 0

    for s in sents:
        n_tok, n_neg, n_unc = lm_counts_basic(s, neg_set, unc_set)
        if is_company_sentence(s, aliases):
            comp_tokens += n_tok; comp_neg += n_neg; comp_unc += n_unc; comp_sents += 1
        elif is_external_sentence(s):
            ext_tokens += n_tok; ext_neg += n_neg; ext_unc += n_unc; ext_sents += 1
        else:
            ext_tokens += n_tok; ext_neg += n_neg; ext_unc += n_unc

    def per1000(cnt, toks): return (1000.0 * cnt / max(toks, 1))
    total_sents = max(len(sents), 1)
    return {
        "company_tokens": comp_tokens,
        "company_neg_cnt": comp_neg,
        "company_unc_cnt": comp_unc,
        "company_neg_per_1000": per1000(comp_neg, comp_tokens),
        "company_unc_per_1000": per1000(comp_unc, comp_tokens),
        "external_tokens": ext_tokens,
        "external_neg_cnt": ext_neg,
        "external_unc_cnt": ext_unc,
        "external_neg_per_1000": per1000(ext_neg, ext_tokens),
        "external_unc_per_1000": per1000(ext_unc, ext_tokens),
        "company_sent_ratio": comp_sents / float(total_sents),
        "external_sent_ratio": ext_sents / float(total_sents),
    }

# =========================
# 6) FinBERT（可選）
# =========================
FINBERT_MODEL_NAME = "ProsusAI/finbert"

def load_finbert():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    tok = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device)
    mdl.eval()
    return tok, mdl, device

def finbert_scores_batch(paras: List[str], tok, mdl, device, batch_size=8, max_length=256) -> List[Dict[str, float]]:
    import torch
    import torch.nn.functional as F
    out = []
    if not paras:
        return out
    for i in range(0, len(paras), batch_size):
        batch = paras[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = mdl(**enc).logits  # [B,3] labels: 0 neg, 1 neu, 2 pos
            prob = F.softmax(logits, dim=1).cpu().numpy()
        for p in prob:
            out.append({"finbert_neg": float(p[0]), "finbert_neu": float(p[1]), "finbert_pos": float(p[2])})
    return out

# =========================
# 7) 價格下載與事件窗計算（yfinance + Stooq 後援；索引扁平化）
# =========================
def fetch_prices(symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """先用 yfinance；失敗改用 Stooq。無論來源為何，都將索引扁平為單一 DatetimeIndex。"""
    def _finalize(df: pd.DataFrame, src_tag: str) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None

        # 取收盤價欄
        close_col = None
        for cand in ["Close", "close", "Adj Close", "adj close"]:
            if cand in df.columns:
                close_col = cand
                break
        if close_col is None and len(df.columns) > 0:
            close_col = df.columns[0]

        df = df.rename(columns={close_col: "close"})[["close"]].copy()

        # —— 扁平化索引：強制變成單一 DatetimeIndex ——
        if isinstance(df.index, pd.MultiIndex):
            names = [n or "" for n in df.index.names]
            if "Date" in names:
                lvl = names.index("Date")
                idx = df.index.get_level_values(lvl)
            else:
                idx = df.index.get_level_values(-1)
            df.index = pd.to_datetime(idx).tz_localize(None)
        else:
            df.index = pd.to_datetime(df.index).tz_localize(None)

        # 篩選日期範圍、排序
        df = df.sort_index()
        df = df.loc[(df.index >= pd.Timestamp(start_date.date())) & (df.index <= pd.Timestamp(end_date.date()))]
        if df.empty:
            return None

        # 報酬
        df["ret"]  = df["close"].pct_change().fillna(0.0).astype(float)
        df["lret"] = np.log1p(df["ret"]).astype(float)

        print("[prices] {} via {}: rows={} range={}..{}".format(
            symbol, src_tag, len(df), df.index.min().date(), df.index.max().date()
        ))
        return df

    # ---- 1) yfinance ----
    try:
        import yfinance as yf
        raw = yf.download(
            symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            interval="1d",
            threads=False
        )
        df = _finalize(raw, "yfinance")
        if df is not None:
            return df
    except Exception as e:
        print("[warn] yfinance fetch failed for {}: {}".format(symbol, e))

    # ---- 2) Stooq 後援 ----
    try:
        stq = (symbol.lower() + ".us") if not symbol.lower().endswith(".us") else symbol.lower()
        url = "https://stooq.com/q/d/l/?s={}&i=d".format(stq)
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        import io as _io
        csv_txt = r.text
        if "Date,Open,High,Low,Close,Volume" not in csv_txt:
            print("[warn] stooq returned unexpected for {}".format(stq))
            return None
        raw = pd.read_csv(_io.StringIO(csv_txt))
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
        raw = raw.dropna(subset=["Date"]).set_index("Date").sort_index()
        raw = raw.rename(columns={"Close": "Close"})
        df = _finalize(raw, "stooq")
        return df
    except Exception as e:
        print("[warn] stooq fetch failed for {}: {}".format(symbol, e))
        return None

def next_trading_day(idx: pd.DatetimeIndex, date_obj: datetime.date) -> Optional[pd.Timestamp]:
    ts = pd.Timestamp(date_obj)
    later = idx[idx > ts]
    if len(later) == 0:
        return None
    return later[0]

def cum_logret(series: pd.Series, n: int) -> Optional[float]:
    if series is None or len(series) == 0:
        return None
    seg = series.iloc[:n]
    if len(seg) < n:
        return None
    return float(seg.sum())

def compute_event_windows(file_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if file_df is None or file_df.empty:
        return pd.DataFrame()

    fdates = pd.to_datetime(file_df["filing_date"])
    start = (fdates.min() - pd.Timedelta(days=40)).to_pydatetime()
    end   = (fdates.max() + pd.Timedelta(days=40)).to_pydatetime()

    px = fetch_prices(ticker, start, end)
    mkt = fetch_prices("SPY", start, end)

    rows = []
    for _, r in file_df.iterrows():
        fd = pd.to_datetime(r["filing_date"]).date()
        event = None
        if px is not None:
            event = next_trading_day(px.index, fd)
        if event is None:
            rows.append({
                "accession": r["accession"],
                "event_date": None,
                "r1d": None, "r3d": None, "r5d": None, "r10d": None, "r20d": None,
                "ar1d": None, "ar3d": None, "ar5d": None, "ar10d": None, "ar20d": None,
                "vol5d": None, "vol20d": None
            })
            continue

        win = px.loc[event:]
        lret = win["lret"] if "lret" in win.columns else None

        if mkt is not None and "lret" in mkt.columns:
            mwin = mkt.reindex(win.index)
            mlret = mwin["lret"]
        else:
            mlret = None

        def ar_n(n):
            rr = cum_logret(lret, n)
            if rr is None:
                return None, None
            mr = cum_logret(mlret, n) if mlret is not None else None
            ar = None if mr is None else float(rr - mr)
            return float(rr), ar

        r1, ar1 = ar_n(1)
        r3, ar3 = ar_n(3)
        r5, ar5 = ar_n(5)
        r10, ar10 = ar_n(10)
        r20, ar20 = ar_n(20)

        vol5 = float(win["lret"].iloc[:5].std()) if lret is not None and len(win) >= 5 else None
        vol20 = float(win["lret"].iloc[:20].std()) if lret is not None and len(win) >= 20 else None

        rows.append({
            "accession": r["accession"],
            "event_date": event.date().isoformat(),
            "r1d": r1, "r3d": r3, "r5d": r5, "r10d": r10, "r20d": r20,
            "ar1d": ar1, "ar3d": ar3, "ar5d": ar5, "ar10d": ar10, "ar20d": ar20,
            "vol5d": vol5, "vol20d": vol20
        })

    return pd.DataFrame(rows)

# =========================
# 8) 主流程
# =========================
def run(ticker, years_back=5, include_forms=("10-K", "10-Q"), use_finbert=False):
    cik, company_title = get_cik_and_title_from_ticker(ticker)
    filings = list_filings(cik, forms=include_forms, years_back=years_back)
    if not filings:
        print("No filings found in range.")
        return

    neg_path, unc_path = ensure_lm_dicts()
    neg_set, unc_set = load_wordset(neg_path), load_wordset(unc_path)

    tok = mdl = device = None
    if use_finbert:
        print("[info] loading FinBERT...")
        tok, mdl, device = load_finbert()

    aliases = make_company_aliases(company_title, ticker)

    para_rows, file_rows = [], []

    for fmeta in tqdm(filings, desc="{} filings".format(ticker)):
        html = download_primary_html(fmeta)
        item1a = extract_item_1a_text(html)
        if not item1a:
            continue
        paras = split_paragraphs(item1a)

        fb_scores = []
        if use_finbert and paras:
            fb_scores = finbert_scores_batch(paras, tok, mdl, device, batch_size=8, max_length=256)

        # 段落級
        for idx, p in enumerate(paras, start=1):
            m = lm_counts(p, neg_set, unc_set)
            ce = company_external_lm_features(p, neg_set, unc_set, aliases)
            row = {
                "ticker": ticker,
                "company_title": company_title,
                "cik": fmeta["cik"],
                "form": fmeta["form"],
                "filing_date": fmeta["filing_date"],
                "accession": fmeta["accession_nodash"],
                "para_idx": idx,
                "para_text": p,
            }
            row.update(m)
            row.update(ce)
            if use_finbert and idx-1 < len(fb_scores):
                row.update(fb_scores[idx-1])
            para_rows.append(row)

        # 文件級（彙總）
        if paras:
            doc_text = "\n".join(paras)
            mdoc = lm_counts(doc_text, neg_set, unc_set)
            mdoc["n_paras"] = len(paras)

            comp_tok = sum(r["company_tokens"] for r in para_rows if r["accession"] == fmeta["accession_nodash"])
            comp_neg = sum(r["company_neg_cnt"] for r in para_rows if r["accession"] == fmeta["accession_nodash"])
            comp_unc = sum(r["company_unc_cnt"] for r in para_rows if r["accession"] == fmeta["accession_nodash"])
            ext_tok  = sum(r["external_tokens"] for r in para_rows if r["accession"] == fmeta["accession_nodash"])
            ext_neg  = sum(r["external_neg_cnt"] for r in para_rows if r["accession"] == fmeta["accession_nodash"])
            ext_unc  = sum(r["external_unc_cnt"] for r in para_rows if r["accession"] == fmeta["accession_nodash"])

            def per1000(cnt, toks): return (1000.0 * cnt / max(toks, 1))
            mdoc.update({
                "company_tokens": comp_tok,
                "company_neg_cnt": comp_neg,
                "company_unc_cnt": comp_unc,
                "company_neg_per_1000": per1000(comp_neg, comp_tok),
                "company_unc_per_1000": per1000(comp_unc, comp_tok),
                "external_tokens": ext_tok,
                "external_neg_cnt": ext_neg,
                "external_unc_cnt": ext_unc,
                "external_neg_per_1000": per1000(ext_neg, ext_tok),
                "external_unc_per_1000": per1000(ext_unc, ext_tok),
                "company_sent_ratio_mean": float(pd.Series(
                    [r["company_sent_ratio"] for r in para_rows if r["accession"] == fmeta["accession_nodash"]]
                ).mean() if len(paras) else 0.0)
            })

            if use_finbert and fb_scores:
                dfb = pd.DataFrame(fb_scores)
                mdoc["finbert_neg_mean"] = float(dfb["finbert_neg"].mean())
                mdoc["finbert_neu_mean"] = float(dfb["finbert_neu"].mean())
                mdoc["finbert_pos_mean"] = float(dfb["finbert_pos"].mean())

            file_row = {
                "ticker": ticker,
                "company_title": company_title,
                "cik": fmeta["cik"],
                "form": fmeta["form"],
                "filing_date": fmeta["filing_date"],
                "accession": fmeta["accession_nodash"],
            }
            file_row.update(mdoc)
            file_rows.append(file_row)

    para_df = pd.DataFrame(para_rows)
    file_df = pd.DataFrame(file_rows)
    out1 = DATA / "{}_item1a_paragraphs.csv".format(ticker)
    out2 = DATA / "{}_item1a_filelevel.csv".format(ticker)
    if len(para_df):
        para_df.to_csv(out1, index=False, encoding="utf-8-sig")
    if len(file_df):
        file_df.to_csv(out2, index=False, encoding="utf-8-sig")

    # ---- 事件窗計算並輸出第三個 CSV ----
    try:
        events_df = compute_event_windows(file_df, ticker)
        if events_df is not None and not events_df.empty:
            merged = file_df.merge(events_df, on="accession", how="left")
            out3 = DATA / "{}_item1a_events.csv".format(ticker)
            merged.to_csv(out3, index=False, encoding="utf-8-sig")
            print("Saved: {}".format(out3))
        else:
            print("No event metrics generated (price data missing?).")
    except Exception as e:
        print("Event window computation failed: {}".format(e))

    print("\nSaved: {}".format(out1 if len(para_df) else "no paragraphs"))
    print("Saved: {}".format(out2 if len(file_df) else "no file-level rows"))

# =========================
# 入口
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, required=True, help="e.g., AAPL, MSFT")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--forms", type=str, default="10-K,10-Q")
    ap.add_argument("--use-finbert", action="store_true", help="Enable FinBERT paragraph probabilities")
    args = ap.parse_args()
    forms = tuple([s.strip().upper() for s in args.forms.split(",") if s.strip()])
    run(args.ticker.upper(), years_back=args.years, include_forms=forms, use_finbert=args.use_finbert)
