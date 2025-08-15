# excel_export.py — 將主要成果彙整成一份 Excel：reports/summary.xlsx
import pathlib
import pandas as pd
import datetime as dt

ROOT = pathlib.Path(__file__).parent.resolve()
DATA = ROOT / "data"
OUTDIR = ROOT / "reports"
OUTDIR.mkdir(exist_ok=True)
OUT = OUTDIR / "summary.xlsx"


def _escape_formula_like(s):
    """任何以 '=' 開頭的字串，都前置單引號，避免被 Excel 視為公式。"""
    if isinstance(s, str) and s.startswith("="):
        return "'" + s
    return s


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    # 僅處理物件欄位，避免破壞數值格式
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].map(_escape_formula_like)
    return df


def autosize(writer, sheet_name, df, freeze_header=True):
    ws = writer.sheets[sheet_name]
    if freeze_header:
        ws.freeze_panes(1, 0)
    # 依欄名與前 500 列估計欄寬
    for i, col in enumerate(df.columns):
        maxlen = max([len(str(col))] + [len(str(v)) for v in df[col].astype(str).values[:500]])
        ws.set_column(i, i, min(max(10, maxlen + 2), 60))


def write_df(writer, name, df):
    name = name[:31]  # Excel 工作表名上限 31 字
    if df is None or df.empty:
        tmp = pd.DataFrame({"info": [f"No data for {name}"]})
        tmp = _sanitize_df(tmp)
        tmp.to_excel(writer, sheet_name=name, index=False)
        autosize(writer, name, tmp)
    else:
        df = _sanitize_df(df)
        df.to_excel(writer, sheet_name=name, index=False)
        autosize(writer, name, df)


def main():
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[info] building Excel @ {OUT} ({timestamp})")

    # 關閉「字串→公式」自動偵測，避免 Excel 移除公式警告
    writer = pd.ExcelWriter(
        OUT,
        engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_formulas": False}},
    )

    # 1) Shortlist & summary（文字版）
    sl_path = DATA / "signal_shortlist.csv"
    sl = pd.read_csv(sl_path) if sl_path.exists() else pd.DataFrame()
    write_df(writer, "signal_shortlist", sl)

    sum_txt = DATA / "signal_summary.txt"
    if sum_txt.exists():
        lines = [l.rstrip("\n") for l in sum_txt.read_text(encoding="utf-8").splitlines()]
        df_txt = pd.DataFrame({"signal_summary": lines})
    else:
        df_txt = pd.DataFrame({"signal_summary": ["(no signal_summary.txt)"]})
    write_df(writer, "signal_summary", df_txt)

    # 2) 每家公司 p<0.05 顯著結果（*_regressions_sig_p05.csv）
    for p in sorted(DATA.glob("*_regressions_sig_p05.csv")):
        ticker = p.name.split("_")[0].upper()
        try:
            df = pd.read_csv(p)
        except Exception as e:
            df = pd.DataFrame({"error": [str(e)]})
        write_df(writer, f"{ticker}_sig", df)

    # 3) 合併 correlations
    corr_list = []
    for p in sorted(DATA.glob("*_analysis_correlations.csv")):
        ticker = p.name.split("_")[0].upper()
        try:
            dfc = pd.read_csv(p)
            dfc["ticker"] = ticker
            corr_list.append(dfc)
        except Exception:
            pass
    corr_all = pd.concat(corr_list, ignore_index=True) if corr_list else pd.DataFrame()
    write_df(writer, "correlations_all", corr_all)

    # 4) 合併後的 events 與摘要
    ev_path = DATA / "all_events_merged.csv"
    ev_sum_path = DATA / "all_events_summary.csv"
    ev = pd.read_csv(ev_path) if ev_path.exists() else pd.DataFrame()
    ev_sum = pd.read_csv(ev_sum_path) if ev_sum_path.exists() else pd.DataFrame()
    write_df(writer, "events_all", ev)
    write_df(writer, "events_summary", ev_sum)

    # 5) Panel 固定效果回歸彙總（新增）
    panel_path = DATA / "panel_summary.csv"
    panel = pd.read_csv(panel_path) if panel_path.exists() else pd.DataFrame()
    write_df(writer, "panel_summary", panel)

    # 6) 封面
    cover = pd.DataFrame(
        {
            "key": [
                "generated_at",
                "project_dir",
                "data_dir",
                "plots_dir",
                "tickers_detected",
                "files_written",
            ],
            "value": [
                timestamp,
                str(ROOT),
                str(DATA),
                str(ROOT / "plots"),
                ", ".join(
                    sorted({p.name.split("_")[0].upper() for p in DATA.glob("*_regressions_sig_p05.csv")})
                )
                or "(none)",
                "signal_shortlist, signal_summary, *_sig, correlations_all, events_all, events_summary, panel_summary",
            ],
        }
    )
    write_df(writer, "README", cover)

    writer.close()
    print(f"[ok] saved {OUT}")


if __name__ == "__main__":
    main()
