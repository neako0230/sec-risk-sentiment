@echo off
setlocal enabledelayedexpansion

REM ===== 批次清單：想加減就改這裡 =====
set TICKERS=AAPL MSFT NVDA AMZN GOOGL META TSLA

for %%T in (%TICKERS%) do (
  echo.
  echo ========= Processing %%T =========
  python run_mvp.py --ticker %%T --years 5 --use-finbert
  if errorlevel 1 echo [warn] run_mvp failed for %%T & rem 繼續
  python analyze_events.py --ticker %%T
  if errorlevel 1 echo [warn] analyze_events failed for %%T
  python regression_test.py --ticker %%T
  if errorlevel 1 echo [warn] regression_test failed for %%T
)

echo.
echo ===== Update cross-ticker summaries & gallery =====
python summarize_signals.py
python plot_signals.py
python make_gallery.py

echo.
echo Done. Open plots\index.html for the gallery.
