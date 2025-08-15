\# SEC 風險段落 × 段落級情緒 × 事件窗 MVP



以 SEC 10-K/10-Q 的 \*\*Item 1A. Risk Factors\*\* 做段落級情緒（LM/FinBERT）與主體對齊（company vs. external），再與公告後事件窗報酬/超額報酬/波動做關聯與可預測性檢驗。



\## 1) 環境

\- Windows + CMD，虛擬環境放在 `.venv/`

\- 需求清單：`requirements.txt`



```bash

python -m venv .venv

.\\.venv\\Scripts\\activate

pip install -r requirements.txt



