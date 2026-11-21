# Agentic AI Research Assistant — Fast Fix

This optimized version fixes the long-loading experience by eliminating the optional additional research pass and using **two local Qwen3 model calls instead of four to six**. Its five LangGraph stages are: query planning (rules), web research (DDGS), AI analysis (Ollama), evidence review (rules), report writing (Ollama).

## Update your existing Windows project
1. Stop your old Streamlit process with **Ctrl+C** in the VS Code terminal.
2. Extract this ZIP to `C:\Users\sathi\agentic-research-assistant`; replace existing project files. **Do not delete `venv`.**
3. In VS Code terminal, launch with ` .\venv\Scripts\python.exe -m streamlit run app.py `, then click Research **once**.

All packages are the same as before. No reinstall is necessary. No PowerShell activation is necessary. If a run fails, the browser shows an error message. The interface displays progress by stage and keeps the most recently generated report.

## Reality check
Your laptop's CPU may still take some time to run Qwen3 4B. The interface now shows its stage instead of a single indefinite spinner, but an unusually slow or blocked Ollama model can still delay completion. You can verify Ollama is responsive with `ollama run qwen3:4b` in a separate terminal.

These are **search snippets**, not fact-checked full articles. Verify citations manually before relying on the content. The app uses free, locally downloaded models and public web search (subject to internet access/rate limits).
