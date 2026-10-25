@echo off
cd /d "%~dp0"
py -3.10 -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
