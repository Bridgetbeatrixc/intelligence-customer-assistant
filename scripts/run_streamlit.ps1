param(
    [int]$Port = 8501
)

Set-Location (Resolve-Path "$PSScriptRoot\..")
python -m streamlit run app.py --server.port $Port --server.headless true
