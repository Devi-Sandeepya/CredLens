Set-Location "$PSScriptRoot\ml-service"
if (!(Test-Path ".venv")) { py -m venv .venv }
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
if (Test-Path "..\data\raw\application_train.csv") {
  & ".\.venv\Scripts\python.exe" -m src.pipeline
  & ".\.venv\Scripts\python.exe" -m src.train
} else {
  & ".\.venv\Scripts\python.exe" -m src.demo_seed
}
& ".\.venv\Scripts\python.exe" -m uvicorn src.app:app --reload --port 8001
