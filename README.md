---
title: ETA Time Prediction API
emoji: 🛵
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# ETA Time Prediction - Backend (FastAPI)

Endpoints: `GET /health`, `POST /predict`, `GET /docs`.
The model (`models:/model/1` on DagsHub MLflow) is loaded once at startup; the preprocessor comes from `models/preprocessor.joblib`.

## Environment variables

| Name | Where | Required | Notes |
|---|---|---|---|
| `DAGSHUB_USER_TOKEN` | HF Space **Secret** | Yes if the DagsHub repo is private (and recommended anyway) | DagsHub access token (DagsHub -> Settings -> Tokens). Never commit it. |
| `DAGSHUB_USERNAME` | Space variable | No | Defaults to `sanketdixit18` |
| `MLFLOW_TRACKING_URI` | Space variable | No | Defaults to `https://dagshub.com/sanketdixit18/ETA-Time-Prediction.mlflow` |
| `MODEL_VERSION` | Space variable | No | Defaults to `1` (what the original `app.py` used) |
| `PORT` | set by Space | No | Defaults to `7860` |

## Run locally (Windows PowerShell)

```powershell
cd eta-backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:DAGSHUB_USER_TOKEN = "<your token>"   # this shell only
python app.py                               # serves on http://127.0.0.1:7860
```

```powershell
Invoke-RestMethod http://127.0.0.1:7860/health
```

## Sample request

```powershell
$body = @{
  ID="test-1"; Delivery_person_ID="WEBRES00DEL00"; Delivery_person_Age="30"; Delivery_person_Ratings="4.6"
  Restaurant_latitude=19.1136; Restaurant_longitude=72.8697
  Delivery_location_latitude=19.1176; Delivery_location_longitude=72.9047
  Order_Date="14-10-2026"; Time_Orderd="12:30:00"; Time_Order_picked="12:40:00"
  Weatherconditions="conditions Sunny"; Road_traffic_density="Medium"; Vehicle_condition=2
  Type_of_order="Meal"; Type_of_vehicle="motorcycle"; multiple_deliveries="1"; Festival="No"; City="Urban"
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:7860/predict -ContentType "application/json" -Body $body
```

Response: `{"prediction": <minutes as float>}`. Bad input gives 422; model not loaded gives 503 (see `/health` for the reason).

## Deploy to Hugging Face Spaces

1. huggingface.co -> New Space -> SDK **Docker** (blank template). Name it e.g. `eta-backend`.
2. Space -> Settings -> Variables and secrets -> add Secret `DAGSHUB_USER_TOKEN`.
3. Push these files (not the tests are required, but harmless): `app.py`, `Dockerfile`, `requirements.txt`, `run_information.json`, `README.md`, `scripts/`, `models/preprocessor.joblib`, `.gitignore`.
   ```powershell
   cd eta-backend
   git init
   git remote add space https://huggingface.co/spaces/<username>/<space-name>
   git add .
   git commit -m "Deploy ETA backend"
   git push space main   # use a HF write token as the password
   ```
4. Wait for the build, then open `https://<username>-<space-name>.hf.space/health`. Expect `{"status":"healthy",...}`.
5. Test `/predict` with the sample above, pointed at the Space URL.

## Tests

```powershell
pip install pytest httpx
pytest -q                       # uses a stub model + the real preprocessor
$env:DAGSHUB_USER_TOKEN="..."; pytest tests/test_live_model.py   # real model, needs network + token
```
